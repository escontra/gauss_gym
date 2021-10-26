import abc
from typing import List, Dict
import dataclasses
import random
import numpy as np
import torch
import json
import trimesh
import viser.transforms as vtf
import pathlib

from gauss_gym import utils
from gauss_gym.utils import path, hf_utils, math_utils, warp_utils


@dataclasses.dataclass
class Mesh:
  # Scene metadata.
  scene_name: str
  filename: str
  filepath: str
  splatpath: str

  # Mesh parameters.
  vertices: np.ndarray
  triangles: np.ndarray

  # Camera parameters. Camera translations are defined in the world frame.
  cam_trans: np.ndarray
  cam_offset: np.ndarray
  cam_quat_xyzw: np.ndarray

  # Rotation matrix from IG coordinate system to original coordinate system.
  ig_to_orig_rot: np.ndarray


def compute_orientations_from_positions(
  positions: np.ndarray, yaw_only: bool = False
) -> np.ndarray:
  """Compute camera orientations from a trajectory of positions.

  This creates a camera coordinate frame where:
  - X axis points in the direction of motion (tangent to the path)
  - Y axis is perpendicular to X in the horizontal plane
  - Z axis points upward (or is computed to complete the right-handed frame)

  Parameters
  ----------
  positions : np.ndarray
      Array of 3D positions with shape (N, 3)
  yaw_only : bool, optional
      If True, only uses XY motion for orientation (keeps Z axis vertical)
      If False, uses full 3D motion direction, by default False

  Returns
  -------
  np.ndarray
      Camera quaternions in xyzw format with shape (N, 4), in OpenCV convention
  """
  # Compute direction vectors from position differences
  if yaw_only:
    delta_xy = positions[1:, :2] - positions[:-1, :2]
    delta_xyz = np.concatenate([delta_xy, np.zeros_like(delta_xy[:, :1])], axis=-1)
  else:
    delta_xyz = positions[1:] - positions[:-1]

  # Repeat first direction to match input length
  delta_xyz = np.concatenate([delta_xyz[:1], delta_xyz], axis=0)
  delta_xyz = delta_xyz / np.linalg.norm(delta_xyz, axis=-1, keepdims=True)
  x_component = delta_xyz

  # Define world Z axis
  world_z = np.zeros_like(x_component)
  world_z[:, 2] = 1

  # Compute Y axis perpendicular to X in horizontal plane
  y_component = np.cross(world_z, x_component)
  y_component = y_component / np.linalg.norm(y_component, axis=-1, keepdims=True)

  # Compute Z axis to complete right-handed coordinate frame
  z_component = np.cross(x_component, y_component)
  z_component = z_component / np.linalg.norm(z_component, axis=-1, keepdims=True)

  # Stack into rotation matrices and convert to quaternions
  cam_rot = np.stack([x_component, y_component, z_component], axis=2)
  cam_quat_xyzw = math_utils.matrix_to_quaternion(torch.tensor(cam_rot))
  cam_quat_xyzw = math_utils.robot_to_opencv(cam_quat_xyzw).cpu().numpy()

  return cam_quat_xyzw


def get_filtered_poses(
  mesh: Mesh,
  grid_diameter: float = 0.5,
  grid_points: int = 20,
  min_ceiling_height: float = 2.0,
) -> Mesh:
  """Filter out invalid camera poses by ray casting to verify mesh geometry below and above.

  For each camera pose, creates a grid of points in a circle around the pose and
  casts rays downward in the -z direction and upward in the +z direction.
  - Downward rays: If any rays miss the mesh (return infinity), that pose is invalid.
  - Upward rays: If any rays hit at distance <= min_ceiling_height, that pose is invalid.
    Infinity (no ceiling) is acceptable.

  Parameters
  ----------
  mesh : Mesh
      Input mesh with camera poses to filter
  grid_diameter : float, optional
      Diameter of the sampling grid around each pose in meters, by default 0.5
  grid_points : int, optional
      Number of points to sample in the grid (will be squared), by default 5
  min_ceiling_height : float, optional
      Minimum acceptable ceiling height in meters, by default 2.0

  Returns
  -------
  Mesh
      New mesh with only valid camera poses
  """
  # Convert mesh to warp format for ray casting
  device = 'cuda:0'
  wp_mesh = warp_utils.convert_to_wp_mesh(mesh.vertices, mesh.triangles, device)

  # Create grid of offsets in xy plane (circle pattern)
  radius = grid_diameter / 2
  grid_1d = np.linspace(-radius, radius, grid_points)
  xx, yy = np.meshgrid(grid_1d, grid_1d)

  # Filter to only points within the circle
  dist = np.sqrt(xx**2 + yy**2)
  circle_mask = dist <= radius
  offsets_xy = np.stack([xx[circle_mask], yy[circle_mask]], axis=-1)

  num_grid_points = offsets_xy.shape[0]
  num_poses = mesh.cam_trans.shape[0]

  # Create ray start positions: each pose position + grid offsets
  # Shape: (num_poses, num_grid_points, 3)
  ray_starts = mesh.cam_trans[:, None, :] + np.concatenate(
    [
      offsets_xy[None, :, :].repeat(num_poses, axis=0),
      np.zeros((num_poses, num_grid_points, 1)),
    ],
    axis=-1,
  )

  # Ray directions: all pointing down (-z)
  ray_directions = np.zeros_like(ray_starts)
  ray_directions[:, :, 2] = -1.0

  # Flatten for ray casting (downward)
  ray_starts_flat = ray_starts.reshape(-1, 3)
  ray_directions_flat = ray_directions.reshape(-1, 3)

  # Convert to torch tensors
  ray_starts_torch = torch.tensor(ray_starts_flat, dtype=torch.float32, device=device)
  ray_directions_torch = torch.tensor(
    ray_directions_flat, dtype=torch.float32, device=device
  )

  # Perform downward ray casting
  ray_hits_down = warp_utils.ray_cast(ray_starts_torch, ray_directions_torch, wp_mesh)

  # Reshape back to (num_poses, num_grid_points, 3)
  ray_hits_down = ray_hits_down.cpu().numpy().reshape(num_poses, num_grid_points, 3)

  # Check for invalid downward hits (infinity or nan)
  invalid_ground_hits = np.isinf(ray_hits_down) | np.isnan(ray_hits_down)
  # A pose has valid ground if ALL downward rays hit (no invalid hits in any dimension)
  valid_ground = ~np.any(invalid_ground_hits, axis=(1, 2))

  # Now cast rays upward for ceiling check
  ray_directions_up = np.zeros_like(ray_starts)
  ray_directions_up[:, :, 2] = 1.0  # Pointing up (+z)

  ray_directions_up_flat = ray_directions_up.reshape(-1, 3)
  ray_directions_up_torch = torch.tensor(
    ray_directions_up_flat, dtype=torch.float32, device=device
  )

  # Perform upward ray casting
  ray_hits_up = warp_utils.ray_cast(ray_starts_torch, ray_directions_up_torch, wp_mesh)

  # Reshape back to (num_poses, num_grid_points, 3)
  ray_hits_up = ray_hits_up.cpu().numpy().reshape(num_poses, num_grid_points, 3)

  # Compute distance to ceiling for each ray
  # Distance is the norm of (hit_point - ray_start)
  ceiling_distances = np.linalg.norm(ray_hits_up - ray_starts, axis=-1)

  # Ceiling is too low if any ray hits at distance <= min_ceiling_height
  # Infinity (no ceiling) is acceptable
  ceiling_too_low = np.any(
    np.isfinite(ceiling_distances) & (ceiling_distances <= min_ceiling_height), axis=1
  )
  valid_ceiling = ~ceiling_too_low

  # Combine both checks
  valid_poses = valid_ground & valid_ceiling

  utils.print(
    f'Filtered poses: {valid_poses.sum()}/{num_poses} valid ({100 * valid_poses.sum() / num_poses:.1f}%) '
    f'[ground: {valid_ground.sum()}, ceiling: {valid_ceiling.sum()}]',
    color='cyan',
  )

  # Return new mesh with filtered poses
  return Mesh(
    scene_name=mesh.scene_name,
    filename=mesh.filename,
    filepath=mesh.filepath,
    splatpath=mesh.splatpath,
    vertices=mesh.vertices,
    triangles=mesh.triangles,
    cam_trans=mesh.cam_trans[valid_poses],
    cam_offset=mesh.cam_offset,
    cam_quat_xyzw=mesh.cam_quat_xyzw[valid_poses],
    ig_to_orig_rot=mesh.ig_to_orig_rot,
  )


class DataIngest(abc.ABC):
  @abc.abstractmethod
  def find_valid_scene_directories(self):
    """Looks in the scene path and finds all valid scene directories."""
    pass

  @abc.abstractmethod
  def download_scenes(self, max_num_scenes: int):
    """Download max number of scenes from scene path.

    Parameters
    ----------
    max_num_scenes : int
        Maximum number of scenes to download. Use -1 to download all scenes.
    """
    pass

  @abc.abstractmethod
  def load_meshes(self) -> List[Mesh]:
    """Extract meshes and return them as a list of RawMesh objects."""
    pass


class GaussGymData(DataIngest):
  def __init__(self, repo_id: str = 'escontra/gauss_gym_data', scene: str = ''):
    """Initialize GaussGymData with HuggingFace repo or local path.

    Parameters
    ----------
    repo_id : str, optional
        Either a HuggingFace repo ID (e.g., 'escontra/gauss_gym_data') or
        a local path prefixed with 'local:' (e.g., 'local:/path/to/data')
    scene : str, optional
        Specific scene name to load from the dataset. If provided, only this
        scene will be discovered and downloaded.
        Example: 'cute_bridge', 'grace_cathedral', 'home_night'
        If None, all scenes will be searched.
    """
    self.is_local = repo_id.startswith('local:')
    if self.is_local:
      self.local_path = path.LocalPath(repo_id[6:])  # Remove 'local:' prefix
      self.repo_id = None
      self.repo_type = None
    else:
      self.repo_id = repo_id
      self.repo_type = 'dataset'
      self.local_path = None

    self.scene = None if scene == '' else scene
    self._valid_scenes = {}
    self._downloaded_scenes = {}

  def find_valid_scene_directories(self):
    """Looks in the HF repo tree or local directory and finds all valid scene directories.

    If a scene was specified in __init__, only that scene is searched.
    """
    self._valid_scenes = {}

    if self.is_local:
      # Walk local filesystem - check root and all subdirectories
      base_path = pathlib.Path(str(self.local_path))
      directories_to_check = [base_path] + list(base_path.rglob('*'))

      for root_path in directories_to_check:
        if not root_path.is_dir():
          continue

        # Get relative path from local_path
        try:
          rel_path = root_path.relative_to(self.local_path)
          root = str(rel_path) if str(rel_path) != '.' else ''
        except ValueError:
          continue

        # If a specific scene was requested, filter by it
        if self.scene is not None:
          if root != self.scene:
            continue

        # Check if this directory is a valid scene
        splatfacto_path = root_path / 'splatfacto'
        meshes_path = root_path / 'meshes'
        valid_scene = splatfacto_path.exists() and meshes_path.exists()

        if valid_scene:
          scene_name = root if root else 'root'
          self._valid_scenes[scene_name] = root
    else:
      # Original HF logic
      for root, dirs, _ in hf_utils.walk(self.repo_id, repo_type=self.repo_type):
        # If a specific scene was requested, filter by it
        if self.scene is not None:
          if root != self.scene:
            continue

        # Check if this directory is a valid scene
        valid_scene = 'splatfacto' in dirs and 'meshes' in dirs
        if valid_scene:
          scene_name = root if root else 'root'
          self._valid_scenes[scene_name] = root

    scene_msg = f' matching {self.scene}' if self.scene else ''
    source = 'local directory' if self.is_local else 'HF repo'
    utils.print(
      f'Found {len(self._valid_scenes)} valid scene(s){scene_msg} in {source}',
      color='green',
    )
    return self._valid_scenes

  def download_scenes(self, max_num_scenes: int):
    """Download max number of scenes from HF repo or set local paths.

    Parameters
    ----------
    max_num_scenes : int
        Maximum number of scenes to download. Use -1 to download all scenes.
    """
    if not self._valid_scenes:
      self.find_valid_scene_directories()

    # Limit to max_num_scenes (-1 means download all)
    all_scenes = list(self._valid_scenes.items())
    scenes_to_download = (
      all_scenes
      if max_num_scenes == -1
      else random.sample(all_scenes, min(max_num_scenes, len(all_scenes)))
    )

    if self.is_local:
      # For local paths, just set the paths without downloading
      for scene_name, scene_path in scenes_to_download:
        utils.print(f'Using local scene: {scene_name}', color='cyan')

        full_local_path = self.local_path
        if scene_path:
          full_local_path = full_local_path / scene_path

        self._downloaded_scenes[scene_name] = full_local_path
    else:
      # Original HF download logic
      for scene_name, scene_path in scenes_to_download:
        utils.print(f'Downloading scene: {scene_name}', color='cyan')

        # Download this specific scene directory
        allow_patterns = f'{scene_path}/**' if scene_path else None

        local_path = hf_utils.snapshot_download(
          repo_id=self.repo_id,
          repo_type=self.repo_type,
          allow_patterns=allow_patterns,
        )

        # Store the local path for this scene
        full_local_path = path.LocalPath(local_path)
        if scene_path:
          full_local_path = full_local_path / scene_path

        self._downloaded_scenes[scene_name] = full_local_path

    action = 'Set' if self.is_local else 'Downloaded'
    utils.print(f'{action} {len(self._downloaded_scenes)} scenes', color='green')

  def load_meshes(self, cams_yaw_only: bool = False) -> List[Mesh]:
    """Extract meshes and return them as a list of RawMesh objects."""
    if not self._downloaded_scenes:
      raise RuntimeError('No scenes downloaded. Call download_scenes() first.')

    meshes = []

    for scene_name, scene_path in self._downloaded_scenes.items():
      utils.print(f'Loading meshes from scene: {scene_name}', color='cyan')
      filenames = [
        f.name for f in (scene_path / 'meshes').glob('*') if f.suffix == '.npz'
      ]
      splat_path = scene_path / 'splatfacto'
      for filename in filenames:
        filepath = scene_path / 'meshes' / filename
        with filepath.open('rb') as f:
          mesh_dict = np.load(f)
          cam_offset = np.array(mesh_dict['offset'])[0].astype(np.float32)
          vertices = np.array(mesh_dict['vertices']).astype(np.float32)
          triangles = np.array(mesh_dict['triangles']).astype(np.uint32)
          curr_cam_trans = np.array(mesh_dict['cam_trans']).astype(np.float32)
          curr_cam_trans = math_utils.smooth_path(
            curr_cam_trans, smoothing_factor=10, resample_num_points=30
          )
          from_ig_rotation = np.array(mesh_dict['from_ig_rotation']).astype(np.float32)

        # Compute orientations from camera translations
        cam_quat_xyzw = compute_orientations_from_positions(
          curr_cam_trans, yaw_only=cams_yaw_only
        )

        mesh = Mesh(
          scene_name=scene_name,
          filename=filename,
          filepath=filepath,
          splatpath=splat_path,
          vertices=vertices,
          triangles=triangles,
          cam_trans=curr_cam_trans,
          cam_offset=cam_offset,
          cam_quat_xyzw=cam_quat_xyzw,
          ig_to_orig_rot=from_ig_rotation,
        )
        mesh = get_filtered_poses(mesh)
        meshes.append(mesh)

    utils.print(f'Loaded {len(meshes)} meshes total', color='green')
    return meshes


class GrandTourData(DataIngest):
  def __init__(self, mission: str = ''):
    """Initialize GrandTourData with HuggingFace repo.

    Parameters
    ----------
    mission : str, optional
        Specific mission name to load scenes from. If provided, only scenes
        from this mission will be discovered and downloaded.
        Example: '2024-10-01-11-47-44'
        If None, all missions will be searched.
    """
    self.repo_id = 'leggedrobotics/grand_tour_dataset'
    self.repo_type = 'dataset'
    self.mission = None if mission == '' else mission
    self._valid_scenes = {}
    self._downloaded_scenes = {}

  def find_valid_scene_directories(self):
    """Looks in the HF repo tree and finds all valid scene directories.

    Valid scenes contain:
    - mesh_filled_decimated.ply file
    - transforms.json file
    - splatfacto directory

    Scenes follow the pattern: <mission>/splats/<chunk_name>
    Missions without a 'splats' directory are automatically skipped.
    If a mission was specified in __init__, only that mission is searched.
    """
    self._valid_scenes = {}

    for root, dirs, files in hf_utils.walk(self.repo_id, repo_type=self.repo_type):
      # Check if we're in a chunk directory (inside a splats directory)
      # This automatically skips missions without splats directories
      if root and '/splats/' in root:
        # If a specific mission was requested, filter by it
        if self.mission is not None:
          if not root.startswith(f'{self.mission}/'):
            continue

        # Check if this directory is a valid scene
        has_mesh = 'mesh_filled_decimated.ply' in files
        has_transforms = 'transforms.json' in files
        has_splatfacto = 'splatfacto' in dirs

        if has_mesh and has_transforms and has_splatfacto:
          scene_name = root if root else 'root'
          self._valid_scenes[scene_name] = root

    mission_msg = f' in mission {self.mission}' if self.mission else ''
    utils.print(
      f'Found {len(self._valid_scenes)} valid scenes{mission_msg} in HF repo',
      color='green',
    )
    return self._valid_scenes

  def download_scenes(self, max_num_scenes: int):
    """Download max number of scenes from HF repo.

    Parameters
    ----------
    max_num_scenes : int
        Maximum number of scenes to download. Use -1 to download all scenes.
    """
    if not self._valid_scenes:
      self.find_valid_scene_directories()

    # Limit to max_num_scenes (-1 means download all)
    all_scenes = list(self._valid_scenes.items())
    scenes_to_download = (
      all_scenes
      if max_num_scenes == -1
      else random.sample(all_scenes, min(max_num_scenes, len(all_scenes)))
    )

    for scene_name, scene_path in scenes_to_download:
      utils.print(f'Downloading scene: {scene_name}', color='cyan')

      # Download this specific scene directory
      allow_patterns = f'{scene_path}/**' if scene_path else None

      local_path = hf_utils.snapshot_download(
        repo_id=self.repo_id,
        repo_type=self.repo_type,
        allow_patterns=allow_patterns,
      )

      # Store the local path for this scene
      full_local_path = path.LocalPath(local_path)
      if scene_path:
        full_local_path = full_local_path / scene_path

      self._downloaded_scenes[scene_name] = full_local_path

    utils.print(f'Downloaded {len(self._downloaded_scenes)} scenes', color='green')

  def load_meshes(self, cams_yaw_only: bool = False) -> List[Mesh]:
    """Extract meshes and return them as a list of Mesh objects.

    This follows the mesh.ply loading logic from gaussian_terrain.py.
    """
    if not self._downloaded_scenes:
      raise RuntimeError('No scenes downloaded. Call download_scenes() first.')

    meshes = []

    for scene_name, scene_path in self._downloaded_scenes.items():
      utils.print(f'Loading mesh from scene: {scene_name}', color='cyan')

      filepath = scene_path / 'mesh_filled_decimated.ply'
      splat_path = scene_path / 'splatfacto'

      # Load the mesh using trimesh
      scene = trimesh.load(str(filepath))
      if isinstance(scene, trimesh.Scene):
        mesh = trimesh.util.concatenate(
          [
            mesh
            for mesh in scene.geometry.values()
            if isinstance(mesh, trimesh.Trimesh)
          ]
        )
      else:
        mesh = scene

      # Center the mesh
      mean_vertex = np.mean(np.array(mesh.vertices), axis=0, keepdims=True)
      mesh.vertices -= mean_vertex

      # Define rotation to IsaacGym coordinate system
      to_ig_euler_xyz = np.array([np.pi, 0.0, 0.0])

      # Get camera poses from transforms.json
      transforms_path = scene_path / 'transforms.json'
      if not transforms_path.exists():
        utils.print(f'transforms.json not found: {transforms_path}', color='yellow')
        continue

      with open(transforms_path, 'r') as f:
        transforms = json.load(f)

      curr_cam_trans = []
      curr_cam_quat_xyzw = []
      for frame in transforms['frames']:
        file_path = path.LocalPath(frame['file_path'])
        if 'front' in file_path.stem:
          # Only use the front camera poses.
          curr_transform = vtf.SE3.from_matrix(np.array(frame['transform_matrix']))
          curr_cam_trans.append(curr_transform.translation())
          curr_cam_quat_xyzw.append(curr_transform.rotation().as_quaternion_xyzw())
          # curr_cam_quat_xyzw.append(curr_transform.rotation().as_quaternion())
          # rot_mat = np.array(frame['transform_matrix'])[:3, :3]
          # curr_cam_quat_xyzw.append(
          #   math_utils.matrix_to_quaternion(torch.tensor(rot_mat)).numpy()
          # )

      utils.print(f'num frames: {len(curr_cam_trans)}', color='cyan')
      curr_cam_trans = np.array(curr_cam_trans)
      curr_cam_trans -= mean_vertex
      curr_cam_quat_xyzw = np.array(curr_cam_quat_xyzw)

      # Apply rotation to align with IsaacGym coordinate system
      # rotation = vtf.SO3.from_rpy_radians(
      #   to_ig_euler_xyz[0], to_ig_euler_xyz[1], to_ig_euler_xyz[2]
      # )
      rotation = (
        vtf.SO3.from_x_radians(to_ig_euler_xyz[0]).inverse()
        @ vtf.SO3.from_y_radians(to_ig_euler_xyz[1]).inverse()
        @ vtf.SO3.from_z_radians(to_ig_euler_xyz[2]).inverse()
      )

      # curr_cam_trans = np.dot(curr_cam_trans, rotation.as_matrix())
      # mesh.vertices = np.dot(mesh.vertices, rotation.as_matrix())
      curr_cam_trans = np.dot(curr_cam_trans, rotation.as_matrix().T)
      mesh.vertices = np.dot(mesh.vertices, rotation.as_matrix().T)
      curr_cam_quat_xyzw = (
        rotation @ vtf.SO3.from_quaternion_xyzw(curr_cam_quat_xyzw)
      ).as_quaternion_xyzw()
      curr_cam_quat_xyzw = (
        math_utils.opengl_to_opencv(torch.tensor(curr_cam_quat_xyzw)).cpu().numpy()
      )

      # Extract mesh data
      vertices = np.array(mesh.vertices).astype(np.float32)
      triangles = np.array(mesh.faces).astype(np.uint32)
      cam_offset = mean_vertex[0]
      from_ig_rotation = np.array(rotation.inverse().as_matrix())

      mesh = Mesh(
        scene_name=scene_name,
        filename='mesh_filled_decimated.ply',
        filepath=filepath,
        splatpath=splat_path,
        vertices=vertices,
        triangles=triangles,
        cam_trans=curr_cam_trans,
        cam_offset=cam_offset,
        cam_quat_xyzw=curr_cam_quat_xyzw,
        ig_to_orig_rot=from_ig_rotation,
      )

      # Filter out invalid poses using ray casting
      mesh = get_filtered_poses(mesh)

      meshes.append(mesh)

    utils.print(f'Loaded {len(meshes)} meshes total', color='green')
    return meshes


class MapAnythingData(DataIngest):
  def __init__(self, scene_name: str = None):
    """Initialize MapAnythingData with HuggingFace repo.

    This loader is designed for the veo_scenes dataset from Map Anything.

    Parameters
    ----------
    scene_name : str, optional
        Specific scene name to load from the dataset. If provided, only this
        scene will be discovered and downloaded.
        Example: 'waterfall', 'hiking_trail', etc.
        If None, all scenes will be searched.
    """
    self.repo_id = 'escontra/veo_scenes'
    self.repo_type = 'dataset'
    self.scene_name = scene_name
    self._valid_scenes = {}
    self._downloaded_scenes = {}

  def find_valid_scene_directories(self):
    """Looks in the HF repo tree and finds all valid scene directories.

    Valid scenes contain:
    - dense_mesh.glb file
    - transforms.json file
    - splatfacto directory

    If a scene_name was specified in __init__, only that scene is searched.
    """
    self._valid_scenes = {}

    for root, dirs, files in hf_utils.walk(self.repo_id, repo_type=self.repo_type):
      # If a specific scene was requested, filter by it
      if self.scene_name is not None:
        if root != self.scene_name:
          continue

      # Check if this directory is a valid scene
      has_mesh = 'dense_mesh.glb' in files
      has_transforms = 'transforms.json' in files
      has_splatfacto = 'splatfacto' in dirs

      if has_mesh and has_transforms and has_splatfacto:
        scene_name = root if root else 'root'
        self._valid_scenes[scene_name] = root

    scene_msg = f' matching {self.scene_name}' if self.scene_name else ''
    utils.print(
      f'Found {len(self._valid_scenes)} valid scene(s){scene_msg} in HF repo',
      color='green',
    )
    return self._valid_scenes

  def download_scenes(self, max_num_scenes: int):
    """Download max number of scenes from HF repo.

    Parameters
    ----------
    max_num_scenes : int
        Maximum number of scenes to download. Use -1 to download all scenes.
    """
    if not self._valid_scenes:
      self.find_valid_scene_directories()

    # Limit to max_num_scenes (-1 means download all)
    scenes_to_download = (
      list(self._valid_scenes.items())
      if max_num_scenes == -1
      else list(self._valid_scenes.items())[:max_num_scenes]
    )

    for scene_name, scene_path in scenes_to_download:
      utils.print(f'Downloading scene: {scene_name}', color='cyan')

      # Download this specific scene directory
      allow_patterns = f'{scene_path}/**' if scene_path else None

      local_path = hf_utils.snapshot_download(
        repo_id=self.repo_id,
        repo_type=self.repo_type,
        allow_patterns=allow_patterns,
      )

      # Store the local path for this scene
      full_local_path = path.LocalPath(local_path)
      if scene_path:
        full_local_path = full_local_path / scene_path

      self._downloaded_scenes[scene_name] = full_local_path

    utils.print(f'Downloaded {len(self._downloaded_scenes)} scenes', color='green')

  def load_meshes(self, cams_yaw_only: bool = False) -> List[Mesh]:
    """Extract meshes and return them as a list of Mesh objects.

    This follows the map_anything.glb loading logic from gaussian_terrain.py.
    """
    if not self._downloaded_scenes:
      raise RuntimeError('No scenes downloaded. Call download_scenes() first.')

    meshes = []

    for scene_name, scene_path in self._downloaded_scenes.items():
      utils.print(f'Loading mesh from scene: {scene_name}', color='cyan')

      filepath = scene_path / 'dense_mesh.glb'
      splat_path = scene_path / 'splatfacto'

      # Load the mesh using trimesh
      scene = trimesh.load(str(filepath))
      if isinstance(scene, trimesh.Scene):
        mesh = trimesh.util.concatenate(
          [
            mesh
            for mesh in scene.geometry.values()
            if isinstance(mesh, trimesh.Trimesh)
          ]
        )
      else:
        mesh = scene

      # Simplify mesh
      mesh = mesh.simplify_quadric_decimation(face_count=10000, aggression=0)
      mean_vertex = np.mean(np.array(mesh.vertices), axis=0, keepdims=True)
      mesh.vertices -= mean_vertex

      # Define rotation to IsaacGym coordinate system (specific to map_anything)
      to_ig_euler_xyz = np.array([-np.pi / 2, 0.0, np.pi])

      # Get camera poses from transforms.json
      transforms_path = scene_path / 'transforms.json'
      if not transforms_path.exists():
        utils.print(f'transforms.json not found: {transforms_path}', color='yellow')
        continue

      with open(transforms_path, 'r') as f:
        transforms = json.load(f)

      curr_cam_trans = []
      for frame in transforms['frames']:
        curr_cam_trans.append(np.array(frame['transform_matrix'])[:3, 3])

      # Skip first 8 frames (specific to map_anything processing)
      curr_cam_trans = curr_cam_trans[8:]
      utils.print(f'num frames: {len(curr_cam_trans)}', color='cyan')
      curr_cam_trans = np.array(curr_cam_trans)
      curr_cam_trans -= mean_vertex

      # Apply rotation to align with IsaacGym coordinate system
      rotation = (
        vtf.SO3.from_x_radians(to_ig_euler_xyz[0]).inverse()
        @ vtf.SO3.from_y_radians(to_ig_euler_xyz[1]).inverse()
        @ vtf.SO3.from_z_radians(to_ig_euler_xyz[2]).inverse()
      )

      curr_cam_trans = np.dot(curr_cam_trans, rotation.as_matrix().T)

      # Apply path smoothing (specific parameters for map_anything)
      curr_cam_trans = math_utils.smooth_path(
        curr_cam_trans, smoothing_factor=3, resample_num_points=30
      )

      mesh.vertices = np.dot(mesh.vertices, rotation.as_matrix().T)

      # Extract mesh data
      vertices = np.array(mesh.vertices).astype(np.float32)
      triangles = np.array(mesh.faces).astype(np.uint32)
      cam_offset = mean_vertex[0]
      from_ig_rotation = np.array(rotation.inverse().as_matrix())

      # Compute orientations from camera translations
      cam_quat_xyzw = compute_orientations_from_positions(
        curr_cam_trans, yaw_only=cams_yaw_only
      )

      mesh = Mesh(
        scene_name=scene_name,
        filename='dense_mesh.glb',
        filepath=filepath,
        splatpath=splat_path,
        vertices=vertices,
        triangles=triangles,
        cam_trans=curr_cam_trans,
        cam_offset=cam_offset,
        cam_quat_xyzw=cam_quat_xyzw,
        ig_to_orig_rot=from_ig_rotation,
      )

      # Filter out invalid poses using ray casting
      mesh = get_filtered_poses(mesh)

      meshes.append(mesh)

    utils.print(f'Loaded {len(meshes)} meshes total', color='green')
    return meshes


class ARKitData(DataIngest):
  def __init__(self, split: str = ''):
    """Initialize ARKitData with HuggingFace repo.

    Parameters
    ----------
    split : str, optional
        Specific split to load scenes from ('training' or 'validation').
        If provided, only scenes from this split will be discovered and downloaded.
        If None, all splits will be searched.
    """
    self.repo_id = 'escontra/gauss_gym_arkit'
    self.repo_type = 'dataset'
    self.split = None if split == '' else split
    self._valid_scenes = {}
    self._downloaded_scenes = {}

  def find_valid_scene_directories(self):
    """Looks in the HF repo tree and finds all valid scene directories.

    Valid scenes contain:
    - lowres_wide.traj file
    - *_3dod_mesh_filled_decimated.ply file
    - splatfacto directory

    Scenes follow the pattern: <split>/<video_id>
    If a split was specified in __init__, only that split is searched.
    """
    self._valid_scenes = {}

    for root, dirs, files in hf_utils.walk(self.repo_id, repo_type=self.repo_type):
      # Check if we're in a video directory (inside a split directory)
      if root and '/' in root:
        # If a specific split was requested, filter by it
        if self.split is not None:
          if not root.startswith(f'{self.split}/'):
            continue

        # Check if this directory is a valid scene
        has_traj = 'lowres_wide.traj' in files
        has_mesh = any(f.endswith('_3dod_mesh_filled_decimated.ply') for f in files)
        has_splatfacto = 'splatfacto' in dirs

        if has_traj and has_mesh and has_splatfacto:
          scene_name = root if root else 'root'
          self._valid_scenes[scene_name] = root

    split_msg = f' in split {self.split}' if self.split else ''
    utils.print(
      f'Found {len(self._valid_scenes)} valid scenes{split_msg} in HF repo',
      color='green',
    )
    return self._valid_scenes

  def download_scenes(self, max_num_scenes: int):
    """Download max number of scenes from HF repo.

    Parameters
    ----------
    max_num_scenes : int
        Maximum number of scenes to download. Use -1 to download all scenes.
    """
    if not self._valid_scenes:
      self.find_valid_scene_directories()

    # Limit to max_num_scenes (-1 means download all)
    all_scenes = list(self._valid_scenes.items())
    scenes_to_download = (
      all_scenes
      if max_num_scenes == -1
      else random.sample(all_scenes, min(max_num_scenes, len(all_scenes)))
    )

    for scene_name, scene_path in scenes_to_download:
      utils.print(f'Downloading scene: {scene_name}', color='cyan')

      # Download this specific scene directory
      allow_patterns = f'{scene_path}/**' if scene_path else None

      local_path = hf_utils.snapshot_download(
        repo_id=self.repo_id,
        repo_type=self.repo_type,
        allow_patterns=allow_patterns,
      )

      # Store the local path for this scene
      full_local_path = path.LocalPath(local_path)
      if scene_path:
        full_local_path = full_local_path / scene_path

      self._downloaded_scenes[scene_name] = full_local_path

    utils.print(f'Downloaded {len(self._downloaded_scenes)} scenes', color='green')

  def _parse_traj_file(self, traj_path: path.LocalPath) -> Dict[str, np.ndarray]:
    """Parse .traj file and return poses as 4x4 transformation matrices.

    The .traj file format (space-delimited):
    - Column 1: timestamp
    - Columns 2-4: rotation (axis-angle representation in radians)
    - Columns 5-7: translation (in meters)

    Parameters
    ----------
    traj_path : path.LocalPath
        Path to the .traj file

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping timestamp strings to 4x4 transformation matrices
    """
    poses = {}

    with open(traj_path, 'r') as f:
      for line in f:
        parts = line.strip().split()
        if len(parts) < 7:
          continue

        timestamp = parts[0]
        axis_angle = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        translation = np.array([float(parts[4]), float(parts[5]), float(parts[6])])

        # Convert axis-angle to rotation matrix using viser transforms
        rotation = vtf.SO3.exp(axis_angle).as_matrix()

        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, -1] = translation

        poses[timestamp] = np.linalg.inv(transform)

    return poses

  def _apply_opengl_transform(self, pose: np.ndarray) -> np.ndarray:
    """Apply transformation to convert from ARKit coordinate system to OpenGL.

    This follows the logic from nerfstudio's arkitscenes_dataparser.py _get_pose method:
    1. Invert y and z axes (columns 1 and 2)
    2. Swap rows: [x, y, z, w] -> [y, x, z, w]
    3. Invert z row

    Parameters
    ----------
    pose : np.ndarray
        4x4 transformation matrix in ARKit coordinate system

    Returns
    -------
    np.ndarray
        4x4 transformation matrix in OpenGL coordinate system
    """
    frame_pose = pose.copy()

    # Invert y and z axes
    frame_pose[0:3, 1:3] *= -1

    # Swap rows to reorder [x, y, z, w] to [y, x, z, w]
    frame_pose = frame_pose[np.array([1, 0, 2, 3]), :]

    # Invert z row
    frame_pose[2, :] *= -1

    return frame_pose

  def load_meshes(
    self, rot_from_positions: bool = True, cams_yaw_only: bool = True
  ) -> List[Mesh]:
    """Extract meshes and return them as a list of Mesh objects.

    This follows similar logic to GrandTourData, loading the mesh and camera poses,
    then applying transformations to convert to IsaacGym coordinate system.
    """
    if not self._downloaded_scenes:
      raise RuntimeError('No scenes downloaded. Call download_scenes() first.')

    meshes = []

    for scene_name, scene_path in self._downloaded_scenes.items():
      utils.print(f'Loading mesh from scene: {scene_name}', color='cyan')

      # Find the mesh file (should be *_3dod_mesh_filled_decimated.ply)
      mesh_files = list(scene_path.glob('*_3dod_mesh_filled_decimated.ply'))
      if not mesh_files:
        utils.print(f'No mesh file found in {scene_name}', color='yellow')
        continue

      filepath = mesh_files[0]
      splat_path = scene_path / 'splatfacto'

      # Load the mesh using trimesh
      scene = trimesh.load(str(filepath))
      if isinstance(scene, trimesh.Scene):
        mesh = trimesh.util.concatenate(
          [
            mesh
            for mesh in scene.geometry.values()
            if isinstance(mesh, trimesh.Trimesh)
          ]
        )
      else:
        mesh = scene

      # Apply ARKit to OpenGL transformation to mesh vertices (same as _load_3d_points)
      # R_x: flips Y and Z axes
      R_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
      mesh.vertices = np.dot(mesh.vertices, R_x.T)

      # R_z: rotates around Z axis (swaps and flips X/Y)
      R_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
      mesh.vertices = np.dot(mesh.vertices, R_z.T)

      # Center the mesh
      mean_vertex = np.mean(np.array(mesh.vertices), axis=0, keepdims=True)
      mesh.vertices -= mean_vertex

      # Load camera poses from .traj file
      traj_path = scene_path / 'lowres_wide.traj'
      if not traj_path.exists():
        utils.print(f'lowres_wide.traj not found: {traj_path}', color='yellow')
        continue

      poses_dict = self._parse_traj_file(traj_path)

      # Apply OpenGL transformation to all poses
      curr_cam_trans = []
      curr_cam_quat_xyzw = []
      for timestamp in sorted(poses_dict.keys(), key=float):
        pose = poses_dict[timestamp]
        opengl_pose = self._apply_opengl_transform(pose)
        # opengl_se3 = vtf.SE3.from_matrix(pose)
        opengl_se3 = vtf.SE3.from_matrix(opengl_pose)
        curr_cam_trans.append(opengl_se3.translation())
        curr_cam_quat_xyzw.append(opengl_se3.rotation().as_quaternion_xyzw())

      # Center poses in ARKit coordinate system before OpenGL transformation
      curr_cam_trans -= mean_vertex
      # for timestamp in poses_dict:
      #   poses_dict[timestamp][:3, 3] -= mean_vertex[0]

      utils.print(f'num frames: {len(curr_cam_trans)}', color='cyan')

      curr_cam_trans = np.array(curr_cam_trans)
      curr_cam_quat_xyzw = np.array(curr_cam_quat_xyzw)

      # Define rotation to IsaacGym coordinate system (same as GrandTourData)
      to_ig_euler_xyz = np.array([np.pi, 0.0, 0.0])

      # Apply rotation to align with IsaacGym coordinate system
      rotation = (
        vtf.SO3.from_x_radians(to_ig_euler_xyz[0]).inverse()
        @ vtf.SO3.from_y_radians(to_ig_euler_xyz[1]).inverse()
        @ vtf.SO3.from_z_radians(to_ig_euler_xyz[2]).inverse()
      )

      curr_cam_trans = np.dot(curr_cam_trans, rotation.as_matrix().T)
      mesh.vertices = np.dot(mesh.vertices, rotation.as_matrix().T)
      curr_cam_trans = math_utils.smooth_path(
        curr_cam_trans,
        smoothing_factor=5,
        resample_num_points=min(max(len(curr_cam_trans), 100), 100),
      )
      if rot_from_positions:
        curr_cam_quat_xyzw = compute_orientations_from_positions(
          curr_cam_trans, yaw_only=cams_yaw_only
        )
      else:
        curr_cam_quat_xyzw = (
          rotation @ vtf.SO3.from_quaternion_xyzw(curr_cam_quat_xyzw)
        ).as_quaternion_xyzw()
        curr_cam_quat_xyzw = (
          math_utils.opengl_to_opencv(torch.tensor(curr_cam_quat_xyzw)).cpu().numpy()
        )

      # Extract mesh data
      vertices = np.array(mesh.vertices).astype(np.float32)
      triangles = np.array(mesh.faces).astype(np.uint32)
      cam_offset = mean_vertex[0]
      from_ig_rotation = np.array(rotation.inverse().as_matrix())

      mesh = Mesh(
        scene_name=scene_name,
        filename=filepath.name,
        filepath=filepath,
        splatpath=splat_path,
        vertices=vertices,
        triangles=triangles,
        cam_trans=curr_cam_trans,
        cam_offset=cam_offset,
        cam_quat_xyzw=curr_cam_quat_xyzw,
        ig_to_orig_rot=from_ig_rotation,
      )

      # Filter out invalid poses using ray casting
      mesh = get_filtered_poses(mesh, min_ceiling_height=1.0, grid_diameter=0.3)
      if mesh.cam_trans.shape[0] < 10:
        continue

      meshes.append(mesh)

    utils.print(f'Loaded {len(meshes)} meshes total', color='green')
    return meshes
