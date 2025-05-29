import numpy as np
import os
import pathlib
import dataclasses
from collections import defaultdict
from isaacgym import gymapi
from typing import Union, Dict, List
import torch
import warp as wp

import legged_gym
from legged_gym.utils import sensors, math


wp.init()


TARGET_IDX_OFFSET = 5


@dataclasses.dataclass
class RawMesh:
  # Scene metadata.
  scene_name: str
  filename: str
  filepath: str

  # Mesh parameters.
  vertices: np.ndarray
  triangles: np.ndarray

  # Camera parameters. Camera translations are defined in the world frame.
  cam_trans: np.ndarray
  cam_offset: np.ndarray

  # Rotation matrix from IG coordinate system to original coordinate system.
  ig_to_orig_rot: np.ndarray


@dataclasses.dataclass
class Mesh(RawMesh):
  # Camera parameters.
  # Camera orientations are defined in the OpenCV camera convention,
  # which is what IsaacGym uses.
  #  +Z (fwd)
  #   \
  #    \------ +X (right)
  #     |
  #     |
  #     +Y (down)
  cam_quat_xyzw: np.ndarray

  # Pose trajectories are front padded so each mesh has an equal number of camera poses.
  # This tensor indicates the first index of each mesh's unpadded poses.
  valid_pose_start_idxs: np.ndarray

def find_valid_scene_directories(scene_paths: List[str]) -> List[str]:
  valid_scenes = []
  for scene_path in scene_paths:
      for root, dirs, _ in os.walk(scene_path):
          if "splatfacto" in dirs and "meshes" in dirs:
              valid_scenes.append(root)
  return valid_scenes


class GaussianTerrain:
  def __init__(self, cfg: Dict, num_robots) -> None:
    self.cfg = cfg
    self.num_robots = num_robots
    self.scene_paths = find_valid_scene_directories(self.cfg["terrain"]["scenes"])

    self._mesh_dict = {}
    self._load_meshes()
    self._process_poses()

    print(f"Loaded {len(self._mesh_dict)} meshes.")

  @property
  def num_meshes(self):
    return len(self._mesh_dict)

  @property
  def scene_keys(self):
    return list(set([k[0] for k in self.mesh_keys]))

  @property
  def mesh_keys(self):
    return [key.split("/") for key in list(self._mesh_dict.keys())]

  def get_scene_mesh_keys(self, scene_name: str):
    return [k for k in self.mesh_keys if k[0] == scene_name]

  def get_mesh(self, scene_name: str, filename: str):
    return self._mesh_dict[os.path.join(scene_name, filename)]

  def get_value(self, key: str):
    values = {}
    for mesh_key in self.mesh_keys:
      values["/".join(mesh_key)] = getattr(
        self._mesh_dict[os.path.join(mesh_key[0], mesh_key[1])], key
      )
    return values

  def _load_mesh(self, scene_path: pathlib.Path, scene: str, filename: str):
    filepath = scene_path / "meshes" / filename
    mesh_dict = np.load(filepath)

    cam_offset = np.array(mesh_dict["offset"])[0].astype(np.float32)
    vertices = np.array(mesh_dict["vertices"]).astype(np.float32)
    triangles = np.array(mesh_dict["triangles"]).astype(np.uint32)
    curr_cam_trans = np.array(mesh_dict["cam_trans"]).astype(np.float32)
    curr_cam_trans = self.smooth_path(
      curr_cam_trans, smoothing_factor=10,
      resample_num_points=30)
    from_ig_rotation = np.array(mesh_dict["from_ig_rotation"]).astype(np.float32)

    return RawMesh(
      scene_name=scene,
      filename=filename,
      filepath=filepath,
      vertices=vertices,
      triangles=triangles,
      cam_trans=curr_cam_trans,
      cam_offset=cam_offset,
      ig_to_orig_rot=from_ig_rotation,
    )

  def _load_meshes(self):
    for scene_path in self.scene_paths:
      scene_path = pathlib.Path(scene_path)
      scene_name = scene_path.name
      filenames = [f.name for f in (scene_path / "meshes").iterdir() if f.suffix == '.npz']
      for filename in filenames:
        self._mesh_dict[os.path.join(scene_name, filename)] = self._load_mesh(
          scene_path, scene_name, filename
        )

  def _process_poses(self):
    max_length = max([m.cam_trans.shape[0] for m in self._mesh_dict.values()])

    for filepath, raw_mesh in self._mesh_dict.items():
      # Compute orientations from camera translations.
      if self.cfg["terrain"]["cams_yaw_only"]:
        delta_xy = raw_mesh.cam_trans[1:, :2] - raw_mesh.cam_trans[:-1, :2]
        delta_xyz = np.concatenate(
          [delta_xy, np.zeros_like(delta_xy[:, :-1])], axis=-1
        )
      else:
        delta_xyz = raw_mesh.cam_trans[1:] - raw_mesh.cam_trans[:-1]
      x_component = delta_xyz / np.linalg.norm(
        delta_xyz, axis=-1, keepdims=True
      )
      z_component = np.zeros_like(x_component)
      z_component[:, 2] = 1
      y_component = np.cross(z_component, x_component)
      y_component = y_component / np.linalg.norm(
        y_component, axis=-1, keepdims=True
      )

      cam_rot = math.matrix_to_quaternion(torch.tensor(np.stack([x_component, y_component, z_component], axis=2)))
      y_rot = math.quat_from_y_rot(np.pi / 2, cam_rot.shape[0])
      z_rot = math.quat_from_z_rot(-np.pi / 2, cam_rot.shape[0])
      to_opencv = math.quat_mul(y_rot, z_rot)
      cam_quat_xyzw = math.quat_mul(cam_rot, to_opencv.detach())

      # Compute valid poses to sample along camera trajectory.
      valid_pose_start_idxs = max_length - raw_mesh.cam_trans.shape[0]
      raw_mesh.cam_trans = np.pad(
        raw_mesh.cam_trans,
        ((max_length - raw_mesh.cam_trans.shape[0], 0), (0, 0)),
        mode="edge",
      )
      cam_quat_xyzw = np.pad(
        cam_quat_xyzw,
        ((max_length - cam_quat_xyzw.shape[0], 0), (0, 0)),
        mode="edge",
      )
      self._mesh_dict[filepath] = Mesh(
        **raw_mesh.__dict__,
        cam_quat_xyzw=cam_quat_xyzw,
        valid_pose_start_idxs=valid_pose_start_idxs,
      )

  def smooth_path(self, poses, smoothing_factor=5, 
                  # resample=True, resample_factor=1.0,
                  resample_num_points: Union[int, None]=None,
      ):
    """Smooths a trajectory of poses using a Savitzky-Golay filter.
    
    Args:
        poses (np.ndarray): Array of poses with shape [N, 3]
        smoothing_factor (int): Window size for smoothing. Larger values create smoother paths.
            Must be odd and at least 3. Default: 5
        resample_num_points (int): Number of points to resample the path to. Default: None
    Returns:
        np.ndarray: Smoothed poses with shape [M, 3] where M depends on resample_num_points
    """
    import scipy.signal
    import scipy.interpolate
    
    # Ensure poses is a numpy array
    poses = np.array(poses)
    
    # Ensure smoothing_factor is odd
    if smoothing_factor % 2 == 0:
        smoothing_factor += 1
    
    # Ensure minimal window size
    smoothing_factor = max(3, smoothing_factor)
    
    # Polynomial order - using 2 for quadratic smoothing
    poly_order = min(2, smoothing_factor - 1)
    
    # Apply Savitzky-Golay filter to each dimension
    smoothed_poses = np.zeros_like(poses)
    for i in range(poses.shape[1]):
        smoothed_poses[:, i] = scipy.signal.savgol_filter(
            poses[:, i], smoothing_factor, poly_order
        )
    
    if resample_num_points is None:
        return smoothed_poses
    
    # Resample the path using spline interpolation
    n_points = poses.shape[0]
    new_n_points = resample_num_points
    
    # Create a parameter along the path (cumulative distance)
    t = np.zeros(n_points)
    for i in range(1, n_points):
        t[i] = t[i-1] + np.linalg.norm(smoothed_poses[i] - smoothed_poses[i-1])
    
    # Normalize parameter to [0, 1]
    if t[-1] > 0:
        t = t / t[-1]
    
    # Create interpolation splines for each dimension
    splines = [
        scipy.interpolate.splrep(t, smoothed_poses[:, i], k=3, s=1)
        for i in range(smoothed_poses.shape[1])
    ]
    
    # Sample new points
    new_t = np.linspace(0, 1, new_n_points)
    resampled_poses = np.zeros((new_n_points, poses.shape[1]))
    for i in range(poses.shape[1]):
        resampled_poses[:, i] = scipy.interpolate.splev(new_t, splines[i])
    
    return resampled_poses


class GaussianSceneManager:
  # Loads meshes and assigns robots to each mesh. Also handles replacement of meshes from multiple scenes.
  # Provides classes for sampling commands along a camera path and detecting when the robot strays off the original path.
  # Also provides visualization tools for debugging.
  def __init__(self, env):
    self._env = env
    self._terrain = GaussianTerrain(env.cfg, env.num_envs)

    self.renderer = sensors.GaussianSplattingRenderer(self._env, self)

    self.command_scale = torch.zeros(
      self._env.num_envs, 1, device=self._env.device, dtype=torch.float32, requires_grad=False
    )
    self.velocity_command = torch.zeros(
      self._env.num_envs, 2, device=self._env.device, dtype=torch.float32, requires_grad=False
    )
    self.heading_command = torch.zeros(
      self._env.num_envs, 1, device=self._env.device, dtype=torch.float32, requires_grad=False
    )
    self.axis_geom = None
    self.velocity_geom = None
    self.heading_geom = None

    # Index of the current robot state in the camera trajectory.
    self.state_idx = torch.zeros(
      self._env.num_envs, device=self._env.device, dtype=torch.int64, requires_grad=False
    )

    self.local_offset = torch.tensor(
        np.array(self._env.cfg["env"]["camera_params"]["cam_xyz_offset"])[None].repeat(self._env.num_envs, 0),
        dtype=torch.float, device=self._env.device, requires_grad=False
    )

    self.cam_rpy_offset = math.quat_mul(
      math.quat_mul(
        math.quat_from_x_rot(self._env.cfg["env"]["camera_params"]["cam_rpy_offset"][0], 1, self._env.device),
        math.quat_from_y_rot(self._env.cfg["env"]["camera_params"]["cam_rpy_offset"][1], 1, self._env.device)),
        math.quat_from_z_rot(self._env.cfg["env"]["camera_params"]["cam_rpy_offset"][2], 1, self._env.device)).detach()

    self.robot_frame_transform = math.quat_mul(
      math.quat_from_z_rot(np.pi / 2, 1, self._env.device),
      math.quat_from_y_rot(-np.pi / 2, 1, self._env.device)
    ).detach()

  def spawn_meshes(self):
    # Add meshes to the environment.
    num_rows = np.floor(np.sqrt(self._terrain.num_meshes))

    curr_x_offset = 0.0
    curr_y_offset = 0.0
    env_origins = []
    all_vertices = []
    all_triangles = []

    for i, (scene, mesh) in enumerate(self._terrain.mesh_keys):
      if i > 0 and i % num_rows == 0:
        curr_x_offset += self._env.cfg["env"]["env_spacing"]
        curr_y_offset = 0.0
      else:
        curr_y_offset += self._env.cfg["env"]["env_spacing"]

      if self._env.cfg["sim"]["up_axis"] == 1:
        env_origin = [curr_x_offset, curr_y_offset, 0.0]
      elif self._env.cfg["sim"]["up_axis"] == 0:
        env_origin = [curr_x_offset, 0.0, curr_y_offset]
      else:
        raise ValueError
      env_origin = np.array(env_origin, dtype=np.float32)
      env_origins.append(env_origin)

      mesh = self._terrain.get_mesh(scene, mesh)
      vertices = env_origin[None] + mesh.vertices
      mesh_params = gymapi.TriangleMeshParams()
      mesh_params.static_friction = self._env.cfg["terrain"]["static_friction"]
      mesh_params.dynamic_friction = self._env.cfg["terrain"]["dynamic_friction"]
      mesh_params.restitution = self._env.cfg["terrain"]["restitution"]
      mesh_params.nb_vertices = vertices.shape[0]
      mesh_params.nb_triangles = mesh.triangles.shape[0]
      self._env.gym.add_triangle_mesh(
        self._env.sim,
        vertices.flatten(order="C"),
        mesh.triangles.flatten(order="C"),
        mesh_params,
      )

      vertices_offset = (
        0 if len(all_vertices) == 0 else np.concatenate(all_vertices).shape[0]
      )
      all_triangles.append(mesh.triangles + vertices_offset)
      all_vertices.append(vertices)

    self.all_vertices_mesh = np.concatenate(all_vertices)
    self.all_triangles_mesh = np.concatenate(all_triangles)
    self.terrain_mesh = sensors.convert_to_wp_mesh(self.all_vertices_mesh, self.all_triangles_mesh, self._env.device)

    self.env_origins = np.array(env_origins)
    self.construct_trajectory_arrays()

  def construct_trajectory_arrays(self):
    # Assign different env origins, cam_trans, quat, and offsets to each environment.
    repeat_factor = int(np.ceil(self._env.num_envs / self._terrain.num_meshes))

    def repeat(x):
      return np.repeat(np.array(x), repeat_factor, axis=0)[: self._env.num_envs]

    cam_trans_orig = np.array(
      list(self._terrain.get_value("cam_trans").values())
    )
    cam_quat_xyzw_orig = np.array(
      list(self._terrain.get_value("cam_quat_xyzw").values())
    )
    env_origins_z0 = np.pad(self.env_origins[:, :2], ((0, 0), (0, 1)), mode="constant", constant_values=0.0)
    cam_trans_orig = cam_trans_orig + env_origins_z0[:, None, :]

    # Use warp to get ground position at each camera.
    directions = np.array([0, 0, -1])[None, None].repeat(cam_trans_orig.shape[0], axis=0).repeat(cam_trans_orig.shape[1], axis=1)
    directions_torch = math.to_torch(directions, device=self._env.device, requires_grad=False)
    cam_trans_orig_torch = math.to_torch(cam_trans_orig, device=self._env.device, requires_grad=False)
    ground_positions_world_frame = sensors.ray_cast(cam_trans_orig_torch.view(-1, 3), directions_torch.reshape(-1, 3), self.terrain_mesh)
    ground_positions_world_frame = ground_positions_world_frame.view(*cam_trans_orig_torch.shape).cpu().numpy()
    ground_positions_world_frame = ground_positions_world_frame - env_origins_z0[:, None, :]
    self.ground_positions = math.to_torch(
      repeat(ground_positions_world_frame),
      device=self._env.device,
      requires_grad=False,
    )

    self.cam_trans_viz = math.to_torch(
      cam_trans_orig, device=self._env.device, requires_grad=False
    )
    self.cam_quat_xyzw_viz = math.to_torch(
      cam_quat_xyzw_orig, device=self._env.device, requires_grad=False
    )

    self.scenes = repeat(list(self._terrain.get_value("scene_name").values()))
    self.cam_trans = math.to_torch(
      repeat(list(self._terrain.get_value("cam_trans").values())),
      device=self._env.device,
      requires_grad=False,
    )
    self.cam_quat_xyzw = math.to_torch(
      repeat(list(self._terrain.get_value("cam_quat_xyzw").values())),
      device=self._env.device,
      requires_grad=False,
    )
    self.ig_to_orig_rot = math.to_torch(
      repeat(list(self._terrain.get_value("ig_to_orig_rot").values())),
      device=self._env.device,
      requires_grad=False,
    )
    self.cam_offset = math.to_torch(
      repeat(list(self._terrain.get_value("cam_offset").values())),
      device=self._env.device,
      requires_grad=False,
    )
    self.valid_pose_start_idxs = math.to_torch(
      repeat(list(self._terrain.get_value("valid_pose_start_idxs").values())),
      device=self._env.device,
      requires_grad=False,
    )
    self.env_origins = math.to_torch(
      repeat(self.env_origins),
      device=self._env.device,
      requires_grad=False,
    )

    # Get start and end env ids for each scene.
    self.scene_start_end_ids = {}
    curr_scene = None
    start, end = 0, 0
    for k in self.scenes:
      if curr_scene is None:
        curr_scene = k
      elif curr_scene != k:
        self.scene_start_end_ids[curr_scene] = (start, end)
        curr_scene = k
        start = end
      end += 1
    self.scene_start_end_ids[curr_scene] = (start, end)


  def get_cam_link_pose_world_frame(self):
    # In the frame of the world.
    return self._env.get_camera_link_state()[:, :3], self._env.get_camera_link_state()[:, 3:7]

  def get_cam_link_velocity_world_frame(self):
    cam_link_lin_vel = self._env.get_camera_link_state()[:, 7:10]
    cam_link_ang_vel = self._env.get_camera_link_state()[:, 10:13]
    return cam_link_lin_vel, cam_link_ang_vel

  def get_cam_pose_world_frame(self):
    cam_link_trans, cam_link_quat = self.get_cam_link_pose_world_frame()
    # Apply xyz offset in the local robot frame.
    cam_trans = cam_link_trans + math.quat_apply(cam_link_quat, self.local_offset)
    cam_quat = math.quat_mul(cam_link_quat, self.cam_rpy_offset.expand(cam_link_quat.shape[0], -1))
    return cam_trans, cam_quat

  def get_cam_velocity_world_frame(self):
    cam_link_lin_vel, cam_link_ang_vel = self.get_cam_link_velocity_world_frame()
    _, cam_link_quat = self.get_cam_link_pose_world_frame()

    # Angular velocity remains the same after offset
    cam_ang_vel = cam_link_ang_vel

    # Linear velocity needs to account for the offset
    # v = v_link + ω × r, where r is the offset vector in world frame
    # Using cross product of angular velocity with position offset
    offset_world = math.quat_apply(cam_link_quat, self.local_offset)
    velocity_from_rotation = torch.cross(cam_link_ang_vel, offset_world)
    cam_lin_vel = cam_link_lin_vel + velocity_from_rotation
    return cam_lin_vel, cam_ang_vel

  def get_cam_link_pose_local_frame(self):
    # The frame local to each environment.
    cam_trans, cam_quat = self.get_cam_link_pose_world_frame()
    cam_trans = cam_trans - self.env_origins
    return cam_trans, cam_quat

  def mesh_id_for_env_id(self, env_id):
    robots_per_mesh = int(
      np.ceil(self._env.num_envs / self._terrain.num_meshes)
    )
    return int(np.floor(env_id / robots_per_mesh))

  def mesh_name_from_id(self, mesh_id):
    return '/'.join(self._terrain.mesh_keys[mesh_id])

  def env_ids_for_mesh_id(self, mesh_id):
    robots_per_mesh = int(
      np.ceil(self._env.num_envs / self._terrain.num_meshes)
    )
    return torch.arange(
      mesh_id * robots_per_mesh,
      min((mesh_id + 1) * robots_per_mesh, self._env.num_envs),
      device=self._env.device,
      dtype=torch.int32,
    )

  @property
  def num_meshes(self):
    return self._terrain.num_meshes

  def _get_nearest_traj_idx(self, monotonic):
    curr_cam_trans, _ = self.get_cam_link_pose_local_frame()
    trans_difference = curr_cam_trans[:, None] - self.cam_trans
    distance = torch.norm(trans_difference, dim=-1)
    if monotonic:
      # Create a mask for valid indices
      num_envs, num_poses = distance.shape
      indices = torch.arange(num_poses, device=distance.device)[None]
      indices = indices.repeat_interleave(num_envs, 0)
      mask = (indices < self.state_idx[:, None]) | (indices > self.state_idx[:, None] + TARGET_IDX_OFFSET)
    
      # Apply mask by setting invalid distances to infinity
      masked_distance = distance.clone()
      masked_distance[mask] = float('inf')

      min_distance_idx = torch.argmin(masked_distance, dim=1)
    else:
      min_distance_idx = torch.argmin(distance, dim=1)

    
    return min_distance_idx

  def _get_nearest_traj_pose(self, monotonic=False):
    nearest_traj_idx = self._get_nearest_traj_idx(monotonic)
    nearest_traj_pos_idx = (
      nearest_traj_idx.unsqueeze(1)
      .unsqueeze(2)
      .expand(-1, 1, self.cam_trans.shape[-1])
    )
    nearest_traj_pos = torch.gather(
      self.cam_trans, dim=1, index=nearest_traj_pos_idx
    ).squeeze(1)
    nearest_traj_quat_idx = (
      nearest_traj_idx.unsqueeze(1)
      .unsqueeze(2)
      .expand(-1, 1, self.cam_quat_xyzw.shape[-1])
    )
    nearest_traj_quat = torch.gather(
      self.cam_quat_xyzw, dim=1, index=nearest_traj_quat_idx
    ).squeeze(1)
    return nearest_traj_pos, nearest_traj_quat, nearest_traj_idx

  def to_robot_frame(self, cam_quat):
    # Camera poses are in OpenCV convention, whereas robot poses are defined 
    #  +Z (up)
    #   |
    #   |
    #   |
    #   +------ +Y (right)
    #    \
    #     \
    #      \
    #       +X (forward)
    return math.quat_mul(cam_quat, self.robot_frame_transform.expand(cam_quat.shape[0], -1))

  def sample_cam_pose(self, env_ids, use_ground_positions=False):
    # Sample a random camera position and orientation from the camera trajectories.
    if use_ground_positions:
      cam_trans_subs = self.ground_positions[env_ids]
    else:
      cam_trans_subs = self.cam_trans[env_ids]
    cam_quat_xyzw_subs = self.cam_quat_xyzw[env_ids]
    rand = torch.rand((len(env_ids),), device=self._env.device)
    low = self.valid_pose_start_idxs[env_ids].to(torch.float32)
    # high = (self.valid_pose_start_idxs[env_ids].to(torch.float32) + cam_trans_subs.shape[1]) / 2
    high = self.valid_pose_start_idxs[env_ids].to(torch.float32) + 2
    # high = (
    #   torch.ones((len(env_ids),), device=self._env.device)
    #   * cam_trans_subs.shape[1]
    #   - 6
    # )
    state_idx = (rand * (high - low) + low).to(torch.int64)
    self.state_idx[env_ids] = state_idx

    trans_idx = (
      state_idx.unsqueeze(1)
      .unsqueeze(2)
      .expand(-1, 1, cam_trans_subs.shape[-1])
    )
    cam_trans = torch.gather(cam_trans_subs, dim=1, index=trans_idx).squeeze(1)

    quat_idx = (
      state_idx.unsqueeze(1)
      .unsqueeze(2)
      .expand(-1, 1, cam_quat_xyzw_subs.shape[-1])
    )
    cam_quat = torch.gather(cam_quat_xyzw_subs, dim=1, index=quat_idx).squeeze(
      1
    )

    # Convert to robot frame.
    robot_quat = self.to_robot_frame(cam_quat)
    return cam_trans, robot_quat

  def check_completed(self, env_ids):
    _, _, nearest_idx = (
      self._get_nearest_traj_pose(monotonic=True)
    )
    past_end = nearest_idx >= (
      self.cam_trans.shape[1] - 4
    )  # Past end of trajectory.

    completion_counter = defaultdict(lambda: 0)
    for env_id in env_ids:
      mesh_id = self.mesh_id_for_env_id(int(env_id))
      mesh_name = self.mesh_name_from_id(mesh_id)
      completion_counter[mesh_name] += int(past_end[int(env_id)])
    return completion_counter

  def check_termination(self):
    # Check if robot is too far from camera trajectory (Indicative of poor rendering).
    curr_cam_link_trans, curr_cam_link_quat = self.get_cam_link_pose_local_frame()
    nearest_cam_trans, nearest_cam_quat, nearest_idx = (
      self._get_nearest_traj_pose(monotonic=True)
    )
    nearest_robot_quat = self.to_robot_frame(nearest_cam_quat)
    past_end = nearest_idx >= (
      self.cam_trans.shape[1] - 4
    )  # Past end of trajectory.
    distance_exceeded = past_end.clone()
    yaw_exceeded = past_end.clone()

    distance = torch.norm((curr_cam_link_trans - nearest_cam_trans)[:, :2], dim=-1)
    distance_exceeded |= distance > self._env.cfg["terrain"]["max_traj_pos_distance"]

    quat_difference = math.quat_mul(curr_cam_link_quat, math.quat_conjugate(nearest_robot_quat))
    _, _, yaw = math.get_euler_xyz(quat_difference)
    yaw_distance = torch.abs(math.wrap_to_pi(yaw))
    yaw_exceeded |= yaw_distance > self._env.cfg["terrain"]["max_traj_yaw_distance_rad"]
    return distance_exceeded, yaw_exceeded

  def sample_commands(self, env_ids, still_env_mask=None):
    """Randommly select commands of some environments

    Args:
        env_ids (List[int]): Environments ids for which new commands are needed
    """
    _, _, nearest_idx = self._get_nearest_traj_pose(monotonic=True)
    # Update the state index to the nearest camera trajectory.
    self.state_idx[:] = nearest_idx

    # Choose a target pose moderately far away. If nearest_idx + 1 is chosen, target
    # velocities fall to 0 as the robot approaches this target.
    target_idx = torch.clamp(nearest_idx + TARGET_IDX_OFFSET, 0, self.cam_trans.shape[1] - 1)
    target_traj_quat_idx = (
      target_idx.unsqueeze(1)
      .unsqueeze(2)
      .expand(-1, 1, self.cam_quat_xyzw.shape[-1])
    )
    target_traj_quat = torch.gather(
      self.cam_quat_xyzw, dim=1, index=target_traj_quat_idx
    ).squeeze(1)
    target_traj_quat = self.to_robot_frame(target_traj_quat)
    _, _, yaw = math.get_euler_xyz(target_traj_quat)
    self.heading_command = math.wrap_to_pi(
      yaw
    )  # Match the orientation of the nearest camera.
    target_traj_pos_idx = (
      target_idx.unsqueeze(1)
      .unsqueeze(2)
      .expand(-1, 1, self.cam_trans.shape[-1])
    )
    target_traj_trans = torch.gather(
      self.cam_trans, dim=1, index=target_traj_pos_idx
    ).squeeze(1)
    curr_cam_link_trans, _ = self.get_cam_link_pose_local_frame()
    pos_delta = target_traj_trans - curr_cam_link_trans

    pos_delta_local = math.quat_rotate_inverse(
      self._env.root_states[:, 3:7], pos_delta
    )[:, :2]
    pos_delta_norm = pos_delta_local / torch.norm(
      pos_delta_local, dim=-1, keepdim=True
    )
    self.command_scale[env_ids] = math.torch_rand_float(
      self._env.command_ranges["lin_vel"][0],
      self._env.command_ranges["lin_vel"][1],
      (len(env_ids), 1),
      device=self._env.device,
    )
    pos_delta_norm *= self.command_scale
    # pos_delta_norm *= (
    #   torch.norm(pos_delta_norm[:, :2], dim=1, keepdim=True) > 0.1
    # )  # Zero out small commands.
    self.velocity_command = pos_delta_norm
    if still_env_mask is not None:
      self.velocity_command[still_env_mask] = 0.0

    return self.heading_command, self.velocity_command

  def debug_vis(self, env):
    if self.axis_geom is None:
      self.axis_geom = sensors.BatchWireframeAxisGeometry(
        np.prod(self.cam_trans_viz.shape[:2]), 0.25, 0.005, 16
      )
    if self.velocity_geom is None:
      self.velocity_geom = sensors.BatchWireframeAxisGeometry(
        self._env.num_envs,
        0.3,
        0.01,
        32,
        color_x=(1, 1, 1),
        color_y=(1, 1, 1),
        color_z=(0, 0, 0),
      )
    if self.heading_geom is None:
      self.heading_geom = sensors.BatchWireframeAxisGeometry(
        self._env.num_envs, 0.2, 0.01, 32, color_x=(1, 1, 0)
      )

    # Draw camera trajectory.
    if self._env.selected_environment < 0:
      self.axis_geom.draw(
        self.cam_trans_viz.reshape(-1, 3),
        self.cam_quat_xyzw_viz.reshape(-1, 4),
        self._env.gym,
        self._env.viewer,
        self._env.envs[0],
      )
    else:
      mesh_idx = self.mesh_id_for_env_id(self._env.selected_environment)
      only_selected_range = [
        mesh_idx * self.cam_trans_viz.shape[1],
        (mesh_idx + 1) * self.cam_trans_viz.shape[1],
      ]
      self.axis_geom.draw(
        self.cam_trans_viz.reshape(-1, 3),
        self.cam_quat_xyzw_viz.reshape(-1, 4),
        self._env.gym,
        self._env.viewer,
        self._env.envs[0],
        only_render_selected=only_selected_range,
      )

    # Draw velocity command.
    velocity_quat = self._env.root_states[:, 3:7]
    velocity_trans = self._env.root_states[:, :3] + math.quat_apply(
      velocity_quat,
      torch.tensor([0, 0, 0.25], device=self._env.device)[None].repeat(
        self._env.num_envs, 1
      ),
    )
    axis_scales = (
      self.velocity_command.cpu().numpy()
      / self._env.command_ranges["lin_vel"][1]
    )
    axis_scales = np.pad(
      axis_scales, ((0, 0), (0, 1)), mode="constant", constant_values=0
    )
    axis_scales = math.to_torch(
      axis_scales, device=self._env.device, requires_grad=False
    )
    self.velocity_geom.draw(
      velocity_trans,
      velocity_quat,
      self._env.gym,
      self._env.viewer,
      self._env.envs[0],
      axis_scales=axis_scales,
      only_render_selected=self._env.selected_environment,
    )

    # Draw heading command.
    heading_trans = self._env.root_states[:, :3] + math.quat_apply(
      self._env.root_states[:, 3:7],
      torch.tensor([0, 0, 0.2], device=self._env.device)[None].repeat(
        self._env.num_envs, 1
      ),
    )
    heading_quat = math.quat_from_euler_xyz(
      torch.zeros_like(self.heading_command),
      torch.zeros_like(self.heading_command),
      self.heading_command,
    )
    axis_scales = math.to_torch(
      np.array([1, 0, 0])[None].repeat(self._env.num_envs, axis=0),
      device=self._env.device,
      requires_grad=False,
    )
    self.heading_geom.draw(
      heading_trans,
      heading_quat,
      self._env.gym,
      self._env.viewer,
      self._env.envs[0],
      axis_scales=axis_scales,
      only_render_selected=self._env.selected_environment,
    )
