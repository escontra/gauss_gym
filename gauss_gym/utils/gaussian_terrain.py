import abc
import numpy as np
import os
import dataclasses
from collections import defaultdict
from typing import Dict, Optional, Tuple
import copy
import torch

from isaacgym import gymapi

from gauss_gym import utils
from gauss_gym.utils import (
  sensors,
  math_utils,
  warp_utils,
  visualization_geometries,
  space,
  scene_ingest,
)
from gauss_gym.rl import utils as rl_utils


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


@dataclasses.dataclass
class EqualLengthMesh(scene_ingest.Mesh):
  max_length: int

  @property
  def valid_pose_start_idxs(self):
    return self.max_length - self.cam_trans.shape[0]

  @property
  def cam_trans_padded(self):
    return np.pad(
      self.cam_trans,
      ((self.max_length - self.cam_trans.shape[0], 0), (0, 0)),
      mode='edge',
    )

  @property
  def cam_quat_xyzw_padded(self):
    return np.pad(
      self.cam_quat_xyzw,
      ((self.max_length - self.cam_quat_xyzw.shape[0], 0), (0, 0)),
      mode='edge',
    )


class GaussianTerrain:
  def __init__(self, cfg: Dict, num_robots) -> None:
    self.cfg = cfg
    self.num_robots = num_robots

    self._mesh_dict = {}

    # Load scenes from multiple data sources based on configuration
    scenes_cfg = copy.deepcopy(cfg['terrain']['scenes'])
    max_num_scenes = scenes_cfg.pop('max_num_scenes')
    cams_yaw_only = cfg['terrain']['cams_yaw_only']

    all_meshes = []

    # Iterate through each data source
    for source_name, source_cfg in scenes_cfg.items():
      split_frac = source_cfg.pop('split_frac')
      class_name = source_cfg.pop('class_name')

      # Calculate number of scenes to load from this source
      num_scenes_from_source = int(max_num_scenes * split_frac)
      if num_scenes_from_source == 0:
        continue

      utils.print(
        f'Loading {num_scenes_from_source} scenes from {source_name}...', color='cyan'
      )

      data_ingest = getattr(scene_ingest, class_name)(**source_cfg)

      # Download and load scenes
      data_ingest.download_scenes(max_num_scenes=num_scenes_from_source)
      meshes = data_ingest.load_meshes(cams_yaw_only=cams_yaw_only)
      all_meshes.extend(meshes)

      utils.print(f'Loaded {len(meshes)} meshes from {source_name}', color='green')

    # Pad all meshes to the same length
    max_length = max([mesh.cam_trans.shape[0] for mesh in all_meshes])
    for mesh in all_meshes:
      self._mesh_dict[os.path.join(mesh.scene_name, mesh.filename)] = EqualLengthMesh(
        **mesh.__dict__,
        max_length=max_length,
      )

    utils.print(f'Loaded {len(self._mesh_dict)} meshes total.', color='green')

  @property
  def num_meshes(self):
    return len(self._mesh_dict)

  @property
  def scene_keys(self):
    return list(set([k[0] for k in self.mesh_keys]))

  @property
  def mesh_keys(self):
    mesh_keys = []
    for key in list(self._mesh_dict.keys()):
      key_split = key.split('/')
      scene = '/'.join(key_split[:-1])
      fname = key_split[-1]
      mesh_keys.append((scene, fname))
    return mesh_keys
    # return [key.split("/") for key in list(self._mesh_dict.keys())]

  def get_scene_mesh_keys(self, scene_name: str):
    return [k for k in self.mesh_keys if k[0] == scene_name]

  def get_mesh(self, scene_name: str, filename: str):
    return self._mesh_dict[os.path.join(scene_name, filename)]

  def get_value(self, key: str):
    values = {}
    for mesh_key in self.mesh_keys:
      values['/'.join(mesh_key)] = getattr(
        self._mesh_dict[os.path.join(mesh_key[0], mesh_key[1])], key
      )
    return values


class GaussianSceneManager:
  # Loads meshes and assigns robots to each mesh. Also handles replacement of meshes from multiple scenes.
  # Provides classes for sampling commands along a camera path and detecting when the robot strays off the original path.
  # Also provides visualization tools for debugging.
  def __init__(self, env):
    self._env = env
    self._terrain = GaussianTerrain(env.cfg, env.num_envs)

    self.renderer = sensors.GaussianSplattingRenderer(self._env, self)

    self.axis_geom = None
    self.closest_axis_geom = None
    self.target_axis_geom = None
    self.velocity_geom = None
    self.heading_geom = None

    self.local_offset = torch.tensor(
      np.array(self._env.cfg['env']['camera_params']['cam_xyz_offset'])[None].repeat(
        self._env.num_envs, 0
      ),
      dtype=torch.float,
      device=self._env.device,
      requires_grad=False,
    )

    self.cam_rpy_offset = math_utils.quat_mul(
      math_utils.quat_mul(
        math_utils.quat_from_x_rot(
          self._env.cfg['env']['camera_params']['cam_rpy_offset'][0],
          1,
          self._env.device,
        ),
        math_utils.quat_from_y_rot(
          self._env.cfg['env']['camera_params']['cam_rpy_offset'][1],
          1,
          self._env.device,
        ),
      ),
      math_utils.quat_from_z_rot(
        self._env.cfg['env']['camera_params']['cam_rpy_offset'][2], 1, self._env.device
      ),
    ).detach()

    self.local_rpy_offset = torch.stack(
      [
        torch.full(
          (self._env.num_envs,),
          self._env.cfg['env']['camera_params']['cam_rpy_offset'][0],
          device=self._env.device,
          requires_grad=False,
        ),
        torch.full(
          (self._env.num_envs,),
          self._env.cfg['env']['camera_params']['cam_rpy_offset'][1],
          device=self._env.device,
          requires_grad=False,
        ),
        torch.full(
          (self._env.num_envs,),
          self._env.cfg['env']['camera_params']['cam_rpy_offset'][2],
          device=self._env.device,
          requires_grad=False,
        ),
      ],
      dim=-1,
    )

    self.robot_frame_transform = math_utils.quat_mul(
      math_utils.quat_from_z_rot(np.pi / 2, 1, self._env.device),
      math_utils.quat_from_y_rot(-np.pi / 2, 1, self._env.device),
    ).detach()

  def spawn_meshes(self):
    # Add meshes to the environment.
    curr_x_offset = 0.0
    curr_y_offset = 0.0
    env_origins = []
    all_vertices = []
    all_triangles = []
    self.all_vertices_orig = defaultdict(dict)
    self.all_triangles_orig = defaultdict(dict)

    mesh_keys_repeated = math_utils.repeat_interleave(
      self._terrain.mesh_keys, self.num_mesh_repeats
    )
    num_rows = int(np.floor(np.sqrt(len(mesh_keys_repeated))))
    buffer = 0.2  # Buffer between meshes to prevent touching

    # Track the maximum extent in x direction for each row
    row_max_x_extent = 0.0

    for i, (scene_name, mesh_name) in enumerate(mesh_keys_repeated):
      mesh = self._terrain.get_mesh(scene_name, mesh_name)

      # Compute bounding box using 99th/1st percentile to exclude outliers (1% buffer)
      x_vertices = mesh.vertices[:, 0]
      y_vertices = mesh.vertices[:, 1]
      x_min = np.percentile(x_vertices, 1)
      x_max = np.percentile(x_vertices, 99)
      y_min = np.percentile(y_vertices, 1)
      y_max = np.percentile(y_vertices, 99)

      # Shift the mesh so its minimum bounds start at the current offset
      # This ensures tight packing
      x_shift = curr_x_offset - x_min
      y_shift = curr_y_offset - y_min

      if self._env.cfg['sim']['up_axis'] == 1:
        env_origin = [x_shift, y_shift, 0.0]
      elif self._env.cfg['sim']['up_axis'] == 0:
        env_origin = [x_shift, 0.0, y_shift]
      else:
        raise ValueError
      env_origin = np.array(env_origin, dtype=np.float32)
      env_origins.append(env_origin)

      # Compute the actual extent after placing the mesh
      placed_x_max = x_max + x_shift
      placed_y_max = y_max + y_shift

      # Track the maximum x extent in this row
      row_max_x_extent = max(row_max_x_extent, placed_x_max)

      # Update offsets for the next mesh
      if (i + 1) % num_rows == 0 and i < len(mesh_keys_repeated) - 1:
        # Next mesh will start a new row
        curr_x_offset = row_max_x_extent + buffer
        curr_y_offset = 0.0
        row_max_x_extent = 0.0
      else:
        # Next mesh will be in the same row
        curr_y_offset = placed_y_max + buffer

      # mesh = self._terrain.get_mesh(scene, mesh)
      vertices = env_origin[None] + mesh.vertices
      mesh_params = gymapi.TriangleMeshParams()
      mesh_params.static_friction = self._env.cfg['terrain']['static_friction']
      mesh_params.dynamic_friction = self._env.cfg['terrain']['dynamic_friction']
      mesh_params.restitution = self._env.cfg['terrain']['restitution']
      mesh_params.nb_vertices = vertices.shape[0]
      mesh_params.nb_triangles = mesh.triangles.shape[0]
      self._env.gym.add_triangle_mesh(
        self._env.sim,
        vertices.flatten(order='C'),
        mesh.triangles.flatten(order='C'),
        mesh_params,
      )

      vertices_offset = (
        0 if len(all_vertices) == 0 else np.concatenate(all_vertices).shape[0]
      )
      all_triangles.append(mesh.triangles + vertices_offset)
      all_vertices.append(vertices)

      self.all_vertices_orig[i // self.num_mesh_repeats][i % self.num_mesh_repeats] = (
        np.array(vertices)
      )
      self.all_triangles_orig[i // self.num_mesh_repeats][i % self.num_mesh_repeats] = (
        np.array(mesh.triangles)
      )

    self.terrain_mesh = warp_utils.convert_to_wp_mesh(
      np.concatenate(all_vertices), np.concatenate(all_triangles), self._env.device
    )

    self.construct_trajectory_arrays(np.array(env_origins))

  def _apply_mesh_repeat(self, x: np.ndarray):
    return np.repeat(x, self.num_mesh_repeats, axis=0)

  def construct_trajectory_arrays(self, env_origins: np.ndarray):
    # Assign different env origins, cam_trans, quat, and offsets to each environment.
    cam_trans_orig = np.array(
      list(self._terrain.get_value('cam_trans_padded').values())
    )
    cam_quat_xyzw_orig = np.array(
      list(self._terrain.get_value('cam_quat_xyzw_padded').values())
    )
    # Repeat_interleave the number of repetitions.
    cam_trans_orig = self._apply_mesh_repeat(cam_trans_orig)
    cam_quat_xyzw_orig = self._apply_mesh_repeat(cam_quat_xyzw_orig)
    env_origins_z0 = np.pad(
      env_origins[:, :2], ((0, 0), (0, 1)), mode='constant', constant_values=0.0
    )
    cam_trans_orig = cam_trans_orig + env_origins_z0[:, None, :]

    num_meshes_with_repeat = len(self._terrain.mesh_keys) * self.num_mesh_repeats

    # Repeat environments evenly.
    base_repeat = max(self._env.num_envs // num_meshes_with_repeat, 1)
    remainder = (
      self._env.num_envs % num_meshes_with_repeat
      if self._env.num_envs > num_meshes_with_repeat
      else 0
    )
    repeat_counts = [base_repeat] * num_meshes_with_repeat
    for i in range(remainder):
      repeat_counts[i] += 1
    repeat_counts = np.array(repeat_counts)

    def repeat(x, mesh_repeat=False):
      x = np.array(x)
      if mesh_repeat:
        x = self._apply_mesh_repeat(x)
      x = np.repeat(x, repeat_counts, axis=0)
      if x.shape[0] > self._env.num_envs:
        x = x[: self._env.num_envs]
      assert x.shape[0] == self._env.num_envs
      return x

    # Repeat counts per mesh.
    self.repeat_counts_per_mesh = repeat_counts
    self.repeat_counts_per_scene = np.array(
      [np.sum(x) for x in np.split(repeat_counts, self.num_meshes)]
    )
    self.repeats_cumsum_per_scene = np.cumsum(self.repeat_counts_per_scene)

    # Use warp to get ground position at each camera.
    directions = (
      np.array([0, 0, -1])[None, None]
      .repeat(cam_trans_orig.shape[0], axis=0)
      .repeat(cam_trans_orig.shape[1], axis=1)
    )
    directions_torch = math_utils.to_torch(
      directions, device=self._env.device, requires_grad=False
    )
    cam_trans_orig_torch = math_utils.to_torch(
      cam_trans_orig, device=self._env.device, requires_grad=False
    )
    ground_positions_world_frame = warp_utils.ray_cast(
      cam_trans_orig_torch.view(-1, 3),
      directions_torch.reshape(-1, 3),
      self.terrain_mesh,
    )
    ground_positions_world_frame = (
      ground_positions_world_frame.view(*cam_trans_orig_torch.shape).cpu().numpy()
    )
    ground_positions_world_frame = (
      ground_positions_world_frame - env_origins_z0[:, None, :]
    )
    self.ground_positions = math_utils.to_torch(
      repeat(ground_positions_world_frame),
      device=self._env.device,
      requires_grad=False,
    )

    self.cam_trans_viz = math_utils.to_torch(
      cam_trans_orig, device=self._env.device, requires_grad=False
    )
    self.cam_quat_xyzw_viz = math_utils.to_torch(
      cam_quat_xyzw_orig, device=self._env.device, requires_grad=False
    )

    self.scenes = repeat([s[0] for s in self._terrain.mesh_keys], mesh_repeat=True)
    self.cam_trans = math_utils.to_torch(
      repeat(
        list(self._terrain.get_value('cam_trans_padded').values()), mesh_repeat=True
      ),
      device=self._env.device,
      requires_grad=False,
    )
    self.cam_quat_xyzw = math_utils.to_torch(
      repeat(
        list(self._terrain.get_value('cam_quat_xyzw_padded').values()), mesh_repeat=True
      ),
      device=self._env.device,
      requires_grad=False,
    )
    self.ig_to_orig_rot = math_utils.to_torch(
      repeat(
        list(self._terrain.get_value('ig_to_orig_rot').values()), mesh_repeat=True
      ),
      device=self._env.device,
      requires_grad=False,
    )
    self.cam_offset = math_utils.to_torch(
      repeat(list(self._terrain.get_value('cam_offset').values()), mesh_repeat=True),
      device=self._env.device,
      requires_grad=False,
    )
    self.valid_pose_start_idxs = math_utils.to_torch(
      repeat(
        list(self._terrain.get_value('valid_pose_start_idxs').values()),
        mesh_repeat=True,
      ),
      device=self._env.device,
      requires_grad=False,
    )
    self.env_origins = math_utils.to_torch(
      repeat(env_origins),
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
    return self._env.get_camera_link_state()[:, :3], self._env.get_camera_link_state()[
      :, 3:7
    ]

  def get_cam_link_velocity_world_frame(self):
    cam_link_lin_vel = self._env.get_camera_link_state()[:, 7:10]
    cam_link_ang_vel = self._env.get_camera_link_state()[:, 10:13]
    return cam_link_lin_vel, cam_link_ang_vel

  def get_cam_pose_world_frame(self):
    cam_link_trans, cam_link_quat = self.get_cam_link_pose_world_frame()
    # Apply xyz offset in the local robot frame.
    cam_trans = cam_link_trans + math_utils.quat_apply(cam_link_quat, self.local_offset)
    # cam_quat = math_utils.quat_mul(cam_link_quat, self.cam_rpy_offset.expand(cam_link_quat.shape[0], -1))

    cam_quat_offset = math_utils.quat_mul(
      math_utils.quat_mul(
        math_utils.quat_from_x_rot(self.local_rpy_offset[:, 0]),
        math_utils.quat_from_y_rot(self.local_rpy_offset[:, 1]),
      ),
      math_utils.quat_from_z_rot(self.local_rpy_offset[:, 2]),
    ).detach()
    cam_quat = math_utils.quat_mul(cam_link_quat, cam_quat_offset)
    return cam_trans, cam_quat

  def get_cam_velocity_world_frame(self):
    cam_link_lin_vel, cam_link_ang_vel = self.get_cam_link_velocity_world_frame()
    _, cam_link_quat = self.get_cam_link_pose_world_frame()

    # Angular velocity remains the same after offset
    cam_ang_vel = cam_link_ang_vel

    # Linear velocity needs to account for the offset
    # v = v_link + ω × r, where r is the offset vector in world frame
    # Using cross product of angular velocity with position offset
    offset_world = math_utils.quat_apply(cam_link_quat, self.local_offset)
    velocity_from_rotation = torch.cross(cam_link_ang_vel, offset_world, dim=-1)
    cam_lin_vel = cam_link_lin_vel + velocity_from_rotation
    return cam_lin_vel, cam_ang_vel

  def get_cam_link_pose_local_frame(self):
    # The frame local to each environment.
    cam_trans, cam_quat = self.get_cam_link_pose_world_frame()
    cam_trans = cam_trans - self.env_origins
    return cam_trans, cam_quat

  def mesh_name_from_id(self, mesh_id):
    return '/'.join(self._terrain.mesh_keys[mesh_id])

  def mesh_id_from_name(self, mesh_name):
    mesh_name_split = mesh_name.split('/')
    mesh_name_index = ('/'.join(mesh_name_split[:-1]), mesh_name_split[-1])
    return self._terrain.mesh_keys.index(mesh_name_index)

  @property
  def num_meshes(self):
    return self._terrain.num_meshes

  @property
  def num_mesh_repeats(self):
    return self._env.cfg['terrain']['num_mesh_repeats']

  @property
  def mesh_names(self):
    return [self.mesh_name_from_id(i) for i in range(self.num_meshes)]

  def robots_in_mesh_id(self, mesh_id):
    return self.repeat_counts_per_scene[mesh_id]

  def mesh_id_for_env_id(self, env_id):
    assert env_id < self._env.num_envs, f'Env id {env_id} is out of range'
    if env_id >= self._env.num_envs:
      return -1
    # Find which mesh instance (across all meshes and repetitions)
    mesh_instance_idx = np.searchsorted(
      np.cumsum(self.repeat_counts_per_mesh), env_id, side='right'
    )
    # Convert to (mesh_id, rep_id)
    mesh_id = mesh_instance_idx // self.num_mesh_repeats
    rep_id = mesh_instance_idx % self.num_mesh_repeats
    return mesh_id, rep_id

  def get_cam_viz_for_env_id(self, env_id):
    mesh_id, rep_idx = self.mesh_id_for_env_id(env_id)
    cam_viz_idx = mesh_id * self.num_mesh_repeats + rep_idx
    return self.cam_trans_viz[cam_viz_idx], self.cam_quat_xyzw_viz[cam_viz_idx]

  def env_ids_for_mesh_id(self, mesh_id):
    assert mesh_id < self.num_meshes, f'Mesh id {mesh_id} is out of range'
    start = np.sum(self.repeat_counts_per_scene[:mesh_id])
    if start >= self._env.num_envs:
      return torch.tensor([], device=self._env.device, dtype=torch.int32)
    end = min(start + self.repeat_counts_per_scene[mesh_id], self._env.num_envs)
    return torch.arange(start, end, device=self._env.device, dtype=torch.int32)

  def _get_nearest_traj_idx(
    self,
    monotonic: Optional[torch.Tensor],
    look_window: int,
    near_state_idx: Optional[torch.Tensor] = None,
  ):
    curr_cam_trans, _ = self.get_cam_link_pose_local_frame()
    trans_difference = curr_cam_trans[:, None] - self.cam_trans
    distance = torch.norm(trans_difference, dim=-1)
    if near_state_idx is not None:
      # Create a mask for valid indices
      num_envs, num_poses = distance.shape
      indices = torch.arange(num_poses, device=distance.device)[None]
      indices = indices.repeat_interleave(num_envs, 0)

      lower_idx = near_state_idx[:, None]
      upper_idx = near_state_idx[:, None]
      if monotonic is None:
        lower_idx = lower_idx - look_window
        upper_idx = upper_idx + look_window
      else:
        # True is increasing, False is decreasing.
        upper_idx = torch.where(monotonic[:, None], upper_idx + look_window, upper_idx)
        lower_idx = torch.where(~monotonic[:, None], lower_idx - look_window, lower_idx)

      mask = (indices < lower_idx) | (indices > upper_idx)

      # Apply mask by setting invalid distances to infinity
      masked_distance = distance.clone()
      masked_distance[mask] = float('inf')

      min_distance_idx = torch.argmin(masked_distance, dim=1)
    else:
      min_distance_idx = torch.argmin(distance, dim=1)

    return min_distance_idx

  def _get_nearest_traj_pose(
    self,
    monotonic: Optional[torch.Tensor] = None,
    look_window: int = 4,
    near_state_idx: Optional[torch.Tensor] = None,
  ):
    nearest_traj_idx = self._get_nearest_traj_idx(
      monotonic, look_window, near_state_idx
    )
    nearest_traj_pos, nearest_traj_quat = self._retrieve_from_trajectory(
      self.cam_trans, self.cam_quat_xyzw, nearest_traj_idx
    )
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
    return math_utils.quat_mul(
      cam_quat, rl_utils.broadcast_left(self.robot_frame_transform, cam_quat)
    )

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
    cam_trans, cam_quat = self._retrieve_from_trajectory(
      cam_trans_subs, cam_quat_xyzw_subs, state_idx
    )
    robot_quat = self.to_robot_frame(cam_quat)
    return cam_trans, robot_quat

  def _retrieve_from_trajectory(self, trans, quat, index):
    """Retrieve pose and trans from trajectory at given index.

    Args:
        trans (torch.Tensor): Pose trajectory. (N, T, 3)
        quat (torch.Tensor): Quaternion trajectory. (N, T, 4)
        index (torch.Tensor): Index to retrieve from trajectory. (N)

    Returns:
        trans (torch.Tensor): Retrieved pose. (N, 3)
        quat (torch.Tensor): Retrieved quaternion. (N, 4)
    """
    trans_idx = index.unsqueeze(1).unsqueeze(2).expand(-1, 1, trans.shape[-1])
    quat_idx = index.unsqueeze(1).unsqueeze(2).expand(-1, 1, quat.shape[-1])
    trans = torch.gather(trans, dim=1, index=trans_idx).squeeze(1)
    quat = torch.gather(quat, dim=1, index=quat_idx).squeeze(1)
    return trans, quat

  def debug_vis(self, env):
    if self.axis_geom is None:
      self.axis_geom = visualization_geometries.BatchWireframeAxisGeometry(
        np.prod(self.cam_trans_viz.shape[:2]), 0.25, 0.005, 16
      )
    if self.closest_axis_geom is None:
      self.closest_axis_geom = visualization_geometries.BatchWireframeAxisGeometry(
        self._env.num_envs,
        0.3,
        0.01,
        24,
        color_x=(1, 0, 0),
        color_y=(1, 0, 0),
        color_z=(1, 0, 0),
      )
    if self.target_axis_geom is None:
      self.target_axis_geom = visualization_geometries.BatchWireframeAxisGeometry(
        self._env.num_envs,
        0.3,
        0.01,
        24,
        color_x=(0, 1, 0),
        color_y=(0, 1, 0),
        color_z=(0, 1, 0),
      )
    if self.velocity_geom is None:
      self.velocity_geom = visualization_geometries.BatchWireframeAxisGeometry(
        self._env.num_envs,
        0.3,
        0.01,
        32,
        color_x=(1, 1, 1),
        color_y=(1, 1, 1),
        color_z=(0, 0, 0),
      )
    if self.heading_geom is None:
      self.heading_geom = visualization_geometries.BatchWireframeAxisGeometry(
        self._env.num_envs, 0.2, 0.01, 32, color_x=(1, 1, 0)
      )

    closest_trans, closest_quat = self._retrieve_from_trajectory(
      self.cam_trans, self.cam_quat_xyzw, self.state_idx
    )
    closest_trans = closest_trans + self.env_origins
    closest_quat = self.to_robot_frame(closest_quat)

    target_trans, target_quat = self._retrieve_from_trajectory(
      self.cam_trans, self.cam_quat_xyzw, self.target_idx
    )
    target_trans = target_trans + self.env_origins
    target_quat = self.to_robot_frame(target_quat)

    # Draw camera trajectory.
    if self._env.selected_environment < 0:
      self.axis_geom.draw(
        self.cam_trans_viz.reshape(-1, 3),
        self.cam_quat_xyzw_viz.reshape(-1, 4),
        self._env.gym,
        self._env.viewer,
        self._env.envs[0],
      )
      self.closest_axis_geom.draw(
        closest_trans.reshape(-1, 3),
        closest_quat.reshape(-1, 4),
        self._env.gym,
        self._env.viewer,
        self._env.envs[0],
      )
      self.target_axis_geom.draw(
        target_trans.reshape(-1, 3),
        target_quat.reshape(-1, 4),
        self._env.gym,
        self._env.viewer,
        self._env.envs[0],
      )
    else:
      mesh_idx, _ = self.mesh_id_for_env_id(self._env.selected_environment)
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
      self.closest_axis_geom.draw(
        closest_trans.reshape(-1, 3),
        closest_quat.reshape(-1, 4),
        self._env.gym,
        self._env.viewer,
        self._env.envs[0],
        only_render_selected=self._env.selected_environment,
      )
      self.target_axis_geom.draw(
        target_trans.reshape(-1, 3),
        target_quat.reshape(-1, 4),
        self._env.gym,
        self._env.viewer,
        self._env.envs[0],
        only_render_selected=self._env.selected_environment,
      )

    # Draw velocity command.
    velocity_quat = self._env.root_states[:, 3:7]
    velocity_trans = self._env.root_states[:, :3] + math_utils.quat_apply(
      velocity_quat,
      torch.tensor([0, 0, 0.25], device=self._env.device)[None].repeat(
        self._env.num_envs, 1
      ),
    )
    axis_scales = (
      self.velocity_command.cpu().numpy() / self._env.command_ranges['lin_vel'][1]
    )
    axis_scales = np.pad(
      axis_scales, ((0, 0), (0, 1)), mode='constant', constant_values=0
    )
    axis_scales = math_utils.to_torch(
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
    heading_trans = self._env.root_states[:, :3] + math_utils.quat_apply(
      self._env.root_states[:, 3:7],
      torch.tensor([0, 0, 0.2], device=self._env.device)[None].repeat(
        self._env.num_envs, 1
      ),
    )
    heading_quat = math_utils.quat_from_euler_xyz(
      torch.zeros_like(self.heading_command),
      torch.zeros_like(self.heading_command),
      self.heading_command,
    )
    axis_scales = math_utils.to_torch(
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


class CommandManager(abc.ABC):
  def __init__(self, env, cfg: Dict):
    self.cfg = cfg
    self.env = env
    self.device = env.device

  @abc.abstractmethod
  def command_space(self) -> space.Space:
    pass

  @abc.abstractmethod
  def command_termination_condition(
    self, scene_manager: GaussianSceneManager
  ) -> torch.Tensor:
    pass

  @abc.abstractmethod
  def reset(
    self, env_ids, scene_manager: GaussianSceneManager
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    pass

  @abc.abstractmethod
  def update(self, env_ids, scene_manager: GaussianSceneManager) -> torch.Tensor:
    pass

  @abc.abstractmethod
  def ignore_command_mask(self, scene_manager: GaussianSceneManager) -> torch.Tensor:
    pass

  @abc.abstractmethod
  def check_completed(
    self, env_ids, scene_manager: GaussianSceneManager
  ) -> Dict[str, list]:
    pass

  @abc.abstractmethod
  def update_curriculum(
    self, scene_manager: GaussianSceneManager, completion_mean: Dict[str, float]
  ) -> Dict[str, float]:
    pass


class VelocityCommandManager(CommandManager):
  TARGET_IDX_OFFSET = 5  # TODO: Make this configurable.

  def __init__(self, env, cfg: Dict):
    super().__init__(env, cfg)

    self.task_cfg = cfg['commands']['velocity']

    self.command_scale = torch.zeros(
      self.env.num_envs,
      1,
      device=self.env.device,
      dtype=torch.float32,
      requires_grad=False,
    )
    self.lin_vel_range = self.task_cfg['lin_vel_range']
    self.ang_vel_yaw_range = self.task_cfg['ang_vel_yaw_range']

    # Current velocity and heading commands.
    self.velocity_command = torch.zeros(
      self.env.num_envs, 2, device=self.device, dtype=torch.float32, requires_grad=False
    )
    self.heading_command = torch.zeros(
      self.env.num_envs, device=self.device, dtype=torch.float32, requires_grad=False
    )
    self.heading_quat = torch.zeros(
      self.env.num_envs, 4, device=self.device, dtype=torch.float32, requires_grad=False
    )
    self.yaw_vel_command = torch.zeros(
      self.env.num_envs, device=self.device, dtype=torch.float32, requires_grad=False
    )

    # Index of the current and target robot states in the camera trajectory.
    self.state_idx = torch.zeros(
      self.env.num_envs, device=self.device, dtype=torch.int64, requires_grad=False
    )
    self.target_idx = torch.zeros(
      self.env.num_envs, device=self.device, dtype=torch.int64, requires_grad=False
    )

    self.still_envs = torch.zeros(
      self.env.num_envs, dtype=torch.bool, device=self.device
    )

    self.is_increasing = torch.ones(
      self.env.num_envs, dtype=torch.bool, device=self.device
    )
    self.success_envs = torch.zeros(
      self.env.num_envs, dtype=torch.bool, device=self.device
    )

    # Queues for tracking completion stats.
    self.started_queue = defaultdict(lambda: [])
    self.ended_queue = defaultdict(lambda: [])
    self.success_queue = defaultdict(lambda: [])

  def command_space(self) -> space.Space:
    return space.Space(
      dtype=torch.float32,
      shape=(3,),
    )

  def command_termination_condition(self, scene_manager: GaussianSceneManager):
    # Check if robot is too far from camera trajectory (Indicative of poor rendering).
    curr_cam_link_trans, curr_cam_link_quat = (
      scene_manager.get_cam_link_pose_local_frame()
    )
    nearest_cam_trans, nearest_cam_quat, _ = scene_manager._get_nearest_traj_pose(
      monotonic=self.is_increasing,
      near_state_idx=self.state_idx,
      look_window=VelocityCommandManager.TARGET_IDX_OFFSET,
    )
    nearest_robot_quat = scene_manager.to_robot_frame(nearest_cam_quat)

    distance = torch.norm((curr_cam_link_trans - nearest_cam_trans)[:, :2], dim=-1)
    distance_exceeded = distance > self.task_cfg['max_traj_pos_distance']

    quat_difference = math_utils.quat_mul(
      curr_cam_link_quat, math_utils.quat_conjugate(nearest_robot_quat)
    )
    _, _, yaw = math_utils.get_euler_xyz(quat_difference)
    yaw_distance = torch.abs(math_utils.wrap_to_pi(yaw))
    yaw_exceeded = yaw_distance > self.task_cfg['max_traj_yaw_distance_rad']
    truncate_condition = torch.logical_or(distance_exceeded, yaw_exceeded)
    if self.task_cfg['truncate_on_success']:
      truncate_condition = torch.logical_or(truncate_condition, self.success_envs)
    return truncate_condition

  def _reset_scales(self, env_ids):
    self.command_scale[env_ids] = math_utils.torch_rand_float(
      *self.lin_vel_range,
      (len(env_ids), 1),
      device=self.device,
    )
    self.still_envs[env_ids] = (
      torch.rand(len(env_ids), device=self.device) < self.task_cfg['still_proportion']
    )

  def reset(
    self, env_ids, scene_manager: GaussianSceneManager
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    self.is_increasing[env_ids] = True
    self.success_envs[env_ids] = False
    self._reset_scales(env_ids)

    # Sample a random camera position and orientation from the camera trajectories.
    if self.task_cfg['use_ground_positions']:
      cam_trans_subs = scene_manager.ground_positions[env_ids]
    else:
      cam_trans_subs = scene_manager.cam_trans[env_ids]
    cam_quat_xyzw_subs = scene_manager.cam_quat_xyzw[env_ids]
    low = scene_manager.valid_pose_start_idxs[env_ids].to(torch.float32)
    high = scene_manager.valid_pose_start_idxs[env_ids].to(torch.float32) + 2
    state_idx = math_utils.sample_uniform(low, high, (len(env_ids),), self.device).to(
      torch.int64
    )
    self.state_idx[env_ids] = state_idx
    pos, quat = scene_manager._retrieve_from_trajectory(
      cam_trans_subs, cam_quat_xyzw_subs, state_idx
    )
    quat = scene_manager.to_robot_frame(quat)
    pos += self.env.env_origins[env_ids]
    pos += self.env.base_init_state[:3][None]
    return pos, quat

  def update(self, env_ids, scene_manager: GaussianSceneManager):
    resample_command_env_ids = (
      (
        self.env.episode_length_buf
        % int(self.task_cfg['resampling_time'] / self.env.dt)
        == 0
      )
      .nonzero(as_tuple=False)
      .flatten()
    )
    self._reset_scales(resample_command_env_ids)

    _, _, nearest_idx = scene_manager._get_nearest_traj_pose(
      monotonic=self.is_increasing,
      near_state_idx=self.state_idx,
      look_window=VelocityCommandManager.TARGET_IDX_OFFSET,
    )
    # Update the state index to the nearest camera trajectory.
    self.state_idx[env_ids] = nearest_idx[env_ids]

    reached_end = torch.logical_and(
      nearest_idx >= (scene_manager.cam_trans.shape[1] - 2), self.is_increasing
    )
    reached_start = torch.logical_and(nearest_idx <= 2, ~self.is_increasing)
    switch_direction = torch.logical_or(reached_end, reached_start)
    self.is_increasing[switch_direction] = ~self.is_increasing[switch_direction]
    if self.task_cfg['success_on_loop']:
      self.success_envs = torch.logical_or(self.success_envs, reached_start)
    else:
      self.success_envs = torch.logical_or(self.success_envs, reached_end)
    new_target_idx = torch.where(
      self.is_increasing,
      nearest_idx + VelocityCommandManager.TARGET_IDX_OFFSET,
      nearest_idx - VelocityCommandManager.TARGET_IDX_OFFSET,
    )
    self.target_idx = torch.clamp(
      new_target_idx, 0, scene_manager.cam_trans.shape[1] - 1
    )

    # Choose a target pose moderately far away. If nearest_idx + 1 is chosen, target
    # velocities fall to 0 as the robot approaches this target.
    target_traj_trans, target_traj_quat = scene_manager._retrieve_from_trajectory(
      scene_manager.cam_trans, scene_manager.cam_quat_xyzw, self.target_idx
    )
    target_traj_quat = scene_manager.to_robot_frame(target_traj_quat)
    _, _, yaw = math_utils.get_euler_xyz(target_traj_quat)
    yaw = torch.where(self.is_increasing, yaw, yaw + torch.pi)
    self.heading_command[:] = yaw
    self.heading_quat[:] = target_traj_quat

    curr_cam_link_trans, _ = scene_manager.get_cam_link_pose_local_frame()
    pos_delta = target_traj_trans - curr_cam_link_trans
    pos_delta_local = math_utils.quat_rotate_inverse(
      self.env.root_states[:, 3:7], pos_delta
    )[:, :2]
    pos_delta_norm = pos_delta_local / torch.norm(pos_delta_local, dim=-1, keepdim=True)
    pos_delta_norm *= self.command_scale
    self.velocity_command[:] = pos_delta_norm

    _, _, heading = math_utils.get_euler_xyz(self.env.base_quat)

    quat_difference = math_utils.quat_mul(
      self.heading_quat, math_utils.quat_conjugate(self.env.base_quat)
    )
    _, _, yaw = math_utils.get_euler_xyz(quat_difference)
    yaw = math_utils.wrap_to_pi(yaw)
    self.yaw_vel_command[:] = torch.clip(yaw, *self.ang_vel_yaw_range)

    ignore_command_mask = self.ignore_command_mask(scene_manager)
    self.velocity_command[ignore_command_mask] = 0.0
    self.yaw_vel_command[ignore_command_mask] = 0.0
    self.velocity_command[:] *= (
      torch.abs(self.velocity_command) >= self.task_cfg['small_lin_vel_threshold']
    ).float()
    self.yaw_vel_command[:] *= (
      torch.abs(self.yaw_vel_command) >= self.task_cfg['small_ang_vel_threshold']
    ).float()
    return torch.cat([self.velocity_command, self.yaw_vel_command[..., None]], dim=-1)

  def ignore_command_mask(self, scene_manager: GaussianSceneManager):
    small_lin_vel_mask = (
      torch.norm(self.velocity_command, dim=-1)
      < self.task_cfg['small_lin_vel_threshold']
    )
    small_ang_vel_mask = (
      torch.abs(self.yaw_vel_command) < self.task_cfg['small_ang_vel_threshold']
    )
    return torch.logical_or(
      torch.logical_and(small_lin_vel_mask, small_ang_vel_mask), self.still_envs
    )

  def check_completed(self, env_ids, scene_manager: GaussianSceneManager):
    curr_completion_stats = {mesh_name: [] for mesh_name in scene_manager.mesh_names}
    for env_id in env_ids:
      env_id = int(env_id)
      mesh_id, _ = scene_manager.mesh_id_for_env_id(env_id)
      mesh_name = scene_manager.mesh_name_from_id(mesh_id)

      num_envs_in_mesh = len(scene_manager.env_ids_for_mesh_id(mesh_id))

      if num_envs_in_mesh == 0:
        continue

      env_started = env_id in self.started_queue[mesh_name]
      env_ended = env_id in self.ended_queue[mesh_name]
      env_completed = env_id in self.success_queue[mesh_name]

      if env_started and not env_ended:
        # Env has started started and not ended yet for this logging period.
        self.ended_queue[mesh_name].append(env_id)
        if self.success_envs[int(env_id)] and not env_completed:
          self.success_queue[mesh_name].append(env_id)
      elif not env_started and not env_ended:
        self.started_queue[mesh_name].append(env_id)

      if (
        len(self.started_queue[mesh_name])
        == len(self.ended_queue[mesh_name])
        == num_envs_in_mesh
      ):
        # Clear the queue and log the completion stats.
        curr_completion_stats[mesh_name] = np.isin(
          self.started_queue[mesh_name], self.success_queue[mesh_name]
        ).tolist()
        self.started_queue[mesh_name] = []
        self.ended_queue[mesh_name] = []
        self.success_queue[mesh_name] = []

    return curr_completion_stats

  def update_curriculum(
    self, scene_manager: GaussianSceneManager, completion_mean: Dict[str, float]
  ):
    return {}

  def _reward_vel_tracking(self, tracking_sigma: float):
    # lin_vel_error = torch.sum(torch.square(self.velocity_command - self.env.filtered_lin_vel[:, :2]), dim=1)
    # return torch.exp(-lin_vel_error / tracking_sigma)
    vel_error = torch.square(self.velocity_command - self.env.filtered_lin_vel[:, :2])
    vel_error = torch.exp(-vel_error / tracking_sigma)
    return vel_error.sum(dim=-1)

  def _reward_yaw_vel_tracking(self, tracking_sigma: float):
    ang_vel_error = torch.square(self.yaw_vel_command - self.env.filtered_ang_vel[:, 2])
    ang_vel_error = torch.exp(-ang_vel_error / tracking_sigma)
    return ang_vel_error


class GoalCommandManager(CommandManager):
  def __init__(self, env, cfg):
    super().__init__(env, cfg)

    self.task_cfg = cfg['commands']['goal']

    self.start_position = torch.zeros(
      self.env.num_envs, 3, device=self.device, dtype=torch.float32, requires_grad=False
    )
    self.start_orientation = torch.zeros(
      self.env.num_envs, 4, device=self.device, dtype=torch.float32, requires_grad=False
    )

    self.goal_position = torch.zeros(
      self.env.num_envs, 3, device=self.device, dtype=torch.float32, requires_grad=False
    )
    self.goal_orientation = torch.zeros(
      self.env.num_envs, 4, device=self.device, dtype=torch.float32, requires_grad=False
    )
    self.remaining_steps = torch.zeros(
      self.env.num_envs, device=self.device, dtype=torch.int64, requires_grad=False
    )

    self.goal_eval_steps = int(self.task_cfg['goal_evaluation_time_s'] / self.env.dt)

    if self.task_cfg['distance_curriculum']:
      self.distance_schedule = self.task_cfg['distance_schedule']
    else:
      self.distance_schedule = [
        self.task_cfg['goal_distance'],
      ] * 2
    self.goal_distance = torch.full(
      (self.env.num_envs,),
      self.distance_schedule[0],
      device=self.device,
      dtype=torch.float32,
      requires_grad=False,
    )

    # Queues for tracking completion stats.
    self.started_queue = defaultdict(lambda: [])
    self.ended_queue = defaultdict(lambda: [])
    self.success_queue = defaultdict(lambda: [])

  def command_space(self):
    # Goal position (2 or 3), goal orientation (1), and remaining time (1).
    return space.Space(
      dtype=torch.float32,
      shape=(5,),
    )

  def command_termination_condition(
    self, scene_manager: GaussianSceneManager
  ) -> torch.Tensor:
    # Which task-related termination conditions should be checked?

    # Should check if the robot is out of the mesh, but probably not here, instead in the env.
    return self.remaining_steps == 0

  def _reset_goal(self, env_ids, scene_manager: GaussianSceneManager):
    self.goal_position[env_ids], self.goal_orientation[env_ids] = self._sample_pose(
      env_ids, scene_manager, is_goal=True
    )
    remaining_time_s = math_utils.sample_uniform(
      *self.task_cfg['goal_window_s'], (len(env_ids),), self.device
    )
    self.remaining_steps[env_ids] = (remaining_time_s / self.env.dt).to(torch.int64)

  def reset(self, env_ids, scene_manager: GaussianSceneManager):
    # Sample new goal position + orientation and remaining time.
    # Sample goal position + orientation from camera trajectory.
    self.start_position[env_ids], self.start_orientation[env_ids] = self._sample_pose(
      env_ids, scene_manager
    )
    self._reset_goal(env_ids, scene_manager)
    return self.start_position[env_ids], self.start_orientation[env_ids]

  def update(self, env_ids, scene_manager: GaussianSceneManager) -> torch.Tensor:
    # Compute the relative position and heading (yaw) to the goal, and pass in remaining time.
    # Get current robot position and orientation
    curr_pos = self.env.root_states[:, :3]
    curr_quat = self.env.root_states[:, 3:7]

    # Compute relative position to goal
    rel_pos_world = self.goal_position - curr_pos
    rel_pos_robot = math_utils.quat_rotate_inverse(curr_quat, rel_pos_world)

    # Compute relative heading (yaw) to goal
    goal_quat = self.goal_orientation
    quat_diff = math_utils.quat_mul(goal_quat, math_utils.quat_conjugate(curr_quat))
    _, _, rel_yaw = math_utils.get_euler_xyz(quat_diff)
    rel_yaw = math_utils.wrap_to_pi(rel_yaw)

    # Get remaining time as fraction
    remaining_time_s = self.remaining_steps.float() * self.env.dt

    # Combine into command: [rel_x, rel_y, rel_z, rel_yaw, remaining_time]
    command = torch.cat(
      [rel_pos_robot, rel_yaw[..., None], remaining_time_s[..., None]], dim=-1
    )

    # Decrement remaining steps
    self.remaining_steps -= 1

    assert (self.remaining_steps >= 0).all()

    return command

  def ignore_command_mask(self, scene_manager: GaussianSceneManager) -> torch.Tensor:
    # Should we ignore the command? Probably never?
    return torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)

  def check_completed(
    self, env_ids, scene_manager: GaussianSceneManager
  ) -> Dict[str, list]:
    goal_reached = self._check_goal_reached()

    curr_completion_stats = {mesh_name: [] for mesh_name in scene_manager.mesh_names}
    for env_id in env_ids:
      env_id = int(env_id)
      mesh_id, _ = scene_manager.mesh_id_for_env_id(env_id)
      mesh_name = scene_manager.mesh_name_from_id(mesh_id)

      num_envs_in_mesh = len(scene_manager.env_ids_for_mesh_id(mesh_id))

      if num_envs_in_mesh == 0:
        continue

      env_started = env_id in self.started_queue[mesh_name]
      env_ended = env_id in self.ended_queue[mesh_name]
      env_completed = env_id in self.success_queue[mesh_name]

      if env_started and not env_ended:
        # Env has started started and not ended yet for this logging period.
        self.ended_queue[mesh_name].append(env_id)
        if goal_reached[int(env_id)] and not env_completed:
          self.success_queue[mesh_name].append(env_id)
      elif not env_started and not env_ended:
        self.started_queue[mesh_name].append(env_id)

      if (
        len(self.started_queue[mesh_name])
        == len(self.ended_queue[mesh_name])
        == num_envs_in_mesh
      ):
        # Clear the queue and log the completion stats.
        curr_completion_stats[mesh_name] = np.isin(
          self.started_queue[mesh_name], self.success_queue[mesh_name]
        ).tolist()
        self.started_queue[mesh_name] = []
        self.ended_queue[mesh_name] = []
        self.success_queue[mesh_name] = []

    return curr_completion_stats

  def update_curriculum(
    self, scene_manager: GaussianSceneManager, completion_mean: Dict[str, float]
  ):
    metrics = {}
    if self.task_cfg['distance_curriculum']:
      for mesh_name, completion in completion_mean.items():
        mesh_id = scene_manager.mesh_id_from_name(mesh_name)
        env_ids = scene_manager.env_ids_for_mesh_id(mesh_id)
        if completion >= self.task_cfg['completion_up_threshold']:
          self.goal_distance[env_ids] += self.task_cfg['distance_up_amount']
          self.goal_distance[env_ids] = torch.clamp(
            self.goal_distance[env_ids], *self.distance_schedule
          )
          self.started_queue[mesh_name] = []
          self.ended_queue[mesh_name] = []
          self.success_queue[mesh_name] = []
        metrics[mesh_name] = self.goal_distance[env_ids[0]].cpu().item()
    return metrics

  def _compute_pos_yaw_error(self):
    curr_pos, goal_pos = self.env.root_states[:, :3], self.goal_position
    if not self.task_cfg['goal_xyz']:
      curr_pos, goal_pos = curr_pos[:, :2], goal_pos[:, :2]
    pos_error = torch.norm(curr_pos - goal_pos, dim=-1)
    _, _, yaw_error = math_utils.get_euler_xyz(
      math_utils.quat_mul(
        self.env.root_states[:, 3:7], math_utils.quat_conjugate(self.goal_orientation)
      )
    )
    yaw_error = torch.abs(math_utils.wrap_to_pi(yaw_error))
    return pos_error, yaw_error

  def _check_goal_reached(self):
    pos_error, yaw_error = self._compute_pos_yaw_error()
    goal_reached = pos_error < self.task_cfg['goal_reached_pos_threshold']
    if not self.task_cfg['goal_reached_pos_only']:
      goal_reached = torch.logical_and(
        goal_reached, yaw_error < self.task_cfg['goal_reached_yaw_threshold']
      )
    return goal_reached

  def _reward_pos_tracking(self):
    is_goal_eval_time = self.remaining_steps <= self.goal_eval_steps
    pos_error, _ = self._compute_pos_yaw_error()
    pos_rew = 1.0 - 0.5 * pos_error
    return pos_rew * is_goal_eval_time

  def _reward_yaw_tracking(self):
    is_goal_eval_time = self.remaining_steps <= self.goal_eval_steps
    _, yaw_error = self._compute_pos_yaw_error()
    yaw_rew = 1 - 0.5 * yaw_error
    return yaw_rew * is_goal_eval_time

  def _reward_stand_at_target(self):
    goal_reached = self._check_goal_reached()
    joint_error = torch.norm(self.env.dof_pos - self.env.default_dof_pos, dim=-1)
    return joint_error * goal_reached

  def _reward_move_in_direction(self):
    vel_robot = self.env.filtered_lin_vel
    rel_pos_robot = math_utils.quat_rotate_inverse(
      self.env.root_states[:, 3:7], self.goal_position - self.env.root_states[:, :3]
    )

    # Normalize vectors
    vel_norm = torch.norm(vel_robot, dim=-1, keepdim=True)
    dir_norm = torch.norm(rel_pos_robot, dim=-1, keepdim=True)

    # Avoid division by zero
    vel_norm = torch.clamp(vel_norm, min=1e-8)
    dir_norm = torch.clamp(dir_norm, min=1e-8)

    vel_unit = vel_robot / vel_norm
    dir_unit = rel_pos_robot / dir_norm

    # Compute cosine similarity (dot product of unit vectors)
    reward = torch.sum(vel_unit * dir_unit, dim=-1)

    return reward

  def _reward_dont_wait(self, vel_threshold: float):
    goal_not_reached = ~self._check_goal_reached()
    not_moving = torch.norm(self.env.filtered_lin_vel[:, :2], dim=-1) < vel_threshold
    return (goal_not_reached * not_moving).float()

  def _sample_pose(
    self, env_ids, scene_manager: GaussianSceneManager, is_goal: bool = False
  ):
    if self.task_cfg['use_ground_positions']:
      cam_trans_subs = scene_manager.ground_positions[env_ids]
    else:
      cam_trans_subs = scene_manager.cam_trans[env_ids]
    cam_quat_xyzw_subs = scene_manager.cam_quat_xyzw[env_ids]

    cam_trans_subs += self.env.env_origins[env_ids][:, None, :]
    cam_trans_subs += self.env.base_init_state[:3][None, None]
    cam_quat_xyzw_subs = scene_manager.to_robot_frame(cam_quat_xyzw_subs)

    low = scene_manager.valid_pose_start_idxs[env_ids].to(torch.float32)
    high = torch.full_like(low, scene_manager.cam_trans.shape[1])
    state_idx = math_utils.sample_uniform(low, high, (len(env_ids),), self.device).to(
      torch.int64
    )

    if is_goal:
      distance = torch.norm(self.start_position[env_ids, None] - cam_trans_subs, dim=-1)
      distance = distance <= self.goal_distance[env_ids][..., None]
      mask = torch.arange(cam_trans_subs.shape[1], device=self.device)
      mask = mask[None].repeat(len(env_ids), 1)
      mask = mask < scene_manager.valid_pose_start_idxs[env_ids][..., None]
      distance[mask] = False
      state_idx = math_utils.sample_true_indices(distance, 1)

    pos, quat = scene_manager._retrieve_from_trajectory(
      cam_trans_subs, cam_quat_xyzw_subs, state_idx
    )
    if is_goal and self.task_cfg['random_heading_goal']:
      quat = math_utils.quat_from_euler_xyz(
        torch.zeros_like(quat[:, 0]),
        torch.zeros_like(quat[:, 1]),
        math_utils.sample_uniform(-np.pi, np.pi, (len(env_ids),), self.device),
      )
    return pos, quat
