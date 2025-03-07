import numpy as np
import viser.transforms as vtf
import os
import pickle
from typing import List, Union
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils.math import (
  wrap_to_pi,
  matrix_to_quaternion,
  quat_apply,
)
from isaacgym.torch_utils import (
  to_torch,
  quat_mul,
  get_euler_xyz,
  quat_from_euler_xyz,
  quat_conjugate,
  quat_rotate_inverse,
  torch_rand_float,
)
from isaacgym import gymapi
import torch
from legged_gym.teacher.sensors import (
  BatchWireframeAxisGeometry,
  GaussianSplattingRenderer,
)
import dataclasses


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


class GaussianTerrain:
  def __init__(self, cfg: LeggedRobotCfg, num_robots) -> None:
    self.cfg = cfg
    self.num_robots = num_robots
    self.type = self.cfg.terrain.mesh_type

    if self.cfg.sim.up_axis == 1:
      height_offset = [0, 0, self.cfg.terrain.height_offset]
    elif self.cfg.sim.up_axis == 0:
      height_offset = [0, self.cfg.terrain.height_offset, 0]
    else:
      raise ValueError
    self._height_offset = np.array(height_offset)[None]

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

  def _load_mesh(self, scene: str, filename: str):
    filepath = os.path.join(
      self.cfg.terrain.scene_root, scene, "meshes", filename
    )
    with open(filepath, "rb") as f:
      mesh_dict = pickle.load(f)

    cam_offset = np.array(mesh_dict["offset"])[0].astype(np.float32)
    vertices = np.array(mesh_dict["vertices"]).astype(np.float32)
    triangles = np.array(mesh_dict["triangles"]).astype(np.uint32)
    curr_cam_trans = (
      np.array(mesh_dict["cam_trans"]).astype(np.float32) + self._height_offset
    )

    return RawMesh(
      scene_name=scene,
      filename=filename,
      filepath=filepath,
      vertices=vertices,
      triangles=triangles,
      cam_trans=curr_cam_trans,
      cam_offset=cam_offset,
    )

  def _load_meshes(self):
    for scene_name in os.listdir(os.path.join(self.cfg.terrain.scene_root)):
      for filename in os.listdir(
        os.path.join(self.cfg.terrain.scene_root, scene_name, "meshes")
      ):
        self._mesh_dict[os.path.join(scene_name, filename)] = self._load_mesh(
          scene_name, filename
        )

  def _process_poses(self):
    max_length = max([m.cam_trans.shape[0] for m in self._mesh_dict.values()])

    for filepath, raw_mesh in self._mesh_dict.items():
      # Compute orientations from camera translations.
      delta_xy = raw_mesh.cam_trans[1:, :2] - raw_mesh.cam_trans[:-1, :2]
      delta_xyz = np.concatenate(
        [delta_xy, np.zeros_like(delta_xy[:, :-1])], axis=-1
      )
      x_component = delta_xyz / np.linalg.norm(
        delta_xyz, axis=-1, keepdims=True
      )
      z_component = np.zeros_like(x_component)
      z_component[:, 2] = 1
      y_component = np.cross(z_component, x_component)
      y_component = y_component / np.linalg.norm(
        y_component, axis=-1, keepdims=True
      )
      cam_rot = vtf.SO3.from_matrix(np.stack([x_component, y_component, z_component], axis=2))
      # Switch to OpenGL camera convention.
      cam_rot = cam_rot @ vtf.SO3.from_y_radians(np.pi / 2)
      cam_rot = cam_rot @ vtf.SO3.from_z_radians(-np.pi / 2)
      cam_quat_xyzw = cam_rot.as_quaternion_xyzw()

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


class GaussianSceneManager:
  # Loads meshes and assigns robots to each mesh. Also handles replacement of meshes from multiple scenes.
  # Provides classes for sampling commands along a camera path and detecting when the robot strays off the original path.
  # Also provides visualization tools for debugging.
  def __init__(self, env):
    self._env = env
    self._terrain = GaussianTerrain(env.cfg, env.num_envs)

    self.renderer = GaussianSplattingRenderer(self._env, self)

    self.mesh_params = gymapi.TriangleMeshParams()
    self.mesh_params.static_friction = self._env.cfg.terrain.static_friction
    self.mesh_params.dynamic_friction = self._env.cfg.terrain.dynamic_friction
    self.mesh_params.restitution = self._env.cfg.terrain.restitution

    self.command_scale = torch.zeros(
      self._env.num_envs, 1, device=self._env.device, dtype=torch.float32
    )
    self.velocity_command = torch.zeros(
      self._env.num_envs, 2, device=self._env.device, dtype=torch.float32
    )
    self.heading_command = torch.zeros(
      self._env.num_envs, 1, device=self._env.device, dtype=torch.float32
    )
    self.axis_geom = None
    self.velocity_geom = None
    self.heading_geom = None

    self.local_offset = torch.tensor(
        np.array(self._env.cfg.env.cam_xyz_offset)[None].repeat(self._env.num_envs, 0),
        dtype=torch.float, device=self._env.device, requires_grad=False
    )

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
        curr_x_offset += self._env.cfg.env.env_spacing
        curr_y_offset = 0.0
      else:
        curr_y_offset += self._env.cfg.env.env_spacing

      if self._env.cfg.sim.up_axis == 1:
        env_origin = [curr_x_offset, curr_y_offset, 0.0]
      elif self._env.cfg.sim.up_axis == 0:
        env_origin = [curr_x_offset, 0.0, curr_y_offset]
      else:
        raise ValueError
      env_origin = np.array(env_origin, dtype=np.float32)
      env_origins.append(env_origin)

      mesh = self._terrain.get_mesh(scene, mesh)
      vertices = env_origin[None] + mesh.vertices
      self.mesh_params.nb_vertices = vertices.shape[0]
      self.mesh_params.nb_triangles = mesh.triangles.shape[0]
      self._env.gym.add_triangle_mesh(
        self._env.sim,
        vertices.flatten(order="C"),
        mesh.triangles.flatten(order="C"),
        self.mesh_params,
      )

      vertices_offset = (
        0 if len(all_vertices) == 0 else np.concatenate(all_vertices).shape[0]
      )
      all_triangles.append(mesh.triangles + vertices_offset)
      all_vertices.append(vertices)

    self.all_vertices_mesh = np.concatenate(all_vertices)
    self.all_triangles_mesh = np.concatenate(all_triangles)

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

    self.cam_trans_viz = to_torch(
      cam_trans_orig, device=self._env.device, requires_grad=False
    )
    self.cam_quat_xyzw_viz = to_torch(
      cam_quat_xyzw_orig, device=self._env.device, requires_grad=False
    )

    # self.cam_trans_viz = to_torch(cam_trans_orig.reshape(-1, 3), device=self._env.device, requires_grad=False)
    # self.cam_quat_xyzw_viz = to_torch(np.roll(cam_quat_wxyz_orig.reshape(-1, 4), -1, axis=-1), device=self._env.device, requires_grad=False)

    self.scenes = repeat(list(self._terrain.get_value("scene_name").values()))
    self.cam_trans = to_torch(
      repeat(list(self._terrain.get_value("cam_trans").values())),
      device=self._env.device,
      requires_grad=False,
    )
    self.cam_quat_xyzw = to_torch(
      repeat(list(self._terrain.get_value("cam_quat_xyzw").values())),
      device=self._env.device,
      requires_grad=False,
    )
    self.cam_offset = to_torch(
      repeat(list(self._terrain.get_value("cam_offset").values())),
      device=self._env.device,
      requires_grad=False,
    )
    self.valid_pose_start_idxs = to_torch(
      repeat(list(self._terrain.get_value("valid_pose_start_idxs").values())),
      device=self._env.device,
      requires_grad=False,
    )
    self.env_origins = to_torch(
      repeat(self.env_origins),
      device=self._env.device,
      requires_grad=False,
    )

  def get_cam_link_pose_world_frame(self):
    # In the frame of the world.
    # Apply xyz offset in the local robot frame.
    cam_offset_world = quat_apply(self._env.get_camera_link_state()[:, 3:7], self.local_offset)
    cam_trans = self._env.get_camera_link_state()[:, :3] + cam_offset_world
    cam_quat = self._env.get_camera_link_state()[:, 3:7]
    return cam_trans, cam_quat

  def get_cam_pose_world_frame(self):
    cam_trans, cam_quat = self.get_cam_link_pose_world_frame()
    cam_so3 = vtf.SO3.from_quaternion_xyzw(cam_quat.cpu().numpy())
    cam_so3 = cam_so3 @ vtf.SO3.from_x_radians(self._env.cfg.env.cam_rpy_offset[0])
    cam_so3 = cam_so3 @ vtf.SO3.from_y_radians(self._env.cfg.env.cam_rpy_offset[1])
    cam_so3 = cam_so3 @ vtf.SO3.from_z_radians(self._env.cfg.env.cam_rpy_offset[2])
    cam_quat = torch.tensor(cam_so3.as_quaternion_xyzw(), device=cam_quat.device, dtype=torch.float, requires_grad=False)
    return cam_trans, cam_quat

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

  def _get_nearest_traj_idx(self):
    curr_cam_trans, _ = self.get_cam_link_pose_local_frame()
    trans_difference = curr_cam_trans[:, None] - self.cam_trans
    distance = torch.norm(trans_difference, dim=-1)
    min_distance_idx = torch.argmin(distance, dim=1)
    return min_distance_idx

  def _get_nearest_traj_pose(self):
    nearest_traj_idx = self._get_nearest_traj_idx()
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
    cam_quat = vtf.SO3.from_quaternion_xyzw(cam_quat.cpu().numpy())
    robot_quat = cam_quat @ vtf.SO3.from_z_radians(np.pi / 2)
    robot_quat = robot_quat @ vtf.SO3.from_y_radians(-np.pi / 2)
    robot_quat = to_torch(robot_quat.as_quaternion_xyzw(), device=self._env.device, requires_grad=False)
    return robot_quat

  def sample_cam_pose(self, env_ids):
    # Sample a random camera position and orientation from the camera trajectories.
    cam_trans_subs = self.cam_trans[env_ids]
    cam_quat_xyzw_subs = self.cam_quat_xyzw[env_ids]
    rand = torch.rand((len(env_ids),), device=self._env.device)
    low = self.valid_pose_start_idxs[env_ids].to(torch.float32)
    high = (
      torch.ones((len(env_ids),), device=self._env.device)
      * cam_trans_subs.shape[1]
      - 3
    )
    state_idx = (rand * (high - low) + low).to(torch.int64)

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

  def check_termination(self):
    # Check if robot is too far from camera trajectory (Indicative of poor rendering).
    curr_cam_link_trans, curr_cam_link_quat = self.get_cam_link_pose_local_frame()
    nearest_cam_trans, nearest_cam_quat, nearest_idx = (
      self._get_nearest_traj_pose()
    )
    nearest_robot_quat = self.to_robot_frame(nearest_cam_quat)
    past_end = nearest_idx >= (
      self.cam_trans.shape[1] - 2
    )  # Past end of trajectory.
    distance_exceeded = yaw_exceeded = past_end

    distance = torch.norm((curr_cam_link_trans - nearest_cam_trans)[:, :2], dim=-1)
    distance_exceeded |= distance > self._env.cfg.env.max_traj_pos_distance

    quat_difference = quat_mul(curr_cam_link_quat, quat_conjugate(nearest_robot_quat))
    _, _, yaw = get_euler_xyz(quat_difference)
    yaw_distance = torch.abs(wrap_to_pi(yaw))
    yaw_exceeded |= yaw_distance > self._env.cfg.env.max_traj_yaw_distance_rad
    return distance_exceeded, yaw_exceeded

  def sample_commands(self, env_ids):
    """Randommly select commands of some environments

    Args:
        env_ids (List[int]): Environments ids for which new commands are needed
    """
    _, nearest_cam_quat, nearest_idx = self._get_nearest_traj_pose()
    nearest_robot_quat = self.to_robot_frame(nearest_cam_quat)
    _, _, yaw = get_euler_xyz(nearest_robot_quat)
    self.heading_command = wrap_to_pi(
      yaw
    )  # Match the orientation of the nearest camera.

    # Choose a target pose moderately far away. If nearest_idx + 1 is chosen, target
    # velocities fall to 0 as the robot approaches this target.
    target_idx = torch.clamp(nearest_idx + 5, 0, self.cam_trans.shape[1] - 1)
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

    pos_delta_local = quat_rotate_inverse(
      self._env.root_states[:, 3:7], pos_delta
    )[:, :2]
    pos_delta_norm = pos_delta_local / torch.norm(
      pos_delta_local, dim=-1, keepdim=True
    )
    self.command_scale[env_ids] = torch_rand_float(
      self._env.command_ranges["lin_vel"][0],
      self._env.command_ranges["lin_vel"][1],
      (len(env_ids), 1),
      device=self._env.device,
    )
    pos_delta_norm *= self.command_scale
    pos_delta_norm *= (
      torch.norm(pos_delta_norm[:, :2], dim=1, keepdim=True) > 0.1
    )  # Zero out small commands.
    self.velocity_command = pos_delta_norm

    return self.heading_command, self.velocity_command

  def debug_vis(self, env):
    if self.axis_geom is None:
      self.axis_geom = BatchWireframeAxisGeometry(
        np.prod(self.cam_trans_viz.shape[:2]), 0.25, 0.005, 16
      )
    if self.velocity_geom is None:
      self.velocity_geom = BatchWireframeAxisGeometry(
        self._env.num_envs,
        0.3,
        0.01,
        32,
        color_x=(1, 1, 1),
        color_y=(1, 1, 1),
        color_z=(0, 0, 0),
      )
    if self.heading_geom is None:
      self.heading_geom = BatchWireframeAxisGeometry(
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
    velocity_trans = self._env.root_states[:, :3] + quat_apply(
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
    axis_scales = to_torch(
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
    heading_trans = self._env.root_states[:, :3] + quat_apply(
      self._env.root_states[:, 3:7],
      torch.tensor([0, 0, 0.2], device=self._env.device)[None].repeat(
        self._env.num_envs, 1
      ),
    )
    heading_quat = quat_from_euler_xyz(
      torch.zeros_like(self.heading_command),
      torch.zeros_like(self.heading_command),
      self.heading_command,
    )
    axis_scales = to_torch(
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
