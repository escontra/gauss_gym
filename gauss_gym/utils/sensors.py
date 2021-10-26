from typing import Tuple
import torch
import numpy as np

from gauss_gym.utils import (
  visualization,
  visualization_geometries,
  math_utils,
  warp_utils,
  batch_gs_renderer,
  mesh_utils,
)
from gauss_gym.rl import utils as rl_utils


class RayCaster:
  def __init__(
    self,
    env,
    width=1.0,
    length=1.6,
    resolution_width=0.07,
    resolution_length=0.1,
    direction: Tuple = (0.0, 0.0, -1.0),
  ):
    self.attachement_pos = (0.0, 0.0, 0.5)
    self.attachement_quat = (0.0, 0.0, 0.0, 1.0)
    self.attach_yaw_only = True
    self.body_attachement_name = 'base'
    self.default_hit_value = 10
    self.terrain_mesh = env.scene_manager.terrain_mesh
    self.num_envs = env.num_envs
    self.device = env.device

    n_steps_y = int(width / resolution_width)
    n_steps_x = int(length / resolution_length)
    n_steps_y = (n_steps_y // 2) * 2
    n_steps_x = (n_steps_x // 2) * 2
    y = torch.linspace(
      start=-width / 2, end=width / 2, steps=n_steps_y, device=self.device
    )
    x = torch.linspace(
      start=-length / 2, end=length / 2, steps=n_steps_x, device=self.device
    )
    grid_x, grid_y = torch.meshgrid(x, y)

    self.ray_starts = torch.stack([grid_x, grid_y, torch.zeros_like(grid_x)], dim=-1)

    self.ray_directions = torch.zeros_like(self.ray_starts)
    self.ray_directions[..., :] = torch.tensor(list(direction), device=self.device)

    self.rays_shape = self.ray_directions.shape[:-1]
    self.num_rays = np.prod(self.rays_shape)

    offset_pos = torch.tensor(list(self.attachement_pos), device=self.device)
    offset_quat = torch.tensor(list(self.attachement_quat), device=env.device)
    self.ray_directions = math_utils.quat_apply(
      offset_quat.repeat(np.prod(self.ray_directions.shape[:-1]), 1),
      self.ray_directions,
    )
    self.ray_starts += offset_pos

    self.ray_starts = self.ray_starts[None].repeat(self.num_envs, 1, 1, 1)
    self.ray_directions = self.ray_directions[None].repeat(self.num_envs, 1, 1, 1)

    self.ray_hits_world = torch.zeros_like(self.ray_starts)
    self.env = env
    self.sphere_geom = None

  def update(self, dt, env_ids=...):
    """Perform raycasting on the terrain.

    Args:
        env_ids (List[int], optional): Subset of environments for which to return the ray hits. Defaults to ....
    """
    states = self.env.root_states[env_ids, :].squeeze(1)
    pos = states[..., :3]
    quats = states[..., 3:7]
    pos = pos[:, None, None].repeat(1, *self.rays_shape, 1)
    quats = quats[:, None, None].repeat(1, *self.rays_shape, 1)
    if self.attach_yaw_only:
      ray_starts_world = (
        math_utils.quat_apply_yaw(quats, self.ray_starts[env_ids]) + pos
      )
      ray_directions_world = self.ray_directions[env_ids]
    else:
      ray_starts_world = math_utils.quat_apply(quats, self.ray_starts[env_ids]) + pos
      ray_directions_world = math_utils.quat_apply(quats, self.ray_directions[env_ids])

    self.ray_hits_world[env_ids] = warp_utils.ray_cast(
      ray_starts_world, ray_directions_world, self.terrain_mesh
    )

  def get_data(self):
    return torch.nan_to_num(
      self.ray_hits_world, posinf=self.default_hit_value, neginf=-self.default_hit_value
    )

  def debug_vis(self, env):
    offset = torch.tensor([0.0, 0.0, 0.01], device=self.device)[None, None, None]
    ray_hits_world_viz = self.ray_hits_world + offset
    if self.sphere_geom is None:
      self.sphere_geom = visualization_geometries.BatchWireframeSphereGeometry(
        self.num_envs * self.num_rays, 0.02, 10, 10, None, color=(0, 1, 0)
      )
    if self.env.selected_environment >= 0:
      only_render_selected_range = [
        self.env.selected_environment * self.num_rays,
        (self.env.selected_environment + 1) * self.num_rays,
      ]
      self.sphere_geom.draw(
        ray_hits_world_viz,
        env.gym,
        env.viewer,
        env.envs[0],
        only_render_selected=only_render_selected_range,
      )
    else:
      self.sphere_geom.draw(ray_hits_world_viz, env.gym, env.viewer, env.envs[0])


class FootDistanceSensor:
  def __init__(self, env):
    """Detect contacts between feet and terrain.

    Args:
        env (Env): The environment.
        contact_method (str): The method to use to detect contacts. Can be "ray" or "force".
        ray_direction (Tuple): The direction of the ray.
        force_window_size (int): The number of timesteps for the force window. If any value
          in the window registers a contact, the foot is considered to be in contact. Helps
          with stability.
    """
    self.feet_edge_pos = env.cfg['asset']['feet_edge_pos']
    self.attach_yaw_only = True
    self.default_hit_value = 10
    self.terrain_mesh = env.scene_manager.terrain_mesh
    self.num_envs = env.num_envs
    self.num_feet = len(env.feet_indices)
    self.device = env.device
    self.env = env

    feet_edge_relative_pos = math_utils.to_torch(
      env.cfg['asset']['feet_edge_pos'], device=env.device, requires_grad=False
    )
    self.num_edge_points = feet_edge_relative_pos.shape[0]
    self.feet_edge_relative_pos = (
      feet_edge_relative_pos.unsqueeze(0)
      .unsqueeze(0)
      .expand(self.num_envs, self.num_feet, self.num_edge_points, 3)
    )
    self.feet_ground_distance = torch.zeros(
      self.num_envs, self.num_feet, self.num_edge_points, device=env.device
    )

  def update(self, dt, env_ids=...):
    """Perform raycasting on the terrain.

    Args:
        env_ids (List[int], optional): Subset of environments for which to return the ray hits. Defaults to ....
    """
    feet_pos, feet_quat, _, _ = self.env.get_feet_state()
    expanded_feet_pos = feet_pos.unsqueeze(2).expand(
      self.num_envs, self.num_feet, self.num_edge_points, 3
    )
    expanded_feet_quat = feet_quat.unsqueeze(2).expand(
      self.num_envs, self.num_feet, self.num_edge_points, 4
    )
    feet_edge_pos = expanded_feet_pos.reshape(-1, 3) + math_utils.quat_rotate(
      expanded_feet_quat.reshape(-1, 4), self.feet_edge_relative_pos.reshape(-1, 3)
    )

    nearest_points = warp_utils.nearest_point(feet_edge_pos, self.terrain_mesh)
    dist = torch.norm(nearest_points - feet_edge_pos, dim=-1)
    self.feet_ground_distance[env_ids] = dist.view(
      self.num_envs, self.num_feet, self.num_edge_points
    )

  def get_data(self):
    return self.feet_ground_distance - self.env.cfg['asset']['feet_contact_radius']

  def debug_vis(self, env):
    return


class FootContactSensor:
  def __init__(
    self,
    env,
    contact_method: str = 'force',
    ray_direction: Tuple = (0.0, 0.0, -1.0),
    force_window_size: int = 2,
  ):
    """Detect contacts between feet and terrain.

    Args:
        env (Env): The environment.
        contact_method (str): The method to use to detect contacts. Can be "ray" or "force".
        ray_direction (Tuple): The direction of the ray.
        force_window_size (int): The number of timesteps for the force window. If any value
          in the window registers a contact, the foot is considered to be in contact. Helps
          with stability.
    """
    self.feet_edge_pos = env.cfg['asset']['feet_edge_pos']
    self.attach_yaw_only = True
    self.default_hit_value = 10
    self.terrain_mesh = env.scene_manager.terrain_mesh
    self.num_envs = env.num_envs
    self.num_feet = len(env.feet_indices)
    self.device = env.device
    self.contact_method = contact_method
    self.env = env
    self.sphere_geom = None

    feet_edge_relative_pos = math_utils.to_torch(
      env.cfg['asset']['feet_edge_pos'], device=env.device, requires_grad=False
    )
    self.num_edge_points = feet_edge_relative_pos.shape[0]
    self.total_edges = self.num_feet * self.num_edge_points
    self.feet_edge_pos = torch.zeros(
      self.num_envs, self.num_feet, self.num_edge_points, 3, device=env.device
    )
    self.feet_edge_relative_pos = (
      feet_edge_relative_pos.unsqueeze(0)
      .unsqueeze(0)
      .expand(self.num_envs, self.num_feet, self.num_edge_points, 3)
    )
    self.feet_contact_viz = torch.zeros(
      self.num_envs,
      self.num_feet,
      self.num_edge_points,
      dtype=torch.bool,
      device=env.device,
    )

    if contact_method == 'ray':
      self.ray_starts = torch.zeros(1, 3, device=self.device)
      self.ray_directions = torch.zeros_like(self.ray_starts)
      self.ray_directions[..., :] = torch.tensor(
        list(ray_direction), device=self.device
      )

      self.ray_starts = self.ray_starts.repeat(self.num_feet, 1)
      self.ray_directions = self.ray_directions.repeat(self.num_feet, 1)

      self.ray_starts = self.ray_starts.repeat(self.num_envs, 1, 1)
      self.ray_directions = self.ray_directions.repeat(self.num_envs, 1, 1)
      self.feet_ground_distance = torch.zeros(
        self.num_envs, self.num_feet, self.num_edge_points, device=env.device
      )
      self.feet_contact = torch.zeros(
        self.num_envs,
        self.num_feet,
        self.num_edge_points,
        dtype=torch.bool,
        device=env.device,
      )
    elif contact_method == 'force':
      self.feet_contact = torch.zeros(
        self.num_envs,
        self.num_feet,
        force_window_size,
        dtype=torch.bool,
        device=env.device,
      )
    else:
      raise ValueError(f'Invalid contact method: {contact_method}')

  def update(self, dt, env_ids=...):
    """Perform raycasting on the terrain.

    Args:
        env_ids (List[int], optional): Subset of environments for which to return the ray hits. Defaults to ....
    """
    feet_pos, feet_quat, _, _ = self.env.get_feet_state()
    expanded_feet_pos = feet_pos.unsqueeze(2).expand(
      self.num_envs, self.num_feet, self.num_edge_points, 3
    )
    expanded_feet_quat = feet_quat.unsqueeze(2).expand(
      self.num_envs, self.num_feet, self.num_edge_points, 4
    )
    feet_edge_pos = expanded_feet_pos.reshape(-1, 3) + math_utils.quat_rotate(
      expanded_feet_quat.reshape(-1, 4), self.feet_edge_relative_pos.reshape(-1, 3)
    )
    self.feet_edge_pos[env_ids] = feet_edge_pos.reshape(
      self.num_envs, self.num_feet, self.num_edge_points, 3
    )

    if self.contact_method == 'ray':
      nearest_points = warp_utils.nearest_point(feet_edge_pos, self.terrain_mesh)
      dist = torch.norm(nearest_points - feet_edge_pos, dim=-1)
      self.feet_ground_distance[env_ids] = dist.view(
        self.num_envs, self.num_feet, self.num_edge_points
      )
      self.feet_contact[env_ids] = (
        self.feet_ground_distance[env_ids]
        < self.env.cfg['asset']['feet_contact_radius']
      )
      self.feet_contact_viz = self.feet_contact.clone()
    elif self.contact_method == 'force':
      contact_forces = self.env.contact_forces[env_ids, self.env.feet_indices, :]
      in_contact = torch.linalg.norm(contact_forces, dim=-1) > 0.1
      self.feet_contact[env_ids] = torch.roll(self.feet_contact[env_ids], -1, dims=-1)
      self.feet_contact[env_ids, :, -1] = in_contact
      self.feet_contact_viz = torch.any(self.feet_contact, dim=-1)[..., None].repeat(
        1, 1, self.num_edge_points
      )

  def get_data(self):
    foot_contact = torch.any(self.feet_contact, dim=-1)
    return foot_contact

  def debug_vis(self, env):
    if self.sphere_geom is None:
      self.sphere_geom = visualization_geometries.BatchWireframeSphereGeometry(
        self.num_envs * self.num_feet * self.num_edge_points,
        self.env.cfg['asset']['feet_contact_radius'],
        16,
        16,
        None,
        color=(1, 1, 0),
      )
    feet_edge_pos = self.feet_edge_pos.view(-1, 3)
    feet_contact = self.feet_contact_viz.view(-1)
    color_red = math_utils.to_torch(
      np.array([1.0, 0.0, 0.0])[None].repeat(feet_contact.shape[0], 0),
      device=self.device,
      requires_grad=False,
    )
    color_green = math_utils.to_torch(
      np.array([0.0, 1.0, 0.0])[None].repeat(feet_contact.shape[0], 0),
      device=self.device,
      requires_grad=False,
    )
    colors = torch.where(feet_contact[..., None], color_green, color_red)
    if self.env.selected_environment >= 0:
      only_render_selected_range = [
        self.env.selected_environment * self.total_edges,
        (self.env.selected_environment + 1) * self.total_edges,
      ]
      self.sphere_geom.draw(
        feet_edge_pos,
        env.gym,
        env.viewer,
        env.envs[0],
        only_render_selected=only_render_selected_range,
        custom_colors=colors,
      )
    else:
      self.sphere_geom.draw(
        feet_edge_pos, env.gym, env.viewer, env.envs[0], custom_colors=colors
      )


class GaussianSplattingRenderer:
  def __init__(self, env, scene_manager):
    self.num_envs = env.num_envs
    self.device = env.device
    self.scene_manager = scene_manager
    self.terrain = scene_manager._terrain

    self.scene_path_map = {}
    for s, f in self.terrain.mesh_keys:
      self.scene_path_map[s] = self.terrain.get_mesh(s, f).splatpath / 'splat.ply'
    self.path_scene_map = {v: k for k, v in self.scene_path_map.items()}
    self._gs_renderer = batch_gs_renderer.MultiSceneRenderer(
      list(self.scene_path_map.values()),
      renderer_gpus=[self.device],
      # renderer_gpus=[1],
      output_gpu=self.device,
    )

    self.viz_state = (None, None)
    self.camera_positions = torch.zeros(
      self.num_envs, 3, device=self.device, requires_grad=False
    )
    self.camera_quats_xyzw = torch.zeros(
      self.num_envs, 4, device=self.device, requires_grad=False
    )
    self.env = env

    downscale_factor = float(self.env.cfg['env']['camera_params']['downscale_factor'])
    self.cam_height = int(
      self.env.cfg['env']['camera_params']['cam_height'] / downscale_factor
    )
    self.cam_width = int(
      self.env.cfg['env']['camera_params']['cam_width'] / downscale_factor
    )
    self.fl_x = self.env.cfg['env']['camera_params']['fl_x'] / downscale_factor
    self.fl_y = self.env.cfg['env']['camera_params']['fl_y'] / downscale_factor
    self.pp_x = self.env.cfg['env']['camera_params']['pp_x'] / downscale_factor
    self.pp_y = self.env.cfg['env']['camera_params']['pp_y'] / downscale_factor

    self.fov = np.arctan2(self.cam_height / 2, self.fl_y)
    self.aspect = self.cam_width / self.cam_height

    # Camera intrinsics randomization.
    self.apply_domain_rand = self.env.cfg['domain_rand']['apply_domain_rand']
    self.camera_refresh_interval_s = self.env.cfg['domain_rand'][
      'camera_refresh_interval_s'
    ]
    self.fl_rand_params = self.env.cfg['domain_rand']['focal_length']
    self.pp_rand_params = self.env.cfg['domain_rand']['principal_point']
    self.camera_pos_rand_params = self.env.cfg['domain_rand']['camera_pos']
    self.camera_rot_rand_params = self.env.cfg['domain_rand']['camera_rot']
    self.refresh_duration_rand_params = self.env.cfg['domain_rand']['refresh_duration']
    self.fl_x_sample = torch.full(
      (self.num_envs,), self.fl_x, device=self.device, dtype=torch.float32
    )
    self.fl_y_sample = torch.full(
      (self.num_envs,), self.fl_y, device=self.device, dtype=torch.float32
    )
    self.pp_x_sample = torch.full(
      (self.num_envs,), self.pp_x, device=self.device, dtype=torch.float32
    )
    self.pp_y_sample = torch.full(
      (self.num_envs,), self.pp_y, device=self.device, dtype=torch.float32
    )
    self.camera_pos_delta_sample = torch.zeros(
      self.num_envs, 3, device=self.device, dtype=torch.float32
    )
    self.camera_quat_delta_sample = torch.zeros(
      self.num_envs, 4, device=self.device, dtype=torch.float32
    )
    self.camera_quat_delta_sample[:, 3] = 1.0
    self.refresh_duration_s_sample = np.mean(self.refresh_duration_rand_params['range'])
    self._maybe_sample_camera_params()

    self.renders = torch.zeros(
      self.num_envs,
      3,
      self.cam_height,
      self.cam_width,
      device=self.device,
      dtype=torch.uint8,
    )
    self.frustrum_geom = None
    self.axis_geom = None
    self.new_frames_acquired = True

    self.local_offset = torch.tensor(
      np.array(self.env.cfg['env']['camera_params']['cam_xyz_offset'])[None].repeat(
        self.num_envs, 0
      ),
      dtype=torch.float,
      device=self.device,
      requires_grad=False,
    )

  def get_gs_renderers(self):
    return {k: self._gs_renderer.renderers[v] for k, v in self.scene_path_map.items()}

  def _maybe_sample_camera_params(self):
    if self.apply_domain_rand and self.fl_rand_params['apply']:
      self.fl_x_sample[:] = math_utils.apply_randomization(
        torch.full((self.num_envs,), self.fl_x, device=self.device), self.fl_rand_params
      )
      self.fl_y_sample[:] = math_utils.apply_randomization(
        torch.full((self.num_envs,), self.fl_y, device=self.device), self.fl_rand_params
      )

    if self.apply_domain_rand and self.pp_rand_params['apply']:
      self.pp_x_sample[:] = math_utils.apply_randomization(
        torch.full((self.num_envs,), self.pp_x, device=self.device), self.pp_rand_params
      )
      self.pp_y_sample[:] = math_utils.apply_randomization(
        torch.full((self.num_envs,), self.pp_y, device=self.device), self.pp_rand_params
      )

    if self.apply_domain_rand and self.camera_pos_rand_params['apply']:
      self.camera_pos_delta_sample[:] = math_utils.apply_randomization(
        torch.zeros(self.num_envs, 3, device=self.device), self.camera_pos_rand_params
      )

    if self.apply_domain_rand and self.camera_rot_rand_params['apply']:
      camera_eulerxyz_delta_sample = math_utils.apply_randomization(
        torch.zeros(self.num_envs, 3, device=self.device), self.camera_rot_rand_params
      )
      self.camera_quat_delta_sample[:] = math_utils.quat_from_euler_xyz(
        camera_eulerxyz_delta_sample[..., 0],
        camera_eulerxyz_delta_sample[..., 1],
        camera_eulerxyz_delta_sample[..., 2],
      )

    if self.apply_domain_rand and self.refresh_duration_rand_params['apply']:
      if not self.env.cfg['multi_gpu'] or self.env.cfg['multi_gpu_global_rank'] == 0:
        self.refresh_duration_s_sample = math_utils.apply_randomization(
          0.0, self.refresh_duration_rand_params
        )

      if self.env.cfg['multi_gpu']:
        self.refresh_duration_s_sample = rl_utils.broadcast_scalar(
          float(self.refresh_duration_s_sample), 0, self.device
        )

  def update(self, dt, env_ids=...):
    """Render images with Gaussian splatting.

    Args:
        env_ids (List[int], optional): Subset of environments for which to return the ray hits. Defaults to ....
    """

    if (
      self.apply_domain_rand
      and self.env.common_step_counter
      % int(self.camera_refresh_interval_s / self.env.dt)
      == 0
    ):
      self._maybe_sample_camera_params()

    cam_trans, cam_quat = self.scene_manager.get_cam_pose_world_frame()
    cam_trans += self.camera_pos_delta_sample
    cam_quat = math_utils.quat_mul(cam_quat, self.camera_quat_delta_sample)
    self.camera_positions[:] = cam_trans
    self.camera_quats_xyzw[:] = cam_quat

    # Refresh rate is determined by the refresh_duration_s.
    if int(self.refresh_duration_s_sample / self.env.dt) == 0:
      should_refresh = True
    else:
      # TODO: We're using `common_step_counter` here. Using `episode_length_buf` would be more accurate,
      # but is considerably slower as every step would require at least a few renders. With `common_step_counter`,
      # we only need to render every `refresh_duration_s` steps.
      should_refresh = (
        (self.env.common_step_counter - 1)
        % int(self.refresh_duration_s_sample / self.env.dt)
      ) == 0
    if not should_refresh:
      self.new_frames_acquired = False
      return

    cam_lin_vel, cam_ang_vel = self.scene_manager.get_cam_velocity_world_frame()

    cam_trans -= self.env.env_origins
    cam_rot = math_utils.quaternion_to_matrix(cam_quat)

    # Convert from IG frame to GS frame for each scene.
    cam_trans = torch.einsum(
      'ij, ijk -> ik', cam_trans, self.scene_manager.ig_to_orig_rot
    )
    cam_trans += self.scene_manager.cam_offset
    cam_rot = torch.einsum(
      'ijk, ikl -> ijl', self.scene_manager.ig_to_orig_rot, cam_rot
    )
    c2ws = torch.eye(4, dtype=torch.float32, device=self.device)[None].repeat(
      cam_trans.shape[0], 1, 1
    )
    c2ws[:, :3, :3] = cam_rot
    c2ws[:, :3, 3] = cam_trans

    scene_poses, scene_linear_velocities, scene_angular_velocities = {}, {}, {}
    scene_fl_x, scene_fl_y, scene_pp_x, scene_pp_y = {}, {}, {}, {}
    for k, (start, end) in self.scene_manager.scene_start_end_ids.items():
      scene_poses[self.scene_path_map[k]] = c2ws[start:end]
      scene_linear_velocities[self.scene_path_map[k]] = cam_lin_vel[start:end]
      scene_angular_velocities[self.scene_path_map[k]] = cam_ang_vel[start:end]
      scene_fl_x[self.scene_path_map[k]] = self.fl_x_sample[start:end]
      scene_fl_y[self.scene_path_map[k]] = self.fl_y_sample[start:end]
      scene_pp_x[self.scene_path_map[k]] = self.pp_x_sample[start:end]
      scene_pp_y[self.scene_path_map[k]] = self.pp_y_sample[start:end]

    renders = self._gs_renderer.batch_render(
      scene_poses,
      fl_x=scene_fl_x,
      fl_y=scene_fl_y,
      pp_x=scene_pp_x,
      pp_y=scene_pp_y,
      h=self.cam_height,
      w=self.cam_width,
      motion_blur_frac=0.2,
      blur_dt=1.0 / 300.0,
      camera_linear_velocity=scene_linear_velocities,
      camera_angular_velocity=scene_angular_velocities,
      minibatch=1024,
    )
    colors = [v[0] for v in renders.values()]
    self.renders[:] = torch.cat(colors, dim=0).permute(0, 3, 1, 2)
    self.new_frames_acquired = True

  def get_data(self):
    if self.new_frames_acquired:
      return self.renders
    else:
      return None

  def debug_vis(self, env):
    if self.frustrum_geom is None:
      self.frustrum_geom = visualization_geometries.BatchWireframeFrustumGeometry(
        self.num_envs,
        0.1,
        0.2,
        self.cam_width,
        self.cam_height,
        self.fl_x,
        self.fl_y,
        0.005,
        32,
      )
      self.axis_geom = visualization_geometries.BatchWireframeAxisGeometry(
        self.num_envs, 0.25, 0.005, 32
      )

    self.frustrum_geom.draw(
      self.camera_positions,
      self.camera_quats_xyzw,
      env.gym,
      env.viewer,
      env.envs[0],
      self.env.selected_environment,
    )
    self.axis_geom.draw(
      self.camera_positions,
      self.camera_quats_xyzw,
      env.gym,
      env.viewer,
      env.envs[0],
      only_render_selected=self.env.selected_environment,
    )
    if self.new_frames_acquired:
      self.viz_state = visualization.update_image(
        self.env, *self.viz_state, self.env.selected_environment, self.renders
      )


class LinkHeightSensor:
  def __init__(self, env, link_names, color=(1, 0, 0), attach_yaw_only=True):
    self.attachement_pos = (0.0, 0.0, 0.0)
    self.attachement_quat = (0.0, 0.0, 0.0, 1.0)
    direction = (0.0, 0.0, -1.0)
    self.attach_yaw_only = attach_yaw_only
    self.link_names = link_names
    self.link_indices = [
      env.gym.find_asset_rigid_body_index(env.robot_asset, name) for name in link_names
    ]
    self.color = color

    self.default_hit_value = 10
    self.terrain_mesh = env.scene_manager.terrain_mesh
    self.num_envs = env.num_envs
    self.device = env.device

    self.ray_starts = torch.zeros(len(self.link_indices), 3, device=self.device)
    self.ray_directions = torch.zeros_like(self.ray_starts)
    self.ray_directions[..., :] = torch.tensor(list(direction), device=self.device)
    self.num_rays = len(self.ray_directions)

    offset_pos = torch.tensor(list(self.attachement_pos), device=self.device)
    offset_quat = torch.tensor(list(self.attachement_quat), device=env.device)
    self.ray_directions = math_utils.quat_apply(
      offset_quat.repeat(len(self.ray_directions), 1), self.ray_directions
    )
    self.ray_starts += offset_pos

    self.ray_starts = self.ray_starts.repeat(self.num_envs, 1, 1)
    self.ray_directions = self.ray_directions.repeat(self.num_envs, 1, 1)

    self.ray_hits_world = torch.zeros(
      self.num_envs, self.num_rays, 3, device=self.device
    )
    self.link_heights = torch.zeros(
      self.num_envs, len(self.link_indices), device=self.device
    )
    self.env = env
    self.sphere_geom = None

  def update(self, dt, env_ids=...):
    """Perform raycasting on the terrain.

    Args:
        env_ids (List[int], optional): Subset of environments for which to return the ray hits. Defaults to ....
    """
    pos = self.env.rigid_body_state.view(self.num_envs, self.env.num_bodies, 13)[
      env_ids, self.link_indices, 0:3
    ]
    quats = self.env.rigid_body_state.view(self.num_envs, self.env.num_bodies, 13)[
      env_ids, self.link_indices, 3:7
    ]

    if self.attach_yaw_only:
      ray_starts_world = (
        math_utils.quat_apply_yaw(quats, self.ray_starts[env_ids]) + pos
      )
      ray_directions_world = self.ray_directions[env_ids]
    else:
      ray_starts_world = math_utils.quat_apply(quats, self.ray_starts[env_ids]) + pos
      ray_directions_world = math_utils.quat_apply(quats, self.ray_directions[env_ids])

    self.ray_hits_world[env_ids] = warp_utils.ray_cast(
      ray_starts_world, ray_directions_world, self.terrain_mesh
    )
    self.link_heights[env_ids] = pos[..., 2] - self.ray_hits_world[..., 2]

  def get_data(self):
    return torch.nan_to_num(
      self.link_heights, posinf=self.default_hit_value, neginf=self.default_hit_value
    )

  def debug_vis(self, env):
    if self.sphere_geom is None:
      self.sphere_geom = visualization_geometries.BatchWireframeSphereGeometry(
        self.num_envs * self.num_rays, 0.02, 20, 20, None, color=self.color
      )
    if self.env.selected_environment >= 0:
      only_render_selected_range = [
        self.env.selected_environment * self.num_rays,
        (self.env.selected_environment + 1) * self.num_rays,
      ]
      self.sphere_geom.draw(
        self.ray_hits_world,
        env.gym,
        env.viewer,
        env.envs[0],
        only_render_selected=only_render_selected_range,
      )
    else:
      self.sphere_geom.draw(self.ray_hits_world, env.gym, env.viewer, env.envs[0])


class MeshHeightSensor:
  def __init__(
    self,
    env,
    link_names,
    ray_direction: Tuple = (0.0, 0.0, -1.0),
    num_samples_x: int = 10,
    num_samples_y: int = 10,
    attach_yaw_only: bool = False,
  ):
    """Detect contacts between feet and terrain.

    Args:
        env (Env): The environment.
        ray_direction (Tuple): The direction of the ray.
        force_window_size (int): The number of timesteps for the force window. If any value
          in the window registers a contact, the foot is considered to be in contact. Helps
          with stability.
        num_samples_x (int): The number of samples to take in the x direction.
        num_samples_y (int): The number of samples to take in the y direction.
    """
    self.attach_yaw_only = attach_yaw_only
    self.default_hit_value = 10
    self.terrain_mesh = env.scene_manager.terrain_mesh
    self.num_envs = env.num_envs
    self.num_links = len(link_names)
    self.link_indices = [
      env.gym.find_asset_rigid_body_index(env.robot_asset, name) for name in link_names
    ]
    self.device = env.device
    self.env = env
    self.sphere_geom = None
    meshes = mesh_utils.get_mesh_for_links(env.asset_path, link_names)
    link_sample_pos = None
    for mesh in meshes:
      # sampled_points = mesh_utils.mesh_sampler_grid(mesh, num_samples_x, num_samples_y, scan_direction='z')
      # print(sampled_points.shape)
      sampled_points, _ = mesh_utils.compute_mesh_ray_points(
        mesh,
        resolution_x=num_samples_x,
        resolution_y=num_samples_y,
        scan_direction='+z',
        visualize=False,
      )
      if link_sample_pos is None:
        link_sample_pos = sampled_points
      else:
        assert link_sample_pos.shape == sampled_points.shape, (
          'All meshes must be the same (Contact escontra).'
        )
        assert np.allclose(link_sample_pos, sampled_points), (
          'All meshes must be the same (Contact escontra).'
        )

    link_sample_relative_pos = math_utils.to_torch(
      link_sample_pos, device=env.device, requires_grad=False
    )

    self.num_sample_pos = link_sample_relative_pos.shape[0]
    self.link_sample_pos = torch.zeros(
      self.num_envs, self.num_links, self.num_sample_pos, 3, device=env.device
    )
    self.link_sample_relative_pos = (
      link_sample_relative_pos.unsqueeze(0)
      .unsqueeze(0)
      .expand(self.num_envs, self.num_links, self.num_sample_pos, 3)
    )

    # self.ray_starts = torch.zeros(self.num_envs, self.num_links, self.num_sample_pos, 3, device=self.device)
    self.ray_directions = torch.zeros_like(self.link_sample_relative_pos)
    self.ray_directions[..., :] = torch.tensor(list(ray_direction), device=self.device)

    self.ray_starts_world = torch.zeros(
      self.num_envs, self.num_links, self.num_sample_pos, 3, device=self.device
    )
    self.ray_hits_world = torch.zeros_like(self.ray_starts_world)
    self.link_heights = torch.zeros(
      self.num_envs, self.num_links, self.num_sample_pos, device=self.device
    )
    self.env = env
    self.sphere_geom = None

  def update(self, dt, env_ids=...):
    """Perform raycasting on the terrain.

    Args:
        env_ids (List[int], optional): Subset of environments for which to return the ray hits. Defaults to ....
    """
    pos = self.env.rigid_body_state.view(self.num_envs, self.env.num_bodies, 13)[
      env_ids, self.link_indices, 0:3
    ]
    quats = self.env.rigid_body_state.view(self.num_envs, self.env.num_bodies, 13)[
      env_ids, self.link_indices, 3:7
    ]

    pos = pos[:, :, None].repeat(1, 1, self.num_sample_pos, 1)
    quats = quats[:, :, None].repeat(1, 1, self.num_sample_pos, 1)

    if self.attach_yaw_only:
      self.ray_starts_world = (
        math_utils.quat_apply_yaw(quats, self.link_sample_relative_pos[env_ids]) + pos
      )
      ray_directions_world = self.ray_directions[env_ids]
    else:
      self.ray_starts_world = (
        math_utils.quat_apply(quats, self.link_sample_relative_pos[env_ids]) + pos
      )
      ray_directions_world = self.ray_directions[env_ids]

    self.ray_hits_world[env_ids] = warp_utils.ray_cast(
      self.ray_starts_world, ray_directions_world, self.terrain_mesh
    )
    self.link_heights[env_ids] = (
      self.ray_starts_world[..., 2] - self.ray_hits_world[..., 2]
    )

  def get_data(self):
    return torch.nan_to_num(
      self.link_heights, posinf=self.default_hit_value, neginf=self.default_hit_value
    )

  def debug_vis(self, env):
    # We are migrating away from the IsaacGym visualizer, and are instead
    # developing utils/viser_visualizer.py. All IsaacGym visualizer
    # features are deprecated.
    return
