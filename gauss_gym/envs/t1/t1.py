import isaacgym  # noqa: F401

import torch
from gauss_gym.envs import LeggedRobot
from gauss_gym.utils import math_utils


class T1(LeggedRobot):
  def _init_buffers(self):
    super()._init_buffers()
    self.phase_offset = torch.zeros(
      (self.num_envs, 2), dtype=torch.float32, device=self.device, requires_grad=False
    )
    self.phase_offset[:, 0] = math_utils.torch_rand_float(
      -torch.pi, torch.pi, (self.num_envs, 1), device=self.device
    ).squeeze(1)
    self.phase_offset[:, 1] = (
      torch.fmod(self.phase_offset[:, 0] + 2 * torch.pi, 2 * torch.pi) - torch.pi
    )
    self.phase = self.phase_offset
    self.phase_dt = (
      2
      * torch.pi
      * self.dt
      * math_utils.torch_rand_float(
        *self.cfg['commands']['gait_frequency'], (self.num_envs, 1), device=self.device
      )
    )

  def step(self, actions, actions_mean=None):
    phase_tp1 = self.episode_length_buf.unsqueeze(1) * self.phase_dt + self.phase_offset
    self.phase = torch.fmod(phase_tp1 + torch.pi, 2 * torch.pi) - torch.pi
    return super().step(actions, actions_mean)

  def _reset_buffers(self, env_ids):
    super()._reset_buffers(env_ids)

  def _resample_commands(self, env_ids):
    self.phase_dt[env_ids] = (
      2
      * torch.pi
      * self.dt
      * math_utils.torch_rand_float(
        *self.cfg['commands']['gait_frequency'], (len(env_ids), 1), device=self.device
      )
    )
    super()._resample_commands(env_ids)

  def _reward_dof_pos_limits(self, soft_dof_pos_limit):
    # Penalize dof positions too close to the limit
    lower = self.dof_pos_limits[:, 0] + 0.5 * (1 - soft_dof_pos_limit) * (
      self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
    )
    upper = self.dof_pos_limits[:, 1] - 0.5 * (1 - soft_dof_pos_limit) * (
      self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
    )
    return torch.sum(((self.dof_pos < lower) | (self.dof_pos > upper)).float(), dim=-1)

  def _reward_feet_roll(self):
    roll, _, _ = math_utils.get_euler_xyz(self.get_feet_state()[1].reshape(-1, 4))
    roll = (roll.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (
      2 * torch.pi
    ) - torch.pi
    return torch.sum(torch.square(roll), dim=-1)

  def _reward_feet_yaw_diff(self):
    _, _, yaw = math_utils.get_euler_xyz(self.get_feet_state()[1].reshape(-1, 4))
    yaw = (yaw.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (
      2 * torch.pi
    ) - torch.pi
    return torch.square((yaw[:, 1] - yaw[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi)

  def _reward_feet_yaw_mean(self):
    _, _, yaw = math_utils.get_euler_xyz(self.get_feet_state()[1].reshape(-1, 4))
    yaw = (yaw.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (
      2 * torch.pi
    ) - torch.pi
    feet_yaw_mean = yaw.mean(dim=-1) + torch.pi * (
      torch.abs(yaw[:, 1] - yaw[:, 0]) > torch.pi
    )
    return torch.square(
      (math_utils.get_euler_xyz(self.base_quat)[2] - feet_yaw_mean + torch.pi)
      % (2 * torch.pi)
      - torch.pi
    )

  def _reward_feet_distance(self, close_feet_threshold: float):
    _, _, base_yaw = math_utils.get_euler_xyz(self.base_quat)
    feet_distance = torch.abs(
      torch.cos(base_yaw)
      * (self.get_feet_state()[0][:, 1, 1] - self.get_feet_state()[0][:, 0, 1])
      - torch.sin(base_yaw)
      * (self.get_feet_state()[0][:, 1, 0] - self.get_feet_state()[0][:, 0, 0])
    )
    # distance_diff = torch.clip(feet_distance - feet_distance_ref, max=0.)
    return (feet_distance < close_feet_threshold) * 1.0

  def _reward_feet_distance_clipped(self, feet_distance_ref):
    _, _, base_yaw = math_utils.get_euler_xyz(self.base_quat)
    feet_distance = torch.abs(
      torch.cos(base_yaw)
      * (self.get_feet_state()[0][:, 1, 1] - self.get_feet_state()[0][:, 0, 1])
      - torch.sin(base_yaw)
      * (self.get_feet_state()[0][:, 1, 0] - self.get_feet_state()[0][:, 0, 0])
    )
    distance_diff = torch.clip(feet_distance - feet_distance_ref, max=0.0)
    return distance_diff

  def _reward_t1_pose(self):
    weights = []
    for name in self.dof_names:
      if 'Head' in name:
        weights.append(1.0)
      elif 'Shoulder' in name:
        weights.append(50.0)
      elif 'Elbow' in name:
        weights.append(50.0)
      elif 'Wrist' in name:
        weights.append(50.0)
      elif 'Hand' in name:
        weights.append(50.0)
      elif 'Waist' in name:
        weights.append(50.0)
      elif 'Hip_Pitch' in name:
        weights.append(0.01)
      elif 'Hip_Roll' in name:
        weights.append(1.0)
      elif 'Hip_Yaw' in name:
        weights.append(5.0)
      elif 'Knee_Pitch' in name:
        weights.append(0.01)
      elif 'Ankle_Pitch' in name:
        weights.append(1.0)
      elif 'Ankle_Roll' in name:
        weights.append(5.0)
      else:
        raise ValueError(f'Unknown dof name: {name}')
    weights = torch.tensor(weights, device=self.device)[None]
    pose_error = torch.square(self.dof_pos - self.default_dof_pos)
    weighted_error = pose_error * weights
    return torch.sum(weighted_error, dim=1)

  def _reward_t1_pose_old(self):
    # Stay close to the default pose.
    weights = []
    for name in self.dof_names:
      if 'Head' in name:
        weights.append(0.03)
      elif 'Shoulder' in name:
        weights.append(1.0)
      elif 'Elbow' in name:
        weights.append(1.0)
      elif 'Waist' in name:
        weights.append(1.0)
      elif 'Hip' in name:
        weights.append(0.03)
      elif 'Knee' in name:
        weights.append(0.03)
      elif 'Ankle_Pitch' in name:
        weights.append(0.03)
      elif 'Ankle_Roll' in name:
        weights.append(0.5)
      elif 'Wrist' in name:
        weights.append(1.0)
      elif 'Hand' in name:
        weights.append(1.0)
      else:
        raise ValueError(f'Unknown dof name: {name}')
    weights = torch.tensor(weights, device=self.device)[None]
    reward = torch.exp(
      -torch.sum(torch.square(self.dof_pos - self.default_dof_pos) * weights, dim=-1)
    )
    return reward

  def _reward_dof_vel_head(self):
    # Penalize dof velocity of the head
    idxs = [self.dof_names.index(name) for name in ['Head_pitch', 'AAHead_yaw']]
    return torch.sum(torch.square(self.dof_vel[:, idxs]), dim=-1)

  def _reward_dof_acc_head(self, method: str = 'mean'):
    # Penalize dof accelerations
    # Use the last dof acc if method is 'last', 'mean' to use the mean of
    # the substeps.
    idxs = [self.dof_names.index(name) for name in ['Head_pitch', 'AAHead_yaw']]
    if method == 'last':
      dof_acc = self.substep_dof_acc[:, -1, :]
    elif method == 'mean':
      dof_acc = torch.mean(self.substep_dof_acc, dim=1)
    else:
      raise ValueError(f'Invalid method: {method}')
    reward = torch.sum(torch.square(dof_acc[:, idxs]), dim=-1)
    return reward

  def _reward_feet_swing(self, swing_period):
    phase_left = (self.phase[:, 0] + torch.pi) / (2 * torch.pi)
    phase_right = (self.phase[:, 1] + torch.pi) / (2 * torch.pi)
    left_swing = (
      torch.abs(phase_left) < swing_period
    )  # & (self.gait_frequency > 1.0e-8)
    right_swing = (
      torch.abs(phase_right) < swing_period
    )  # & (self.gait_frequency > 1.0e-8)
    return (left_swing & ~self.feet_contact[:, 0]).float() + (
      right_swing & ~self.feet_contact[:, 1]
    ).float()

  def _reward_feet_phase(self, swing_height: float, track_rew: bool = True):
    """
    Reward for tracking the desired foot height based on gait phase.
    Based on MuJoCo Playground's implementation.
    """

    foot_sample_heights = self.sensors['foot_height_raycaster_grid'].get_data()
    foot_sample_heights = torch.min(foot_sample_heights, dim=-1)[0]
    foot_z_left = foot_sample_heights[:, 0]
    foot_z_right = foot_sample_heights[:, 1]

    # Use playground phase directly (already in -π to π range)
    rz_left = math_utils.get_rz(self.phase[:, 0], swing_height)
    rz_right = math_utils.get_rz(self.phase[:, 1], swing_height)

    if track_rew:
      # Calculate height tracking errors
      error_left = torch.square(foot_z_left - rz_left)
      error_right = torch.square(foot_z_right - rz_right)

      # Combine errors and apply exponential reward
      total_error = error_left + error_right
      return torch.exp(-total_error / 0.01)
    else:
      should_contact_left = rz_left < 0.02
      should_contact_right = rz_right < 0.02

      is_contact_left = self.feet_contact[:, 0]
      is_contact_right = self.feet_contact[:, 1]

      exceed_des_height_left = foot_z_left > rz_left
      exceed_des_height_right = foot_z_right > rz_right

      on_track_left = torch.where(
        should_contact_left, is_contact_left, exceed_des_height_left
      )
      on_track_right = torch.where(
        should_contact_right, is_contact_right, exceed_des_height_right
      )

      return 0.5 * (on_track_left + on_track_right).float()
