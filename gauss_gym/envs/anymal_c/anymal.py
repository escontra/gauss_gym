import isaacgym  # noqa: F401

import torch

import gauss_gym
from gauss_gym.envs import LeggedRobot
from gauss_gym.utils import config


class Anymal(LeggedRobot):
  cfg: config.Config

  def __init__(self, cfg):
    super().__init__(cfg)

    # load actuator network
    if self.cfg['control']['use_actuator_network']:
      actuator_network_path = self.cfg['control']['actuator_net_file'].format(
        GAUSS_GYM_ROOT_DIR=gauss_gym.GAUSS_GYM_ROOT_DIR
      )
      self.actuator_network = torch.jit.load(actuator_network_path).to(self.device)

  def reset_idx(self, env_ids, time_out_buf):
    metrics = super().reset_idx(env_ids, time_out_buf)
    # Additionaly empty actuator network hidden states
    self.sea_hidden_state_per_env[:, env_ids] = 0.0
    self.sea_cell_state_per_env[:, env_ids] = 0.0
    return metrics

  def _init_buffers(self):
    super()._init_buffers()
    # Additionally initialize actuator network hidden state tensors
    self.sea_input = torch.zeros(
      self.num_envs * self.num_actions, 1, 2, device=self.device, requires_grad=False
    )
    self.sea_hidden_state = torch.zeros(
      2, self.num_envs * self.num_actions, 8, device=self.device, requires_grad=False
    )
    self.sea_cell_state = torch.zeros(
      2, self.num_envs * self.num_actions, 8, device=self.device, requires_grad=False
    )
    self.sea_hidden_state_per_env = self.sea_hidden_state.view(
      2, self.num_envs, self.num_actions, 8
    )
    self.sea_cell_state_per_env = self.sea_cell_state.view(
      2, self.num_envs, self.num_actions, 8
    )

  def _compute_torques(self, actions, dof_stiffness, dof_damping):
    # Choose between pd controller and actuator network
    if self.cfg['control']['use_actuator_network']:
      with torch.inference_mode():
        actions = actions * self.motor_strength_multiplier
        actions = actions + self.motor_error
        self.sea_input[:, 0, 0] = (
          actions + self.default_dof_pos - self.dof_pos
        ).flatten()
        self.sea_input[:, 0, 1] = self.dof_vel.flatten()
        torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = (
          self.actuator_network(
            self.sea_input, (self.sea_hidden_state, self.sea_cell_state)
          )
        )
      return torques
    else:
      # pd controller
      return super()._compute_torques(actions, dof_stiffness, dof_damping)
