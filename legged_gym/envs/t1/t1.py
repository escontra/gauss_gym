# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from time import time
import numpy as np
import os

from isaacgym import torch_utils as tu
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot
from legged_gym.teacher import ObsManager
import legged_gym.teacher.observations as O

class T1(LeggedRobot):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
      super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
      obs_groups_cfg = {
          "teacher_observations": {
              # optional parameters: scale, clip([min, max]), noise
              "add_noise": True,  # turns off the noise in all observations
              "is_recurrent": True,
              "projected_gravity": {"func": O.projected_gravity, "noise": self.cfg.noise.noise_scales.gravity},
              "base_ang_vel": {"func": O.base_ang_vel, "noise": self.cfg.noise.noise_scales.ang_vel, "scale": self.obs_scales.ang_vel},
              "velocity_commands": {"func": O.velocity_commands, "scale": self.commands_scale},
              "gait_progress": {"func": O.gait_progress},
              "base_lin_vel": {"func": O.base_lin_vel, "noise": self.cfg.noise.noise_scales.lin_vel, "scale": self.obs_scales.lin_vel},
              "dof_pos": {"func": O.dof_pos, "noise": self.cfg.noise.noise_scales.dof_pos, "scale": self.obs_scales.dof_pos},
              "dof_vel": {"func": O.dof_vel, "noise": self.cfg.noise.noise_scales.dof_vel, "scale": self.obs_scales.dof_vel},
              "actions": {"func": O.actions},
              "ray_cast": {
                  "func": O.ray_cast,
                  "noise": self.cfg.noise.noise_scales.height_measurements,
                  "sensor": "raycast_grid",
                  "clip": (-1.0, 1.0),
                  "scale": self.obs_scales.height_measurements
              },
          },
          "student_observations": {
              # optional parameters: scale, clip([min, max]), noise
              "add_noise": self.cfg.noise.add_noise,  # turns off the noise in all observations
              "is_recurrent": True,
              "projected_gravity": {"func": O.projected_gravity, "noise": self.cfg.noise.noise_scales.gravity},
              "base_ang_vel": {"func": O.base_ang_vel, "noise": self.cfg.noise.noise_scales.ang_vel, "scale": self.obs_scales.ang_vel},
              "velocity_commands": {"func": O.velocity_commands, "scale": self.commands_scale},
              "gait_progress": {"func": O.gait_progress},
              "dof_pos": {"func": O.dof_pos, "noise": self.cfg.noise.noise_scales.dof_pos, "scale": self.obs_scales.dof_pos},
              "dof_vel": {"func": O.dof_vel, "noise": self.cfg.noise.noise_scales.dof_vel, "scale": self.obs_scales.dof_vel},
              "actions": {"func": O.actions},
              "images": {"func": O.gs_render, "sensor": "gs_renderer"},
          },
      }
      self.obs_manager = ObsManager(self, obs_groups_cfg)


    def _init_buffers(self):
        super()._init_buffers()
        self.gait_frequency = torch.ones(self.num_envs, dtype=torch.float, device=self.device)*self.cfg.commands.gait_frequency
        self.gait_process = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)


    def step(self, actions):
        self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.gait_frequency, 1.0)
        obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras = super().step(actions)
        return obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras

    def _reward_survival(self):
        # Reward survival
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_root_acc(self):
        # Penalize root accelerations
        return torch.sum(torch.square((self.last_root_vel - self.root_states[:, 7:13]) / self.dt), dim=-1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        lower = self.dof_pos_limits[:, 0] + 0.5 * (1 - self.cfg.rewards.soft_dof_pos_limit) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        upper = self.dof_pos_limits[:, 1] - 0.5 * (1 - self.cfg.rewards.soft_dof_pos_limit) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        return torch.sum(((self.dof_pos < lower) | (self.dof_pos > upper)).float(), dim=-1)

    def _reward_torque_tiredness(self):
        # Penalize torque tiredness
        return torch.sum(torch.square(self.torques / self.torque_limits).clip(max=1.0), dim=-1)

    def _reward_power(self):
        # Penalize power
        return torch.sum((self.torques * self.dof_vel).clip(min=0.0), dim=-1)

    def _reward_feet_slip(self):
        # Penalize feet velocities when contact
        return (
            torch.sum(
                torch.square((self.last_feet_pos - self.get_feet_pos_quat()[0]) / self.dt).sum(dim=-1) * self.feet_contact.float(),
                dim=-1,
            )
            * (self.episode_length_buf > 1).float()
        )

    def _reward_feet_vel_z(self):
        return torch.sum(torch.square((self.last_feet_pos - self.get_feet_pos_quat()[0]) / self.dt)[:, :, 2], dim=-1)

    def _reward_feet_roll(self):
        roll, _, _ = tu.get_euler_xyz(self.get_feet_pos_quat()[1].reshape(-1, 4))
        roll = (roll.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        return torch.sum(torch.square(roll), dim=-1)

    def _reward_feet_yaw_diff(self):
        _, _, yaw = tu.get_euler_xyz(self.get_feet_pos_quat()[1].reshape(-1, 4))
        yaw = (yaw.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        return torch.square((yaw[:, 1] - yaw[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi)

    def _reward_feet_yaw_mean(self):
        _, _, yaw = tu.get_euler_xyz(self.get_feet_pos_quat()[1].reshape(-1, 4))
        yaw = (yaw.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        feet_yaw_mean = yaw.mean(dim=-1) + torch.pi * (torch.abs(yaw[:, 1] - yaw[:, 0]) > torch.pi)
        return torch.square((tu.get_euler_xyz(self.base_quat)[2] - feet_yaw_mean + torch.pi) % (2 * torch.pi) - torch.pi)

    def _reward_feet_distance(self):
        _, _, base_yaw = tu.get_euler_xyz(self.base_quat)
        feet_distance = torch.abs(
            torch.cos(base_yaw) * (self.get_feet_pos_quat()[0][:, 1, 1] - self.get_feet_pos_quat()[0][:, 0, 1])
            - torch.sin(base_yaw) * (self.get_feet_pos_quat()[0][:, 1, 0] - self.get_feet_pos_quat()[0][:, 0, 0])
        )
        return torch.clip(self.cfg.rewards.feet_distance_ref - feet_distance, min=0.0, max=0.1)

    def _reward_feet_swing(self):
        left_swing = (torch.abs(self.gait_process - 0.25) < 0.5 * self.cfg.rewards.swing_period) & (self.gait_frequency > 1.0e-8)
        right_swing = (torch.abs(self.gait_process - 0.75) < 0.5 * self.cfg.rewards.swing_period) & (self.gait_frequency > 1.0e-8)
        return (left_swing & ~self.feet_contact[:, 0]).float() + (right_swing & ~self.feet_contact[:, 1]).float()