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


from isaacgym import torch_utils as tu
from isaacgym import gymtorch, gymapi, gymutil

import torch
from legged_gym.envs import LeggedRobot

class T1(LeggedRobot):

    def _init_buffers(self):
        super()._init_buffers()
        self.gait_frequency = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.gait_process = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def step(self, actions):
        self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.gait_frequency, 1.0)
        obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras = super().step(actions)
        return obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        self.gait_frequency[env_ids] = tu.torch_rand_float(
            *self.cfg["commands"]["gait_frequency"], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.gait_frequency[self.still_envs] = 0.0
        self.gait_frequency[self.get_small_command_mask()] = 0.0

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        lower = self.dof_pos_limits[:, 0] + 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        upper = self.dof_pos_limits[:, 1] - 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        return torch.sum(((self.dof_pos < lower) | (self.dof_pos > upper)).float(), dim=-1)

    def _reward_feet_roll(self):
        roll, _, _ = tu.get_euler_xyz(self.get_feet_state()[1].reshape(-1, 4))
        roll = (roll.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        return torch.sum(torch.square(roll), dim=-1)

    def _reward_feet_yaw_diff(self):
        _, _, yaw = tu.get_euler_xyz(self.get_feet_state()[1].reshape(-1, 4))
        yaw = (yaw.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        return torch.square((yaw[:, 1] - yaw[:, 0] + torch.pi) % (2 * torch.pi) - torch.pi)

    def _reward_feet_yaw_mean(self):
        _, _, yaw = tu.get_euler_xyz(self.get_feet_state()[1].reshape(-1, 4))
        yaw = (yaw.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        feet_yaw_mean = yaw.mean(dim=-1) + torch.pi * (torch.abs(yaw[:, 1] - yaw[:, 0]) > torch.pi)
        return torch.square((tu.get_euler_xyz(self.base_quat)[2] - feet_yaw_mean + torch.pi) % (2 * torch.pi) - torch.pi)

    def _reward_feet_distance(self):
        _, _, base_yaw = tu.get_euler_xyz(self.base_quat)
        feet_distance = torch.abs(
            torch.cos(base_yaw) * (self.get_feet_state()[0][:, 1, 1] - self.get_feet_state()[0][:, 0, 1])
            - torch.sin(base_yaw) * (self.get_feet_state()[0][:, 1, 0] - self.get_feet_state()[0][:, 0, 0])
        )
        return torch.clip(self.cfg["rewards"]["feet_distance_ref"] - feet_distance, min=0.0, max=0.1)

    def _reward_feet_swing(self):
        left_swing = (torch.abs(self.gait_process - 0.25) < 0.5 * self.cfg["rewards"]["swing_period"]) & (self.gait_frequency > 1.0e-8)
        right_swing = (torch.abs(self.gait_process - 0.75) < 0.5 * self.cfg["rewards"]["swing_period"]) & (self.gait_frequency > 1.0e-8)
        return (left_swing & ~self.feet_contact[:, 0]).float() + (right_swing & ~self.feet_contact[:, 1]).float()