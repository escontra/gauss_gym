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

# from tkinter import Image
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from warnings import WarningMessage
import numpy as np
import os

from isaacgym import torch_utils as tu
from isaacgym import gymtorch, gymapi, gymutil

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.gaussian_terrain import GaussianSceneManager
from legged_gym.utils.math import wrap_to_pi, apply_randomization
from legged_gym.teacher import sensors
from legged_gym.teacher import observation_groups
from legged_gym.teacher.observation_manager import ObsManager
from legged_gym.utils import config


class LeggedRobot(BaseTask):
    def __init__(self, cfg: config.Config, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.reward_scales = dict(self.cfg.rewards.scales)
        self.command_ranges = dict(self.cfg.commands.ranges)
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.distance_exceeded_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.yaw_exceeded_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.initial_camera_set = False
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

        # Added observation manager to compute teacher observations more flexible
        self.sensors = {
            "raycast_grid": sensors.RayCaster(self),
            # "gs_renderer": self.scene_manager.renderer,
            "base_height_raycaster": sensors.LinkHeightSensor(self, [self.cfg.asset.base_link_name], color=(0.5, 0.0, 0.5)),
            "hip_height_raycaster": sensors.LinkHeightSensor(self, self.cfg.asset.hip_link_names, color=(1.0, 0.41, 0.71)),
            "foot_contact_sensor": sensors.FootContactSensor(self)}
        self.obs_manager = ObsManager(self, [getattr(observation_groups, group) for group in self.cfg.observations.observation_groups])

    def obs_group_size(self, group_name):
        return self.obs_manager.obs_dims_per_group[group_name]

    def obs_group_size_per_name(self, group_name):
        return self.obs_manager.obs_dims_per_group_obs[group_name]

    def clip_position_action_by_torque_limit(self, actions_scaled):
        """ For position control, scaled actions should be in the coordinate of robot default dof pos
        """
        dof_vel = self.dof_vel
        dof_pos_ = self.dof_pos - self.default_dof_pos
        p_limits_low = (-self.torque_limits) + self.dof_damping * dof_vel
        p_limits_high = (self.torque_limits) + self.dof_damping * dof_vel
        actions_low = (p_limits_low/self.dof_stiffness) + dof_pos_
        actions_high = (p_limits_high/self.dof_stiffness) + dof_pos_
        actions_scaled_clipped = torch.clip(actions_scaled, actions_low, actions_high)
        return actions_scaled_clipped

    def pre_physics_step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        if isinstance(clip_actions, (tuple, list)):
            clip_actions = torch.tensor(clip_actions, device=self.device)

        if getattr(self.cfg.normalization, "clip_actions_delta", None) is not None:
            actions = torch.clip(
                actions,
                self.last_actions - self.cfg.normalization.clip_actions_delta,
                self.last_actions + self.cfg.normalization.clip_actions_delta,
            )
        
        # some customized action clip methods to bound the action output
        if self.cfg.normalization.clip_actions_method == "tanh":
            actions = (torch.tanh(actions) * clip_actions).to(self.device)
        elif self.cfg.normalization.clip_actions_method == "hard":
            actions = torch.clip(
                actions, self.clip_actions_low, self.clip_actions_high,
            )
        else:
            actions = torch.clip(actions, -clip_actions, clip_actions)

        action_scale = self.cfg.control.action_scale
        if isinstance(action_scale, (tuple, list)):
            action_scale = torch.tensor(action_scale, device= self.sim_device)

        actions_scaled_clipped = actions * action_scale
        if self.cfg.control.computer_clip_torque:
            if self.cfg.control.control_type == "P":
                actions_scaled_clipped = self.clip_position_action_by_torque_limit(actions_scaled_clipped)
            else:
                raise NotImplementedError

        self.last_contact_forces[:] = self.contact_forces

        return actions, actions_scaled_clipped

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions, actions_scaled_clipped = self.pre_physics_step(actions)
        self.actions[:] = actions
        # step physics and render each frame
        self.render()
        for dec_i in range(self.cfg.control.decimation):
            self.pre_decimation_step(dec_i)
            self.torques = self._compute_torques(actions, actions_scaled_clipped).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.post_decimation_step(dec_i)
        self.post_physics_step()
        return self.obs_dict["student_observations"], self.obs_dict["teacher_observations"], self.rew_buf, self.reset_buf, self.extras

    def pre_decimation_step(self, dec_i):
        self.last_dof_vel[:] = self.dof_vel

    def post_decimation_step(self, dec_i):
        self.substep_torques[:, dec_i, :] = self.torques
        self.substep_dof_vel[:, dec_i, :] = self.dof_vel
        self.substep_exceed_dof_pos_limits[:, dec_i, :] = (self.dof_pos < self.dof_pos_limits[:, 0]) | (self.dof_pos > self.dof_pos_limits[:, 1])
        self.substep_exceed_dof_pos_limit_abs[:, dec_i, :] = torch.clip(torch.maximum(
            self.dof_pos_limits[:, 0] - self.dof_pos,
            self.dof_pos - self.dof_pos_limits[:, 1],
        ), min= 0) # make sure the value is non-negative

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = tu.quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = tu.quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = tu.quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        for sensor in self.sensors.values():
            sensor.update(-1)
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
        self.feet_contact[:] = self.sensors["foot_contact_sensor"].get_data()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_feet_pos[:] = self.get_feet_pos_quat()[0]
        self.last_contacts[:] = self.feet_contact[:]
        self.last_torques[:] = self.torques[:]

        self.obs_dict = self.obs_manager.compute_obs(self)        

    def _update_physics_curriculum(self):

        if self.cfg.domain_rand.dof_friction_curriculum.apply:
            if (self.common_step_counter - 1) % self.cfg.domain_rand.dof_friction_curriculum.update_every == 0:
                start = self.cfg.domain_rand.dof_friction_curriculum.start
                end = self.cfg.domain_rand.dof_friction_curriculum.end
                progress = np.clip(self.common_step_counter / self.cfg.domain_rand.dof_friction_curriculum.steps, 0., 1.)
                curr_end = start + (end - start) * progress
                print(f'Friction Curriculum [{self.common_step_counter}] progress: {progress}, start: {start}, curr_end: {curr_end}, end: {end}')
                for i in range(self.num_envs):
                    dof_props = self.gym.get_actor_dof_properties(
                        self.envs[i],
                        self.actor_handles[i])
                    if self.cfg.domain_rand.dof_friction_curriculum.sample_type == "end":
                        new_friction = curr_end
                    elif self.cfg.domain_rand.dof_friction_curriculum.sample_type == "uniform":
                        lower_bound = min(start, curr_end)
                        upper_bound = max(start, curr_end)
                        new_friction = np.random.uniform(lower_bound, upper_bound)
                    else:
                        raise ValueError(f"Invalid sample type: {self.cfg.domain_rand.dof_friction_curriculum.sample_type}")
                    dof_props["friction"].fill(new_friction)
                    self.dof_friction_curriculum_values[i] = new_friction
                    self.gym.set_actor_dof_properties(
                        self.envs[i],
                        self.actor_handles[i],
                        dof_props)

    def check_termination(self):
        """ Check if environments need to be reset."""
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= (self.root_states[:, 2] - self.measured_base_heights_mesh[..., 0]) < self.cfg.rewards.terminate_height

        distance_exceeded, yaw_exceeded = self.scene_manager.check_termination()
        self.distance_exceeded_buf = distance_exceeded
        self.yaw_exceeded_buf = yaw_exceeded

        self.reset_buf |= (self.distance_exceeded_buf | self.yaw_exceeded_buf)
        self.time_out_buf |= (self.distance_exceeded_buf | self.yaw_exceeded_buf)

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        
        self._fill_extras(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # Sample commands to track camera trajectory.
        self._resample_commands(env_ids)
        self._reset_buffers(env_ids)

    def _fill_extras(self, env_ids):
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.extras["episode"]['rew_frame_' + key] = torch.nanmean(self.episode_sums[key][env_ids] / self.episode_length_buf[env_ids])
            self.episode_sums[key][env_ids] = 0.
        # log power related info
        self.extras["episode"]["max_power_throughout_episode"] = self.max_power_per_timestep[env_ids].max().cpu().item()
        # log whether the episode ends by timeout or dead, or by reaching the goal
        self.extras["episode"]["timeout_ratio"] = self.time_out_buf.float().sum() / self.reset_buf.float().sum()
        self.extras["episode"]["distance_exceeded_ratio"] = self.distance_exceeded_buf.float().sum() / self.reset_buf.float().sum()
        self.extras["episode"]["yaw_exceeded_ratio"] = self.yaw_exceeded_buf.float().sum() / self.reset_buf.float().sum()
        self.extras["episode"]["num_terminated"] = self.reset_buf.float().sum()
        for i in range(len(self.feet_indices)):
            self.extras["episode"][f"{self.feet_names[i]}_contact_force"] = torch.mean(self.contact_forces[:, self.feet_indices[i], 2])
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf | self.distance_exceeded_buf | self.yaw_exceeded_buf

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        # print('---')
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            orig_rew = self.reward_functions[i]()
            rew = orig_rew * self.reward_scales[name]
            # print(name, orig_rew.float().mean(), self.reward_scales[name])
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.scene_manager = GaussianSceneManager(self)
        self.scene_manager.spawn_meshes()
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        for idx_range in self.feet_rigid_body_shape_indices_range:
            for s in range(*idx_range):
                if self.cfg.domain_rand.foot_friction.apply:
                  props[s].friction = apply_randomization(0.0, self.cfg.domain_rand.foot_friction)
                if self.cfg.domain_rand.foot_compliance.apply:
                  props[s].compliance = apply_randomization(0.0, self.cfg.domain_rand.foot_compliance)
                if self.cfg.domain_rand.foot_restitution.apply:
                  props[s].restitution = apply_randomization(0.0, self.cfg.domain_rand.foot_restitution)

        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            # allow config to override torque limits
            if hasattr(self.cfg.control, "torque_limits"):
                if not isinstance(self.cfg.control.torque_limits, (tuple, list)):
                    self.torque_limits = torch.ones(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
                    self.torque_limits *= self.cfg.control.torque_limits
                else:
                    self.torque_limits = torch.tensor(self.cfg.control.torque_limits, dtype=torch.float, device=self.device, requires_grad=False)

        return props

    def _process_rigid_body_props(self, props, env_id):
        for j in range(self.num_bodies):
            if j == self.base_link_index:
                if self.cfg.domain_rand.base_com_x.apply:
                    props[j].com.x, self.base_mass_scaled[env_id, 0] = apply_randomization(
                        props[j].com.x, self.cfg.domain_rand.base_com_x, return_noise=True
                    )
                if self.cfg.domain_rand.base_com_y.apply:
                    props[j].com.y, self.base_mass_scaled[env_id, 1] = apply_randomization(
                        props[j].com.y, self.cfg.domain_rand.base_com_y, return_noise=True
                    )
                if self.cfg.domain_rand.base_com_z.apply:
                    props[j].com.z, self.base_mass_scaled[env_id, 2] = apply_randomization(
                        props[j].com.z, self.cfg.domain_rand.base_com_z, return_noise=True
                    )
                if self.cfg.domain_rand.base_mass.apply:
                    props[j].mass, self.base_mass_scaled[env_id, 3] = apply_randomization(
                        props[j].mass, self.cfg.domain_rand.base_mass, return_noise=True
                    )
                    props[j].invMass = 1.0 / props[j].mass
            else:
                if self.cfg.domain_rand.other_com.apply:
                    props[j].com.x = apply_randomization(props[j].com.x, self.cfg.domain_rand.other_com)
                    props[j].com.y = apply_randomization(props[j].com.y, self.cfg.domain_rand.other_com)
                    props[j].com.z = apply_randomization(props[j].com.z, self.cfg.domain_rand.other_com)
                if self.cfg.domain_rand.other_mass.apply:
                    props[j].mass = apply_randomization(props[j].mass, self.cfg.domain_rand.other_mass)
                    props[j].invMass = 1.0 / props[j].mass
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """

        # Resample command scales. Commands are updated every timestep to guide
        # the robot along the camera trajectory.
        resample_command_env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(resample_command_env_ids)

        self._update_physics_curriculum()

        # Resample sensor latency.
        resample_latency_env_ids = (self.episode_length_buf % int(self.cfg.observations.latency_resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self.obs_manager.resample_sensor_latency(resample_latency_env_ids)

        # log max power across current env step
        self.max_power_per_timestep = torch.maximum(
            self.max_power_per_timestep,
            torch.max(torch.sum(self.substep_torques * self.substep_dof_vel, dim= -1), dim= -1)[0],
        )

        self.measured_base_heights_mesh = self.sensors["base_height_raycaster"].get_data()[..., 2]
        self.measured_hip_heights_mesh = self.sensors["hip_height_raycaster"].get_data()[..., 2]

        self._push_robots()
        self._kick_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        heading_command, velocity_command = self.scene_manager.sample_commands(env_ids)
        if self.cfg.commands.heading_command:
            self.commands[:, 3] = heading_command
        else:
            raise NotImplementedError()
        self.commands[:, :2] = velocity_command
        if self.cfg.commands.heading_command:
            forward = tu.quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        still_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.commands.still_proportion
        still_envs = env_ids[still_mask]
        self.commands[still_envs, :] = 0.0

        return still_envs

    def _compute_torques_original(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.dof_stiffness*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.dof_damping*self.dof_vel
            friction = torch.min(self.dof_friction, torques.abs()) * torch.sign(torques)
            torques = torques - friction
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _compute_torques(self, actions, actions_scaled_clipped):
        if not self.cfg.control.computer_clip_torque:
            return self._compute_torques_original(actions)
        else:
            if self.cfg.control.control_type == "P":
                torques = self.dof_stiffness * (actions_scaled_clipped + self.default_dof_pos - self.dof_pos) \
                    - self.dof_damping * self.dof_vel
                friction = torch.min(self.dof_friction, torques.abs()) * torch.sign(torques)
                torques = torques - friction
            else:
                raise NotImplementedError
            if self.cfg.control.motor_clip_torque:
                torques = torch.clip(
                    torques,
                    -self.torque_limits * self.cfg.control.motor_clip_torque,
                    self.torque_limits * self.cfg.control.motor_clip_torque,
                )
            return torques

    def _reset_buffers(self, env_ids):
        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_feet_pos[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.last_contacts[env_ids] = False
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.last_torques[env_ids] = 0.
        self.max_power_per_timestep[env_ids] = 0.

        # Reset observation buffers.
        self.obs_manager.reset_buffers(env_ids)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """

        if self.cfg.domain_rand.init_dof_pos.apply:
          self.dof_pos[env_ids] = apply_randomization(
              self.default_dof_pos.expand(len(env_ids), -1),
              self.cfg.domain_rand.init_dof_pos
          )
        else:
          self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # Sample starting poses.
        cam_trans, cam_quat = self.scene_manager.sample_cam_pose(env_ids, use_ground_positions=True)
        self.root_states[env_ids, :3] += cam_trans
        self.root_states[env_ids, 3:7] = cam_quat
        # Linear velocity is 7:10, angular velocity is 10:13.
        if self.cfg.domain_rand.init_base_lin_vel_xy.apply:
          self.root_states[env_ids, 7:9] = apply_randomization(
              torch.zeros(len(env_ids), 2, dtype=torch.float, device=self.device),
              self.cfg.domain_rand.init_base_lin_vel_xy
          )
          self.root_states[env_ids, 9:13] = 0.
        else:
          self.root_states[env_ids, 7:13] = 0.
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _kick_robots(self):
        """ Random kick the robots. Emulates an impulse by setting a randomized base velocity. """
        if self.common_step_counter % np.ceil(self.cfg.domain_rand.kick_interval_s / self.dt) == 0:
            if self.cfg.domain_rand.kick_lin_vel.apply:
              self.root_states[:, 7:10] = apply_randomization(self.root_states[:, 7:10], self.cfg.domain_rand.kick_lin_vel)
            if self.cfg.domain_rand.kick_ang_vel.apply:
              self.root_states[:, 10:13] = apply_randomization(self.root_states[:, 10:13], self.cfg.domain_rand.kick_ang_vel)
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        if self.common_step_counter % np.ceil(self.cfg.domain_rand.push_interval_s / self.dt) == 0:
            if self.cfg.domain_rand.push_force.apply:
              self.pushing_forces[:, self.base_link_index, :] = apply_randomization(
                  torch.zeros_like(self.pushing_forces[:, 0, :]),
                  self.cfg.domain_rand.push_force,
              )
            if self.cfg.domain_rand.push_torque.apply:
              self.pushing_torques[:, self.base_link_index, :] = apply_randomization(
                  torch.zeros_like(self.pushing_torques[:, 0, :]),
                  self.cfg.domain_rand.push_torque,
              )
        elif self.common_step_counter % np.ceil(self.cfg.domain_rand.push_interval_s / self.dt) == np.ceil(
            self.cfg.domain_rand.push_duration_s / self.dt
        ):
            self.pushing_forces[:, self.base_link_index, :].zero_()
            self.pushing_torques[:, self.base_link_index, :].zero_()

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.pushing_forces),
            gymtorch.unwrap_tensor(self.pushing_torques),
            gymapi.LOCAL_SPACE,
        )
        
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    def get_camera_link_state(self):
      """Get the position of the camera link."""
      camera_link_state = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.camera_link_indices]
      camera_link_state = camera_link_state.squeeze(1)
      return camera_link_state 

    def get_feet_pos_quat(self):
      feet_pos = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
      feet_quat = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 3:7]
      return feet_pos, feet_quat

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = tu.to_torch(tu.get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = tu.to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contact_forces = torch.zeros_like(self.contact_forces)
        self.last_feet_pos = torch.zeros_like(self.get_feet_pos_quat()[0])
        self.feet_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.pushing_forces = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        self.pushing_torques = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        self.dof_friction_curriculum_values = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.base_lin_vel = tu.quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = tu.quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = tu.quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.measured_heights = 0
        self.measured_base_heights_mesh = 0
        self.measured_hip_heights_mesh = 0
        self.substep_torques = torch.zeros(self.num_envs, self.cfg.control.decimation, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.substep_dof_vel = torch.zeros(self.num_envs, self.cfg.control.decimation, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.substep_exceed_dof_pos_limits = torch.zeros(self.num_envs, self.cfg.control.decimation, self.num_dof, dtype=torch.bool, device=self.device, requires_grad=False)
        self.substep_exceed_dof_pos_limit_abs = torch.zeros(self.num_envs, self.cfg.control.decimation, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.max_power_per_timestep = torch.zeros(self.num_envs, dtype= torch.float32, device= self.device)


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        self.feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        camera_link_names = [s for s in body_names if self.cfg.asset.camera_link_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        if hasattr(self.cfg.asset, "front_hip_names"):
            front_hip_names = getattr(self.cfg.asset, "front_hip_names")
            self.front_hip_indices = torch.zeros(len(front_hip_names), dtype=torch.long, device=self.device, requires_grad=False)
            for i, name in enumerate(front_hip_names):
                self.front_hip_indices[i] = self.gym.find_asset_dof_index(self.robot_asset, name)
        else:
            front_hip_names = []

        if hasattr(self.cfg.asset, "rear_hip_names"):
            rear_hip_names = getattr(self.cfg.asset, "rear_hip_names")
            self.rear_hip_indices = torch.zeros(len(rear_hip_names), dtype=torch.long, device=self.device, requires_grad=False)
            for i, name in enumerate(rear_hip_names):
                self.rear_hip_indices[i] = self.gym.find_asset_dof_index(self.robot_asset, name)
        else:
            rear_hip_names = []

        self.num_actions = self.num_dofs

        hip_names = front_hip_names + rear_hip_names
        if len(hip_names) > 0:
            self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
            for i, name in enumerate(hip_names):
                self.hip_indices[i] = self.gym.find_asset_dof_index(self.robot_asset, name)

        self.feet_indices = torch.zeros(len(self.feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.feet_names)):
            self.feet_indices[i] = self.gym.find_asset_rigid_body_index(self.robot_asset, self.feet_names[i])

        self.camera_link_indices = torch.zeros(len(camera_link_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(camera_link_names)):
            self.camera_link_indices[i] = self.gym.find_asset_rigid_body_index(self.robot_asset, camera_link_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_asset_rigid_body_index(self.robot_asset, penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_asset_rigid_body_index(self.robot_asset, termination_contact_names[i])

        self.base_link_index = self.gym.find_asset_rigid_body_index(self.robot_asset, self.cfg.asset.base_link_name)

        # Rigid body shape indices corresponding to the feet.
        shape_indices = self.gym.get_asset_rigid_body_shape_indices(self.robot_asset)
        self.feet_rigid_body_shape_indices_range = []
        for i in range(len(self.feet_names)):
            start = shape_indices[self.feet_indices[i]].start
            count = shape_indices[self.feet_indices[i]].count
            self.feet_rigid_body_shape_indices_range.append((start, start + count))

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = tu.to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.clip_actions = None
        clip_actions_low = []
        clip_actions_high = []
        if self.cfg.normalization.clip_actions_method == 'hard':
          sim_sdk_map = self.cfg.normalization.sim_sdk_map
          const_dof_range = self.cfg.normalization.const_dof_range
          dof_pos_redundancy = self.cfg.normalization.dof_pos_redundancy
            
        self.dof_stiffness = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_damping = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_friction = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.default_dof_pos = torch.zeros(1, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        # joint positions offsets and PD gains
        for i in range(self.num_dofs):
            default_joint_angle = self.cfg.init_state.default_joint_angles[self.dof_names[i]]
            self.default_dof_pos[:, i] = default_joint_angle
            if self.cfg.normalization.clip_actions_method == 'hard':
                clip_actions_low.append(
                    (const_dof_range[sim_sdk_map[self.dof_names[i]] + "_min"] + dof_pos_redundancy - default_joint_angle) / self.cfg.control.action_scale)
                clip_actions_high.append(
                    (const_dof_range[sim_sdk_map[self.dof_names[i]] + "_max"] - dof_pos_redundancy - default_joint_angle) / self.cfg.control.action_scale)

            found = False
            for name in self.cfg.control.stiffness:
                if name in self.dof_names[i]:
                    self.dof_stiffness[:, i] = self.cfg.control.stiffness[name]
                    self.dof_damping[:, i] = self.cfg.control.damping[name]
                    found = True
            if not found:
                raise ValueError(f"PD gain of joint {self.dof_names[i]} were not defined")
        if self.cfg.normalization.clip_actions_method == 'hard':
            self.clip_actions_low = torch.tensor(clip_actions_low, device=self.device)
            self.clip_actions_high = torch.tensor(clip_actions_high, device=self.device)
        if self.cfg.domain_rand.dof_stiffness.apply:
            self.dof_stiffness = apply_randomization(self.dof_stiffness, self.cfg.domain_rand.dof_stiffness)
        if self.cfg.domain_rand.dof_damping.apply:
            self.dof_damping = apply_randomization(self.dof_damping, self.cfg.domain_rand.dof_damping)
        if self.cfg.domain_rand.dof_friction.apply:
            self.dof_friction = apply_randomization(self.dof_friction, self.cfg.domain_rand.dof_friction)

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.base_mass_scaled = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += tu.torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props_randomized = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props_randomized)
            actor_handle = self.gym.create_actor(env_handle, self.robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props_randomized = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props_randomized)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props_randomized = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props_randomized, recomputeInertia=True)
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        self.env_origins = self.scene_manager.env_origins

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        for sensor in self.sensors.values():
          sensor.debug_vis(self)
        self.scene_manager.debug_vis(self)
        if not self.headless:
            if self.selected_environment >= 0 and (self.selected_environment_changed or not self.initial_camera_set):
                # Point at the middle of the camera trajectory.
                mesh_id = self.scene_manager.mesh_id_for_env_id(self.selected_environment)
                cam_trans = self.scene_manager.cam_trans_viz[mesh_id]
                lookat = cam_trans[cam_trans.shape[0] // 2].cpu().numpy()
                pos = (np.array(self.cfg.viewer.pos) - (self.cfg.viewer.lookat) + lookat).tolist()
                self.set_camera(pos, lookat.tolist())
                self.initial_camera_set = True

    #------------ reward functions----------------

    def _reward_alive(self):
        return 1.

    def _reward_energy_substeps(self):
        # (n_envs, n_substeps, n_dofs) 
        # square sum -> (n_envs, n_substeps)
        # mean -> (n_envs,)
        return torch.mean(torch.sum(torch.square(self.substep_torques * self.substep_dof_vel), dim=-1), dim=-1)

    def _reward_energy(self):
        return torch.sum(torch.square(self.torques * self.dof_vel), dim=1)

    def _reward_exceed_dof_pos_limits(self):
        return self.substep_exceed_dof_pos_limits.to(torch.float32).sum(dim=-1).mean(dim=-1)

    def _reward_exceed_torque_limits_i(self):
        """ Indicator function """
        max_torques = torch.abs(self.substep_torques).max(dim= 1)[0]
        exceed_torque_each_dof = max_torques > (self.torque_limits*self.cfg.rewards.soft_torque_limit)
        exceed_torque = exceed_torque_each_dof.any(dim= 1)
        return exceed_torque.to(torch.float32)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=-1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_base_heights_mesh, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
  
    def _reward_hip_height(self):
        # Penalize hip height away from target
        hip_heights = self.root_states[:, 2].unsqueeze(1) - self.measured_hip_heights_mesh
        hip_height_error = hip_heights - self.cfg.rewards.base_height_target
        hip_height_error = torch.square(hip_height_error).sum(dim=-1)
        return hip_height_error
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=-1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=-1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=-1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=-1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 1.0, dim=-1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=-1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=-1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=-1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=-1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (x axes)
        return torch.exp(-torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0]) / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (y axes)
        return torch.exp(-torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1]) / self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.feet_contact[:]
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        # rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        # feet_air_time = self.feet_air_time
        # if self.cfg["rewards"]["feet_air_time_max_time"] is not None:
        #     feet_air_time = feet_air_time.clip(max=self.cfg["rewards"]["feet_air_time_max_time"])
        if self.cfg.rewards.feet_air_time_max_time is not None:
          time_diff = torch.where(
              self.feet_air_time > self.cfg.rewards.feet_air_time_max_time,
              -1.0,
              self.feet_air_time - self.cfg.rewards.feet_air_time_min_time
          )
        else:
          time_diff = self.feet_air_time - self.cfg.rewards.feet_air_time_min_time
        # if self.cfg["rewards"]["feet_air_time_max_time"] is not None:
        #   print('CLIPPING BY MAX TIME')
          # time_diff = time_diff.clip(max=self.cfg["rewards"]["feet_air_time_max_time"])
        rew_airTime = torch.sum(time_diff * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=-1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=-1) * (torch.norm(self.commands[:, :2], dim=-1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=-1)

    def _reward_feet_slip(self):
        # Penalize feet velocities when contact
        return (
            torch.sum(
                torch.square((self.last_feet_pos - self.get_feet_pos_quat()[0]) / self.dt).sum(dim=-1) * self.feet_contact.float(),
                dim=-1,
            )
            * (self.episode_length_buf > 1).float()
        )

    def _reward_root_acc(self):
        # Penalize root accelerations
        return torch.sum(torch.square((self.last_root_vel - self.root_states[:, 7:13]) / self.dt), dim=-1)

    def _reward_survival(self):
        # Reward survival
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_torque_tiredness(self):
        # Penalize torque tiredness
        return torch.sum(torch.square(self.torques / self.torque_limits).clip(max=1.0), dim=-1)

    def _reward_power(self):
        # Penalize power
        return torch.sum((self.torques * self.dof_vel).clip(min=0.0), dim=-1)

    def _reward_feet_vel_z(self):
        return torch.sum(torch.square((self.last_feet_pos - self.get_feet_pos_quat()[0]) / self.dt)[:, :, 2], dim=-1)

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)

    def _reward_exceed_torque_limits_l1norm(self):
        """ square function for exceeding part """
        exceeded_torques = torch.abs(self.substep_torques) - (self.torque_limits*self.cfg.rewards.soft_torque_limit)
        exceeded_torques[exceeded_torques < 0.] = 0.
        # sum along decimation axis and dof axis
        return torch.norm(exceeded_torques, p=1, dim=-1).sum(dim=1)
