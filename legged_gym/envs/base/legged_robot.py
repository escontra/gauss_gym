import numpy as np
import copy
import functools
import os
from itertools import chain

from isaacgym import torch_utils as tu
from isaacgym import gymtorch, gymapi, gymutil

import torch

import legged_gym
from legged_gym.envs.base import base_task
from legged_gym.utils import (
  sensors,
  observation_groups,
  observation_manager,
  gaussian_terrain,
  math,
  timer,
  space
)


class LeggedRobot(base_task.BaseTask):
    def __init__(self, cfg):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
        """
        self.cfg = cfg
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        super().__init__(self.cfg)
        self.dt = self.cfg["control"]["decimation"] * self.sim_params.dt
        self.command_ranges = dict(self.cfg["commands"]["ranges"])
        self.max_episode_length_s = self.cfg["env"]["episode_length_s"]
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.distance_exceeded_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.yaw_exceeded_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.initial_camera_set = False
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

        # Added observation manager to compute teacher observations more flexible
        self.sensors = {
            "raycast_grid": sensors.RayCaster(self),
            "foot_height_raycaster": sensors.LinkHeightSensor(self, self.feet_names, color=(0.5, 0.0, 0.5)),
            "base_height_raycaster": sensors.LinkHeightSensor(self, [self.cfg["asset"]["base_link_name"]], color=(0.5, 0.0, 0.5)),
            "hip_height_raycaster": sensors.LinkHeightSensor(self, self.cfg["asset"]["hip_link_names"], color=(1.0, 0.41, 0.71)),
            "foot_contact_sensor": sensors.FootContactSensor(self)}
        obs_groups = observation_groups.observation_groups_from_dict(self.cfg["observations"])

        # Maybe add renderer to sensors.
        all_observations = list(chain.from_iterable(obs_group.observations for obs_group in obs_groups))
        need_renderer = self.cfg["env"]["force_renderer"] or observation_groups.CAMERA_IMAGE in all_observations
        if need_renderer:
            self.sensors["gs_renderer"] = self.scene_manager.renderer

        self.obs_manager = observation_manager.ObsManager(self, obs_groups)


    def clip_position_action_by_torque_limit(self, actions_scaled, dof_stiffness, dof_damping):
      """For position control, scaled actions should be in the coordinate of robot default dof pos"""
      dof_vel = self.dof_vel
      dof_pos_ = self.dof_pos - self.default_dof_pos
      p_limits_low = (-self.torque_limits) + dof_damping * dof_vel
      p_limits_high = (self.torque_limits) + dof_damping * dof_vel
      actions_low = (p_limits_low / dof_stiffness) + dof_pos_
      actions_high = (p_limits_high / dof_stiffness) + dof_pos_
      actions_scaled_clipped = torch.clip(
        actions_scaled, actions_low, actions_high
      )
      return actions_scaled_clipped

    def obs_space(self):
        return self.obs_manager.obs_dims_per_group_obs

    def action_space(self):
        act_space = {
            'actions': space.Space(
                shape=(self.num_actions,),
                dtype=np.float32
            )
        }
        if self.cfg["commands"]["command_gains"]:
            act_space['stiffness'] = space.Space(
                shape=(self.num_actions,),
                low=self.default_dof_stiffness[0].cpu().numpy() * self.cfg["commands"]["command_gains_stiffness_range"][0],
                high=self.default_dof_stiffness[0].cpu().numpy() * self.cfg["commands"]["command_gains_stiffness_range"][1],
                dtype=np.float32
            )
            act_space['damping'] = space.Space(
                shape=(self.num_actions,),
                low=self.default_dof_damping[0].cpu().numpy() * self.cfg["commands"]["command_gains_damping_range"][0],
                high=self.default_dof_damping[0].cpu().numpy() * self.cfg["commands"]["command_gains_damping_range"][1],
                dtype=np.float32
            )
        return act_space

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        if isinstance(actions, dict):
            act = actions['actions']
            if 'stiffness' in actions:
                dof_stiffness = actions['stiffness']
            else:
                dof_stiffness = self.default_dof_stiffness.expand(self.num_envs, -1)
            if 'damping' in actions:
                dof_damping = actions['damping']
            else:
                dof_damping = self.default_dof_damping.expand(self.num_envs, -1)
        else:
            act = actions
            dof_stiffness = self.default_dof_stiffness.expand(self.num_envs, -1)
            dof_damping = self.default_dof_damping.expand(self.num_envs, -1)

        # some customized action clip methods to bound the action output
        if self.cfg["normalization"]["clip_actions_method"] == "normal":
            act = torch.clip(act, -self.clip_actions, self.clip_actions)
        elif self.cfg["normalization"]["clip_actions_method"] == "tanh":
            act = (torch.tanh(act) * self.clip_actions).to(self.device)
        elif self.cfg["normalization"]["clip_actions_method"] == "hard":
            act = torch.clip(act, self.clip_actions_low, self.clip_actions_high)
        else:
            raise NotImplementedError(f"Clip action method {self.cfg['normalization']['clip_actions_method']} is not implemented")

        self.actions[:] = act
        self.stiffness[:] = dof_stiffness
        self.damping[:] = dof_damping

        # step physics and render each frame
        self.render()
        with timer.section("physics_step"):
            for dec_i in range(self.cfg["control"]["decimation"]):
                self.pre_decimation_step(dec_i)
                self.torques = self._compute_torques(act, dof_stiffness, dof_damping).view(self.torques.shape)
                with timer.section("simulate"):
                    self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
                    self.gym.simulate(self.sim)
                    if self.device == 'cpu':
                        self.gym.fetch_results(self.sim, True)
                    self.gym.refresh_dof_state_tensor(self.sim)
                self.post_decimation_step(dec_i)
        self.post_physics_step()
        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras

    @timer.section("pre_decimation_step")
    def pre_decimation_step(self, dec_i):
        self.last_dof_vel[:] = self.dof_vel

    @timer.section("post_decimation_step")
    def post_decimation_step(self, dec_i):
        self.substep_torques[:, dec_i, :] = self.torques
        self.substep_dof_vel[:, dec_i, :] = self.dof_vel
        self.substep_exceed_dof_pos_limits[:, dec_i, :] = (self.dof_pos < self.dof_pos_limits[:, 0]) | (self.dof_pos > self.dof_pos_limits[:, 1])
        self.substep_exceed_dof_pos_limit_abs[:, dec_i, :] = torch.clip(torch.maximum(
            self.dof_pos_limits[:, 0] - self.dof_pos,
            self.dof_pos - self.dof_pos_limits[:, 1],
        ), min= 0) # make sure the value is non-negative

    @timer.section("post_physics_step")
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._draw_debug_vis() if needed
        """
        with timer.section("refresh_sim"):
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
          with timer.section("update_sensors"):
            for sensor in self.sensors.values():
                sensor.update(-1)
          if self.viewer and self.enable_viewer_sync and self.debug_viz:
              self._draw_debug_vis()
          self.feet_contact[:] = self.sensors["foot_contact_sensor"].get_data()

          # Initialize buffers with initial values when necessary.
          self._initialize_prev_reset_buffer(self.prev_reset)

          # Update base velocity moving average.
          self.filtered_lin_vel[:] = self.base_lin_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_lin_vel[:] * (
              1.0 - self.cfg["normalization"]["filter_weight"]
          )
          self.filtered_ang_vel[:] = self.base_ang_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_ang_vel[:] * (
              1.0 - self.cfg["normalization"]["filter_weight"]
          )

          # log max power across current env step
          self.max_power_per_timestep = torch.maximum(
              self.max_power_per_timestep,
              torch.max(torch.sum(self.substep_torques * self.substep_dof_vel, dim= -1), dim= -1)[0],
          )

        # Push and kick robots.
        self._push_robots()
        self._kick_robots()

        # Update physics curriculum.
        self._update_physics_curriculum()

        # Check if is first contact.
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact_filt = torch.logical_or(self.feet_contact[:], self.last_contacts) 
        self.first_contact = (self.feet_air_time > 0.) * contact_filt
        self.last_contact = (self.feet_contact_time > 0.) * ~contact_filt
        self.feet_air_time += self.dt
        self.feet_contact_time += self.dt
        self.swing_peak = torch.maximum(
            self.swing_peak,
            self.sensors["foot_height_raycaster"].get_data())

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()

        self.swing_peak *= ~contact_filt
        self.feet_air_time *= ~contact_filt
        self.feet_contact_time *= contact_filt
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_contacts[:] = self.feet_contact[:]
        self.last_torques[:] = self.torques[:]
        self.last_contact_forces[:] = self.contact_forces

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.prev_reset = env_ids

        # Resample command scales. Commands are updated every timestep to guide
        # the robot along the camera trajectory. Call after `reset_idx` to
        # sample a new command at the start of each episode.
        resample_command_env_ids = (self.episode_length_buf % int(self.cfg["commands"]["resampling_time"] / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(resample_command_env_ids)

        # Resample sensor latency.
        self.obs_manager.resample_sensor_latency()

        # Compute observations.
        self.obs_dict = self.obs_manager.compute_obs(self)        

    def _update_physics_curriculum(self):

        if self.cfg["curriculum"]["dof_friction"]["apply"] and self.cfg["curriculum"]["apply_curriculum"]:
            update_every = self.cfg["curriculum"]["dof_friction"]["update_every"] * self.cfg["runner"]["num_steps_per_env"]
            if (self.common_step_counter - 1) % update_every == 0:
                start = self.cfg["curriculum"]["dof_friction"]["start"]
                end = self.cfg["curriculum"]["dof_friction"]["end"]
                total_steps = self.cfg["curriculum"]["dof_friction"]["epochs"] * self.cfg["runner"]["num_steps_per_env"]
                progress = np.clip(self.common_step_counter / total_steps, 0., 1.)
                curr_end = start + (end - start) * progress
                curr_epoch = self.common_step_counter // self.cfg["runner"]["num_steps_per_env"]
                print(f'Friction Curriculum [{curr_epoch}] progress: {progress*100:.2f}%, start: {start:.3f}, curr_end: {curr_end:.3f}, end: {end:.3f}')
                for i in range(self.num_envs):
                    dof_props = self.gym.get_actor_dof_properties(
                        self.envs[i],
                        self.actor_handles[i])
                    if self.cfg["curriculum"]["dof_friction"]["sample_type"] == "end":
                        new_friction = curr_end
                    elif self.cfg["curriculum"]["dof_friction"]["sample_type"] == "uniform":
                        new_friction = np.random.uniform(min(start, curr_end), max(start, curr_end))
                    else:
                        raise ValueError(f"Invalid sample type: {self.cfg['curriculum']['dof_friction']['sample_type']}")
                    dof_props["friction"].fill(new_friction)
                    self.dof_friction_curriculum_values[i] = new_friction
                    self.gym.set_actor_dof_properties(
                        self.envs[i],
                        self.actor_handles[i],
                        dof_props)

    @timer.section("check_termination")
    def check_termination(self):
        """ Check if environments need to be reset."""
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.sensors["base_height_raycaster"].get_data()[..., 0] < self.cfg["termination"]["base_height_threshold"]
        self.reset_buf |= self.projected_gravity[:, -1] > self.cfg["termination"]["projected_gravity_z_threshold"]

        distance_exceeded, yaw_exceeded = self.scene_manager.check_termination()
        self.distance_exceeded_buf = distance_exceeded
        self.yaw_exceeded_buf = yaw_exceeded

        self.reset_buf |= (self.distance_exceeded_buf | self.yaw_exceeded_buf)
        self.time_out_buf |= (self.distance_exceeded_buf | self.yaw_exceeded_buf)

    @timer.section("reset_idx")
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
        self._reset_buffers(env_ids)
        self._resample_commands(env_ids)

    def _fill_extras(self, env_ids):
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = (torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s).cpu().item()
            self.episode_sums[key][env_ids] = 0.
        self.extras["episode"]["distance_traveled_x"] = torch.mean(torch.abs(self.root_states[env_ids, 0] - self.init_positions[env_ids, 0])).cpu().item()
        self.extras["episode"]["distance_traveled_y"] = torch.mean(torch.abs(self.root_states[env_ids, 1] - self.init_positions[env_ids, 1])).cpu().item()
        self.extras["episode"]["distance_traveled_z"] = torch.mean(torch.abs(self.root_states[env_ids, 2] - self.init_positions[env_ids, 2])).cpu().item()
        # log power related info
        self.extras["episode"]["max_power_throughout_episode"] = self.max_power_per_timestep[env_ids].max().cpu().item()
        # log whether the episode ends by timeout or dead, or by reaching the goal
        self.extras["episode"]["timeout_ratio"] = (self.time_out_buf.float().sum() / self.reset_buf.float().sum()).cpu().item()
        self.extras["episode"]["distance_exceeded_ratio"] = (self.distance_exceeded_buf.float().sum() / self.reset_buf.float().sum()).cpu().item()
        self.extras["episode"]["yaw_exceeded_ratio"] = (self.yaw_exceeded_buf.float().sum() / self.reset_buf.float().sum()).cpu().item()
        self.extras["episode"]["num_terminated"] = self.reset_buf.float().sum().cpu().item()
        self.extras["episode"]["feet_swing_peak"] = torch.mean(self.swing_peak[env_ids]).cpu().item()
        self.extras["episode"]["feet_air_time"] = torch.mean(self.feet_air_time[env_ids]).cpu().item()
        self.extras["episode"]["feet_contact_time"] = torch.mean(self.feet_contact_time[env_ids]).cpu().item()
        self.extras["completion_counter"] = self.scene_manager.check_completed(env_ids)
        for i in range(len(self.feet_indices)):
            self.extras["episode"][f"{self.feet_names[i]}_contact_force"] = (torch.mean(self.contact_forces[:, self.feet_indices[i], 2])).cpu().item()
        # send timeout info to the algorithm
        if self.cfg["env"]["send_timeouts"]:
            self.extras["time_outs"] = self.time_out_buf | self.distance_exceeded_buf | self.yaw_exceeded_buf

    @timer.section("compute_reward")
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for name, scale, function in zip(self.reward_names, self.reward_scales, self.reward_functions):
            orig_rew = function()
            assert orig_rew.shape == self.rew_buf.shape, f'{orig_rew.shape} != {self.rew_buf.shape}'
            rew = orig_rew * scale
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.cfg["rewards"]:
            rew = self._reward_termination() * self.cfg["rewards"]["termination"]["scale"] * self.dt
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        sim_cfg = dict(self.cfg["sim"])
        sim_device = self.cfg["sim_device"]
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(sim_device)
        _, self.graphics_device_id = gymutil.parse_device_str(self.cfg["graphics_device"])

        # env device is GPU only if sim is on GPU, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type == "cuda":
            self.device = sim_device
        else:
            self.device = "cpu"

        # graphics device for rendering, -1 for no rendering
        self.headless = self.cfg["headless"]
        if self.headless and not self.cfg["runner"]["record_video"]:
            self.graphics_device_id = -1

        self.sim_params = gymapi.SimParams()

        # assign general sim parameters
        self.sim_params.dt = sim_cfg["dt"]
        self.sim_params.num_client_threads = sim_cfg.get("num_client_threads", 0)
        self.sim_params.use_gpu_pipeline = sim_device_type == "cuda"
        self.sim_params.substeps = sim_cfg.get("substeps", 2)

        # assign up-axis
        if sim_cfg["up_axis"] == 1:
            self.up_axis_idx = 2
            self.sim_params.up_axis = gymapi.UP_AXIS_Z
        elif sim_cfg["up_axis"] == 0:
            self.up_axis_idx = 1
            self.sim_params.up_axis = gymapi.UP_AXIS_Y
        else:
            raise ValueError(f"Invalid physics up-axis: {sim_cfg['up_axis']}")

        # assign gravity
        self.sim_params.gravity = gymapi.Vec3(*sim_cfg["gravity"])

        # configure physics parameters
        if sim_cfg["physics_engine"] == "SIM_PHYSX":
            self.physics_engine = gymapi.SIM_PHYSX
            # set the parameters
            if "physx" in sim_cfg:
                for opt in sim_cfg["physx"].keys():
                    if opt == "contact_collection":
                        setattr(self.sim_params.physx, opt, gymapi.ContactCollection(sim_cfg["physx"][opt]))
                    else:
                        setattr(self.sim_params.physx, opt, sim_cfg["physx"][opt])
                setattr(self.sim_params.physx, "use_gpu", sim_device_type == "cuda")
        elif sim_cfg["physics_engine"] == "SIM_FLEX":
            self.physics_engine = gymapi.SIM_FLEX
            # set the parameters
            if "flex" in sim_cfg:
                for opt in sim_cfg["flex"].keys():
                    setattr(self.sim_params.flex, opt, sim_cfg["flex"][opt])
        else:
            raise ValueError(f"Invalid physics engine backend: {sim_cfg['physics_engine']}")

        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        # The scene manager is responsible for managing the meshes, acquiring sample commands, and other scene-related tasks.
        self.scene_manager = gaussian_terrain.GaussianSceneManager(self)
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
        foot_friction = None
        if self.cfg["domain_rand"]["foot_friction"]["apply"] and self.cfg["domain_rand"]["apply_domain_rand"]:
            foot_friction = math.apply_randomization(0.0, self.cfg["domain_rand"]["foot_friction"])

        for s in self.feet_shape_indices:
            if self.cfg["domain_rand"]["foot_friction"]["apply"] and self.cfg["domain_rand"]["apply_domain_rand"]:
              props[s].friction = foot_friction
              # props[s].friction = math.apply_randomization(0.0, self.cfg["domain_rand"]["foot_friction"])
            if self.cfg["domain_rand"]["foot_compliance"]["apply"] and self.cfg["domain_rand"]["apply_domain_rand"]:
              props[s].compliance = math.apply_randomization(0.0, self.cfg["domain_rand"]["foot_compliance"])
            if self.cfg["domain_rand"]["foot_restitution"]["apply"] and self.cfg["domain_rand"]["apply_domain_rand"]:
              props[s].restitution = math.apply_randomization(0.0, self.cfg["domain_rand"]["foot_restitution"])

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
            # allow config to override torque limits
            if "torque_limits" in self.cfg["control"]:
                if not isinstance(self.cfg["control"]["torque_limits"], (tuple, list)):
                    self.torque_limits = torch.ones(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
                    self.torque_limits *= self.cfg["control"]["torque_limits"]
                else:
                    self.torque_limits = torch.tensor(self.cfg["control"]["torque_limits"], dtype=torch.float, device=self.device, requires_grad=False)

        if self.cfg["domain_rand"]["dof_friction_ig_property"]["apply"] and self.cfg["domain_rand"]["apply_domain_rand"]:
            for i in range(len(props)):
                props["friction"][i] = math.apply_randomization(props["friction"][i], self.cfg["domain_rand"]["dof_friction_ig_property"])

        if self.cfg["domain_rand"]["dof_armature_ig_property"]["apply"] and self.cfg["domain_rand"]["apply_domain_rand"]:
            for i in range(len(props)):
                props["armature"][i] = math.apply_randomization(0.0, self.cfg["domain_rand"]["dof_armature_ig_property"])

        return props

    def _process_rigid_body_props(self, props, env_id):
        for j in range(self.num_bodies):
            if j == self.base_link_index:
                if self.cfg["domain_rand"]["base_com_x"]["apply"] and self.cfg["domain_rand"]["apply_domain_rand"]:
                    props[j].com.x, self.base_mass_scaled[env_id, 0] = math.apply_randomization(
                        props[j].com.x, self.cfg["domain_rand"]["base_com_x"], return_noise=True
                    )
                if self.cfg["domain_rand"]["base_com_y"]["apply"] and self.cfg["domain_rand"]["apply_domain_rand"]:
                    props[j].com.y, self.base_mass_scaled[env_id, 1] = math.apply_randomization(
                        props[j].com.y, self.cfg["domain_rand"]["base_com_y"], return_noise=True
                    )
                if self.cfg["domain_rand"]["base_com_z"]["apply"] and self.cfg["domain_rand"]["apply_domain_rand"]:
                    props[j].com.z, self.base_mass_scaled[env_id, 2] = math.apply_randomization(
                        props[j].com.z, self.cfg["domain_rand"]["base_com_z"], return_noise=True
                    )
                if self.cfg["domain_rand"]["base_mass"]["apply"] and self.cfg["domain_rand"]["apply_domain_rand"]:
                    props[j].mass, self.base_mass_scaled[env_id, 3] = math.apply_randomization(
                        props[j].mass, self.cfg["domain_rand"]["base_mass"], return_noise=True
                    )
                    props[j].invMass = 1.0 / props[j].mass
            else:
                if self.cfg["domain_rand"]["other_com"]["apply"] and self.cfg["domain_rand"]["apply_domain_rand"]:
                    props[j].com.x = math.apply_randomization(props[j].com.x, self.cfg["domain_rand"]["other_com"])
                    props[j].com.y = math.apply_randomization(props[j].com.y, self.cfg["domain_rand"]["other_com"])
                    props[j].com.z = math.apply_randomization(props[j].com.z, self.cfg["domain_rand"]["other_com"])
                if self.cfg["domain_rand"]["other_mass"]["apply"] and self.cfg["domain_rand"]["apply_domain_rand"]:
                    props[j].mass = math.apply_randomization(props[j].mass, self.cfg["domain_rand"]["other_mass"])
                    props[j].invMass = 1.0 / props[j].mass
        return props
    
    def get_small_command_mask(self):
        small_lin_vel_mask = torch.norm(self.commands[:, :2], dim=-1) < self.cfg["commands"]["small_lin_vel_threshold"]
        small_ang_vel_mask = torch.norm(self.commands[:, 2:3], dim=-1) < self.cfg["commands"]["small_ang_vel_threshold"]
        return torch.logical_and(small_lin_vel_mask, small_ang_vel_mask)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.still_envs[env_ids] = torch.rand(len(env_ids), device=self.device) < self.cfg["commands"]["still_proportion"]
        heading_command, velocity_command = self.scene_manager.sample_commands(env_ids, still_env_mask=self.still_envs)
        self.commands[:, :2] = velocity_command
        if self.cfg["commands"]["heading_command"]:
            self.commands[:, 3] = heading_command
            forward = tu.quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*math.wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
            self.commands[:, 2] = torch.clip(self.commands[:, 2], *self.cfg["commands"]["ranges"]["ang_vel_yaw"])
        else:
            raise NotImplementedError()

        # Set small commands or still environments to zero.
        self.commands[self.still_envs, :] = 0.0
        self.commands[self.get_small_command_mask(), :3] = 0.0

    @timer.section("compute_torques")
    def _compute_torques(self, actions, dof_stiffness, dof_damping):
        control_type = self.cfg["control"]["control_type"]
        assert control_type == "P", "Only P controller is supported for now."

        actions = actions * self.action_scale

        # Perturbation of DOF stiffness and damping.
        pert_dof_stiffness = dof_stiffness * self.dof_stiffness_multiplier
        pert_dof_damping = dof_damping * self.dof_damping_multiplier

        if self.cfg["control"]["computer_clip_torque"]:
            actions = self.clip_position_action_by_torque_limit(actions, pert_dof_stiffness, pert_dof_damping)

        # Perturbation of motor strength.
        actions = actions * self.motor_strength_multiplier

        torques = (
            pert_dof_stiffness
            * (actions + self.default_dof_pos - self.dof_pos)
            - pert_dof_damping
            * self.dof_vel
        )
        friction = torch.min(self.dof_friction, torques.abs()) * torch.sign(torques)
        torques = torques - friction

        if self.cfg["control"]["motor_clip_torque"]:
            torques = torch.clip(
                torques,
                -self.torque_limits * self.cfg["control"]["motor_clip_torque"],
                self.torque_limits * self.cfg["control"]["motor_clip_torque"],
            )

        return torques

    def _reset_buffers(self, env_ids):
        # reset buffers
        self.last_root_vel[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.swing_peak[env_ids] = 0.
        self.feet_contact_time[env_ids] = 0.
        self.last_contacts[env_ids] = False
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.last_torques[env_ids] = 0.
        self.max_power_per_timestep[env_ids] = 0.
        self.filtered_lin_vel[env_ids] = 0.0
        self.filtered_ang_vel[env_ids] = 0.0
        self.still_envs[env_ids] = False

        # Reset observation buffers.
        self.obs_manager.reset_buffers(env_ids)

    def _initialize_prev_reset_buffer(self, env_ids):
        # Prevent spikes in rewards caused by a reset buffer.
        if len(env_ids) == 0:
          return
        self.last_root_vel[env_ids] = self.root_states[env_ids, 7:13]
        self.last_actions[env_ids] = self.actions[env_ids]
        self.last_dof_vel[env_ids] = self.dof_vel[env_ids]
        self.last_contacts[env_ids] = self.feet_contact[env_ids]
        self.last_torques[env_ids] = self.torques[env_ids]
        self.filtered_lin_vel[env_ids] = self.base_lin_vel[env_ids]
        self.filtered_ang_vel[env_ids] = self.base_ang_vel[env_ids]

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """

        if self.cfg["domain_rand"]["init_dof_pos"]["apply"] and self.cfg["domain_rand"]["apply_domain_rand"]:
          self.dof_pos[env_ids] = math.apply_randomization(
              self.default_dof_pos.expand(len(env_ids), -1),
              self.cfg["domain_rand"]["init_dof_pos"]
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
    
        rot_quat = cam_quat
        if self.cfg["domain_rand"]["init_base_yaw"]["apply"] and self.cfg["domain_rand"]["apply_domain_rand"]:
          rand_yaw = math.apply_randomization(
              torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
              self.cfg["domain_rand"]["init_base_yaw"]
          )
          rand_quat = math.quat_from_euler_xyz(torch.zeros_like(rand_yaw), torch.zeros_like(rand_yaw), rand_yaw)
          rot_quat = math.quat_mul(rot_quat, rand_quat)
        self.root_states[env_ids, 3:7] = rot_quat
        
        # Linear velocity is 7:10, angular velocity is 10:13.
        if self.cfg["domain_rand"]["init_base_lin_vel_xy"]["apply"] and self.cfg["domain_rand"]["apply_domain_rand"]:
          self.root_states[env_ids, 7:9] = math.apply_randomization(
              torch.zeros(len(env_ids), 2, dtype=torch.float, device=self.device),
              self.cfg["domain_rand"]["init_base_lin_vel_xy"]
          )
          self.root_states[env_ids, 9:13] = 0.
        else:
          self.root_states[env_ids, 7:13] = 0.
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.init_positions[env_ids] = self.root_states[env_ids, 0:3].clone()

    def _kick_robots(self):
        """ Random kick the robots. Emulates an impulse by setting a randomized base velocity. """
        kick_interval_steps = np.ceil(self.cfg["domain_rand"]["kick_interval_s"] / self.dt)
        if self.common_step_counter % kick_interval_steps == 0:
            if self.cfg["domain_rand"]["kick_lin_vel"]["apply"] and self.cfg["domain_rand"]["apply_domain_rand"]:
              self.root_states[:, 7:10] = math.apply_randomization(self.root_states[:, 7:10], self.cfg["domain_rand"]["kick_lin_vel"])
            if self.cfg["domain_rand"]["kick_ang_vel"]["apply"] and self.cfg["domain_rand"]["apply_domain_rand"]:
              self.root_states[:, 10:13] = math.apply_randomization(self.root_states[:, 10:13], self.cfg["domain_rand"]["kick_ang_vel"])
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        push_duration_steps = np.ceil(self.cfg["domain_rand"]["push_duration_s"] / self.dt)
        push_interval_steps = np.ceil(self.cfg["domain_rand"]["push_interval_s"] / self.dt)
        push_interval_reached = self.common_step_counter % push_interval_steps == 0
        push_duration_reached = self.common_step_counter % push_interval_steps == push_duration_steps
        if push_interval_reached:
            if self.cfg["domain_rand"]["push_force"]["apply"] and self.cfg["domain_rand"]["apply_domain_rand"]:
              self.pushing_forces[:, self.base_link_index, :] = math.apply_randomization(
                  torch.zeros_like(self.pushing_forces[:, 0, :]),
                  self.cfg["domain_rand"]["push_force"],
              )
            if self.cfg["domain_rand"]["push_torque"]["apply"] and self.cfg["domain_rand"]["apply_domain_rand"]:
              self.pushing_torques[:, self.base_link_index, :] = math.apply_randomization(
                  torch.zeros_like(self.pushing_torques[:, 0, :]),
                  self.cfg["domain_rand"]["push_torque"],
              )
        elif push_duration_reached:
            self.pushing_forces[:, self.base_link_index, :].zero_()
            self.pushing_torques[:, self.base_link_index, :].zero_()

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.pushing_forces),
            gymtorch.unwrap_tensor(self.pushing_torques),
            gymapi.LOCAL_SPACE,
        )

    def get_camera_link_state(self):
      """Get the position of the camera link."""
      camera_link_state = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.camera_link_indices]
      camera_link_state = camera_link_state.squeeze(1)
      return camera_link_state 

    def get_feet_state(self):
      feet_state = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices]
      feet_pos = feet_state[:, :, 0:3]
      feet_quat = feet_state[:, :, 3:7]
      feet_vel = feet_state[:, :, 7:10]
      feet_ang_vel = feet_state[:, :, 10:13]
      return feet_pos, feet_quat, feet_vel, feet_ang_vel

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
        self.stiffness = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.damping = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands = torch.zeros(self.num_envs, self.cfg["commands"]["num_commands"], dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.swing_peak = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_contact_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contact_forces = torch.zeros_like(self.contact_forces)
        self.feet_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.pushing_forces = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        self.pushing_torques = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        self.dof_friction_curriculum_values = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.base_lin_vel = tu.quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = tu.quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.filtered_lin_vel = self.base_lin_vel.clone()
        self.filtered_ang_vel = self.base_ang_vel.clone()
        self.projected_gravity = tu.quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.substep_torques = torch.zeros(self.num_envs, self.cfg["control"]["decimation"], self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.substep_dof_vel = torch.zeros(self.num_envs, self.cfg["control"]["decimation"], self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.substep_exceed_dof_pos_limits = torch.zeros(self.num_envs, self.cfg["control"]["decimation"], self.num_dof, dtype=torch.bool, device=self.device, requires_grad=False)
        self.substep_exceed_dof_pos_limit_abs = torch.zeros(self.num_envs, self.cfg["control"]["decimation"], self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.max_power_per_timestep = torch.zeros(self.num_envs, dtype= torch.float32, device= self.device)
        self.still_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.init_positions = self.root_states[:, 0:3].clone()
        self.prev_reset = torch.arange(self.num_envs, device=self.device)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """

        reward_dict = copy.deepcopy(self.cfg["rewards"])
        self.reward_functions = []
        self.reward_scales = []
        self.reward_names = []
        self.only_positive_rewards = reward_dict.pop("only_positive_rewards", False)

        for name, rew_cfg in reward_dict.items():
            if name == "termination":
                # Termination reward is added after clipping in ``compute_reward``.
                continue
            scale = rew_cfg.pop("scale")
            if scale == 0:
                continue
            scale *= self.dt
            self.reward_functions.append(functools.partial(
                getattr(self, f'_reward_{name}'),
                **rew_cfg
            ))
            self.reward_scales.append(scale)
            self.reward_names.append(name)

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_names}

    def deploy_config(self):
        """Useful information for deployment scripts."""
        return {
            'num_actions': self.num_actions,
            'dof_names': self.dof_names,
            'feet_names': self.feet_names,
            'base_link_name': self.cfg["asset"]["base_link_name"],
            'camera_link_name': self.cfg["asset"]["camera_link_name"],
        }

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg["asset"]["file"].format(GAUSS_GYM_ROOT_DIR=legged_gym.GAUSS_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg["asset"]["default_dof_drive_mode"]
        asset_options.collapse_fixed_joints = self.cfg["asset"]["collapse_fixed_joints"]
        asset_options.replace_cylinder_with_capsule = self.cfg["asset"]["replace_cylinder_with_capsule"]
        asset_options.flip_visual_attachments = self.cfg["asset"]["flip_visual_attachments"]
        asset_options.fix_base_link = self.cfg["asset"]["fix_base_link"]
        asset_options.density = self.cfg["asset"]["density"]
        asset_options.angular_damping = self.cfg["asset"]["angular_damping"]
        asset_options.linear_damping = self.cfg["asset"]["linear_damping"]
        asset_options.max_angular_velocity = self.cfg["asset"]["max_angular_velocity"]
        asset_options.max_linear_velocity = self.cfg["asset"]["max_linear_velocity"]
        asset_options.armature = self.cfg["asset"]["armature"]
        asset_options.thickness = self.cfg["asset"]["thickness"]
        asset_options.disable_gravity = self.cfg["asset"]["disable_gravity"]

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
        self.feet_names = [s for s in body_names if self.cfg["asset"]["foot_name"] in s]
        camera_link_names = [s for s in body_names if self.cfg["asset"]["camera_link_name"] in s]
        penalized_contact_names = []
        for name in self.cfg["asset"]["penalize_contacts_on"]:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg["asset"]["terminate_after_contacts_on"]:
            termination_contact_names.extend([s for s in body_names if name in s])

        if "front_hip_names" in self.cfg["asset"]:
            front_hip_names = self.cfg["asset"]["front_hip_names"]
            self.front_hip_indices = torch.zeros(len(front_hip_names), dtype=torch.long, device=self.device, requires_grad=False)
            for i, name in enumerate(front_hip_names):
                self.front_hip_indices[i] = self.gym.find_asset_dof_index(self.robot_asset, name)
        else:
            front_hip_names = []

        if "rear_hip_names" in self.cfg["asset"]:
            rear_hip_names = self.cfg["asset"]["rear_hip_names"]
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

        self.base_link_index = self.gym.find_asset_rigid_body_index(self.robot_asset, self.cfg["asset"]["base_link_name"])

        # Rigid body shape indices corresponding to the feet.
        shape_indices = self.gym.get_asset_rigid_body_shape_indices(self.robot_asset)
        self.feet_rigid_body_shape_indices_dict = {}
        self.feet_shape_indices = []
        for foot in self.feet_names:
            body_indices = self.gym.find_asset_rigid_body_index(self.robot_asset, foot)
            shape_range = list(range(
                shape_indices[body_indices].start, shape_indices[body_indices].start + shape_indices[body_indices].count))
            self.feet_rigid_body_shape_indices_dict[foot] = shape_range
            self.feet_shape_indices += shape_range

        base_init_state_list = self.cfg["init_state"]["pos"] + self.cfg["init_state"]["rot"] + self.cfg["init_state"]["lin_vel"] + self.cfg["init_state"]["ang_vel"]
        self.base_init_state = tu.to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.clip_actions = None
        clip_actions_low = []
        clip_actions_high = []
        if self.cfg["normalization"]["clip_actions_method"] == 'hard':
          sim_sdk_map = self.cfg["normalization"]["sim_sdk_map"]
          const_dof_range = self.cfg["normalization"]["const_dof_range"]
          dof_pos_redundancy = self.cfg["normalization"]["dof_pos_redundancy"]

        self.default_dof_stiffness = torch.zeros(1, self.num_dofs, dtype=torch.float, device=self.device)
        self.default_dof_damping = torch.zeros(1, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_friction = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.default_dof_pos = torch.zeros(1, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        # joint positions offsets and PD gains
        for i in range(self.num_dofs):
            default_joint_angle = self.cfg["init_state"]["default_joint_angles"][self.dof_names[i]]
            self.default_dof_pos[:, i] = default_joint_angle
            if self.cfg["normalization"]["clip_actions_method"] == 'hard':
                clip_actions_low.append(
                    (const_dof_range[sim_sdk_map[self.dof_names[i]] + "_min"] + dof_pos_redundancy - default_joint_angle) / self.cfg["control"]["action_scale"])
                clip_actions_high.append(
                    (const_dof_range[sim_sdk_map[self.dof_names[i]] + "_max"] - dof_pos_redundancy - default_joint_angle) / self.cfg["control"]["action_scale"])

            found = False
            for name in self.cfg["control"]["stiffness"]:
                if name in self.dof_names[i]:
                    self.default_dof_stiffness[:, i] = self.cfg["control"]["stiffness"][name]
                    self.default_dof_damping[:, i] = self.cfg["control"]["damping"][name]
                    found = True
            if not found:
                raise ValueError(f"PD gain of joint {self.dof_names[i]} were not defined")
        if self.cfg["normalization"]["clip_actions_method"] == 'hard':
            self.clip_actions_low = torch.tensor(clip_actions_low, device=self.device)
            self.clip_actions_high = torch.tensor(clip_actions_high, device=self.device)
        self.motor_strength_multiplier = torch.ones_like(self.dof_friction)
        if self.cfg["domain_rand"]["motor_strength"]["apply"] and self.cfg["domain_rand"]["apply_domain_rand"]:
            self.motor_strength_multiplier = math.apply_randomization(self.motor_strength_multiplier, self.cfg["domain_rand"]["motor_strength"])
        self.dof_stiffness_multiplier = torch.ones_like(self.dof_friction)
        if self.cfg["domain_rand"]["dof_stiffness"]["apply"] and self.cfg["domain_rand"]["apply_domain_rand"]:
            self.dof_stiffness_multiplier = math.apply_randomization(self.dof_stiffness_multiplier, self.cfg["domain_rand"]["dof_stiffness"])
        self.dof_damping_multiplier = torch.ones_like(self.dof_friction)
        if self.cfg["domain_rand"]["dof_damping"]["apply"] and self.cfg["domain_rand"]["apply_domain_rand"]:
            self.dof_damping_multiplier = math.apply_randomization(self.dof_damping_multiplier, self.cfg["domain_rand"]["dof_damping"])
        if self.cfg["domain_rand"]["dof_damping_ankles"]["apply"] and self.cfg["domain_rand"]["apply_domain_rand"]:
            ankle_names = self.cfg["domain_rand"]["dof_damping_ankles"]["ankle_names"]
            ankle_indices = torch.zeros(len(ankle_names), dtype=torch.long, device=self.device, requires_grad=False)
            for i, name in enumerate(ankle_names):
                ankle_indices[i] = self.gym.find_asset_dof_index(self.robot_asset, name)
            self.dof_damping_multiplier[:, ankle_indices] = math.apply_randomization(self.dof_damping_multiplier[:, ankle_indices], self.cfg["domain_rand"]["dof_damping_ankles"])
        if self.cfg["domain_rand"]["dof_friction"]["apply"] and self.cfg["domain_rand"]["apply_domain_rand"]:
            self.dof_friction = math.apply_randomization(self.dof_friction, self.cfg["domain_rand"]["dof_friction"])

        # Action clipping and scaling.
        self.clip_actions = self.cfg["normalization"]["clip_actions"]
        if isinstance(self.clip_actions, (tuple, list)):
            self.clip_actions = torch.tensor(self.clip_actions, device=self.device)
        self.action_scale = self.cfg["control"]["action_scale"]
        if isinstance(self.action_scale, (tuple, list)):
            self.action_scale = torch.tensor(self.action_scale, device=self.device)

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

            actor_handle = self.gym.create_actor(env_handle, self.robot_asset, start_pose, self.cfg["asset"]["name"], i, self.cfg["asset"]["self_collisions"], 0)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props_randomized = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props_randomized, recomputeInertia=True)
            rigid_shape_props_randomized = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props_randomized)
            dof_props_randomized = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props_randomized)
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
                pos = (np.array(self.cfg["viewer"]["pos"]) - (self.cfg["viewer"]["lookat"]) + lookat).tolist()
                self.set_camera(pos, lookat.tolist())
                self.initial_camera_set = True

    #------------ reward functions----------------

    def _reward_alive(self):
        return 1.

    def _reward_energy_substeps(self):
        # (n_envs, n_substeps, n_dofs) 
        # square sum -> (n_envs, n_substeps)
        # mean -> (n_envs,)
        return torch.mean(torch.sum(torch.abs(self.substep_torques) * torch.abs(self.substep_dof_vel), dim=-1), dim=-1)

    def _reward_energy(self):
        # return torch.sum(torch.square(self.torques * self.dof_vel), dim=1)
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=-1)

    def _reward_exceed_dof_pos_limits(self):
        return self.substep_exceed_dof_pos_limits.to(torch.float32).sum(dim=-1).mean(dim=-1)

    def _reward_exceed_torque_limits_i(self, soft_torque_limit):
        """ Indicator function """
        max_torques = torch.abs(self.substep_torques).max(dim= 1)[0]
        exceed_torque_each_dof = max_torques > (self.torque_limits * soft_torque_limit)
        exceed_torque = exceed_torque_each_dof.any(dim= 1)
        return exceed_torque.to(torch.float32)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.filtered_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.filtered_ang_vel[:, :2]), dim=-1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self, height_target):
        # Penalize base height away from target
        base_height = self.sensors["base_height_raycaster"].get_data()
        base_height_error = base_height - height_target
        base_height_error = torch.square(base_height_error).sum(dim=-1)
        return base_height_error

    def _reward_base_height_l1(self, min_height, max_height):
        # Penalize base height away from target
        base_height = self.sensors["base_height_raycaster"].get_data()
        max_height = max_height if max_height > 0 else 1e6
        height_diff = torch.clip(base_height, max=max_height) - min_height
        base_height_error = height_diff.sum(dim=-1)
        return base_height_error
  
    def _reward_hip_height(self, height_target):
        # Penalize hip height away from target
        hip_heights = self.sensors["hip_height_raycaster"].get_data()
        hip_height_error = hip_heights - height_target
        hip_height_error = torch.square(hip_height_error).sum(dim=-1)
        return hip_height_error

    def _reward_hip_height_l1(self, min_height, max_height):
        hip_heights = self.sensors["hip_height_raycaster"].get_data()
        max_height = max_height if max_height > 0 else 1e6
        height_diff = torch.clip(hip_heights, max=max_height) - min_height
        hip_height_error = height_diff.sum(dim=-1)
        return hip_height_error

    def _reward_torques(self):
        # Penalize torques
        return torch.norm(self.torques, p=2, dim=-1) + self.torques.abs().sum(dim=-1)
        # return torch.sqrt(torch.sum(torch.square(self.torques), dim=-1)) + torch.sum(torch.abs(self.torques), dim=-1)
        # return torch.sum(torch.square(self.torques), dim=-1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=-1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=-1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        action_diff = self.last_actions - self.actions
        return torch.norm(action_diff, p=2, dim=-1) + action_diff.abs().sum(dim=-1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 1.0, dim=-1).to(torch.float32)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self, soft_dof_pos_limit):
        # Penalize dof positions too close to the limit
        m = (self.dof_pos_limits[:, 0] + self.dof_pos_limits[:, 1]) / 2
        r = self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        soft_dof_pos_limits = torch.zeros_like(self.dof_pos_limits)
        soft_dof_pos_limits[:, 0] = m - 0.5 * r * soft_dof_pos_limit
        soft_dof_pos_limits[:, 1] = m + 0.5 * r * soft_dof_pos_limit
        out_of_limits = -(self.dof_pos - soft_dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - soft_dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=-1)

    def _reward_dof_vel_limits(self, soft_dof_vel_limit):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits * soft_dof_vel_limit).clip(min=0., max=1.), dim=-1)

    def _reward_torque_limits(self, soft_torque_limit):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits * soft_torque_limit).clip(min=0.), dim=-1)

    def _reward_tracking_lin_vel(self, tracking_sigma):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.filtered_lin_vel[:, :2]), dim=-1)
        return torch.exp(-lin_vel_error / tracking_sigma)

    def _reward_tracking_lin_vel_x(self, tracking_sigma):
        # Tracking of linear velocity commands (x axes)
        return torch.exp(-torch.square(self.commands[:, 0] - self.filtered_lin_vel[:, 0]) / tracking_sigma)

    def _reward_tracking_lin_vel_y(self, tracking_sigma):
        # Tracking of linear velocity commands (y axes)
        return torch.exp(-torch.square(self.commands[:, 1] - self.filtered_lin_vel[:, 1]) / tracking_sigma)
    
    def _reward_tracking_ang_vel(self, tracking_sigma):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.filtered_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / tracking_sigma)

    def _reward_feet_air_time(self, min_air_time, max_air_time):
        # Reward long steps
        max_air_time = max_air_time if max_air_time > 0 else 1e6
        time_diff = torch.clip(self.feet_air_time, max=max_air_time) - min_air_time
        rew_air_time = torch.sum(time_diff * self.first_contact, dim=1) # reward only on first contact with the ground
        rew_air_time *= ~torch.logical_or(self.get_small_command_mask(), self.still_envs)
        return rew_air_time

    def _reward_feet_contact_time(self, min_contact_time, max_contact_time):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        max_contact_time = max_contact_time if max_contact_time > 0 else 1e6
        time_diff = torch.clip(self.feet_contact_time, max=max_contact_time) - min_contact_time
        rew_contact_time = torch.sum(time_diff * self.last_contact, dim=1) # reward only on contact end with the ground.
        return rew_contact_time
    
    def _reward_stumble(self, multiplier):
        # Penalize feet hitting vertical surfaces
        lateral_forces = self.contact_forces[:, self.feet_indices, :2]
        vertical_forces = self.contact_forces[:, self.feet_indices, 2]
        hit_vertical_surface = (
            torch.norm(lateral_forces, dim=-1) >
            multiplier * torch.abs(vertical_forces))
        return torch.sum(hit_vertical_surface, dim=-1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=-1) * self.get_small_command_mask()

    def _reward_feet_contact_forces(self, max_contact_force):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  max_contact_force).clip(min=0.), dim=-1)

    def _reward_feet_slip(self):
        # Penalize feet velocities when contact
        _, _, feet_vel, _ = self.get_feet_state()
        vel_xy = feet_vel[..., :2]
        vel_xy_norm_sq = torch.sum(torch.square(vel_xy), dim=-1)
        return torch.sum(vel_xy_norm_sq * self.feet_contact, dim=-1) # * ~self.get_small_command_mask()

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
        _, _, feet_vel, _ = self.get_feet_state()
        return torch.sum(torch.square(feet_vel)[:, :, 2], dim=-1)

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return dof_error

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)

    def _reward_delta_torques(self):
        torques_diff = self.last_torques - self.torques
        return torch.norm(torques_diff, p=2, dim=-1) + torques_diff.abs().sum(dim=-1)

    def _reward_exceed_torque_limits_l1norm(self, soft_torque_limit):
        """ square function for exceeding part """
        exceeded_torques = torch.abs(self.substep_torques) - (self.torque_limits * soft_torque_limit)
        exceeded_torques[exceeded_torques < 0.] = 0.
        # sum along decimation axis and dof axis
        return torch.norm(exceeded_torques, p=1, dim=-1).sum(dim=1)

    def _reward_a1_pose(self):
      # Stay close to the default pose.
      weights = []
      for name in self.dof_names:
          if 'hip' in name:
              weights.append(1.0)
          elif 'thigh' in name:
              weights.append(1.0)
          elif 'calf' in name:
              weights.append(0.1)
          else:
              raise ValueError(f"Unknown dof name: {name}")
      weights = torch.tensor(weights, device=self.device)[None]
      reward =  torch.exp(-torch.sum(torch.square(self.dof_pos - self.default_dof_pos) * weights, dim=-1))
      return reward
    
    def _reward_feet_clearance(self, clearance_height):
      _, _, feet_vel, _ = self.get_feet_state()
      vel_xy = feet_vel[..., :2]
      vel_norm = torch.linalg.norm(vel_xy, dim=-1)
      feet_z = self.sensors["foot_height_raycaster"].get_data()
      delta = torch.abs(feet_z - clearance_height)
      return torch.sum(delta * vel_norm, dim=-1)

    def _reward_feet_clearance_clipped(self, clearance_height):
      height_diff = self.swing_peak - clearance_height
      height_diff = torch.clip(height_diff * self.first_contact, max=0.)
      reward = height_diff.sum(dim=-1)
      reward *= ~torch.logical_or(self.get_small_command_mask(), self.still_envs)
      return reward

    def _reward_feet_clearance_2(self, clearance_height):
      _, _, feet_vel, _ = self.get_feet_state()
      vel_xy = feet_vel[..., :2]
      vel_norm = torch.linalg.norm(vel_xy, dim=-1)
      feet_z = self.sensors["foot_height_raycaster"].get_data()
      delta = torch.abs(feet_z - clearance_height)
      reward = torch.sum(delta * vel_norm, dim=-1)
      reward *= ~torch.logical_or(self.get_small_command_mask(), self.still_envs)
      return reward

    def _reward_no_fly(self):
        single_contact = torch.sum(1.*self.feet_contact, dim=-1)==1
        return 1. * single_contact

    def _reward_feet_height(self, max_foot_height):
      nonzero_command = ~self.get_small_command_mask()
      error = (self.swing_peak / max_foot_height) - 1.0
      return torch.sum(torch.square(error) * self.first_contact, dim=-1) * nonzero_command