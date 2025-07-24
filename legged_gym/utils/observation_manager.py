import torch
from typing import List
from legged_gym.utils import observation_groups, math, timer, space, math
from legged_gym.rl import utils as rl_utils
import collections


class ObsManager:
    def __init__(self, env, obs_groups_cfg: List[observation_groups.ObservationGroup]):
        self.env = env
        self.device = env.device
        self.obs_dict = {}
        self.obs_dims_per_group_obs = collections.defaultdict(dict)
        self.obs_buffers_per_group = collections.defaultdict(dict)
        self.latency_buffers_per_group = collections.defaultdict(dict)
        self.delayed_frames_per_group = collections.defaultdict(dict)
        self.noise_level_per_group = collections.defaultdict(dict)
        self.obs_available_for_timestep = collections.defaultdict(dict)
        self.obs_group_cfg = obs_groups_cfg
        self.initialize_env_ids = []
        for obs_group in obs_groups_cfg:
            if obs_group is None:
                continue
            sync_latency_names, sync_latency_buffer = [], None
            if obs_group.sync_latency is not None:
                sync_latency_names = [obs.name for obs in obs_group.sync_latency]
            for obs in obs_group.observations:
                if not isinstance(obs, observation_groups.Observation):
                  raise ValueError(f"Observation {obs} is not an instance of Observation")
                if obs.ignore_in_observation_manager:
                  continue
                latency_range = obs.latency_range if obs_group.add_latency else (0., 0.)
                function = obs.func
                # if function is a string evaluate it, note: it must be imported in the manager module
                example_obs = function(env, obs)
                obs_shape = example_obs.shape
                obs_dtype = example_obs.dtype
                self.obs_dims_per_group_obs[obs_group.name][obs.name] = space.Space(shape=obs_shape[1:], dtype=obs_dtype)

                buffer_length = int(latency_range[1] / env.dt) + 1
                buffer_length += int(obs.refresh_duration / env.dt)
                self.obs_buffers_per_group[obs_group.name][obs.name] = torch.zeros(
                   (
                    buffer_length,
                    env.num_envs,
                    *obs_shape[1:]
                   ),
                   dtype=obs_dtype,
                   device=self.device,
                )
                if obs.name in sync_latency_names:
                    if sync_latency_buffer is None:
                        sync_latency_buffer = self._sample_latency_buffer(latency_range, env.num_envs)
                    self.latency_buffers_per_group[obs_group.name][obs.name] = sync_latency_buffer
                else:
                    self.latency_buffers_per_group[obs_group.name][obs.name] = self._sample_latency_buffer(latency_range, env.num_envs)
                if obs.noise_range:
                    self.noise_level_per_group[obs_group.name][obs.name] = math.sample_uniform(
                      *obs.noise_range, (env.num_envs,), self.device
                    )
                self.delayed_frames_per_group[obs_group.name][obs.name] = torch.zeros_like(
                    self.latency_buffers_per_group[obs_group.name][obs.name],
                    dtype=torch.long, device=self.device)
                self.obs_available_for_timestep[obs_group.name][obs.name] = torch.zeros((buffer_length, env.num_envs), dtype=torch.bool, device=self.device)
  
    def _sample_latency_buffer(self, latency_range, size):
        return math.torch_rand_float(
          *latency_range,
          (size, 1),
          device=self.device,
        ).flatten()

    def resample_sensor_latency(self, env_ids=None):
        for obs_group in self.obs_group_cfg:
            if not obs_group.add_latency:
               continue
            if env_ids is None:
              _int = int(obs_group.latency_resampling_interval_s / self.env.dt)
              env_ids = (self.env.episode_length_buf % _int == 0).nonzero(as_tuple=False).flatten()
            if len(env_ids) == 0:
               continue
            sync_latency_names, sync_latency_buffer = [], None
            if obs_group.sync_latency is not None:
                sync_latency_names = [obs.name for obs in obs_group.sync_latency]
            for obs in obs_group.observations:
                if obs.ignore_in_observation_manager:
                  continue
                latency_range = obs.latency_range if obs_group.add_latency else (0., 0.)
                if obs.name in sync_latency_names:
                    if sync_latency_buffer is None:
                        sync_latency_buffer = self._sample_latency_buffer(latency_range, len(env_ids))
                    self.latency_buffers_per_group[obs_group.name][obs.name][env_ids] = sync_latency_buffer
                else:
                    self.latency_buffers_per_group[obs_group.name][obs.name][env_ids] = self._sample_latency_buffer(
                        latency_range, len(env_ids))

    def reset_buffers(self, env_ids):
        self.resample_sensor_latency(env_ids=env_ids)
        for obs_group in self.obs_group_cfg:
            for obs in obs_group.observations:
                if obs.ignore_in_observation_manager:
                  continue
                self.obs_buffers_per_group[obs_group.name][obs.name][:, env_ids] = 0
                self.delayed_frames_per_group[obs_group.name][obs.name][env_ids] = 0
                self.obs_available_for_timestep[obs_group.name][obs.name][:, env_ids] = False
                if obs.noise_range:
                    self.noise_level_per_group[obs_group.name][obs.name][env_ids] = math.sample_uniform(
                      *obs.noise_range, (len(env_ids),), self.device
                    )
        self.initialize_env_ids = env_ids

    @timer.section("compute_obs")
    def compute_obs(self, env):
        self.obs_dict = {}
        for obs_group in self.obs_group_cfg:
            self.obs_dict[obs_group.name] = {}
            for obs in obs_group.observations:
                if obs.ignore_in_observation_manager:
                  continue
                new_obs = obs.func(env, obs)
                # Initialize the observation buffer with repeated observations.
                if len(self.initialize_env_ids) > 0 and new_obs is not None:
                  fill_obs = new_obs[self.initialize_env_ids].unsqueeze(0)
                  self.obs_buffers_per_group[obs_group.name][obs.name][:, self.initialize_env_ids] = fill_obs
                  self.obs_available_for_timestep[obs_group.name][obs.name][:, self.initialize_env_ids] = True

                # Update buffer with observations.
                self.obs_available_for_timestep[obs_group.name][obs.name] = torch.roll(
                    self.obs_available_for_timestep[obs_group.name][obs.name], -1, dims=0)
                self.obs_buffers_per_group[obs_group.name][obs.name] = torch.roll(
                    self.obs_buffers_per_group[obs_group.name][obs.name], -1, dims=0) 
                if new_obs is not None:
                  self.obs_buffers_per_group[obs_group.name][obs.name][-1] = new_obs
                  self.obs_available_for_timestep[obs_group.name][obs.name][-1, :] = True
                else:
                  self.obs_buffers_per_group[obs_group.name][obs.name][-1] = torch.zeros_like(
                      self.obs_buffers_per_group[obs_group.name][obs.name][-1])
                  self.obs_available_for_timestep[obs_group.name][obs.name][-1, :] = False

                # Whether to refresh the observation delay buffer.
                if int(obs.refresh_duration / self.env.dt) == 0:
                  delay_refresh_mask = torch.ones_like(self.env.episode_length_buf, device=self.device, dtype=torch.bool)
                else:
                  delay_refresh_mask = (self.env.episode_length_buf % int(obs.refresh_duration / self.env.dt)) == 0

                # Get the delayed frames.
                buffer_delayed_frames = ((self.latency_buffers_per_group[obs_group.name][obs.name] / self.env.dt) + 1).to(int)

                self.delayed_frames_per_group[obs_group.name][obs.name] = torch.where(
                    delay_refresh_mask,
                    buffer_delayed_frames,
                    self.delayed_frames_per_group[obs_group.name][obs.name] + 1,
                )
                self.delayed_frames_per_group[obs_group.name][obs.name] = torch.clip(
                    self.delayed_frames_per_group[obs_group.name][obs.name],
                    0,
                    self.obs_buffers_per_group[obs_group.name][obs.name].shape[0],
                )

                if new_obs is None:
                    # Don't return the observation if it's not available.
                    continue

                # Get the nearest valid index for which the observation is available. Only changes index for observations
                # which sometimes return `None`.
                indices = self.obs_buffers_per_group[obs_group.name][obs.name].shape[0] - self.delayed_frames_per_group[obs_group.name][obs.name]
                is_valid_index = self.obs_available_for_timestep[obs_group.name][obs.name]
                rows = torch.arange(is_valid_index.shape[0], device=self.device).unsqueeze(1)
                dists = (rows - indices.unsqueeze(0)).to(torch.float).abs()
                dists = dists.masked_fill(~is_valid_index, float('inf'))
                indices_nearest = dists.argmin(dim=0)
                no_true = ~is_valid_index.any(dim=0)
                if no_true.any():
                  raise ValueError(f"No valid index found for observation {obs.name}")
                obs_value = self.obs_buffers_per_group[obs_group.name][obs.name][
                    indices_nearest,
                    torch.arange(self.env.num_envs, device= self.device),
                ].clone()

                # Add noise, scale, and clip if needed.
                if obs_group.add_noise and obs.noise_range:
                    obs_value = self._add_uniform_noise(obs_value, self.noise_level_per_group[obs_group.name][obs.name])
                if obs.clip:
                    obs_value = obs_value.clip(min=obs.clip[0], max=obs.clip[1])
                if obs.scale is not None:
                    scale = obs.scale
                    if isinstance(scale, list):
                       scale = torch.tensor(scale, device=obs_value.device)[None]
                    obs_value *= scale
                self.obs_dict[obs_group.name][obs.name] = obs_value
        self.initialize_env_ids = []
        return self.obs_dict

    def _add_uniform_noise(self, obs, noise_level: torch.Tensor):
        noise_level = rl_utils.broadcast_right(noise_level, obs)
        return obs + (2 * torch.rand_like(obs) - 1) * noise_level
