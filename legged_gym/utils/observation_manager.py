import torch
from typing import List
from legged_gym.utils import observation_groups, math, timer, space


class ObsManager:
    def __init__(self, env, obs_groups_cfg: List[observation_groups.ObservationGroup]):
        self.env = env
        self.device = env.device
        self.obs_dims_per_group_obs = {}
        self.obs_dict = {}
        self.obs_buffers_per_group = {}
        self.latency_buffers_per_group = {}
        self.delayed_frames_per_group = {}
        self.obs_group_cfg = obs_groups_cfg
        for obs_group in obs_groups_cfg:
            if obs_group is None:
                continue
            self.obs_dims_per_group_obs[obs_group.name] = {}
            self.obs_buffers_per_group[obs_group.name] = {}
            self.latency_buffers_per_group[obs_group.name] = {}
            self.delayed_frames_per_group[obs_group.name] = {}
            sync_latency_names, sync_latency_buffer = [], None
            if obs_group.sync_latency is not None:
                sync_latency_names = [obs.name for obs in obs_group.sync_latency]
            for obs in obs_group.observations:
                if not isinstance(obs, observation_groups.Observation):
                  raise ValueError(f"Observation {obs} is not an instance of Observation")
                latency_range = obs.latency_range if obs_group.add_latency else (0., 0.)
                function = obs.func
                # if function is a string evaluate it, note: it must be imported in the manager module
                example_obs = function(env, obs)
                obs_shape = example_obs.shape
                obs_dtype = example_obs.dtype
                self.obs_dims_per_group_obs[obs_group.name][obs.name] = space.Space(shape=obs_shape[1:], dtype=example_obs.cpu().numpy().dtype)

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
                self.delayed_frames_per_group[obs_group.name][obs.name] = torch.zeros_like(
                    self.latency_buffers_per_group[obs_group.name][obs.name],
                    dtype=torch.long, device=self.device)
  
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
                self.obs_buffers_per_group[obs_group.name][obs.name][:, env_ids] = 0.
                self.delayed_frames_per_group[obs_group.name][obs.name][env_ids] = 0

    @timer.section("compute_obs")
    def compute_obs(self, env):
        self.obs_dict = {}
        for obs_group in self.obs_group_cfg:
            self.obs_dict[obs_group.name] = {}
            for obs in obs_group.observations:
                new_obs = obs.func(env, obs)
                # Add new observation to buffer.
                self.obs_buffers_per_group[obs_group.name][obs.name] = torch.cat([
                    self.obs_buffers_per_group[obs_group.name][obs.name][1:],
                    new_obs.unsqueeze(0),
                ], dim=0)

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
                obs_value = self.obs_buffers_per_group[obs_group.name][obs.name][
                    -self.delayed_frames_per_group[obs_group.name][obs.name],
                    torch.arange(self.env.num_envs, device= self.device),
                ].clone()

                # Add noise, scale, and clip if needed.
                if obs_group.add_noise and obs.noise:
                    obs_value = self._add_uniform_noise(obs_value, obs.noise)
                if obs.clip:
                    obs_value = obs_value.clip(min=obs.clip[0], max=obs.clip[1])
                if obs.scale is not None:
                    scale = obs.scale
                    if isinstance(scale, list):
                       scale = torch.tensor(scale, device=obs_value.device)[None]
                    obs_value *= scale
                self.obs_dict[obs_group.name][obs.name] = obs_value
        return self.obs_dict

    def _add_uniform_noise(self, obs, noise_level):
        return obs + (2 * torch.rand_like(obs) - 1) * noise_level
