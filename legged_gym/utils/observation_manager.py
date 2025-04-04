import torch
from legged_gym.utils import observation_groups
from typing import List
from isaacgym.torch_utils import torch_rand_float


class ObsManager:
    def __init__(self, env, obs_groups_cfg: List[observation_groups.ObservationGroup]):
        self.env = env
        self.device = env.device
        self.obs_per_group = {}
        self.obs_dims_per_group = {}
        self.obs_dims_per_group_func = {}
        self.obs_dims_per_group_obs = {}
        self.obs_dict = {}
        self.obs_buffers_per_group = {}
        self.latency_buffers_per_group = {}
        self.delayed_frames_per_group = {}
        self.obs_group_cfg = obs_groups_cfg
        for obs_group in obs_groups_cfg:
            if obs_group is None:
                continue
            self.obs_per_group[obs_group.name] = []
            obs_dim = 0
            self.obs_dims_per_group_func[obs_group.name] = {}
            self.obs_dims_per_group_obs[obs_group.name] = {}
            self.obs_buffers_per_group[obs_group.name] = {}
            self.latency_buffers_per_group[obs_group.name] = {}
            self.delayed_frames_per_group[obs_group.name] = {}
            for obs in obs_group.observations:
                if not isinstance(obs, observation_groups.Observation):
                  raise ValueError(f"Observation {obs} is not an instance of Observation")
                latency_range = obs.latency_range if obs_group.add_latency else (0., 0.)
                function = obs.func
                # if function is a string evaluate it, note: it must be imported in the manager module
                self.obs_per_group[obs_group.name].append((obs.name, function, obs))
                example_obs = function(env, obs)
                obs_shape = example_obs.shape
                obs_dtype = example_obs.dtype
                delta = obs_shape[1]
                self.obs_dims_per_group_func[obs_group.name][obs.name] = (obs_dim, obs_dim + delta)
                self.obs_dims_per_group_obs[obs_group.name][obs.name] = (obs_shape[1:], obs_dtype)
                obs_dim += delta

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
                self.latency_buffers_per_group[obs_group.name][obs.name] = torch_rand_float(
                  latency_range[0],
                  latency_range[1],
                  (env.num_envs, 1),
                  device=self.device,
                ).flatten()
                self.delayed_frames_per_group[obs_group.name][obs.name] = torch.zeros_like(
                    self.latency_buffers_per_group[obs_group.name][obs.name],
                    dtype=torch.long, device=self.device)
            self.obs_dims_per_group[obs_group.name] = obs_dim

    def resample_sensor_latency(self, env_ids=None, episode_duration_steps=None):
        assert env_ids is not None or episode_duration_steps is not None, "Either env_ids or episode_duration_steps and dt must be provided"
        assert not (env_ids is not None and episode_duration_steps is not None), "Either env_ids or episode_duration_steps and dt must be provided"
        for obs_group in self.obs_group_cfg:
            if not obs_group.add_latency:
               continue
            if env_ids is not None:
              resample_latency_env_ids = env_ids
            else:
              resample_latency_env_ids = (episode_duration_steps % int(obs_group.latency_resampling_interval_s / self.env.dt) == 0).nonzero(as_tuple=False).flatten()
            if len(resample_latency_env_ids) == 0:
               continue
            for obs in obs_group.observations:
                latency_range = obs.latency_range if obs_group.add_latency else (0., 0.)
                self.latency_buffers_per_group[obs_group.name][obs.name][resample_latency_env_ids] = torch_rand_float(
                  latency_range[0],
                  latency_range[1],
                  (len(resample_latency_env_ids), 1),
                  device=self.device,
                ).flatten()

    def reset_buffers(self, env_ids):
        self.resample_sensor_latency(env_ids=env_ids)
        for obs_group in self.obs_group_cfg:
            for obs in obs_group.observations:
                self.obs_buffers_per_group[obs_group.name][obs.name][:, env_ids] = 0.
                self.delayed_frames_per_group[obs_group.name][obs.name][env_ids] = 0

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
