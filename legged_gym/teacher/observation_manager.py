import torch
from legged_gym.teacher.observation_groups import ObservationGroup, Observation
from typing import List
from isaacgym.torch_utils import torch_rand_float


class ObsManager:
    def __init__(self, env, obs_groups_cfg: List[ObservationGroup]):
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
                if not isinstance(obs, Observation):
                  raise ValueError(f"Observation {obs} is not an instance of Observation")
                if not obs_group.add_noise:  # turn off all noise
                  obs.noise = None
                if not obs_group.add_latency:
                  obs.latency_range = (0., 0.)
                function = obs.func
                # if function is a string evaluate it, note: it must be imported in the manager module
                if hasattr(obs, "dofs"):
                  obs.dof_indices, _ = env.robot.find_dofs(obs.dofs)
                if hasattr(obs, "bodies"):
                  obs.body_indices, _ = env.robot.find_bodies(obs.bodies)
                self.obs_per_group[obs_group.name].append((obs.name, function, obs))
                example_obs = function(env, obs)
                obs_shape = example_obs.shape
                obs_dtype = example_obs.dtype
                delta = obs_shape[1]
                self.obs_dims_per_group_func[obs_group.name][obs.name] = (obs_dim, obs_dim + delta)
                self.obs_dims_per_group_obs[obs_group.name][obs.name] = (obs_shape[1:], obs_dtype)
                obs_dim += delta

                buffer_length = int(obs.latency_range[1] / env.dt) + 1
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
                  obs.latency_range[0],
                  obs.latency_range[1],
                  (env.num_envs, 1),
                  device=self.device,
                ).flatten()
                self.delayed_frames_per_group[obs_group.name][obs.name] = torch.zeros_like(
                    self.latency_buffers_per_group[obs_group.name][obs.name],
                    dtype=torch.long, device=self.device)
            self.obs_dims_per_group[obs_group.name] = obs_dim

    def resample_sensor_latency(self, env_ids):
        if len(env_ids) == 0:
            return
        for obs_group in self.obs_group_cfg:
            for obs in obs_group.observations:
                self.latency_buffers_per_group[obs_group.name][obs.name][env_ids] = torch_rand_float(
                  obs.latency_range[0],
                  obs.latency_range[1],
                  (len(env_ids), 1),
                  device=self.device,
                ).flatten()

    def reset_buffers(self, env_ids):
        self.resample_sensor_latency(env_ids)
        for obs_group in self.obs_group_cfg:
            for obs in obs_group.observations:
                self.obs_buffers_per_group[obs_group.name][obs.name][:, env_ids] = 0.
                self.delayed_frames_per_group[obs_group.name][obs.name][env_ids] = 0

    def compute_obs(self, env):
        self.obs_dict = {}
        for group, function_list in self.obs_per_group.items():
            self.obs_dict[group] = {}
            for name,function, params in function_list:
                new_obs = function(env, params)
                # Add new observation to buffer.
                self.obs_buffers_per_group[group][name] = torch.cat([
                    self.obs_buffers_per_group[group][name][1:],
                    new_obs.unsqueeze(0),
                ], dim=0)

                # Whether to refresh the observation delay buffer.
                if int(params.refresh_duration / self.env.dt) == 0:
                  delay_refresh_mask = torch.ones_like(self.env.episode_length_buf, device=self.device, dtype=torch.bool)
                else:
                  delay_refresh_mask = (self.env.episode_length_buf % int(params.refresh_duration / self.env.dt)) == 0

                # Get the delayed frames.
                buffer_delayed_frames = ((self.latency_buffers_per_group[group][name] / self.env.dt) + 1).to(int)

                self.delayed_frames_per_group[group][name] = torch.where(
                    delay_refresh_mask,
                    torch.minimum(
                        buffer_delayed_frames,
                        self.delayed_frames_per_group[group][name] + 1,
                    ),
                    self.delayed_frames_per_group[group][name] + 1,
                )
                self.delayed_frames_per_group[group][name] = torch.clip(
                    self.delayed_frames_per_group[group][name],
                    0,
                    self.obs_buffers_per_group[group][name].shape[0],
                )
                obs = self.obs_buffers_per_group[group][name][
                    -self.delayed_frames_per_group[group][name],
                    torch.arange(self.env.num_envs, device= self.device),
                ].clone()

                # Add noise, scale, and clip if needed.
                noise = params.noise
                clip = params.clip
                scale = params.scale
                if noise:
                    obs = self._add_uniform_noise(obs, noise)
                if clip:
                    obs = obs.clip(min=clip[0], max=clip[1])
                if scale is not None:
                    if isinstance(scale, list):
                       scale = torch.tensor(scale, device=obs.device)[None]
                    obs = scale * obs
                self.obs_dict[group][name] = obs
        return self.obs_dict

    def _add_uniform_noise(self, obs, noise_level):
        return obs + (2 * torch.rand_like(obs) - 1) * noise_level
