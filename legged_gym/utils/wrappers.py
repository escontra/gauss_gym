from typing import Dict, Optional
import torch
import copy

from legged_gym.utils import space, observation_groups
from legged_gym.envs.base import legged_robot
from legged_gym.rl import experience_buffer
from legged_gym.rl import utils as rl_utils
from legged_gym.utils import math


class Wrapper:

  def __init__(self, env: legged_robot.LeggedRobot):
    self.env = env

  def __len__(self):
    return self.env.num_envs

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self.env, name)
    except AttributeError:
      raise ValueError(name)


class ImageEncoderWrapper(Wrapper):

  def __init__(self, env: legged_robot.LeggedRobot, image_encoder_key: str, image_encoder: torch.nn.Module):
    super().__init__(env)
    self.env = env
    self.image_encoder = image_encoder
    self.image_encoder_key = image_encoder_key
    self.image_encoder_space = space.Space(
      torch.float32,
      (self.image_encoder.rnn_state_size,),
      -torch.inf,
      torch.inf)

    self.buffer, self.curr_buffer_idx = None, 0
    self.device = next(self.image_encoder.parameters()).device

    # Used during deployment.
    self.image_encoder_rnn_state = None
    self.image_encoder_hidden_states = self.image_encoder.reset(
      torch.zeros(self.env.num_envs, dtype=torch.bool), None)
    self.dones = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
    self.noise_level = math.sample_uniform(
      *observation_groups.IMAGE_ENCODER_LATENT.noise_range,
      (self.env.num_envs,),
      self.device
    )

  def init_image_encoder_replay_buffer(self, num_steps_per_env: int):
    self.buffer = experience_buffer.ExperienceBuffer(
      num_steps_per_env,
      self.env.num_envs,
      self.device,
    )
    for k, v in self.obs_space().items():
      self.buffer.add_buffer(k, v)
    self.buffer.add_buffer("dones", (), dtype=bool)
    if self.image_encoder.is_recurrent:
      self.buffer.add_buffer(
        f"{self.image_encoder_key}_hidden_states", (self.image_encoder_hidden_states,), is_hidden_state=True)

  def obs_space(self):
    new_obs_space = copy.deepcopy(self.env.obs_space())
    for obs_group in self.env.obs_groups:
      if observation_groups.IMAGE_ENCODER_LATENT in obs_group.observations:
        new_obs_space[obs_group.name][
          observation_groups.IMAGE_ENCODER_LATENT.name
        ] = self.image_encoder_space
    return new_obs_space

  def add_latent_to_obs_dict(self, obs_dict: Dict[str, Dict[str, torch.Tensor]], latent: torch.Tensor):
    for obs_group in self.env.obs_groups:
      if observation_groups.IMAGE_ENCODER_LATENT in obs_group.observations:
        if self.env.obs_groups[self.env.obs_groups.index(obs_group)].add_noise:
          noise_level = rl_utils.broadcast_right(self.noise_level, latent)
          latent = latent + (2 * torch.rand_like(latent) - 1) * noise_level
        obs_dict[obs_group.name][observation_groups.IMAGE_ENCODER_LATENT.name] = latent
    return obs_dict

  def reset(self):
    self.curr_buffer_idx = 0
    obs_dict = self.env.reset()
    self.noise_level[:] = math.sample_uniform(
      *observation_groups.IMAGE_ENCODER_LATENT.noise_range,
      (self.env.num_envs,),
      self.device
    )

    if self.buffer is not None:
      self.buffer.update_data(
        f"{self.image_encoder_key}_hidden_states", self.curr_buffer_idx,
        (self.image_encoder_hidden_states,),
        is_hidden_state=True
      )
      for k, v in obs_dict.items():
        self.buffer.update_data(k, self.curr_buffer_idx, v)
      self.buffer.update_data("dones", self.curr_buffer_idx, self.dones)
      self.curr_buffer_idx = 1

    with torch.no_grad():
      self.env.image_encoder_dists, self.image_encoder_rnn_state, self.image_encoder_hidden_states = self.image_encoder(
        obs_dict[self.image_encoder_key],
        self.image_encoder_hidden_states,
        rnn_only=not self.env.use_viser
      )
    return self.add_latent_to_obs_dict(obs_dict, self.image_encoder_rnn_state)

  def _check_observation_available(self, obs_dict, key: str, obs_space: Dict[str, Dict[str, space.Space]]):
    if key not in obs_dict:
      return False
    for obs_name in obs_space.keys():
      if obs_name not in obs_dict[key]:
        return False
    return True

  def step(self, actions: Dict[str, torch.tensor]):
    obs_dict, rew, done, infos = self.env.step(actions)
    done_ids = done.nonzero(as_tuple=False).flatten()
    self.noise_level[done_ids] = math.sample_uniform(
      *observation_groups.IMAGE_ENCODER_LATENT.noise_range,
      (len(done_ids),),
      self.device
    )
    self.dones = torch.logical_or(self.dones, done)

    if self._check_observation_available(obs_dict, self.image_encoder_key, self.image_encoder.obs_space):
      if self.buffer is not None:
        # Update the buffer.
        self.buffer.update_data(
          f"{self.image_encoder_key}_hidden_states",
          self.curr_buffer_idx,
          (self.image_encoder_hidden_states,),
          is_hidden_state=True
        )
        for k, v in obs_dict.items():
          self.buffer.update_data(k, self.curr_buffer_idx, v)
        self.buffer.update_data("dones", self.curr_buffer_idx, self.dones)
        self.curr_buffer_idx += 1

      # Reset dones in between observations.
      self.image_encoder_hidden_states = self.image_encoder.reset(self.dones, self.image_encoder_hidden_states)
      self.dones[:] = False

      # Encode the new image.
      with torch.no_grad():
        self.env.image_encoder_dists, self.image_encoder_rnn_state, self.image_encoder_hidden_states = self.image_encoder(
          obs_dict[self.image_encoder_key],
          self.image_encoder_hidden_states,
          rnn_only=not self.env.use_viser
        )

      if self.buffer is not None:
        if self.curr_buffer_idx == self.buffer.horizon_length:
          infos[f'{self.image_encoder_key}_buffer'] = self.buffer
          self.curr_buffer_idx = 0

    return self.add_latent_to_obs_dict(obs_dict, self.image_encoder_rnn_state), rew, done, infos
