import functools
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, List
from legged_gym.rl.modules import resnet, outs
from legged_gym.rl.modules.actor_critic import get_activation
from legged_gym.utils import math


def split_and_pad_trajectories(tensor, dones):
    """ Splits trajectories at done indices. Then concatenates them and padds with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example: 
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]    
            
    Assumes that the inputy has the following dimension order: [time, number of envs, aditional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1),trajectory_lengths_list)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)


    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories, masks):
    """ Does the inverse operation of  split_and_pad_trajectories()
    """
    # Need to transpose before and after the masking to have proper reshaping
    return trajectories.transpose(1, 0)[masks.transpose(1, 0)].view(-1, trajectories.shape[0], trajectories.shape[-1]).transpose(1, 0)


def get_mlp_cnn_keys(obs):
  mlp_keys, cnn_keys = [], []
  num_mlp_obs = 0
  for k, v in obs.items():
    if len(v[0]) == 1:
      mlp_keys.append(k)
      num_mlp_obs += v[0][0]
    elif len(v[0]) == 3:
      cnn_keys.append(k)
    else:
      raise ValueError(f'Observation {k} has unexpected shape: {v.shape}')

  return mlp_keys, cnn_keys, num_mlp_obs

IMAGE_EMBEDDING_DIM = 256

class ActorCritic(torch.nn.Module):
  is_recurrent = False
  def __init__(self, num_act, num_obs, num_privileged_obs, init_std):
    super().__init__()
    self.critic = torch.nn.Sequential(
      torch.nn.Linear(num_privileged_obs, 256),
      torch.nn.ELU(),
      torch.nn.Linear(256, 256),
      torch.nn.ELU(),
      torch.nn.Linear(256, 128),
      torch.nn.ELU(),
      torch.nn.Linear(128, 1),
    )
    self.actor = torch.nn.Sequential(
      torch.nn.Linear(num_obs, 256),
      torch.nn.ELU(),
      torch.nn.Linear(256, 128),
      torch.nn.ELU(),
      torch.nn.Linear(128, 128),
      torch.nn.ELU(),
      torch.nn.Linear(128, num_act),
    )
    self.logstd = torch.nn.parameter.Parameter(
      torch.full((1, num_act), fill_value=np.log(init_std)), requires_grad=True
    )

  def act(self, obs, masks=None, hidden_states=None):
    action_mean = self.actor(obs)
    action_std = torch.exp(self.logstd).expand_as(action_mean)
    return torch.distributions.Normal(action_mean, action_std)

  def est_value(self, privileged_obs, masks=None, hidden_states=None, upd_state=True):
    return self.critic(privileged_obs).squeeze(-1)

  def reset(self, dones=None):
     pass


class RecurrentModel(torch.nn.Module):
  is_recurrent = True
  def __init__(
        self,
        output_size,
        obs_space,
        layer_activation="elu",
        hidden_layer_sizes=[256, 128, 64],
        recurrent_state_size=256,
        symlog_inputs=False,
        head: Dict[str, Any] = None):
    super().__init__()
    self.mlp_keys, self.cnn_keys, self.num_mlp_obs = get_mlp_cnn_keys(obs_space)
    self.obs_size = self.num_mlp_obs + len(self.cnn_keys) * IMAGE_EMBEDDING_DIM
    self.recurrent_state_size = recurrent_state_size
    self.memory = Memory(self.obs_size, type='lstm', num_layers=1, hidden_size=self.recurrent_state_size)

    print(f'MLP Keys: {self.mlp_keys}')
    print(f'MLP Num Obs: {self.num_mlp_obs}')
    print(f'CNN Keys: {self.cnn_keys}')

    layers = []
    input_size = self.recurrent_state_size
    for size in hidden_layer_sizes:
      layers.append(torch.nn.Linear(input_size, size))
      layers.append(get_activation(layer_activation))
      input_size = size
    layers.append(outs.Head(input_size, output_size, **head))
    self.model = torch.nn.Sequential(*layers)
    if self.cnn_keys:
      # self.cnn_c = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes=IMAGE_EMBEDDING_DIM)
      self.cnn = resnet.NatureCNN(3, IMAGE_EMBEDDING_DIM)
    else:
      self.cnn = nn.Identity()

    self.symlog_inputs = symlog_inputs

  def reset(self, dones=None):
    self.memory.reset(dones)

  def get_hidden_states(self):
    return (self.memory.hidden_states,)

  def process_obs(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:

    features = torch.cat([obs[k] for k in self.mlp_keys], dim=-1)
    if self.symlog_inputs:
      features = math.symlog(features)
    
    if self.cnn_keys:
      cnn_features = []
      for k in self.cnn_keys:
        cnn_obs = obs[k]
        if cnn_obs.shape[-1] in [1, 3]:
          cnn_obs = permute_cnn_obs(cnn_obs)
        if cnn_obs.dtype == torch.uint8:
          cnn_obs = cnn_obs.float() / 255.0

        batch_dims = cnn_obs.shape[:-3]
        spatial_dims = cnn_obs.shape[-3:]
        cnn_obs = cnn_obs.reshape(-1, *spatial_dims)  # Shape: [M*N*L*O, C, H, W]
        cnn_feat = self.cnn(cnn_obs)
        cnn_feat = cnn_feat.reshape(*batch_dims, *cnn_feat.shape[1:])
        cnn_features.append(cnn_feat)

      cnn_features = torch.cat(cnn_features, dim=-1)
      features = torch.cat([features, cnn_features], dim=-1)
    return features

  def __call__(self,
               obs: Dict[str, torch.Tensor],
               masks: Optional[torch.Tensor]=None,
               hidden_states: Optional[torch.Tensor]=None,
               update_state: bool=True,
               return_states: bool=False) -> Union[outs.Output, Tuple[outs.Output, Dict[str, torch.Tensor]]]:
    processed_obs = self.process_obs(obs)
    rnn_state = self.memory(processed_obs, masks, hidden_states, update_state)
    dist = self.model(rnn_state.squeeze(0))
    if return_states:
        return dist, {'recurrent_state': rnn_state.squeeze(0), 'obs': processed_obs}
    return dist

  def stats(self):
    stats = {}
    for layer in self.model:
      if hasattr(layer, 'stats'):
        stats.update(layer.stats())
    return stats


class Memory(torch.nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        self.rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = self.rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

        # Learnable initial hidden state (size remains [L, 1, H])
        self.initial_hidden_state = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_size), requires_grad=True)
        if self.rnn_cls is nn.LSTM:
            self.initial_cell_state = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_size), requires_grad=True)

        # Initialize hidden_states to None for lazy initialization
        self.hidden_states = None

    @torch.jit.ignore
    def _process_batch_mode(self, input: torch.Tensor, masks: Optional[torch.Tensor], hidden_states: Optional[torch.Tensor]):
        assert hidden_states is not None, "Hidden states not passed to memory module during policy update"
        assert masks is not None, "Masks not passed to memory module during policy update"
        out, _ = self.rnn(input, hidden_states)
        out = unpad_trajectories(out, masks)
        return out

    def forward(self, input: torch.Tensor, masks: Optional[torch.Tensor]=None, hidden_states: Optional[torch.Tensor]=None, upd_state: bool=True) -> torch.Tensor:
        batch_mode = masks is not None
        if batch_mode:  # Batch (update) mode.
            return self._process_batch_mode(input, masks, hidden_states)
        else:  # Inference mode
            if self.hidden_states is None:
                num_envs = input.shape[0]
                init_h = self.initial_hidden_state.to(input.device)
                if self.rnn_cls is nn.LSTM:
                    init_c = self.initial_cell_state.to(input.device)
                    self.hidden_states = (
                        init_h.repeat(1, num_envs, 1),
                        init_c.repeat(1, num_envs, 1)
                    )
                else: # GRU
                    self.hidden_states = init_h.repeat(1, num_envs, 1)

            out, new_hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
            if upd_state:
                self.hidden_states = new_hidden_states
            return out

    def reset(self, dones=None):
        assert self.hidden_states is not None, "Reset called before forward pass! Hidden states not initialized."

        if dones is None or dones.sum() == 0:
            return

        if isinstance(self.hidden_states, tuple): # LSTM
            init_h = self.initial_hidden_state.detach().clone().to(self.hidden_states[0].device)
            init_c = self.initial_cell_state.detach().clone().to(self.hidden_states[1].device)
            self.hidden_states[0][..., dones, :] = init_h
            self.hidden_states[1][..., dones, :] = init_c
        else: # GRU
            init_h = self.initial_hidden_state.detach().clone().to(self.hidden_states.device)
            self.hidden_states[..., dones, :] = init_h


def get_policy_jitted(policy, cfg):
    memory_module = policy.memory
    actor_layers = policy.model[:-1]
    actor_mean_net = policy.model[-1].mean_net
    mlp_keys = policy.mlp_keys
    cnn_keys = policy.cnn_keys
    cnn_model = policy.cnn


    @torch.jit.script
    def policy(observations: Dict[str, torch.Tensor], mlp_keys: List[str], cnn_keys: List[str], symlog_inputs: bool) -> torch.Tensor:
        if symlog_inputs:
          features = torch.cat([math.symlog(observations[k]) for k in mlp_keys], dim=-1)
        else:
          features = torch.cat([observations[k] for k in mlp_keys], dim=-1)
        if cnn_keys:
          cnn_features = []
          for k in cnn_keys:
            cnn_obs = observations[k]
            if cnn_obs.shape[-1] in [1, 3]:
              cnn_obs = permute_cnn_obs(cnn_obs)
            if cnn_obs.dtype == torch.uint8:
              cnn_obs = cnn_obs.float() / 255.0

            orig_batch_size = cnn_obs.shape[0]
            cnn_obs = cnn_obs.reshape(-1, cnn_obs.shape[-3], cnn_obs.shape[-2], cnn_obs.shape[-1])  # Shape: [M*N*L*O, C, H, W]
            cnn_feat = cnn_model(cnn_obs)
            cnn_feat = cnn_feat.reshape(orig_batch_size, cnn_feat.shape[-1])
            cnn_features.append(cnn_feat)

          cnn_features = torch.cat(cnn_features, dim=-1)
          features = torch.cat([features, cnn_features], dim=-1)
        # features = models.process_obs(observations, mlp_keys, cnn_keys, symlog_inputs, cnn_model)
        input_a = memory_module(features, None, None).squeeze(0)
        latent = actor_layers(input_a)
        mean = actor_mean_net(latent)
        return mean

    return functools.partial(policy, mlp_keys=mlp_keys, cnn_keys=cnn_keys, symlog_inputs=cfg["symlog_inputs"])


def permute_cnn_obs(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 5:
        return x.permute(0, 1, 4, 2, 3)  # [B, T, H, W, C] -> [B, T, C, H, W]
    elif x.dim() == 4:
        return x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    elif x.dim() == 3:
        return x.permute(2, 0, 1)  # [H, W, C] -> [H, W]
    else:
        raise ValueError(f"Unexpected shape: {x.shape}")