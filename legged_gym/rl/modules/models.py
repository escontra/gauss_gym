import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
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
        symlog_inputs=False,
        head: Dict[str, Any] = None):
    super().__init__()
    self.mlp_keys, self.cnn_keys, self.num_mlp_obs = get_mlp_cnn_keys(obs_space)
    latent_dim = self.num_mlp_obs + len(self.cnn_keys) * IMAGE_EMBEDDING_DIM

    print(f'MLP Keys: {self.mlp_keys}')
    print(f'MLP Num Obs: {self.num_mlp_obs}')
    print(f'CNN Keys: {self.cnn_keys}')

    self.memory = Memory(latent_dim, type='lstm', num_layers=1, hidden_size=hidden_layer_sizes[0])
    layers = []
    input_size = hidden_layer_sizes[0]
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
          cnn_obs = cnn_obs.permute(*range(len(cnn_obs.shape)-3), -1, -3, -2)
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
               update_state: bool=True) -> outs.Output:
    processed_obs = self.process_obs(obs)
    rnn_state = self.memory(processed_obs, masks, hidden_states, update_state)
    return self.model(rnn_state.squeeze(0))

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
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
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
            out, new_hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
            if upd_state:
                self.hidden_states = new_hidden_states
            return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0
