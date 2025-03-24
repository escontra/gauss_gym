import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Dict, List, Optional
from legged_gym.rl.modules import resnet
from legged_gym.rl.modules.actor_critic import get_activation

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

class ActorCriticRecurrent(torch.nn.Module):
  is_recurrent = True
  def __init__(self, num_act, num_obs, num_privileged_obs, init_std, mu_activation=None):
    super().__init__()
    self.mlp_keys_c, self.cnn_keys_c, self.num_mlp_obs_c = get_mlp_cnn_keys(num_privileged_obs)
    self.mlp_keys_a, self.cnn_keys_a, self.num_mlp_obs_a = get_mlp_cnn_keys(num_obs)
    latent_c_dim = self.num_mlp_obs_c + len(self.cnn_keys_c) * IMAGE_EMBEDDING_DIM
    latent_a_dim = self.num_mlp_obs_a + len(self.cnn_keys_a) * IMAGE_EMBEDDING_DIM

    self.memory_c = Memory(latent_c_dim, type='lstm', num_layers=1, hidden_size=256)
    self.critic = torch.nn.Sequential(
      torch.nn.Linear(256, 256),
      torch.nn.ELU(),
      torch.nn.Linear(256, 256),
      torch.nn.ELU(),
      torch.nn.Linear(256, 128),
      torch.nn.ELU(),
      torch.nn.Linear(128, 1),
    )
    if self.cnn_keys_c:
      # self.cnn_c = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes=IMAGE_EMBEDDING_DIM)
      self.cnn_c = resnet.NatureCNN(3, IMAGE_EMBEDDING_DIM)
    else:
      self.cnn_c = nn.Identity()

    self.memory_a = Memory(latent_a_dim, type='lstm', num_layers=1, hidden_size=256)
    actor_layers = [
      torch.nn.Linear(256, 256),
      torch.nn.ELU(),
      torch.nn.Linear(256, 128),
      torch.nn.ELU(),
      torch.nn.Linear(128, 128),
      torch.nn.ELU(),
      torch.nn.Linear(128, num_act),
    ]
    if mu_activation is not None:
      actor_layers.append(get_activation(mu_activation))
      print('APPLYING TANH ACTIVATION')
    self.actor = torch.nn.Sequential(*actor_layers)
    if self.cnn_keys_a:
      self.cnn_a = resnet.NatureCNN(3, IMAGE_EMBEDDING_DIM)
    else:
      self.cnn_a = nn.Identity()

    self.logstd = torch.nn.parameter.Parameter(
      torch.full((1, num_act), fill_value=np.log(init_std)), requires_grad=True
    )

  def process_obs(self, obs: Dict[str, torch.Tensor], mlp_keys: List[str], cnn_keys: List[str], cnn_model: nn.Module):

    features = torch.cat([obs[k] for k in mlp_keys], dim=-1)
    if cnn_keys:
      cnn_features = []
      for k in cnn_keys:
        cnn_obs = obs[k]
        if cnn_obs.shape[-1] in [1, 3]:
          cnn_obs = cnn_obs.permute(*range(len(cnn_obs.shape)-3), -1, -3, -2)
        if cnn_obs.dtype == torch.uint8:
          cnn_obs = cnn_obs.float() / 255.0

        batch_dims = cnn_obs.shape[:-3]
        spatial_dims = cnn_obs.shape[-3:]
        cnn_obs = cnn_obs.reshape(-1, *spatial_dims)  # Shape: [M*N*L*O, C, H, W]
        cnn_feat = cnn_model(cnn_obs)
        cnn_feat = cnn_feat.reshape(*batch_dims, *cnn_feat.shape[1:])
        cnn_features.append(cnn_feat)

      cnn_features = torch.cat(cnn_features, dim=-1)
      features = torch.cat([features, cnn_features], dim=-1)
    return features

  def reset(self, dones=None):
    self.memory_a.reset(dones)
    self.memory_c.reset(dones)

  def act(self, obs, masks=None, hidden_states=None):
    features = self.process_obs(obs, self.mlp_keys_a, self.cnn_keys_a, self.cnn_a)
    input_a = self.memory_a(features, masks, hidden_states)
    action_mean = self.actor(input_a.squeeze(0))
    action_std = torch.exp(self.logstd).expand_as(action_mean)
    return torch.distributions.Normal(action_mean, action_std)

  def est_value(self, privileged_obs, masks=None, hidden_states=None, upd_state=True):
    privileged_obs = self.process_obs(privileged_obs, self.mlp_keys_c, self.cnn_keys_c, self.cnn_c)
    input_c = self.memory_c(privileged_obs, masks, hidden_states, upd_state)
    return self.critic(input_c.squeeze(0)).squeeze(-1)

  def get_hidden_states(self):
    return self.memory_a.hidden_states, self.memory_c.hidden_states

  @torch.no_grad()
  def clip_std(self, min=None, max=None):
    log_min = torch.log(min) if min is not None else None
    log_max = torch.log(max) if max is not None else None
    self.logstd.copy_(self.logstd.clip(min=log_min, max=log_max))


class Memory(torch.nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None
    
    def forward(self, input, masks=None, hidden_states=None, upd_state=True):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, new_hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
            if upd_state:
                self.hidden_states = new_hidden_states
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0
