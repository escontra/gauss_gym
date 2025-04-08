import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any
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

class ActorCriticRecurrent(torch.nn.Module):
  is_recurrent = True
  def __init__(
        self,
        num_act,
        num_obs,
        num_privileged_obs,
        layer_activation="elu",
        policy_hidden_layer_sizes=[256, 128, 64],
        value_hidden_layer_sizes=[256, 256, 128],
        symlog_inputs=False,
        policy_head: Dict[str, Any] = None,
        value_head: Dict[str, Any] = None):
    super().__init__()
    self.mlp_keys_c, self.cnn_keys_c, self.num_mlp_obs_c = get_mlp_cnn_keys(num_privileged_obs)
    self.mlp_keys_a, self.cnn_keys_a, self.num_mlp_obs_a = get_mlp_cnn_keys(num_obs)
    latent_c_dim = self.num_mlp_obs_c + len(self.cnn_keys_c) * IMAGE_EMBEDDING_DIM
    latent_a_dim = self.num_mlp_obs_a + len(self.cnn_keys_a) * IMAGE_EMBEDDING_DIM

    print(f'Actor MLP Keys: {self.mlp_keys_a}')
    print(f'Actor MLP Num Obs: {self.num_mlp_obs_a}')
    print(f'Actor CNN Keys: {self.cnn_keys_a}')
    print(f'Critic MLP Keys: {self.mlp_keys_c}')
    print(f'Critic MLP Num Obs: {self.num_mlp_obs_c}')
    print(f'Critic CNN Keys: {self.cnn_keys_c}')

    self.memory_c = Memory(latent_c_dim, type='lstm', num_layers=1, hidden_size=value_hidden_layer_sizes[0])
    critic_layers = []
    input_size = value_hidden_layer_sizes[0]
    for size in value_hidden_layer_sizes:
      critic_layers.append(torch.nn.Linear(input_size, size))
      critic_layers.append(get_activation(layer_activation))
      input_size = size
    self.critic = torch.nn.Sequential(*critic_layers)
    self.critic_head = Head(input_size, 1, **value_head)
    if self.cnn_keys_c:
      # self.cnn_c = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes=IMAGE_EMBEDDING_DIM)
      self.cnn_c = resnet.NatureCNN(3, IMAGE_EMBEDDING_DIM)
    else:
      self.cnn_c = nn.Identity()

    self.memory_a = Memory(latent_a_dim, type='lstm', num_layers=1, hidden_size=policy_hidden_layer_sizes[0])
    actor_layers = []
    input_size = policy_hidden_layer_sizes[0]
    for size in policy_hidden_layer_sizes:
      actor_layers.append(torch.nn.Linear(input_size, size))
      actor_layers.append(get_activation(layer_activation))
      input_size = size
    self.actor = torch.nn.Sequential(*actor_layers)
    self.actor_head = Head(input_size, num_act, **policy_head)
    if self.cnn_keys_a:
      self.cnn_a = resnet.NatureCNN(3, IMAGE_EMBEDDING_DIM)
    else:
      self.cnn_a = nn.Identity()

    self.input_transform = math.symlog if symlog_inputs else lambda x: x

  def process_obs(self, obs: Dict[str, torch.Tensor], mlp_keys: List[str], cnn_keys: List[str], cnn_model: nn.Module):

    features = torch.cat([self.input_transform(obs[k]) for k in mlp_keys], dim=-1)
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
    return self.actor_head(self.actor(input_a.squeeze(0)))

  def est_value(self, privileged_obs, masks=None, hidden_states=None, upd_state=True):
    privileged_obs = self.process_obs(privileged_obs, self.mlp_keys_c, self.cnn_keys_c, self.cnn_c)
    input_c = self.memory_c(privileged_obs, masks, hidden_states, upd_state)
    return self.critic_head(self.critic(input_c.squeeze(0)))

  def get_hidden_states(self):
    return self.memory_a.hidden_states, self.memory_c.hidden_states

  @torch.no_grad()
  def clip_std(self, min=None, max=None):
    if self.actor_head.impl == 'normal_logstdparam':
      log_min = torch.log(min) if min is not None else None
      log_max = torch.log(max) if max is not None else None
      self.actor_head.logstd.copy_(self.actor_head.logstd.clip(min=log_min, max=log_max))


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


class Head(torch.nn.Module):

  def __init__(
      self,
      input_size: int,
      output_size: int,
      output_type: str,
      init_std: float = 1.0,
      minstd: float = 1.0,
      maxstd: float = 1.0,
      unimix: float = 0.0,
      bins: int = 255,
      outscale: float = 1.0,
      **kw):
    super().__init__()
    self.input_size = input_size
    self.output_size = output_size
    self.impl = output_type
    self.kw = kw
    self.init_std = init_std
    self.minstd = minstd
    self.maxstd = maxstd
    self.unimix = unimix
    self.bins = bins
    self.outscale = outscale
    if self.impl == 'mse':
      self.projection_net = nn.Linear(input_size, output_size, **self.kw)
      self._init_layer(self.projection_net)
    elif self.impl == 'symexp_twohot':
      self.projection_net = nn.Linear(input_size, output_size * self.bins, **self.kw)
      self._init_layer(self.projection_net)
    elif self.impl == 'bounded_normal':
      self.mean_net = nn.Linear(input_size, output_size, **self.kw)
      self.stddev_net = nn.Linear(input_size, output_size, **self.kw)
      self._init_layer(self.mean_net)
      self._init_layer(self.stddev_net)
    elif self.impl == 'normal_logstd':
      self.mean_net = nn.Linear(input_size, output_size, **self.kw)
      self.stddev_net = nn.Linear(input_size, output_size, **self.kw)
      self._init_layer(self.mean_net)
      self._init_layer(self.stddev_net)
    elif self.impl == 'normal_logstdparam':
      self.mean_net = nn.Linear(input_size, output_size, **self.kw)
      self.logstd = torch.nn.parameter.Parameter(
        torch.full((1, output_size), fill_value=np.log(self.init_std)), requires_grad=True
      )

  def _init_layer(self, layer):
    torch.nn.init.trunc_normal_(layer.weight)
    layer.weight.data *= self.outscale
    torch.nn.init.zeros_(layer.bias)

  def __call__(self, x):
    if not hasattr(self, self.impl):
      raise NotImplementedError(self.impl)
    output = getattr(self, self.impl)(x)
    return output

  def mse(self, x):
    pred = self.projection_net(x)
    return outs.MSE(pred)

  def symexp_twohot(self, x):
    logits = self.projection_net(x)
    logits = logits.reshape((*x.shape[:-1], self.output_size, self.bins))
    if self.bins % 2 == 1:
      half = torch.linspace(-20, 0, (self.bins - 1) // 2 + 1, dtype=torch.float32)
      half = math.symexp(half)
      bins = torch.cat([half, -half[:-1].flip(0)], 0)
    else:
      half = torch.linspace(-20, 0, self.bins // 2, dtype=torch.float32)
      half = math.symexp(half)
      bins = torch.cat([half, -half.flip(0)], 0)
    return outs.TwoHot(logits, bins)

  def bounded_normal(self, x):
    mean = self.mean_net(x)
    stddev = self.stddev_net(x)
    lo, hi = self.minstd, self.maxstd
    stddev = (hi - lo) * torch.sigmoid(stddev + 2.0) + lo
    output = outs.Normal(torch.tanh(mean), stddev)
    return output

  def normal_logstd(self, x):
    mean = self.mean_net(x)
    stddev = torch.exp(self.stddev_net(x))
    lo, hi = self.minstd, self.maxstd
    stddev = (hi - lo) * torch.sigmoid(stddev + 2.0) + lo
    output = outs.Normal(mean, stddev)
    return output

  def normal(self, x):
    mean = self.mean_net(x)
    stddev = self.stddev_net(x)
    lo, hi = self.minstd, self.maxstd
    stddev = (hi - lo) * torch.sigmoid(stddev + 2.0) + lo
    output = outs.Normal(mean, stddev)
    return output

  def normal_logstdparam(self, x):
    mean = self.mean_net(x)
    output = outs.Normal(mean, torch.exp(self.logstd))
    with torch.no_grad():
      log_min = np.log(self.minstd)
      log_max = np.log(self.maxstd)
      self.logstd.copy_(self.logstd.clip(min=log_min, max=log_max))
    return output
