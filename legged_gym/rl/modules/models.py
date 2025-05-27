import numpy as np
import functools
import torch
import torch.nn as nn
from torchvision import transforms
from typing import Dict, Any, Optional, Tuple, List, Callable, Union

from legged_gym.utils import math, space
from legged_gym.rl.modules import resnet, outs
from legged_gym.rl.modules.dino import backbones
from legged_gym.rl.modules.actor_critic import get_activation
from legged_gym.rl.modules import normalizers


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


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
  # TODO: Add option to process 2D observations with CNN or MLP. Currently only supports MLP.
  mlp_keys, mlp_2d_reshape_keys, cnn_keys = [], [], []
  num_mlp_obs = 0
  for k, v in obs.items():
    if len(v.shape) == 1:
      mlp_keys.append(k)
      num_mlp_obs += np.prod(v.shape)
    elif len(v.shape) == 2:
      mlp_keys.append(k)
      mlp_2d_reshape_keys.append(k)
      num_mlp_obs += np.prod(v.shape)
    elif len(v.shape) == 3:
      cnn_keys.append(k)
    else:
      raise ValueError(f'Observation {k} has unexpected shape: {v.shape}')

  return mlp_keys, mlp_2d_reshape_keys, cnn_keys, num_mlp_obs


class ImageFeature(torch.nn.Module):
  def __init__(self, obs_space: Dict[str, space.Space], model_type: str = 'cnn_simple', embedding_dim: int = 256, dino_pretrained: bool = True):
    super().__init__()
    self._obs_space = obs_space
    self.embedding_dim = embedding_dim
    _, _, self.cnn_keys, _ = get_mlp_cnn_keys(obs_space)
    self.model_type = model_type

    print('ImageFeature\n\tCNN Keys: ')
    for key in self.cnn_keys:
      print(f'\t\t{key}: {obs_space[key].shape}')

    encoder_dict = {}
    for key in self.cnn_keys:
      if self.model_type == 'cnn_simple':
        encoder_dict[key] = resnet.NatureCNN(3, embedding_dim)
      elif self.model_type == 'cnn_resnet':
        encoder_dict[key] = resnet.ResNet(
            resnet.BasicBlock, [2, 2, 2, 2], num_classes=embedding_dim)
      elif self.model_type.startswith('dino'):
        dino = getattr(backbones, self.model_type)(pretrained=dino_pretrained)
        encoder_dict[key] = nn.Sequential(
          transforms.Normalize(
              mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
          dino,
          nn.Linear(dino.embed_dim, embedding_dim)
        )
      else:
        raise ValueError(f'Unknown model type: {model_type}')
    self.dummy_param = nn.Parameter(torch.zeros(1))
    self.encoder = nn.ModuleDict(encoder_dict)

  def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
    out_features = {}
    for key in self.cnn_keys:
      cnn_obs = obs[key]
      if cnn_obs.shape[-1] in [1, 3]:
        cnn_obs = permute_cnn_obs(cnn_obs)
      if cnn_obs.dtype == torch.uint8:
        cnn_obs = cnn_obs.float() / 255.0

      batch_dims = cnn_obs.shape[:-3]
      spatial_dims = cnn_obs.shape[-3:]
      cnn_obs = cnn_obs.reshape(-1, *spatial_dims)  # Shape: [M*N*L*O, C, H, W]
      feat_list = []
      for i in range(0, cnn_obs.shape[0], 2048):
        cnn_feat = self.encoder[key](cnn_obs[i:i+2048])
        feat_list.append(cnn_feat)
      cnn_feat = torch.cat(feat_list, dim=0)
      # cnn_feat = self.encoder[key](cnn_obs)
      cnn_feat = cnn_feat.reshape(*batch_dims, *cnn_feat.shape[1:])
      out_features[key] = cnn_feat
    return out_features

  def obs_space(self) -> Dict[str, space.Space]:
    return {key: self._obs_space[key] for key in self.cnn_keys}

  def modified_obs_space(self) -> Dict[str, space.Space]:
    modified_obs_space = {}
    for key in self.cnn_keys:
      modified_obs_space[key] = space.Space(np.float32, (self.embedding_dim,), -np.inf, np.inf)
    return modified_obs_space


IMAGE_EMBEDDING_DIM = 256


class RecurrentModel(torch.nn.Module):
  is_recurrent = True
  def __init__(
        self,
        action_space: Dict[str, space.Space],
        obs_space: Dict[str, space.Space],
        dont_normalize_keys: List[str] = [],
        layer_activation="elu",
        hidden_layer_sizes=[256, 128, 64],
        recurrent_state_size=256,
        symlog_inputs=False,
        normalize_obs=True,
        max_abs_value=None,
        head: Dict[str, Any] = None):
    super().__init__()
    self.obs_space = obs_space
    self.dont_normalize_keys = dont_normalize_keys
    self.action_space = action_space
    self.mlp_keys, self.mlp_2d_reshape_keys, self.cnn_keys, self.num_mlp_obs = get_mlp_cnn_keys(obs_space)
    self.obs_size = self.num_mlp_obs + len(self.cnn_keys) * IMAGE_EMBEDDING_DIM
    self.recurrent_state_size = recurrent_state_size
    self.memory = Memory(self.obs_size, type='lstm', num_layers=1, hidden_size=self.recurrent_state_size)
    self.normalize_obs = normalize_obs
    self.max_abs_value = max_abs_value
    if self.normalize_obs:
      self.obs_normalizer = normalizers.DictNormalizer(obs_space, max_abs_value=self.max_abs_value, dont_normalize_keys=dont_normalize_keys)

    print('RecurrentModel:')
    print(f'\tMLP Keys: {self.mlp_keys}')
    print(f'\tMLP Num Obs: {self.num_mlp_obs}')
    print(f'\tCNN Keys: {self.cnn_keys}')
    print(f'\tNormalizing obs: {self.normalize_obs}')

    layers = []
    input_size = self.recurrent_state_size
    for size in hidden_layer_sizes:
      layers.append(torch.nn.Linear(input_size, size))
      layers.append(get_activation(layer_activation))
      input_size = size
    self.model = torch.nn.Sequential(*layers)
    heads = {k: outs.Head(input_size, v, **head[k]) for k, v in action_space.items()}
    self.heads = nn.ModuleDict(heads)
    if self.cnn_keys:
      # self.cnn_c = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes=IMAGE_EMBEDDING_DIM)
      self.cnn = resnet.NatureCNN(3, IMAGE_EMBEDDING_DIM)
    else:
      self.cnn = nn.Identity()

    self.symlog_inputs = symlog_inputs

  def reset(self, dones: torch.Tensor, hidden_states: Optional[Tuple[torch.Tensor, ...]]=None) -> Tuple[torch.Tensor, ...]:
    return self.memory.reset(dones, hidden_states)

  def process_obs(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
    if self.normalize_obs:
      obs = apply_normalizer(
        obs,
        self.obs_normalizer.mean,
        self.obs_normalizer.std,
        max_abs_value=self.max_abs_value,
        dont_normalize_keys=self.dont_normalize_keys)
    features = process_mlp_features(obs, self.mlp_keys, self.mlp_2d_reshape_keys, self.symlog_inputs)
    
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
  
  def update_normalizer(self, obs: Dict[str, torch.Tensor]):
    if self.normalize_obs:
      self.obs_normalizer.update(obs)

  def forward(self,
              obs: Dict[str, torch.Tensor],
              hidden_states: Tuple[torch.Tensor, ...],
              masks: Optional[torch.Tensor]=None,
              mean_only: bool=False) ->Tuple[Dict[str, Union[outs.Output, torch.Tensor]], Optional[Tuple[torch.Tensor, ...]]]:
    processed_obs = self.process_obs(obs)
    rnn_state, new_hidden_states = self.memory(processed_obs, hidden_states, masks)
    model_state = self.model(rnn_state.squeeze(0))
    if mean_only:
        outs = []
        for k in self.heads:
          outs.append(self.heads[k].mean_net(model_state))
        return tuple(outs), new_hidden_states
    else:
        dists = {k: self.heads[k](model_state) for k in self.heads}
        return dists, new_hidden_states

  def stats(self):
    stats = {}
    for layer in self.model:
      if hasattr(layer, 'stats'):
        stats.update(layer.stats())
    for name, head in self.heads.items():
      if hasattr(head, 'stats'):
        stats.update({f'{name}_{k}': v for k, v in head.stats().items()})
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

    def forward(
          self,
          input: torch.Tensor,
          hidden_states: Tuple[torch.Tensor, ...],
          masks: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if masks is not None:  # Batch (update) mode.
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
            return out, None
        else:  # Inference mode
            out, new_hidden_states = self.rnn(input.unsqueeze(0), hidden_states)
            return out, new_hidden_states

    def reset(self, dones, hidden_states=None):
        if self.rnn_cls == nn.LSTM:
          if hidden_states is not None:
            hidden_states[0][..., dones, :] = self.initial_hidden_state.detach().clone().to(hidden_states[0].device)
            hidden_states[1][..., dones, :] = self.initial_cell_state.detach().clone().to(hidden_states[1].device)
            return hidden_states
          else:
            return(
              self.initial_hidden_state.repeat(1, dones.shape[0], 1),
              self.initial_cell_state.repeat(1, dones.shape[0], 1)
            )
        elif self.rnn_cls == nn.GRU:
          if hidden_states is not None:
            hidden_states[0][..., dones, :] = self.initial_hidden_state.detach().clone().to(hidden_states[0].device)
            return hidden_states
          else:
            return (self.initial_hidden_state.repeat(1, dones.shape[0], 1),)
        else:
          raise ValueError(f"Unknown RNN class: {self.rnn_cls}")


def apply_normalizer(
        observations: Dict[str, torch.Tensor],
        normalizer_mean: Optional[Dict[str, torch.Tensor]],
        normalizer_std: Optional[Dict[str, torch.Tensor]],
        use_mean_offset: bool = True,
        dont_normalize_keys: List[str] = [],
        max_abs_value: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
    normalized_observations = {}

    for k, v in observations.items():
        if k in dont_normalize_keys:
            normalized_observations[k] = v
            continue
        if use_mean_offset and normalizer_mean is not None:
            v = v - normalizer_mean[k].detach()
        if normalizer_std is not None:
            v = v / normalizer_std[k].detach()
        if max_abs_value is not None:
            v = torch.clamp(v, -max_abs_value, max_abs_value)
        normalized_observations[k] = v
    return normalized_observations
    

def process_mlp_features(observations: Dict[str, torch.Tensor], mlp_keys: List[str], mlp_2d_reshape_keys: List[str], symlog_inputs: bool) -> torch.Tensor:
    for k in mlp_2d_reshape_keys:
        observations[k] = observations[k].reshape(*observations[k].shape[:-2], -1)
    features = torch.cat([observations[k] for k in mlp_keys], dim=-1)
    if symlog_inputs:
        features = math.symlog(features)
    return features


def permute_cnn_obs(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 5:
        return x.permute(0, 1, 4, 2, 3)  # [B, T, H, W, C] -> [B, T, C, H, W]
    elif x.dim() == 4:
        return x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    elif x.dim() == 3:
        return x.permute(2, 0, 1)  # [H, W, C] -> [H, W]
    else:
        raise ValueError(f"Unexpected shape: {x.shape}")


def get_policy_jitted(policy: RecurrentModel) -> Callable[[Dict[str, torch.Tensor]], torch.Tensor]:
    memory_module = policy.memory
    actor_layers = policy.model[:-1]
    actor_mean_net = policy.model[-1].mean_net
    mlp_keys = policy.mlp_keys
    cnn_keys = policy.cnn_keys
    cnn_model = policy.cnn
    symlog_inputs = policy.symlog_inputs
    max_abs_value: Optional[float] = None
    normalizer_mean: Optional[Dict[str, torch.Tensor]] = None
    normalizer_std: Optional[Dict[str, torch.Tensor]] = None
    if policy.normalize_obs:
      max_abs_value = policy.max_abs_value
      normalizer_mean = {k: v.data for k, v in policy.obs_normalizer.mean.items()}
      normalizer_std = {k: v.data for k, v in policy.obs_normalizer.std.items()}

    @torch.jit.script
    def policy(
          observations: Dict[str, torch.Tensor],
          mlp_keys: List[str],
          cnn_keys: List[str],
          symlog_inputs: bool,
          normalizer_mean: Optional[Dict[str, torch.Tensor]],
          normalizer_std: Optional[Dict[str, torch.Tensor]],
          max_abs_value: Optional[float]) -> torch.Tensor:

        observations = apply_normalizer(observations, normalizer_mean, normalizer_std, max_abs_value=max_abs_value)
        features = process_mlp_features(observations, mlp_keys, symlog_inputs)

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

        input_a = memory_module(features, None, None).squeeze(0)
        latent = actor_layers(input_a)
        mean = actor_mean_net(latent)
        return mean

    return functools.partial(
       policy,
       mlp_keys=mlp_keys,
       cnn_keys=cnn_keys,
       symlog_inputs=symlog_inputs,
       normalizer_mean=normalizer_mean,
       normalizer_std=normalizer_std,
       max_abs_value=max_abs_value)

