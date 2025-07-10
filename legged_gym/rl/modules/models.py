import abc
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from typing import Dict, Any, Optional, Tuple, List, Union

from legged_gym.utils import math, space
from legged_gym.rl import utils
from legged_gym.rl.modules import resnet, outs
from legged_gym.rl.modules.dino import backbones
from legged_gym.rl.modules.actor_critic import get_activation
from legged_gym.rl.modules import normalizers


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


# Type aliases.
_RNN_STATE = Tuple[torch.Tensor, torch.Tensor]
_DIST = Any  # TODO: Replace with a more specific type.


class RecurrentModel(abc.ABC):
  is_recurrent = True

  @abc.abstractmethod
  def reset(self, dones: torch.Tensor, hidden_states: Optional[Tuple[torch.Tensor, ...]]=None) -> Tuple[torch.Tensor, ...]:
    pass

  @abc.abstractmethod
  def update_normalizer(self, obs: Dict[str, torch.Tensor]):
    pass

  @abc.abstractmethod
  def stats(self) -> Dict[str, torch.Tensor]:
    pass

  @abc.abstractmethod
  def forward(self,
              obs: Dict[str, torch.Tensor],
              hidden_states: Tuple[torch.Tensor, ...],
              masks: Optional[torch.Tensor]=None,
              mean_only: bool=False) ->Tuple[Dict[str, Union[outs.Output, torch.Tensor]], Optional[Tuple[torch.Tensor, ...]]]:
    pass

  @abc.abstractmethod
  def flatten_parameters(self):
    pass


class RecurrentCNNModel(torch.nn.Module, RecurrentModel):
  def __init__(
        self,
        action_space: Dict[str, space.Space],
        obs_space: Dict[str, space.Space],
        head: Dict[str, Any],
        dont_normalize_keys: Optional[List[str]] = None,
        layer_activation: str = "elu",
        hidden_layer_sizes: Optional[List[int]] = [256, 128, 64],
        recurrent_state_size: int =256,
        recurrent_proj_size: int = 0,
        recurrent_skip_connection: bool = False,
        symlog_inputs: bool = False,
        normalize_obs: bool = True,
        max_abs_value: Optional[float] = None,
        model_type: str = 'cnn_simple',
        embedding_dim: int = 256,
        dino_pretrained: bool = True,
        torchaug_transforms: bool = True):

    super().__init__()
    print(f'{self.__class__.__name__}:')
    self.obs_space = obs_space
    self.image_feature_model = ImageFeature(
      obs_space,
      model_type=model_type,
      embedding_dim=embedding_dim,
      dino_pretrained=dino_pretrained,
      torchaug_transforms=torchaug_transforms)
    self.cnn_keys = self.image_feature_model.cnn_keys
    self.mlp_obs_space = {**obs_space, **self.image_feature_model.modified_obs_space()}
    dont_normalize_keys = dont_normalize_keys or []
    self.recurrent_model = RecurrentMLPModel(
      action_space=action_space,
      obs_space=self.mlp_obs_space,
      dont_normalize_keys=dont_normalize_keys + list(self.image_feature_model.obs_space().keys()),
      head=head,
      layer_activation=layer_activation,
      hidden_layer_sizes=hidden_layer_sizes,
      recurrent_state_size=recurrent_state_size,
      recurrent_proj_size=recurrent_proj_size,
      recurrent_skip_connection=recurrent_skip_connection,
      symlog_inputs=symlog_inputs,
      normalize_obs=normalize_obs,
      max_abs_value=max_abs_value)

  @torch.jit.export
  def reset(
      self,
      dones: torch.Tensor,
      hidden_states: Optional[_RNN_STATE]=None) -> _RNN_STATE:
    return self.recurrent_model.reset(dones, hidden_states)

  @torch.jit.ignore
  def update_normalizer(self, obs: Dict[str, torch.Tensor]):
    self.recurrent_model.update_normalizer(obs)

  @torch.jit.ignore
  def stats(self):
    stats = {}
    for layer in self.recurrent_model.model:
      if hasattr(layer, 'stats'):
        stats.update(layer.stats())
    for name, head in self.recurrent_model.heads.items():
      if hasattr(head, 'stats'):
        stats.update({f'{name}_{k}': v for k, v in head.stats().items()})
    return stats

  @torch.jit.ignore
  def encoder_parameters(self):
    return list(self.image_feature_model.parameters())

  @torch.jit.ignore
  def rnn_parameters(self):
    return (
      list(self.recurrent_model.memory.parameters())
      + list(self.recurrent_model.rnn_proj.parameters())
    )

  @torch.jit.ignore
  def decoder_parameters(self):
    return (
      list(self.recurrent_model.model.parameters())
      + list(self.recurrent_model.heads.parameters())
    )

  def forward(self,
              obs: Dict[str, torch.Tensor],
              hidden_states: _RNN_STATE,
              masks: Optional[torch.Tensor]=None,
              unpad: bool=True,
              rnn_only: bool=False,
              ) -> Tuple[
                 Optional[Dict[str, _DIST]],
                 torch.Tensor,
                 Optional[_RNN_STATE]]:
    new_obs: Dict[str, torch.Tensor] = {}
    for k, v in obs.items():
      new_obs[k] = v
    image_feature_obs = self.image_feature_model(obs)
    for k, v in image_feature_obs.items():
      new_obs[k] = v
    return self.recurrent_model.forward(
      new_obs, hidden_states, masks, unpad, rnn_only)

  @torch.jit.ignore
  def flatten_parameters(self):
    self.recurrent_model.flatten_parameters()

  @property
  def rnn_state_size(self) -> int:
    return self.recurrent_model.rnn_state_size

  @property
  def output_dist_names(self) -> List[str]:
    return self.recurrent_model.output_dist_names


class ImageFeature(torch.nn.Module):
  def __init__(
      self,
      obs_space: Dict[str, space.Space],
      model_type: str = 'cnn_simple',
      embedding_dim: int = 256,
      dino_pretrained: bool = True,
      torchaug_transforms: bool = True):
    super().__init__()
    self._obs_space = obs_space
    self.embedding_dim = embedding_dim
    _, _, self.cnn_keys, _ = utils.get_mlp_cnn_keys(obs_space)
    self.model_type = model_type

    print(f'{self.__class__.__name__}:')
    print('\tCNN Keys:')
    encoder_dict = {}
    if self.cnn_keys is not None:
      for key in self.cnn_keys:
        print(f'\t\t{key}: {obs_space[key].shape}')
        if self.model_type == 'cnn_simple':
          encoder_dict[key] = resnet.NatureCNN(3, embedding_dim)
        elif self.model_type == 'cnn_resnet':
          encoder_dict[key] = resnet.ResNet(
              resnet.BasicBlock, [2, 2, 2, 2], num_classes=embedding_dim)
        elif self.model_type.startswith('dino'):
          dino = getattr(backbones, self.model_type)(pretrained=dino_pretrained)
          encoder_dict[key] = nn.Sequential(
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
            dino,
            nn.Linear(dino.embed_dim, embedding_dim)
          )
        elif self.model_type.startswith('timm'):
          import timm
          timm_model = timm.create_model(self.model_type[5:], pretrained=dino_pretrained, num_classes=0)
          cfg = timm_model.default_cfg
          encoder_dict[key] = nn.Sequential(
            transforms.Normalize(mean=cfg['mean'], std=cfg['std']),
            timm_model,
            nn.Linear(timm_model.num_features, embedding_dim)
          )
        else:
          raise ValueError(f'Unknown model type: {model_type}')
    if encoder_dict:
      self.encoder = nn.ModuleDict(encoder_dict)
    else:
      self.encoder = None

    if torchaug_transforms:
      import torchaug
      self.perturbation = torchaug.transforms.SequentialTransform(
        [
          torchaug.transforms.RandomPhotometricDistort(
              brightness=(0.6, 1.4), 
              contrast=(0.6, 1.4),
              saturation=(0.6, 1.4),
              hue=(-0.05, 0.05),
              p_transform=0.5,
              p=0.5,
              batch_transform=True,
          ),
          torchaug.transforms.RandomAutocontrast(
            p=0.5,
            batch_transform=True,
          ),
          torchaug.transforms.RandomGaussianBlur(
            kernel_size=(3, 3),
            sigma=(0.2, 1.0),
            p=0.5,
            batch_transform=True,
          ),
        ],
        inplace=False,
        batch_inplace=False,
        batch_transform=True,
        permute_chunks=False,
      )
    else:
      self.perturbation = transforms.ColorJitter(
        brightness=(0.6, 1.4), 
        contrast=(0.6, 1.4),
        saturation=(0.6, 1.4),
        hue=0
      )
    print(f'Perturbations: \n {self.perturbation}')

  @torch.jit.ignore
  def apply_perturbations(self, cnn_obs: torch.Tensor) -> torch.Tensor:
    if self.training:
      return self.perturbation(cnn_obs)
    else:
      return cnn_obs

  def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out_features = {}
    for key, encoder in self.encoder.items():
      cnn_obs = obs[key]
      if cnn_obs.dtype == torch.uint8:
        cnn_obs = cnn_obs.float() / 255.0

      batch_dims = cnn_obs.shape[:-3]
      spatial_dims = cnn_obs.shape[-3:]
      cnn_obs = utils.flatten_batch(cnn_obs, spatial_dims) # Shape: [M*N*L*O, C, H, W]
      cnn_obs = self.apply_perturbations(cnn_obs)
      cnn_feat = encoder(cnn_obs)
      cnn_feat = utils.unflatten_batch(cnn_feat, batch_dims)
      out_features[key] = cnn_feat
    return out_features

  @torch.jit.unused
  def obs_space(self) -> Dict[str, space.Space]:
    if self.cnn_keys is None:
      return {}
    return {key: self._obs_space[key] for key in self.cnn_keys}

  @torch.jit.unused
  def modified_obs_space(self) -> Dict[str, space.Space]:
    if self.cnn_keys is None:
      return {}
    modified_obs_space = {}
    for key in self.cnn_keys:
      modified_obs_space[key] = space.Space(torch.float32, (self.embedding_dim,), -torch.inf, torch.inf)
    return modified_obs_space


class RecurrentMLPModel(torch.nn.Module, RecurrentModel):
  def __init__(
        self,
        action_space: Dict[str, space.Space],
        obs_space: Dict[str, space.Space],
        head: Dict[str, Any],
        project_dims: Optional[Dict[str, int]] = None,
        dont_normalize_keys: Optional[List[str]] = None,
        layer_activation: str = "elu",
        hidden_layer_sizes: Optional[List[int]] = [256, 128, 64],
        recurrent_state_size: int =256,
        recurrent_proj_size: int = 0,
        recurrent_skip_connection: bool = False,
        symlog_inputs: bool = False,
        normalize_obs: bool = True,
        max_abs_value: Optional[float] = None):
    super().__init__()
    if project_dims is None:
      project_dims = {}

    self.obs_space = obs_space
    self.obs_keys = list(obs_space.keys())
    self.dont_normalize_keys: List[str] = dont_normalize_keys or ['_DEFAULT_UNUSED']
    self.action_space = action_space

    input_projectors = {}
    for k, v in project_dims.items():
      input_projectors[k] = nn.Sequential(
        torch.nn.Linear(int(np.prod(obs_space[k].shape)), v),
        get_activation(layer_activation)
      )
    if input_projectors:
      self.input_projectors = nn.ModuleDict(input_projectors)
    else:
      self.input_projectors = None

    self.mlp_keys, mlp_2d_reshape_keys, self.cnn_keys, self.num_mlp_obs = utils.get_mlp_cnn_keys(obs_space, project_dims)
    self.mlp_2d_reshape_keys: List[str] = mlp_2d_reshape_keys or ['_DEFAULT_UNUSED']
    assert self.cnn_keys is None or len(self.cnn_keys) == 0, f"CNN keys are not supported for {self.__class__.__name__}, got: [{self.cnn_keys}]"
    self.obs_size = self.num_mlp_obs
    self.recurrent_state_size = recurrent_state_size
    self.recurrent_proj_size = recurrent_proj_size
    self.recurrent_skip_connection = recurrent_skip_connection
    self.memory = Memory(self.obs_size, type='lstm', num_layers=1, hidden_size=self.recurrent_state_size)
    self.normalize_obs = normalize_obs
    self.max_abs_value = max_abs_value
    print(f'{self.__class__.__name__}:')
    if self.normalize_obs:
      self.obs_normalizer = normalizers.DictNormalizer(obs_space, max_abs_value=self.max_abs_value, dont_normalize_keys=dont_normalize_keys)
    print('\tMLP Keys:')
    if self.mlp_keys is not None:
      for key in self.mlp_keys:
        print(f'\t\t{key}: {obs_space[key].shape}')
    print(f'\tMLP Num Obs: {self.num_mlp_obs}')
    print(f'\tNormalizing obs: {self.normalize_obs}')

    if self.recurrent_proj_size > 0:
      self.rnn_proj = torch.nn.Linear(self.recurrent_state_size, self.recurrent_proj_size)
    else:
      self.rnn_proj = torch.nn.Identity()

    layers = []
    input_size = self.rnn_state_size
    for size in hidden_layer_sizes:
      layers.append(torch.nn.Linear(input_size, size))
      layers.append(get_activation(layer_activation))
      input_size = size
    self.model = torch.nn.Sequential(*layers)
    heads = {}
    for k, v in action_space.items():
      heads[k] = getattr(outs, head[k]['class_name'])(input_size, v, **head[k]['params'])
    self.heads = nn.ModuleDict(heads)
    print('\tOutput Heads')
    for k, v in heads.items():
      print(f'\t\t{k}: {v.output_size} {v.__class__.__name__}')

    self.symlog_inputs = symlog_inputs

  @property
  def output_dist_names(self) -> List[str]:
    dist_names = []
    for k, v in self.heads.items():
      if isinstance(v, outs.VoxelGridDecoderHead):
        dist_names.append(f'{k}/occupancy_grid')
        dist_names.append(f'{k}/centroid_grid')
      else:
        dist_names.append(k)
    return dist_names

  @torch.jit.export
  def reset(
      self,
      dones: torch.Tensor,
      hidden_states: Optional[Tuple[torch.Tensor, torch.Tensor]]=None) -> Tuple[torch.Tensor, torch.Tensor]:
    return self.memory.reset(dones, hidden_states)

  def process_obs(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
    normalizer_mean = self.obs_normalizer.mean()
    normalizer_std = self.obs_normalizer.std()
    processed_obs: Dict[str, torch.Tensor] = {}
    for k in self.mlp_keys:
      processed_obs[k] = obs[k]
      if self.normalize_obs and k not in self.dont_normalize_keys:
        if normalizer_mean is not None:
            processed_obs[k] = processed_obs[k] - normalizer_mean[k].detach()
        if normalizer_std is not None:
            processed_obs[k] = processed_obs[k] / normalizer_std[k].detach()
        if self.max_abs_value is not None:
            processed_obs[k] = torch.clamp(processed_obs[k], -self.max_abs_value, self.max_abs_value)
      if k in self.mlp_2d_reshape_keys:
        processed_obs[k] = utils.flatten_obs(processed_obs[k], self.obs_space[k].shape)
      if self.symlog_inputs:
        processed_obs[k] = math.symlog(processed_obs[k])
    if self.input_projectors is not None:
      for k, net in self.input_projectors.items():
        processed_obs[k] = net(processed_obs[k])
    features = torch.cat([processed_obs[k] for k in self.mlp_keys], dim=-1)
    return features

  @torch.jit.unused
  def update_normalizer(self, obs: Dict[str, torch.Tensor], multi_gpu: bool = False):
    if self.normalize_obs:
      self.obs_normalizer.update(obs)
      if multi_gpu:
        utils.sync_state_dict(self.obs_normalizer, 0)

  def forward(self,
              obs: Dict[str, torch.Tensor],
              hidden_states: _RNN_STATE,
              masks: Optional[torch.Tensor]=None,
              unpad: bool=True,
              rnn_only: bool=False,
              ) -> Tuple[
                 Optional[Dict[str, _DIST]],
                 torch.Tensor,
                 Optional[_RNN_STATE]]:
    processed_obs = self.process_obs(obs)

    # RNN.
    rnn_state, new_hidden_states = self.memory(processed_obs, hidden_states, masks, unpad=unpad)
    rnn_state = rnn_state.squeeze(0)
    rnn_state = self.rnn_proj(rnn_state)

    if self.recurrent_skip_connection:
      # Concatenate the processed obs with the rnn state.
      if masks is not None and unpad:
        processed_obs = utils.unpad_trajectories(processed_obs, masks)
      rnn_state = torch.cat([rnn_state, processed_obs], dim=-1)

    # Heads.
    if rnn_only:
      dists = None
    else:
      model_state: torch.Tensor = self.model(rnn_state)
      dists: Dict[str, Any] = {k: v(model_state) for k, v in self.heads.items()}
    return dists, rnn_state, new_hidden_states

  @torch.jit.unused
  def stats(self):
    stats = {}
    for layer in self.model:
      if hasattr(layer, 'stats'):
        stats.update(layer.stats())
    for name, head in self.heads.items():
      if hasattr(head, 'stats'):
        stats.update({f'{name}_{k}': v for k, v in head.stats().items()})
    return stats

  @torch.jit.unused
  def flatten_parameters(self):
    self.memory.rnn.flatten_parameters()

  @property
  def rnn_state_size(self):
    rnn_size = self.recurrent_proj_size if self.recurrent_proj_size > 0 else self.recurrent_state_size
    if self.recurrent_skip_connection:
      rnn_size += self.num_mlp_obs
    return rnn_size


class Memory(torch.nn.Module):
    def __init__(self, input_size: List[int], type: str='lstm', num_layers: int=1, hidden_size: int=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.is_lstm = rnn_cls is nn.LSTM
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.rnn.flatten_parameters()

        # Learnable initial hidden state (size remains [L, 1, H])
        self.initial_hidden_state = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_size), requires_grad=True)
        if rnn_cls is nn.LSTM:
            self.initial_cell_state = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_size), requires_grad=True)

    def forward(
          self,
          input: torch.Tensor,
          hidden_states: _RNN_STATE,
          masks: Optional[torch.Tensor]=None,
          unpad: bool=True) -> Tuple[torch.Tensor, Optional[_RNN_STATE]]:
        if masks is not None:  # Batch (update) mode.
            out, _ = self.rnn(input, hidden_states)
            if unpad:
              out = utils.unpad_trajectories(out, masks)
            return out, None
        else:  # Inference mode
            out, new_hidden_states = self.rnn(input.unsqueeze(0), hidden_states)
            return out, new_hidden_states

    def reset(
            self,
            dones: torch.Tensor,
            hidden_states: Optional[_RNN_STATE]=None
          ) -> _RNN_STATE:
        if self.is_lstm:
          if hidden_states is not None:
            hidden_states[0][:, dones, :] = self.initial_hidden_state.detach().clone().to(hidden_states[0].device)
            hidden_states[1][:, dones, :] = self.initial_cell_state.detach().clone().to(hidden_states[1].device)
            return hidden_states
          else:
            return(
              self.initial_hidden_state.repeat(1, dones.shape[0], 1),
              self.initial_cell_state.repeat(1, dones.shape[0], 1)
            )
        else:
          raise ValueError("GRU NOT SUPPORTED")
