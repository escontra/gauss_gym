from typing import List, Dict, Any, Callable, Optional
import torch
import torch.utils._pytree as pytree
import dataclasses

from legged_gym.rl import utils


@dataclasses.dataclass
class MiniBatch:
  obs: Dict[str, Any]
  obs_sym: Dict[str, Any]
  hidden_states: Dict[str, Any]
  hidden_states_sym: Dict[str, Any]
  rl_values: Dict[str, Any]
  network_batch: Dict[str, Any]
  network_sym_batch: Dict[str, Any]

  def __repr__(self):
    repr_str = 'Obs:'
    for obs_group in self.obs.keys():
      repr_str += f'\n\t{obs_group}:'
      for obs_name in self.obs[obs_group].keys():
        repr_str += f'\n\t\t{obs_name}: {self.obs[obs_group][obs_name].shape}'
    repr_str += '\n\n'
    repr_str += 'Obs Sym:'
    for obs_group in self.obs_sym.keys():
      repr_str += f'\n\t{obs_group}:'
      for obs_name in self.obs[obs_group].keys():
        repr_str += f'\n\t\t{obs_name}: {self.obs[obs_group][obs_name].shape}'
    repr_str += '\n\n'
    repr_str += 'Hidden States:'
    for hidden_state_group in self.hidden_states.keys():
      repr_str += f'\n\t{hidden_state_group}: {len(self.hidden_states[hidden_state_group])}'
    repr_str += '\n\n'
    repr_str += 'RL Values:'
    for rl_value_group in self.rl_values.keys():
      repr_str += f'\n\t{rl_value_group}: {type(self.rl_values[rl_value_group])}'
    repr_str += '\n\n'
    repr_str += 'Network Batch:'
    for network_group in self.network_batch.keys():
      repr_str += f'\n\t{network_group}'
    repr_str += '\n\n'
    repr_str += 'Network Sym Batch:'
    for network_group in self.network_sym_batch.keys():
      repr_str += f'\n\t{network_group}'
    return repr_str


class ExperienceBuffer:
  def __init__(self, horizon_length, num_envs, device):
    self.tensor_dict = {}
    self.horizon_length = horizon_length
    self.num_envs = num_envs
    self.device = device

  def add_buffer(self, name, shape, dtype=None, is_hidden_state: bool = False):
    if is_hidden_state:
      if shape is None or shape == (None, None):
        return
      shape = (
        shape[0]
        if isinstance(shape[0], tuple)
        else (shape[0],)
      )
      self.tensor_dict[name] = [
        torch.zeros(self.horizon_length, *shape[i].shape, device=self.device)
        for i in range(len(shape))
      ]
    elif isinstance(shape, tuple):
      self.tensor_dict[name] = torch.zeros(
        self.horizon_length,
        self.num_envs,
        *shape,
        dtype=dtype,
        device=self.device,
      )
    elif isinstance(shape, dict):
      self.tensor_dict[name] = {}
      for obs_name, obs_group in shape.items():
        self.tensor_dict[name][obs_name] = torch.zeros(
          self.horizon_length,
          self.num_envs,
          *obs_group.shape,
          dtype=obs_group.dtype,
          device=self.device,
        )
    else:
      raise ValueError(f'Unsupported shape type: {type(shape)}')

  def update_data(self, name, idx, data, is_hidden_state: bool = False):
    if is_hidden_state:
      if data is None or data == (None, None):
        return
      data = (
        data[0]
        if isinstance(data[0], tuple)
        else (data[0],)
      )
      for i in range(len(data)):
        self.tensor_dict[name][i][idx].copy_(data[i].clone().detach())
    elif isinstance(data, torch.Tensor):
      self.tensor_dict[name][idx].copy_(data)
    elif isinstance(data, dict):
      for k, v in data.items():
        self.tensor_dict[name][k][idx].copy_(v)
    else:
      raise ValueError(f'Unsupported data type: {type(data)}')

  def __len__(self):
    return len(self.tensor_dict)

  def __getitem__(self, buf_name):
    return self.tensor_dict[buf_name]

  def keys(self):
    return self.tensor_dict.keys()

  def _split_and_pad_obs(self, obs_key, dones_key, hidden_states_key=None):
    obs_split = pytree.tree_map(
      lambda x: utils.split_and_pad_trajectories(x, self.tensor_dict[dones_key]),
      self.tensor_dict[obs_key],
    )
    obs = pytree.tree_map(
      lambda x: x[0].detach(), obs_split, is_leaf=lambda x: isinstance(x, tuple)
    )
    traj_masks = pytree.tree_map(
      lambda x: x[1].detach(), obs_split, is_leaf=lambda x: isinstance(x, tuple)
    )
    traj_masks = list(traj_masks.values())[0]

    hidden_states, last_was_done = None, None
    if hidden_states_key:
      last_was_done = torch.zeros_like(self.tensor_dict[dones_key], dtype=torch.bool)
      last_was_done[1:] = self.tensor_dict[dones_key][:-1]
      last_was_done[0] = True
      hidden_states = [
        saved_hidden_states.permute(2, 0, 1, 3)[last_was_done.permute(1, 0)].transpose(1, 0)
        for saved_hidden_states in self.tensor_dict[hidden_states_key]
      ]

    return obs, traj_masks, hidden_states, last_was_done

  def reccurent_mini_batch_generator(
      self,
      num_learning_epochs,
      num_mini_batches,
      last_value_items: Optional[List],
      obs_groups: List[str],
      hidden_states_keys: List[str],
      networks: Dict[str, torch.nn.Module],
      networks_sym: Dict[str, torch.nn.Module],
      obs_sym_groups: List[str],
      symm_key: str,
      symmetry_fn: Callable[[str, str], Callable[[torch.Tensor], torch.Tensor]],
      symmetry_flip_latents: bool,
      dones_key: str,
      rl_keys: List[str],
  ):

    # Symmetry-augmented observations computed during the environment step.
    symms = {}
    if symm_key in self.tensor_dict and self.tensor_dict[symm_key]:
      symms, _, _, _ = self._split_and_pad_obs(symm_key, dones_key)

    obs, obs_sym, hidden_states, last_was_done = {}, {}, {}, None
    for obs_group in obs_groups:
      hs_key = None
      if obs_group in hidden_states_keys:
        hs_key = f"{obs_group}_hidden_states"
      obs[obs_group], traj_masks, hidden_states[obs_group], curr_last_was_done = self._split_and_pad_obs(
        obs_group, dones_key, hs_key)
      if last_was_done is None and curr_last_was_done is not None:
        last_was_done = curr_last_was_done
      if obs_group in obs_sym_groups:
        obs_sym[obs_group] = {}
        for key, value in obs[obs_group].items():
          if key in symms:
            obs_sym[obs_group][key] = symms[key]
          else:
            obs_sym[obs_group][key] = symmetry_fn(obs_group, key)(value).detach()

    mini_batch_size = self.num_envs // num_mini_batches
    for _ in range(num_learning_epochs):
      first_traj = 0
      with torch.no_grad():
        if last_value_items is not None:
          last_value_network = last_value_items[0]
          last_value_args = last_value_items[1:]
          last_value = list(last_value_network(*last_value_args)[0].values())[0].pred().detach()
      for i in range(num_mini_batches):
        start = i * mini_batch_size
        stop = (i + 1) * mini_batch_size
        last_traj = first_traj + torch.sum(last_was_done[:, start:stop]).item()

        masks_batch = traj_masks[:, first_traj:last_traj]
        obs_batch, obs_sym_batch, hidden_states_batch, hidden_states_sym_batch = {}, {}, {}, {}
        for obs_group in obs_groups:
          obs_batch[obs_group] = pytree.tree_map(lambda x: x[:, first_traj:last_traj], obs[obs_group])
          if obs_group in obs_sym_groups:
            obs_sym_batch[obs_group] = pytree.tree_map(lambda x: x[:, first_traj:last_traj], obs_sym[obs_group])
          if obs_group in hidden_states_keys:
            hs = pytree.tree_map(lambda x: x[:, first_traj:last_traj], hidden_states[obs_group])
            hidden_states_batch[obs_group] = hs
            if symmetry_flip_latents:
              hidden_states_sym_batch[obs_group] = utils.mirror_latent(hs)
            else:
              hidden_states_sym_batch[obs_group] = hs

        rl_values_batch = {}
        if last_value_items is not None:
          rl_values_batch['last_value'] = pytree.tree_map(lambda x: x[start:stop], last_value)
        rl_values_batch['masks'] = masks_batch
        for key in rl_keys:
          rl_values_batch[key] = pytree.tree_map(lambda x: x[:, start:stop], self.tensor_dict[key])

        with torch.no_grad():
          network_batch, network_sym_batch = {}, {}
          for key, network in networks.items():
            assert key in obs_groups, f'Network key [{key}] not in obs_groups: [{obs_groups}]'
            network_batch[key], _, _ = network(
              obs_batch[key],
              masks=masks_batch,
              hidden_states=hidden_states_batch[key])
          for key, network in networks_sym.items():
            assert key in obs_sym_groups, f'Network key [{key}] not in obs_sym_groups: [{obs_sym_groups}]'
            network_sym_batch[key], _, _ = network(
              obs_sym_batch[key],
              masks=masks_batch,
              hidden_states=hidden_states_sym_batch[key])

        yield MiniBatch(
          obs=obs_batch,
          obs_sym=obs_sym_batch,
          hidden_states=hidden_states_batch,
          hidden_states_sym=hidden_states_sym_batch,
          rl_values=rl_values_batch,
          network_batch=network_batch,
          network_sym_batch=network_sym_batch,
        )
        first_traj = last_traj
