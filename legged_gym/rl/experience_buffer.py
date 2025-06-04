import torch
import torch.utils._pytree as pytree

from legged_gym.rl import utils


class ExperienceBuffer:
  def __init__(self, horizon_length, num_envs, device):
    self.tensor_dict = {}
    self.horizon_length = horizon_length
    self.num_envs = num_envs
    self.device = device

  def add_buffer(self, name, shape, dtype=None):
    if isinstance(shape, tuple):
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
          dtype=getattr(torch, obs_group.dtype.name),
          device=self.device,
        )

  def add_hidden_state_buffers(self, name, hidden_states):
    if hidden_states is None or hidden_states == (None, None):
      return
    hidden_states = (
      hidden_states[0]
      if isinstance(hidden_states[0], tuple)
      else (hidden_states[0],)
    )
    # make a tuple out of GRU hidden state sto match the LSTM format
    # hid_a = (
    #   hidden_states[0]
    #   if isinstance(hidden_states[0], tuple)
    #   else (hidden_states[0],)
    # )
    # hid_c = (
    #   hidden_states[1]
    #   if isinstance(hidden_states[1], tuple)
    #   else (hidden_states[1],)
    # )

    self.tensor_dict[name] = [
      torch.zeros(self.horizon_length, *hidden_states[i].shape, device=self.device)
      for i in range(len(hidden_states))
    ]

  def update_hidden_state_buffers(self, name, idx, hidden_states):
    if hidden_states is None or hidden_states == (None, None):
      return
    # hidden_states = hidden_states[0]
    hidden_states = (
      hidden_states[0]
      if isinstance(hidden_states[0], tuple)
      else (hidden_states[0],)
    )

    # make a tuple out of GRU hidden state sto match the LSTM format
    # hid_a = (
    #   hidden_states[0]
    #   if isinstance(hidden_states[0], tuple)
    #   else (hidden_states[0],)
    # )
    # hid_c = (
    #   hidden_states[1]
    #   if isinstance(hidden_states[1], tuple)
    #   else (hidden_states[1],)
    # )

    for i in range(len(hidden_states)):
      self.tensor_dict[name][i][idx].copy_(hidden_states[i].clone().detach())

  def update_data(self, name, idx, data):
    if isinstance(data, dict):
      for k, v in data.items():
        self.tensor_dict[name][k][idx].copy_(v)
    else:
      self.tensor_dict[name][idx].copy_(data)

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
      old_policy_network,
      value_network,
      last_value_obs,
      last_value_hidden_states,
      image_encoder_obs_key,
      policy_obs_key,
      value_obs_key,
      dones_key,
      symmetry,
      symmetry_fn,
      symm_key):

    # Symmetry-augmented observations computed during the environment step.
    symms = None
    if symm_key in self.tensor_dict and self.tensor_dict[symm_key]:
      symms, _, _, _ = self._split_and_pad_obs(symm_key, dones_key)

    policy_obs, traj_masks, policy_hidden_states, last_was_done = self._split_and_pad_obs(
      policy_obs_key, dones_key, f"{policy_obs_key}_hidden_states")
    value_obs, _, value_hidden_states, _ = self._split_and_pad_obs(
      value_obs_key, dones_key, f"{value_obs_key}_hidden_states")
    image_encoder_obs, _, image_encoder_hidden_states, _ = self._split_and_pad_obs(
      image_encoder_obs_key, dones_key, f"{image_encoder_obs_key}_hidden_states")

    if symmetry:
      # Symmetry-augmented observations.
      image_encoder_obs_sym = {}
      for key, value in image_encoder_obs.items():
        image_encoder_obs_sym[key] = symmetry_fn(image_encoder_obs_key, key)(value).detach()

      policy_obs_sym = {}
      for key, value in policy_obs.items():
        if symms is not None and key in symms:
          policy_obs_sym[key] = symms[key]
        else:
          policy_obs_sym[key] = symmetry_fn(policy_obs_key, key)(value).detach()

    mini_batch_size = self.num_envs // num_mini_batches
    for _ in range(num_learning_epochs):
        first_traj = 0
        with torch.no_grad():
          # Used for value bootstrap.
          last_value = value_network(last_value_obs, last_value_hidden_states)[0]['value'].pred().detach()
        for i in range(num_mini_batches):
            start = i * mini_batch_size
            stop = (i + 1) * mini_batch_size
            last_traj = first_traj + torch.sum(last_was_done[:, start:stop]).item()

            masks_batch = traj_masks[:, first_traj:last_traj]
            image_encoder_obs_batch = pytree.tree_map(lambda x: x[:, first_traj:last_traj], image_encoder_obs)
            policy_obs_batch = pytree.tree_map(lambda x: x[:, first_traj:last_traj], policy_obs)
            value_obs_batch = pytree.tree_map(lambda x: x[:, first_traj:last_traj], value_obs)
            policy_hidden_states_batch = pytree.tree_map(lambda x: x[:, first_traj:last_traj], policy_hidden_states)
            value_hidden_states_batch = pytree.tree_map(lambda x: x[:, first_traj:last_traj], value_hidden_states)
            image_encoder_hidden_states_batch = pytree.tree_map(lambda x: x[:, first_traj:last_traj], image_encoder_hidden_states)
            if symmetry:
              symmetry_obs_batch = {
                f'{policy_obs_key}_sym': pytree.tree_map(lambda x: x[:, first_traj:last_traj], policy_obs_sym),
              }
              symmetry_obs_batch[f'{image_encoder_obs_key}_sym'] = pytree.tree_map(lambda x: x[:, first_traj:last_traj], image_encoder_obs_sym)
            else:
              symmetry_obs_batch = {}

            dones_batch = self.tensor_dict[dones_key][:, start:stop]
            time_outs_batch = self.tensor_dict["time_outs"][:, start:stop]
            rewards_batch = self.tensor_dict["rewards"][:, start:stop]
            actions_batch = {k: self.tensor_dict["actions"][k][:, start:stop] for k in self.tensor_dict["actions"].keys()}
            last_value_batch = pytree.tree_map(lambda x: x[start:stop], last_value)
            old_values_batch = self.tensor_dict["values"][:, start:stop]
            with torch.no_grad():
              old_dist_batch, _, _ = old_policy_network(
                policy_obs_batch,
                masks=masks_batch,
                hidden_states=policy_hidden_states_batch
              )
              if symmetry:
                symmetry_obs_batch['old_dists_sym'], _, _ = old_policy_network(
                  symmetry_obs_batch[f'{policy_obs_key}_sym'],
                  masks=masks_batch,
                  hidden_states=policy_hidden_states_batch
                )

            yield {
              policy_obs_key: policy_obs_batch,
              f"{policy_obs_key}_hidden_states": policy_hidden_states_batch,
              value_obs_key: value_obs_batch,
              f"{value_obs_key}_hidden_states": value_hidden_states_batch,
              image_encoder_obs_key: image_encoder_obs_batch,
              f"{image_encoder_obs_key}_hidden_states": image_encoder_hidden_states_batch,
              "actions": actions_batch,
              "last_value": last_value_batch,
              "masks": masks_batch,
              "dones": dones_batch,
              "time_outs": time_outs_batch,
              "rewards": rewards_batch,
              "old_values": old_values_batch,
              "old_dists": old_dist_batch,
              **symmetry_obs_batch,
            }
            
            first_traj = last_traj
