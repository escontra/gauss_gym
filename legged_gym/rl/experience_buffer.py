import torch


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
          *obs_group[0],
          dtype=obs_group[1],
          device=self.device,
        )

  def add_hidden_state_buffers(self, hidden_states):
    if hidden_states is None or hidden_states == (None, None):
      return
    # make a tuple out of GRU hidden state sto match the LSTM format
    hid_a = (
      hidden_states[0]
      if isinstance(hidden_states[0], tuple)
      else (hidden_states[0],)
    )
    hid_c = (
      hidden_states[1]
      if isinstance(hidden_states[1], tuple)
      else (hidden_states[1],)
    )

    self.tensor_dict["hid_a"] = [
      torch.zeros(self.horizon_length, *hid_a[i].shape, device=self.device)
      for i in range(len(hid_a))
    ]
    self.tensor_dict["hid_c"] = [
      torch.zeros(self.horizon_length, *hid_c[i].shape, device=self.device)
      for i in range(len(hid_c))
    ]

  def update_hidden_state_buffers(self, idx, hidden_states):
    if hidden_states is None or hidden_states == (None, None):
      return

    # make a tuple out of GRU hidden state sto match the LSTM format
    hid_a = (
      hidden_states[0]
      if isinstance(hidden_states[0], tuple)
      else (hidden_states[0],)
    )
    hid_c = (
      hidden_states[1]
      if isinstance(hidden_states[1], tuple)
      else (hidden_states[1],)
    )

    for i in range(len(hid_a)):
      self.tensor_dict["hid_a"][i][idx].copy_(hid_a[i].clone().detach())
      self.tensor_dict["hid_c"][i][idx].copy_(hid_c[i].clone().detach())

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
