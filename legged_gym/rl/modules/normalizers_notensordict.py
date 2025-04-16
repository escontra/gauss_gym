from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.utils._pytree as pytree

def _create_param_tree(node):
    if isinstance(node, tuple):
        # Leaf node: create the zero tensor wrapped in nn.Parameter
        shape, dtype = node
        # Ensure shape is a tuple/list if it's a single int
        if isinstance(shape, int):
            shape = (shape,)
        return nn.Parameter(torch.zeros(shape, dtype=dtype))
    elif isinstance(node, dict):
        # Dictionary node: recurse on values, wrap result in nn.ParameterDict
        return nn.ParameterDict({k: _create_param_tree(v) for k, v in node.items()})
    elif isinstance(node, list):
         # List node: recurse on items, wrap result in nn.ParameterList
         return nn.ParameterList([_create_param_tree(item) for item in node])
    else:
        # Handle other potential types in obs_tree if necessary
        raise TypeError(f"Unsupported type in obs_tree structure: {type(node)}")


def param_container_to_plain_structure(container):
    """
    Recursively converts nn.ParameterDict or nn.ParameterList
    to plain Python dicts or lists, extracting tensor data.
    """
    if isinstance(container, nn.ParameterDict):
        new_dict = {}
        for key, value in container.items():
            if isinstance(value, nn.Parameter):
                new_dict[key] = value.data  # Extract the tensor data
            elif isinstance(value, (nn.ParameterDict, nn.ParameterList)):
                new_dict[key] = param_container_to_plain_structure(value) # Recurse
            else:
                new_dict[key] = value # Keep non-parameter/container items as is
        return new_dict
    elif isinstance(container, nn.ParameterList):
        new_list = []
        for item in container:
            if isinstance(item, nn.Parameter):
                new_list.append(item.data) # Extract the tensor data
            elif isinstance(item, (nn.ParameterDict, nn.ParameterList)):
                new_list.append(param_container_to_plain_structure(item)) # Recurse
            else:
                new_list.append(item) # Keep non-parameter/container items as is
        return new_list
    elif isinstance(container, nn.Parameter):
         # Handle case where the top-level item is a Parameter itself
         return container.data
    else:
        # If it's already not a parameter container, return it directly
        return container


class PyTreeNormalizer(nn.Module):

  def __init__(self,
               obs_tree,
               max_abs_value: Optional[float] = None,
               std_min_value: float = 1e-6,
               std_max_value: float = 1e6,
               use_mean_offset: bool = True):
    super().__init__()
    self.max_abs_value = max_abs_value
    self.std_min_value = std_min_value
    self.std_max_value = std_max_value
    self.use_mean_offset = use_mean_offset
    self.count = nn.Parameter(torch.zeros(size=(), dtype=torch.int32), requires_grad=False)
    self.mean = _create_param_tree(obs_tree)
    self.std = _create_param_tree(obs_tree)
    self.summed_variance = _create_param_tree(obs_tree)
    # self.mean = tensordict.TensorDictParams(
    #   tensordict.TensorDict(
    #     pytree.tree_map(
    #       lambda x: torch.zeros(x[0], dtype=x[1]),
    #       obs_tree,
    #       is_leaf=lambda x: isinstance(x, tuple))), no_convert=True)
    # self.std = tensordict.TensorDictParams(
    #   tensordict.TensorDict(
    #     pytree.tree_map(
    #       lambda x: torch.ones(x[0], dtype=x[1]),
    #       obs_tree,
    #       is_leaf=lambda x: isinstance(x, tuple))), no_convert=True)
    # self.summed_variance = tensordict.TensorDictParams(
    #   tensordict.TensorDict(
    #     pytree.tree_map(
    #       lambda x: torch.zeros(x[0], dtype=x[1]),
    #       obs_tree,
    #       is_leaf=lambda x: isinstance(x, tuple))), no_convert=True)

  def normalize(self, obs_tree):
    def normalize_leaf(data: torch.Tensor,
                       mean: torch.Tensor,
                       std: torch.Tensor,
                       max_abs_value: Optional[float] = None,
                       use_mean_offset: bool = True
                       ) -> torch.Tensor:
      if not torch.is_floating_point(data):
        return data
      if use_mean_offset:
        data = data - mean
      data = data / std
      if max_abs_value is not None:
        data = torch.clamp(data, -max_abs_value, max_abs_value)
      return data

    return pytree.tree_map(
      lambda data, mean, std: normalize_leaf(
        data, mean, std,
        max_abs_value=self.max_abs_value,
        use_mean_offset=self.use_mean_offset
      ),
      obs_tree,
      param_container_to_plain_structure(self.mean),
      param_container_to_plain_structure(self.std))
      
  def denormalize(self, obs_tree):
    def denormalize_leaf(data: torch.Tensor,
                         mean: torch.Tensor,
                         std: torch.Tensor,
                         use_mean_offset: bool = True) -> torch.Tensor:
      if not torch.is_floating_point(data):
        return data
      data = data * std
      if use_mean_offset:
        data = data + mean
      return data
    
    return pytree.tree_map(
      lambda data, mean, std: denormalize_leaf(
        data, mean, std,
        use_mean_offset=self.use_mean_offset
      ),
      obs_tree,
      self.mean,
      self.std)

