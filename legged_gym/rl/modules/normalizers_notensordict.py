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

def recursive_apply_normalize(obs_node, mean_node, std_node, normalize_leaf_fn_with_args):
    """
    Recursively applies a function to corresponding leaves of three tree structures.
    Assumes all trees have the same structure.
    """
    if isinstance(obs_node, torch.Tensor):
        # Base case: Found corresponding leaves, apply the function
        # Ensure mean and std are also tensors here if needed
        if not (isinstance(mean_node, torch.Tensor) and isinstance(std_node, torch.Tensor)):
              raise TypeError(f"Leaf type mismatch: expected Tensors, got {type(mean_node)}, {type(std_node)}")
        return normalize_leaf_fn_with_args(obs_node, mean_node, std_node)

    elif isinstance(obs_node, dict):
        # Recursive step for dictionaries
        if not (isinstance(mean_node, dict) and isinstance(std_node, dict)):
            raise TypeError("Structure mismatch: expected dicts")
        new_dict = {}
        for key in obs_node:
            if key not in mean_node or key not in std_node:
                raise KeyError(f"Key '{key}' missing in mean or std structure during normalization")
            new_dict[key] = recursive_apply_normalize(
                obs_node[key], mean_node[key], std_node[key], normalize_leaf_fn_with_args
            )
        return new_dict

    elif isinstance(obs_node, (list, tuple)):
          # Recursive step for lists/tuples
        if not (isinstance(mean_node, type(obs_node)) and isinstance(std_node, type(obs_node))):
            raise TypeError(f"Structure mismatch: expected {type(obs_node)}")
        if not (len(obs_node) == len(mean_node) == len(std_node)):
              raise ValueError("Structure mismatch: sequence lengths differ")

        new_seq = [
            recursive_apply_normalize(o, m, s, normalize_leaf_fn_with_args)
            for o, m, s in zip(obs_node, mean_node, std_node)
        ]
        return type(obs_node)(new_seq) # Return same sequence type

    else:
        # Handle other potential structures or raise an error
        raise TypeError(f"Unsupported node type in tree structure: {type(obs_node)}")



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

    self.mean_dict = None
    self.std_dict = None

  def normalize(self, obs_tree):
    def normalize_fn_for_node(data, mean, std):
        return normalize_leaf(
            data, mean, std,
            max_abs_value=self.max_abs_value,
            use_mean_offset=self.use_mean_offset
        )

    if self.mean_dict is None:
      self.mean_dict = param_container_to_plain_structure(self.mean)
    if self.std_dict is None:
      self.std_dict = param_container_to_plain_structure(self.std)

    normalized_obs_tree = recursive_apply_normalize(
        obs_tree,
        self.mean_dict,
        self.std_dict,
        normalize_fn_for_node # Pass the function reference
    )

    return normalized_obs_tree

