from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.utils._pytree as pytree
import tensordict


def _validate_batch_shapes(batch,
                           reference_sample,
                           batch_dims: Tuple[int, ...]) -> None:
  """Verifies shapes of the batch leaves against the reference sample.

  Checks that batch dimensions are the same in all leaves in the batch.
  Checks that non-batch dimensions for all leaves in the batch are the same
  as in the reference sample.

  Arguments:
    batch: the nested batch of data to be verified.
    reference_sample: the nested array to check non-batch dimensions.
    batch_dims: a Tuple of indices of batch dimensions in the batch shape.

  Returns:
    None.
  """
  def validate_node_shape(reference_sample: torch.Tensor,
                          batch: torch.Tensor) -> None:
    expected_shape = batch_dims + reference_sample.shape
    assert batch.shape == expected_shape, f'{batch.shape} != {expected_shape}'

  pytree.tree_map(validate_node_shape, reference_sample, batch)


class ObservationNormalizer(nn.Module):

  def __init__(self,
               obs_tree,
               max_abs_value: Optional[float] = None,
               std_min_value: float = 1e-6,
               std_max_value: float = 1e6):
    super().__init__()
    self.max_abs_value = max_abs_value
    self.std_min_value = std_min_value
    self.std_max_value = std_max_value

    self.count = nn.Parameter(torch.zeros(size=(), dtype=torch.int32), requires_grad=False)
    self.mean = tensordict.TensorDictParams(
      tensordict.TensorDict(
        pytree.tree_map(
          lambda x: torch.zeros(x[0], dtype=x[1]),
          obs_tree,
          is_leaf=lambda x: isinstance(x, tuple))), no_convert=True)
    self.std = tensordict.TensorDictParams(
      tensordict.TensorDict(
        pytree.tree_map(
          lambda x: torch.ones(x[0], dtype=x[1]),
          obs_tree,
          is_leaf=lambda x: isinstance(x, tuple))), no_convert=True)
    self.summed_variance = tensordict.TensorDictParams(
      tensordict.TensorDict(
        pytree.tree_map(
          lambda x: torch.zeros(x[0], dtype=x[1]),
          obs_tree,
          is_leaf=lambda x: isinstance(x, tuple))), no_convert=True)

  def update(self, obs_tree, validate_shapes: bool = True):
    assert pytree.tree_structure(obs_tree) == pytree.tree_structure(self.mean.to_dict())
    batch_leaves = pytree.tree_leaves(obs_tree)
    if not batch_leaves:
      # Empty batch.
      return
    batch_shape = batch_leaves[0].shape
    # We assume the batch dimensions always go first.
    batch_dims = batch_shape[:len(batch_shape) -
                            pytree.tree_leaves(self.mean.to_dict())[0].ndim]
    batch_axis = list(range(len(batch_dims)))
    step_increment = torch.prod(torch.tensor(batch_dims))

    # Update the item count.
    self.count.data = self.count.data + step_increment

    # Validation is important. If the shapes don't match exactly, but are
    # compatible, arrays will be silently broadcasted resulting in incorrect
    # statistics.
    if validate_shapes:
      _validate_batch_shapes(obs_tree, self.mean.to_dict(), batch_dims)

    def _compute_node_statistics(
        mean: torch.Tensor,
        summed_variance: torch.Tensor,
        batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
      # The mean and the sum of past variances are updated with Welford's
      # algorithm using batches (see https://stackoverflow.com/q/56402955).
      diff_to_old_mean = batch - mean
      mean_update = torch.sum(diff_to_old_mean, dim=batch_axis) / self.count
      mean = mean + mean_update

      diff_to_new_mean = batch - mean
      variance_update = diff_to_old_mean * diff_to_new_mean
      variance_update = torch.sum(variance_update, axis=batch_axis)
      summed_variance = summed_variance + variance_update
      return mean, summed_variance

    updated_stats = pytree.tree_map(_compute_node_statistics, self.mean.to_dict(),
                                    self.summed_variance.to_dict(), obs_tree)
    # Extract `mean` and `summed_variance` from `updated_stats` nest.
    self.mean.update(pytree.tree_map(lambda _, x: x[0], self.mean.to_dict(), updated_stats))
    self.summed_variance.update(pytree.tree_map(lambda _, x: x[1], self.mean.to_dict(), updated_stats))

    def compute_std(summed_variance: torch.Tensor,
                    std: torch.Tensor) -> torch.Tensor:
      assert isinstance(summed_variance, torch.Tensor)
      # Summed variance can get negative due to rounding errors.
      summed_variance = torch.clip(summed_variance, min=0)
      std = torch.sqrt(summed_variance / self.count)
      std = torch.clip(std, self.std_min_value, self.std_max_value)
      return std

    self.std.update(pytree.tree_map(compute_std, self.summed_variance.to_dict(), self.std.to_dict()))

  def normalize(self, obs_tree):
    def normalize_leaf(data: torch.Tensor,
                       mean: torch.Tensor,
                       std: torch.Tensor,
                       max_abs_value: Optional[float] = None
                       ) -> torch.Tensor:
      if not torch.is_floating_point(data):
        return data
      data = (data - mean) / std
      if max_abs_value is not None:
        data = torch.clamp(data, -max_abs_value, max_abs_value)
      return data

    return pytree.tree_map(
      lambda data, mean, std: normalize_leaf(data, mean, std, self.max_abs_value),
      obs_tree,
      self.mean.to_dict(),
      self.std.to_dict())
      
  def denormalize(self, obs_tree):
    def denormalize_leaf(data: torch.Tensor,
                         mean: torch.Tensor,
                         std: torch.Tensor) -> torch.Tensor:
      if not torch.is_floating_point(data):
        return data
      return data * std + mean
    
    return pytree.tree_map(
      denormalize_leaf,
      obs_tree,
      self.mean.to_dict(),
      self.std.to_dict())
