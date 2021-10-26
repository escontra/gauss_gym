import numpy as np
from typing import List, Optional
import torch


@torch.jit.script
class Space:
  def __init__(
    self,
    dtype: torch.dtype,
    shape: List[int],
    low: Optional[torch.Tensor] = None,
    high: Optional[torch.Tensor] = None,
  ):
    self._dtype = dtype
    self._low = self._infer_low(dtype, shape, low, high)
    self._high = self._infer_high(dtype, shape, low, high)
    self._shape = shape

  @property
  def dtype(self) -> torch.dtype:
    return self._dtype

  @property
  def shape(self) -> List[int]:
    return self._shape

  @property
  def low(self) -> torch.Tensor:
    return self._low

  @property
  def high(self) -> torch.Tensor:
    return self._high

  @torch.jit.unused
  def __eq__(self, other):
    return (
      self._dtype == other.dtype
      and self._shape == other.shape
      and np.all(self._low == other.low)
      and np.all(self._high == other.high),
    )

  @torch.jit.unused
  def __repr__(self):
    return f'Space({self.dtype}, shape={self.shape}, low={self.low}, high={self.high})'

  @torch.jit.unused
  def __contains__(self, value):
    value = np.asarray(value)
    if np.issubdtype(self.dtype, str):
      return np.issubdtype(value.dtype, str)
    if value.shape != self.shape:
      return False
    if (value > self.high).any():
      return False
    if (value < self.low).any():
      return False
    if value.dtype != self.dtype:
      return False
    return True

  @torch.jit.unused
  def sample(self):
    low, high = self.low, self.high
    if np.issubdtype(self.dtype, np.floating):
      low = np.maximum(np.ones(self.shape) * np.finfo(self.dtype).min, low)
      high = np.minimum(np.ones(self.shape) * np.finfo(self.dtype).max, high)
    return np.random.uniform(low, high, self.shape).astype(self.dtype)

  @torch.jit.unused
  def _infer_low(
    self,
    dtype: torch.dtype,
    shape: List[int],
    low: Optional[torch.Tensor],
    high: Optional[torch.Tensor],
  ) -> torch.Tensor:
    if low is not None:
      if not isinstance(low, torch.Tensor):
        low = torch.tensor(low, dtype=dtype)
      try:
        return torch.broadcast_to(low, shape)
      except ValueError:
        raise ValueError(f'Cannot broadcast {low} to shape {shape}')
    elif dtype.is_floating_point:
      return -torch.inf * torch.ones(shape)
    elif dtype in (torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8):
      return torch.iinfo(dtype).min * torch.ones(shape, dtype=dtype)
    elif dtype == torch.bool:
      return torch.zeros(shape, dtype=dtype)
    else:
      raise ValueError('Cannot infer low bound from shape and dtype.')

  @torch.jit.unused
  def _infer_high(
    self,
    dtype: torch.dtype,
    shape: List[int],
    low: Optional[torch.Tensor],
    high: Optional[torch.Tensor],
  ) -> torch.Tensor:
    if high is not None:
      if not isinstance(high, torch.Tensor):
        high = torch.tensor(high, dtype=dtype)
      try:
        return torch.broadcast_to(high, shape)
      except ValueError:
        raise ValueError(f'Cannot broadcast {high} to shape {shape}')
    elif dtype.is_floating_point:
      return torch.inf * torch.ones(shape)
    elif dtype in (torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8):
      return torch.iinfo(dtype).max * torch.ones(shape, dtype=dtype)
    elif dtype == torch.bool:
      return torch.ones(shape, dtype=dtype)
    else:
      raise ValueError('Cannot infer high bound from shape and dtype.')
