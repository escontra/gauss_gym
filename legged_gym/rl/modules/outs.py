from typing import Tuple, Union, Dict, List, Optional
import functools
import numpy as np
import torch.utils._pytree as pytree
import torch
import torch.nn as nn

from legged_gym.utils import math, space
from legged_gym.rl.modules import decoder
from legged_gym.rl import utils

i32 = torch.int32
f32 = torch.float32


@torch.jit.script
class Output(object):

  @torch.jit.unused
  def __repr__(self):
    name = type(self).__name__
    pred = self.pred()
    return f'{name}({pred.dtype}, shape={pred.shape})'

  @torch.jit.ignore
  def pred(self) -> torch.Tensor:
    raise NotImplementedError

  @torch.jit.ignore
  def loss(self, target: torch.Tensor) -> torch.Tensor:
    return -self.logp(target.detach())

  @torch.jit.ignore
  def sample(self, shape: Union[Tuple[int], Tuple[()]]=()) -> torch.Tensor:
    raise NotImplementedError

  @torch.jit.ignore
  def logp(self, event: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError

  @torch.jit.ignore
  def prob(self, event: torch.Tensor) -> torch.Tensor:
    return torch.exp(self.logp(event))

  @torch.jit.ignore
  def entropy(self) -> torch.Tensor:
    raise NotImplementedError

  @torch.jit.ignore
  def kl(self, other: 'Output') -> torch.Tensor:
    raise NotImplementedError

  def forward(self):
    return self.pred()


class Agg(Output):

  def __init__(self, output, dims, agg=torch.sum):
    self.output = output
    self.axes = [-i for i in range(1, dims + 1)]
    self.agg = agg

  def __repr__(self):
    name = type(self.output).__name__
    pred = self.pred()
    dims = len(self.axes)
    return f'{name}({pred.dtype}, shape={pred.shape}, agg={dims})'

  def pred(self):
    return self.output.pred()

  def loss(self, target):
    loss = self.output.loss(target)
    return self.agg(loss, self.axes)

  def sample(self, shape=()):
    return self.output.sample(shape)

  def logp(self, event):
    return self.output.logp(event).sum(self.axes)

  def prob(self, event):
    return self.output.prob(event).sum(self.axes)

  def entropy(self):
    entropy = self.output.entropy()
    return self.agg(entropy, self.axes)

  def kl(self, other):
    assert isinstance(other, Agg), other
    kl = self.output.kl(other.output)
    return self.agg(kl, self.axes)


class Frozen:

  def __init__(self, output):
    self.output = output

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      fn = getattr(self.output, name)
    except AttributeError:
      raise ValueError(name)
    return functools.partial(self._wrapper, fn)

  def _wrapper(self, fn, *args, **kwargs):
    result = fn(*args, **kwargs)
    result = result.detach()
    return result


class Concat:

  def __init__(self, outputs, midpoints, dim):
    assert len(midpoints) == len(outputs) - 1
    self.outputs = outputs
    self.midpoints = tuple(midpoints)
    self.dim = dim

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      fns = [getattr(x, name) for x in self.outputs]
    except AttributeError:
      raise ValueError(name)
    return functools.partial(self._wrapper, fns)

  def _wrapper(self, fns, *args, **kwargs):
    los = (None,) + self.midpoints
    his = self.midpoints + (None,)
    results = []
    for fn, lo, hi in zip(fns, los, his):
      segment = [slice(None, None, None)] * (self.dim + 1)
      segment[self.dim] = slice(lo, hi, None)
      segment = tuple(segment)
      a, kw = pytree.tree_map(lambda x: x[segment], (args, kwargs))
      results.append(fn(*a, **kw))
    return pytree.tree_map(lambda *xs: torch.cat(xs, self.dim), *results)


@torch.jit.script
class MSE:

  def __init__(self, mean: torch.Tensor):
    self.mean = mean.to(torch.float32)

  def pred(self) -> torch.Tensor:
    return self.mean

  @torch.jit.unused
  def loss(self, target: torch.Tensor) -> torch.Tensor:
    assert target.dtype == torch.float32, target.dtype
    assert self.mean.shape == target.shape, (self.mean.shape, target.shape)
    return torch.square(self.mean - target.to(f32).detach())

  @torch.jit.export
  def __repr__(self):
    return f'MSE(mean={self.mean.shape})'


class Huber(Output):

  def __init__(self, mean, eps=1.0):
    # Soft Huber loss or Charbonnier loss.
    self.mean = mean.to(f32)
    self.eps = eps

  def pred(self):
    return self.mean

  def loss(self, target):
    assert target.dtype == torch.float32, target.dtype
    assert self.mean.shape == target.shape, (self.mean.shape, target.shape)
    dist = self.mean - target.detach()
    return torch.sqrt(torch.square(dist) + torch.square(self.eps)) - self.eps


@torch.jit.script
class Normal:

  def __init__(self, mean: torch.Tensor, stddev: torch.Tensor):
    self.mean = mean.to(torch.float32)
    self.stddev = torch.broadcast_to(stddev.to(torch.float32), self.mean.shape)

  def pred(self) -> torch.Tensor:
    return self.mean

  @torch.jit.unused
  def sample(self, shape: Tuple[int, ...]=()):
    sample = torch.randn(shape + self.mean.shape, dtype=f32, device=self.mean.device)
    return sample * self.stddev + self.mean

  @torch.jit.unused
  def logp(self, event: torch.Tensor) -> torch.Tensor:
    assert event.dtype == torch.float32, event.dtype
    return torch.distributions.Normal(self.mean, self.stddev).log_prob(event)

  @torch.jit.unused
  def entropy(self) -> torch.Tensor:
    return 0.5 * torch.log(2 * torch.pi * torch.square(self.stddev)) + 0.5

  @torch.jit.unused
  def kl(self, other: 'Normal') -> torch.Tensor:
    # assert isinstance(other, type(self)), (self, other)
    return 0.5 * (
        torch.square(self.stddev / other.stddev) +
        torch.square(other.mean - self.mean) / torch.square(other.stddev) +
        2 * torch.log(other.stddev) - 2 * torch.log(self.stddev) - 1)

  @torch.jit.unused
  def loss(self, target: torch.Tensor) -> torch.Tensor:
    return -self.logp(target.detach())

  @torch.jit.export
  def __repr__(self):
    return f'Normal(mean={self.mean.shape}, stddev={self.stddev.shape})'

  @torch.jit.unused
  def repeat(self, num_augs: int, dim: int=1) -> 'Normal':
    repeats = [1] * self.mean.ndim
    repeats[dim] = num_augs
    mean = self.mean.repeat(*repeats)
    stddev = self.stddev.repeat(*repeats)
    return Normal(mean, stddev)


@torch.jit.script
class Binary:

  def __init__(self, logit: torch.Tensor):
    self.logit = logit.to(torch.float32)

  def pred(self) -> torch.Tensor:
    return (self.logit > 0)

  @torch.jit.unused
  def logp(self, event: torch.Tensor) -> torch.Tensor:
    event = event.to(torch.float32)
    logp = torch.nn.functional.logsigmoid(self.logit)
    lognotp = torch.nn.functional.logsigmoid(-self.logit)
    return event * logp + (1 - event) * lognotp

  @torch.jit.unused
  def sample(self, shape: Optional[List[int]]=None) -> torch.Tensor:
    prob = torch.nn.functional.sigmoid(self.logit)
    shape = shape or ()
    return torch.bernoulli(prob, -1, shape + self.logit.shape)

  @torch.jit.unused
  def loss(self, target: torch.Tensor) -> torch.Tensor:
    return -self.logp(target.detach())

  @torch.jit.export
  def __repr__(self):
    return f'Binary(logit={self.logit.shape})'


@torch.jit.script
class TwoHot:

  def __init__(
      self,
      logits: torch.Tensor,
      bins: torch.Tensor,
    ):
    logits = logits.to(torch.float32)
    assert logits.shape[-1] == len(bins), (logits.shape, len(bins))
    assert bins.dtype == torch.float32, bins.dtype
    self.logits = logits
    self.probs = torch.nn.functional.softmax(logits, dim=-1)
    self.bins = bins.clone().detach().requires_grad_(False)

  def pred(self):
    # The naive implementation results in a non-zero result even if the bins
    # are symmetric and the probabilities uniform, because the sum operation
    # goes left to right, accumulating numerical errors. Instead, we use a
    # symmetric sum to ensure that the predicted rewards and values are
    # actually zero at initialization.
    # return self.unsquash((self.probs * self.bins).sum(-1))
    n = self.logits.shape[-1]
    if n % 2 == 1:
      m = (n - 1) // 2
      p1 = self.probs[..., :m]
      p2 = self.probs[..., m: m + 1]
      p3 = self.probs[..., m + 1:]
      b1 = self.bins[..., :m]
      b2 = self.bins[..., m: m + 1]
      b3 = self.bins[..., m + 1:]
      wavg = (p2 * b2).sum(-1) + ((p1 * b1).flip(dims=[-1]) + (p3 * b3)).sum(-1)
      return wavg
    else:
      p1 = self.probs[..., :n // 2]
      p2 = self.probs[..., n // 2:]
      b1 = self.bins[..., :n // 2]
      b2 = self.bins[..., n // 2:]
      wavg = ((p1 * b1).flip(dims=[-1]) + (p2 * b2)).sum(-1)
      return wavg

  @torch.jit.unused
  def loss(self, target):
    assert target.dtype == torch.float32, target.dtype
    target = target.detach()
    below = (self.bins <= target[..., None]).to(torch.int32).sum(-1) - 1
    above = len(self.bins) - (
        self.bins > target[..., None]).to(torch.int32).sum(-1)
    below = torch.clip(below, 0, len(self.bins) - 1)
    above = torch.clip(above, 0, len(self.bins) - 1)
    equal = (below == above)
    dist_to_below = torch.where(equal, 1, torch.abs(self.bins[below] - target))
    dist_to_above = torch.where(equal, 1, torch.abs(self.bins[above] - target))
    total = dist_to_below + dist_to_above
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    target = (
        torch.nn.functional.one_hot(below, len(self.bins)) * weight_below[..., None] +
        torch.nn.functional.one_hot(above, len(self.bins)) * weight_above[..., None])
    log_pred = self.logits - torch.logsumexp(
        self.logits, dim=-1, keepdims=True)
    return -(target * log_pred).sum(-1)


@torch.jit.script
class VoxelDist:
  def __init__(self, occupancy_logit_grid: torch.Tensor, centroid_grid: torch.Tensor):
    self.occupancy_dist = Binary(occupancy_logit_grid)
    self.centroid_dist = MSE(centroid_grid)

  def pred(self) -> Tuple[torch.Tensor, torch.Tensor]:
    return (self.occupancy_dist.pred(), self.centroid_dist.pred())

  @torch.jit.unused
  def occupancy_grid_loss(self, event: torch.Tensor) -> torch.Tensor:
    return self.occupancy_dist.loss(event)

  @torch.jit.unused
  def centroid_grid_loss(self, event: torch.Tensor) -> torch.Tensor:
    return self.centroid_dist.loss(event)

  @torch.jit.export
  def __repr__(self):
    return f'VoxelDist(occupancy_grid={self.occupancy_dist.logit.shape}, centroid_grid={self.centroid_dist.mean.shape})'


class Categorical(Output):

  def __init__(self, logits, unimix=0.0):
    logits = logits.to(f32)
    if unimix:
      probs = torch.nn.functional.softmax(logits, -1)
      uniform = torch.ones_like(probs) / probs.shape[-1]
      probs = (1 - unimix) * probs + unimix * uniform
      logits = torch.log(probs)
    self.logits = logits

  def pred(self):
    return torch.argmax(self.logits, -1)

  def sample(self, shape=()):
    return torch.distributions.Categorical(logits=self.logits).sample(shape)

  def logp(self, event):
    onehot = torch.nn.functional.one_hot(event, self.logits.shape[-1])
    return (torch.nn.functional.log_softmax(self.logits, -1) * onehot).sum(-1)

  def entropy(self):
    logprob = torch.nn.functional.log_softmax(self.logits, -1)
    prob = torch.nn.functional.softmax(self.logits, -1)
    entropy = -(prob * logprob).sum(-1)
    return entropy

  def kl(self, other):
    logprob = torch.nn.functional.log_softmax(self.logits, -1)
    logother = torch.nn.functional.log_softmax(other.logits, -1)
    prob = torch.nn.functional.softmax(self.logits, -1)
    return (prob * (logprob - logother)).sum(-1)


class OneHot(Output):

  def __init__(self, logits, unimix=0.0):
    self.dist = Categorical(logits, unimix)

  def pred(self):
    index = self.dist.pred()
    return self._onehot_with_grad(index)

  def sample(self, shape=()):
    index = self.dist.sample(shape)
    return self._onehot_with_grad(index)

  def logp(self, event):
    return (torch.nn.functional.log_softmax(self.dist.logits, -1) * event).sum(-1)

  def entropy(self):
    return self.dist.entropy()

  def kl(self, other):
    return self.dist.kl(other.dist)

  def _onehot_with_grad(self, index):
    # Straight through gradients.
    value = torch.nn.functional.one_hot(index, self.dist.logits.shape[-1], dtype=f32)
    probs = torch.nn.functional.softmax(self.dist.logits, -1)
    value = value + (probs - value)
    return value


class NormalLogSTDHead(nn.Module):

  def __init__(
      self,
      input_size: int,
      output_space: space.Space,
      init_std: float = 1.0,
      minstd: float = 1.0,
      maxstd: float = 1.0,
      bounded: bool = False,
      outscale: Optional[float] = None,
    ):
    super().__init__()

    self.input_size = input_size
    self.output_size = output_space.shape
    self.init_std = init_std
    self.minstd = minstd
    self.maxstd = maxstd
    mean_net = nn.Linear(input_size, int(np.prod(self.output_size)))
    if outscale is not None:
      utils.init_linear(mean_net, scale=outscale)
    if bounded:
      mean_net = nn.Sequential(
        mean_net,
        nn.Tanh()
      )
    self.mean_net = mean_net
    self.logstd = torch.nn.parameter.Parameter(
      torch.full(
        (1, int(np.prod(self.output_size))),
        fill_value=math.std_to_logstd(torch.tensor(self.init_std), self.minstd, self.maxstd),
        dtype=torch.float32),
      requires_grad=True
    )

  @torch.jit.unused
  def stats(self) -> Dict[str, torch.Tensor]:
    return {
      'logstd': self.logstd.detach(),
      'std': math.logstd_to_std(self.logstd, self.minstd, self.maxstd).detach()
    }

  def forward(self, x: torch.Tensor) -> Normal:
    mean = self.mean_net(x)
    std = math.logstd_to_std(self.logstd, self.minstd, self.maxstd)
    mean = utils.reshape_output(mean, self.output_size)
    std = utils.reshape_output(std, self.output_size)
    output = Normal(mean, std)
    return output


class TwoHotHead(nn.Module):
  def __init__(
      self,
      input_size: int,
      output_space: space.Space,
      bins: int,
      outscale: float = 1.0,
    ):
    super().__init__()
    self.output_space = output_space
    self.output_size = output_space.shape
    self.num_bins = bins
    self.outscale = outscale
    self.projection_net = nn.Linear(input_size, int(np.prod(self.output_size) * self.num_bins)) #, **self.kw)
    utils.init_linear(self.projection_net, scale=self.outscale)

  def forward(self, x: torch.Tensor) -> TwoHot:
    if self.num_bins % 2 == 1:
      half = torch.linspace(-20, 0, (self.num_bins - 1) // 2 + 1, dtype=torch.float32, device=x.device)
      half = math.symexp(half)
      bins = torch.cat([half, -half[:-1].flip(0)], dim=0)
    else:
      half = torch.linspace(-20, 0, self.num_bins // 2, dtype=torch.float32, device=x.device)
      half = math.symexp(half)
      bins = torch.cat([half, -half.flip(0)], dim=0)

    logits = self.projection_net(x)
    logits = utils.reshape_output(logits, (int(np.prod(self.output_size)), self.num_bins))
    return TwoHot(logits, bins)


class MSEHead(nn.Module):
  def __init__(
      self,
      input_size: int,
      output_space: space.Space,
      outscale: float = 1.0,
    ):
    super().__init__()

    self.input_size = input_size
    self.output_size = output_space.shape
    self.outscale = outscale

    self.projection_net = nn.Linear(input_size, int(np.prod(self.output_size)))
    utils.init_linear(self.projection_net, scale=self.outscale)

  def forward(self, x: torch.Tensor) -> MSE:
    pred = self.projection_net(x)
    pred = utils.reshape_output(pred, self.output_size)
    output = MSE(pred)
    return output


class VoxelGridDecoderHead(nn.Module):
  def __init__(
      self,
      input_size: int,
      output_space: space.Space,
    ):
    super().__init__()
    self.input_size = input_size
    self.output_size = output_space.shape

    self.decoder = decoder.ConvDecoder3D(
      in_channels=input_size,
      out_resolution=self.output_size,
      out_channels=input_size,
      mults=(2, 3),
      act_final=True
    )
    self.occupancy_grid_projection = nn.Conv3d(
      in_channels=input_size,
      out_channels=1,
      kernel_size=1,
      stride=1
    )
    self.centroid_grid_projection = nn.Sequential(
      nn.Conv3d(
        in_channels=input_size,
        out_channels=1,
        kernel_size=1,
        stride=1
      ),
      nn.Sigmoid()
    )

  def forward(self, x: torch.Tensor) -> VoxelDist:
    latent_grid = self.decoder(x)
    batch_shape, obs_shape = latent_grid.shape[:-4], latent_grid.shape[-4:]
    latent_grid = utils.flatten_batch(latent_grid, obs_shape)
    occupancy_grid = self.occupancy_grid_projection(latent_grid).squeeze(1)
    centroid_grid = self.centroid_grid_projection(latent_grid).squeeze(1)
    occupancy_grid = utils.unflatten_batch(occupancy_grid, batch_shape)
    centroid_grid = utils.unflatten_batch(centroid_grid, batch_shape)
    return VoxelDist(occupancy_grid, centroid_grid)
