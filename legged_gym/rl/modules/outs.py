from typing import Tuple, Union, Dict
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

  def __init__(self, mean: torch.Tensor, squash=None):
    self.mean = mean.to(torch.float32)
    self.squash = squash

  def pred(self) -> torch.Tensor:
    return self.mean

  @torch.jit.unused
  def loss(self, target: torch.Tensor) -> torch.Tensor:
    assert target.dtype == torch.float32, target.dtype
    assert self.mean.shape == target.shape, (self.mean.shape, target.shape)
    squash = self.squash or (lambda x: x)
    return torch.square(self.mean - (squash(target.to(f32))).detach())


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
  def sample(self, shape: Tuple[int, ...]=()) -> torch.Tensor:
    prob = torch.nn.functional.sigmoid(self.logit)
    return torch.bernoulli(prob, -1, shape + self.logit.shape)

  @torch.jit.unused
  def loss(self, target: torch.Tensor) -> torch.Tensor:
    return -self.logp(target.detach())


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


class TwoHot(Output):

  def __init__(self, logits, bins, squash=None, unsquash=None):
    logits = logits.to(f32)
    assert logits.shape[-1] == len(bins), (logits.shape, len(bins))
    assert bins.dtype == f32, bins.dtype
    self.logits = logits
    self.probs = torch.nn.functional.softmax(logits, dim=-1)
    self.bins = bins.clone().detach().requires_grad_(False)

    self.squash = squash or (lambda x: x)
    self.unsquash = unsquash or (lambda x: x)

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
      return self.unsquash(wavg)
    else:
      p1 = self.probs[..., :n // 2]
      p2 = self.probs[..., n // 2:]
      b1 = self.bins[..., :n // 2]
      b2 = self.bins[..., n // 2:]
      wavg = ((p1 * b1).flip(dims=[-1]) + (p2 * b2)).sum(-1)
      return self.unsquash(wavg)

  def loss(self, target):
    assert target.dtype == f32, target.dtype
    target = self.squash(target).detach()
    below = (self.bins <= target[..., None]).to(i32).sum(-1) - 1
    above = len(self.bins) - (
        self.bins > target[..., None]).to(i32).sum(-1)
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
        self.logits, -1, keepdims=True)
    return -(target * log_pred).sum(-1)


class Head(torch.nn.Module):

  def __init__(
      self,
      input_size: int,
      output_space: space.Space,
      output_type: str,
      init_std: float = 1.0,
      minstd: float = 1.0,
      maxstd: float = 1.0,
      unimix: float = 0.0,
      bins: int = 255,
      outscale: float = 1.0,
    ):
      # **kw):
    super().__init__()
    self.input_size = input_size
    self.output_size = output_space.shape
    self.impl = output_type
    # self.kw = kw
    self.init_std = init_std
    self.minstd = minstd
    self.maxstd = maxstd
    self.unimix = unimix
    self.bins = bins
    self.outscale = outscale
    if self.impl == 'voxel_grid_decoder':
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
      self._forward_method = self.voxel_grid_decoder
    elif self.impl == 'mse':
      self.projection_net = nn.Linear(input_size, int(np.prod(self.output_size))) #, **self.kw)
      self._init_layer(self.projection_net)
      self._forward_method = self.mse
    # elif self.impl == 'symexp_twohot':
    #   self.projection_net = nn.Linear(input_size, int(np.prod(self.output_size) * self.bins)) #, **self.kw)
    #   self._init_layer(self.projection_net)
    # elif self.impl == 'bounded_normal':
    #   self.mean_net = nn.Linear(input_size, int(np.prod(self.output_size))) #, **self.kw)
    #   self.stddev_net = nn.Linear(input_size, int(np.prod(self.output_size))) #, **self.kw)
    #   self._init_layer(self.mean_net)
    #   self._init_layer(self.stddev_net)
    # elif self.impl == 'normal_logstd':
    #   self.mean_net = nn.Linear(input_size, int(np.prod(self.output_size))) #, **self.kw)
    #   self.stddev_net = nn.Linear(input_size, int(np.prod(self.output_size))) #, **self.kw)
    #   self._init_layer(self.mean_net)
    #   self._init_layer(self.stddev_net)
    # elif self.impl == 'normal_logstdparam':
    #   self.mean_net = nn.Linear(input_size, int(np.prod(self.output_size))) #, **self.kw)
    #   self.logstd = torch.nn.parameter.Parameter(
    #     torch.full((1, int(np.prod(self.output_size))), fill_value=np.log(self.init_std)), requires_grad=True
    #   )
    elif self.impl in ('normal_logstdparam_unclipped', 'bounded_normal_logstdparam_unclipped'):
      mean_net = nn.Linear(input_size, int(np.prod(self.output_size))) #, **self.kw)
      self._forward_method = self.normal_logstdparam_unclipped
      if self.impl == 'bounded_normal_logstdparam_unclipped':
        mean_net = nn.Sequential(
          mean_net,
          nn.Tanh()
        )
        self._forward_method = self.bounded_normal_logstdparam_unclipped
      self.mean_net = mean_net
      self.logstd = torch.nn.parameter.Parameter(
        torch.full(
          (1, int(np.prod(self.output_size))),
          fill_value=math.std_to_logstd(torch.tensor(self.init_std), self.minstd, self.maxstd),
          dtype=torch.float32),
        requires_grad=True
      )
    else:
      raise NotImplementedError(self.impl)

  @torch.jit.unused
  def _init_layer(self, layer):
    torch.nn.init.trunc_normal_(layer.weight)
    layer.weight.data *= self.outscale
    torch.nn.init.zeros_(layer.bias)

  def forward(self, x: torch.Tensor) -> Normal:
    # if not hasattr(self, self.impl):
    #   raise NotImplementedError(self.impl)
    # output = getattr(self, self.impl)(x)
    output = self._forward_method(x)
    return output

  def voxel_grid_decoder(self, x):
    latent_grid = self.decoder(x)
    bshape = latent_grid.shape[:-4]
    latent_grid = latent_grid.reshape((-1, *latent_grid.shape[-4:]))
    occupancy_grid = self.occupancy_grid_projection(latent_grid).squeeze(-1)
    centroid_grid = self.centroid_grid_projection(latent_grid).squeeze(-1)
    occupancy_grid = occupancy_grid.reshape((*bshape, *occupancy_grid.shape[-3:]))
    centroid_grid = centroid_grid.reshape((*bshape, *centroid_grid.shape[-3:]))
    occupancy_dist = Binary(occupancy_grid)
    centroid_dist = MSE(centroid_grid)
    return occupancy_dist, centroid_dist

  def mse(self, x):
    pred = self.projection_net(x)
    pred = pred.reshape((*pred.shape[:-1], *self.output_size))
    return MSE(pred)

  # def symexp_twohot(self, x):
  #   logits = self.projection_net(x)
  #   logits = logits.reshape((*x.shape[:-1], self.output_size, self.bins))
  #   if self.bins % 2 == 1:
  #     half = torch.linspace(-20, 0, (self.bins - 1) // 2 + 1, dtype=torch.float32, device=x.device)
  #     half = math.symexp(half)
  #     bins = torch.cat([half, -half[:-1].flip(0)], 0)
  #   else:
  #     half = torch.linspace(-20, 0, self.bins // 2, dtype=torch.float32, device=x.device)
  #     half = math.symexp(half)
  #     bins = torch.cat([half, -half.flip(0)], 0)
  #   return TwoHot(logits, bins)

  # def bounded_normal(self, x):
  #   mean = self.mean_net(x)
  #   stddev = self.stddev_net(x)
  #   mean = mean.reshape((*mean.shape[:-1], *self.output_size))
  #   stddev = stddev.reshape((*stddev.shape[:-1], *self.output_size))
  #   lo, hi = self.minstd, self.maxstd
  #   stddev = (hi - lo) * torch.sigmoid(stddev + 2.0) + lo
  #   output = Normal(torch.tanh(mean), stddev)
  #   return output

  # def normal_logstd(self, x):
  #   mean = self.mean_net(x)
  #   stddev = torch.exp(self.stddev_net(x))
  #   mean = mean.reshape((*mean.shape[:-1], *self.output_size))
  #   stddev = stddev.reshape((*stddev.shape[:-1], *self.output_size))
  #   lo, hi = self.minstd, self.maxstd
  #   stddev = (hi - lo) * torch.sigmoid(stddev + 2.0) + lo
  #   output = Normal(mean, stddev)
  #   return output

  # def normal(self, x):
  #   mean = self.mean_net(x)
  #   stddev = self.stddev_net(x)
  #   mean = mean.reshape((*mean.shape[:-1], *self.output_size))
  #   stddev = stddev.reshape((*stddev.shape[:-1], *self.output_size))
  #   lo, hi = self.minstd, self.maxstd
  #   stddev = (hi - lo) * torch.sigmoid(stddev + 2.0) + lo
  #   output = Normal(mean, stddev)
  #   return output

  # def normal_logstdparam(self, x):
  #   mean = self.mean_net(x)
  #   with torch.no_grad():
  #     log_min = np.log(self.minstd)
  #     log_max = np.log(self.maxstd)
  #     self.logstd.copy_(self.logstd.clip(min=log_min, max=log_max))
  #   mean = mean.reshape((*mean.shape[:-1], *self.output_size))
  #   std = torch.exp(self.logstd)
  #   std = std.reshape((*std.shape[:-1], *self.output_size))
  #   output = Normal(mean, std)
  #   return output

  def normal_logstdparam_unclipped(self, x):
    mean = self.mean_net(x)
    std = math.logstd_to_std(self.logstd, self.minstd, self.maxstd)
    mean = mean.reshape((*mean.shape[:-1], *self.output_size))
    std = std.reshape((*std.shape[:-1], *self.output_size))
    output = Normal(mean, std)
    return output

  def bounded_normal_logstdparam_unclipped(self, x):
    mean = self.mean_net(x)
    std = math.logstd_to_std(self.logstd, self.minstd, self.maxstd)
    mean = mean.reshape((*mean.shape[:-1], *self.output_size))
    std = std.reshape((*std.shape[:-1], *self.output_size))
    output = Normal(mean, std)
    return output

  # def stats(self):
  #   if self.impl == 'normal_logstdparam':
  #     return {
  #       'logstd': self.logstd.detach().cpu().numpy(),
  #       'std': torch.exp(self.logstd).detach().cpu().numpy()
  #     }
  #   elif self.impl == 'normal_logstdparam_unclipped':
  #     return {
  #       'logstd': self.logstd.detach().cpu().numpy(),
  #       'std': math.logstd_to_std(self.logstd, self.minstd, self.maxstd).detach().cpu().numpy()
  #     }
  #   else:
  #     return {}

class NormalLogSTDHead(nn.Module):

  def __init__(
      self,
      input_size: int,
      output_space: space.Space,
      init_std: float = 1.0,
      minstd: float = 1.0,
      maxstd: float = 1.0,
      bounded: bool = False,
    ):
    super().__init__()

    self.input_size = input_size
    self.output_size = output_space.shape
    self.init_std = init_std
    self.minstd = minstd
    self.maxstd = maxstd
    mean_net = nn.Linear(input_size, int(np.prod(self.output_size))) #, **self.kw)
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

    self.projection_net = nn.Linear(input_size, int(np.prod(self.output_size))) #, **self.kw)
    self._init_layer(self.projection_net)

  @torch.jit.unused
  def _init_layer(self, layer):
    torch.nn.init.trunc_normal_(layer.weight)
    layer.weight.data *= self.outscale
    torch.nn.init.zeros_(layer.bias)

  def forward(self, x: torch.Tensor) -> MSE:
    pred = self.projection_net(x)
    pred = utils.reshape_output(pred, self.output_size)
    return MSE(pred)


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

  def forward(self, x: torch.Tensor) -> Tuple[Binary, MSE]:
    latent_grid = self.decoder(x)
    bshape = latent_grid.shape[:-4]
    latent_grid = latent_grid.reshape((-1, *latent_grid.shape[-4:]))
    occupancy_grid = self.occupancy_grid_projection(latent_grid).squeeze(-1)
    centroid_grid = self.centroid_grid_projection(latent_grid).squeeze(-1)
    occupancy_grid = occupancy_grid.reshape((*bshape, *occupancy_grid.shape[-3:]))
    centroid_grid = centroid_grid.reshape((*bshape, *centroid_grid.shape[-3:]))
    occupancy_dist = Binary(occupancy_grid)
    centroid_dist = MSE(centroid_grid)
    return occupancy_dist, centroid_dist
