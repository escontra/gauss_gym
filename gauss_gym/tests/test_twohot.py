import numpy as np
import jax
import jax.numpy as jnp
import torch
from gauss_gym.utils import math_utils


i32 = jnp.int32
f32 = jnp.float32
sg = jax.lax.stop_gradient


def transform_to_logits(target, bins):
  assert target.dtype == f32, target.dtype
  target = sg(target)
  below = (bins <= target[..., None]).astype(i32).sum(-1) - 1
  above = len(bins) - (bins > target[..., None]).astype(i32).sum(-1)
  below = jnp.clip(below, 0, len(bins) - 1)
  above = jnp.clip(above, 0, len(bins) - 1)
  equal = below == above
  dist_to_below = jnp.where(equal, 1, jnp.abs(bins[below] - target))
  dist_to_above = jnp.where(equal, 1, jnp.abs(bins[above] - target))
  total = dist_to_below + dist_to_above
  weight_below = dist_to_above / total
  weight_above = dist_to_below / total
  target = (
    jax.nn.one_hot(below, len(bins)) * weight_below[..., None]
    + jax.nn.one_hot(above, len(bins)) * weight_above[..., None]
  )
  return target


def transform_to_probs_jax(target, bins, debug=False):
  if debug:
    print('jaxfn')
  assert target.dtype == f32, target.dtype
  target = sg(target)
  below = (bins <= target[..., None]).astype(i32).sum(-1) - 1
  above = len(bins) - (bins > target[..., None]).astype(i32).sum(-1)
  below = jnp.clip(below, 0, len(bins) - 1)
  above = jnp.clip(above, 0, len(bins) - 1)
  equal = below == above
  dist_to_below = jnp.where(equal, 1, jnp.abs(bins[below] - target))
  dist_to_above = jnp.where(equal, 1, jnp.abs(bins[above] - target))
  total = dist_to_below + dist_to_above
  weight_below = dist_to_above / total
  weight_above = dist_to_below / total
  if debug:
    print('target', target.dtype, target.shape)
    print('bins', bins.dtype, bins.shape)
    print('below', below.dtype, np.array(below))
    print('above', above.dtype, np.array(above))
    print('equal', equal.dtype, np.array(equal))
    print('dist_to_below', dist_to_below.dtype, np.array(dist_to_below))
    print('dist_to_above', dist_to_above.dtype, np.array(dist_to_above))
    print('total', total.dtype, np.array(total))
    print('weight_below', weight_below.dtype, np.array(weight_below))
    print('weight_above', weight_above.dtype, np.array(weight_above))
  target = (
    jax.nn.one_hot(below, len(bins)) * weight_below[..., None]
    + jax.nn.one_hot(above, len(bins)) * weight_above[..., None]
  ).sum(-2)
  return target.astype(f32)


def transform_to_probs_pt(target, bins, debug=False):
  if debug:
    print('ptfn')
  assert target.dtype == torch.float32, target.dtype
  target = target.detach()
  below = (bins <= target[..., None]).to(torch.int64).sum(-1) - 1
  above = len(bins) - (bins > target[..., None]).to(torch.int64).sum(-1)
  below = torch.clip(below, 0, len(bins) - 1)
  above = torch.clip(above, 0, len(bins) - 1)
  equal = below == above
  dist_to_below = torch.where(equal, 1, torch.abs(bins[below] - target))
  dist_to_above = torch.where(equal, 1, torch.abs(bins[above] - target))
  total = dist_to_below + dist_to_above
  weight_below = dist_to_above / total
  weight_above = dist_to_below / total
  if debug:
    print('target', target.dtype, target.shape)
    print('bins', bins.dtype, bins.shape)
    print('below', below.dtype, below.cpu().numpy())
    print('above', above.dtype, above.cpu().numpy())
    print('equal', equal.dtype, equal.cpu().numpy())
    print('dist_to_below', dist_to_below.dtype, dist_to_below.cpu().numpy())
    print('dist_to_above', dist_to_above.dtype, dist_to_above.cpu().numpy())
    print('total', total.dtype, total.cpu().numpy())
    print('weight_below', weight_below.dtype, weight_below.cpu().numpy())
    print('weight_above', weight_above.dtype, weight_above.cpu().numpy())
  target = (
    torch.nn.functional.one_hot(below, len(bins)) * weight_below[..., None]
    + torch.nn.functional.one_hot(above, len(bins)) * weight_above[..., None]
  ).sum(-2)
  return target.to(torch.float32)


def get_bins_torch(num_bins):
  if num_bins % 2 == 1:
    half = torch.linspace(
      -20, 0, (num_bins - 1) // 2 + 1, dtype=torch.float32, device='cpu'
    )
    half = math_utils.symexp(half)
    bins = torch.cat([half, -half[:-1].flip(0)], dim=0)
  else:
    half = torch.linspace(-20, 0, num_bins // 2, dtype=torch.float32, device='cpu')
    half = math_utils.symexp(half)
    bins = torch.cat([half, -half.flip(0)], dim=0)
  return bins


def symexp(x):
  return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def get_bins_jax(num_bins):
  if num_bins % 2 == 1:
    half = jnp.linspace(-20, 0, (num_bins - 1) // 2 + 1, dtype=f32)
    half = symexp(half)
    bins = jnp.concatenate([half, -half[:-1][::-1]], 0)
  else:
    half = jnp.linspace(-20, 0, num_bins // 2, dtype=f32)
    half = symexp(half)
    bins = jnp.concatenate([half, -half[::-1]], 0)
  return bins


def transform_from_probs_jax(probs, bins):
  n = probs.shape[-1]
  if n % 2 == 1:
    m = (n - 1) // 2
    p1 = probs[..., :m]
    p2 = probs[..., m : m + 1]
    p3 = probs[..., m + 1 :]
    b1 = bins[..., :m]
    b2 = bins[..., m : m + 1]
    b3 = bins[..., m + 1 :]
    wavg = (p2 * b2).sum(-1) + ((p1 * b1)[..., ::-1] + (p3 * b3)).sum(-1)
    return wavg
  else:
    p1 = probs[..., : n // 2]
    p2 = probs[..., n // 2 :]
    b1 = bins[..., : n // 2]
    b2 = bins[..., n // 2 :]
    wavg = ((p1 * b1)[..., ::-1] + (p2 * b2)).sum(-1)
    return wavg


def transform_from_probs_pt(probs, bins):
  n = probs.shape[-1]
  if n % 2 == 1:
    m = (n - 1) // 2
    p1 = probs[..., :m]
    p2 = probs[..., m : m + 1]
    p3 = probs[..., m + 1 :]
    b1 = bins[..., :m]
    b2 = bins[..., m : m + 1]
    b3 = bins[..., m + 1 :]
    wavg = (p2 * b2).sum(-1) + ((p1 * b1).flip(dims=[-1]) + (p3 * b3)).sum(-1)
    return wavg
  else:
    p1 = probs[..., : n // 2]
    p2 = probs[..., n // 2 :]
    b1 = bins[..., : n // 2]
    b2 = bins[..., n // 2 :]
    wavg = ((p1 * b1).flip(dims=[-1]) + (p2 * b2)).sum(-1)
    return wavg


if __name__ == '__main__':
  num_bins = 14
  bins_pt = get_bins_torch(num_bins).to('cpu')
  bins_jax = get_bins_jax(num_bins)
  print(np.allclose(bins_pt.cpu().numpy(), np.array(bins_jax), rtol=1e-5, atol=1e-8))
  print(bins_pt.shape, bins_jax.shape)
  # Finding: getting bins is the same for both torch and jax.

  # sc_np = np.linspace(bins_pt.numpy()[0], bins_pt.numpy()[-1], 100, dtype=np.float32)[..., None]
  values = [-10.0, 10.0, 12.0, 100.0]
  sc_np = np.array(values, dtype=np.float32)[..., None]
  sc_pt = torch.tensor(values, dtype=torch.float32)[..., None].to('cpu')
  target_jax = transform_to_probs_jax(sc_np, bins_jax, debug=False)
  target_pt = transform_to_probs_pt(sc_pt, bins_pt, debug=False)
  print(np.allclose(target_jax, target_pt.cpu().numpy()))
  print(target_jax.shape, target_pt.shape)
  # Finding: getting target is the same for both torch and jax.

  print('target sums')
  print(target_jax.sum(-1))
  print(target_pt.sum(-1).cpu().numpy())

  print('softmax transform')
  print(target_pt)
  # target_pt_softmax = torch.log_softmax(target_pt, dim=-1)
  # target_pt_softmax = target_pt + torch.logsumexp(target_pt, dim=-1, keepdim=True)
  target_pt_softmax = torch.where(target_pt <= 0, -1.5, target_pt)
  target_pt_softmax = torch.nn.functional.softmax(target_pt_softmax, dim=-1)
  # target
  print(target_pt_softmax)

  # twohot_jax = TwoHot(target_jax, bins_jax)
  pred_jax = transform_from_probs_jax(target_jax, bins_jax)
  pred_pt = transform_from_probs_pt(target_pt, bins_pt)
  pred_pt_softmax = transform_from_probs_pt(target_pt_softmax, bins_pt)
  print(np.allclose(pred_jax, pred_pt.cpu().numpy()))
  print(pred_jax.shape, pred_pt.shape)

  print(pred_jax)
  print(pred_pt.cpu().numpy())
  print(pred_pt_softmax.cpu().numpy())
  # Finding: getting twohot is the same for both torch and jax.
