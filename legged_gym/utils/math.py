import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional, Dict, Any, Tuple


@torch.jit.script
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)


@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles


@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return (upper - lower) * torch.rand(*shape, device=device) + lower

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return torch.roll(standardize_quaternion(out), -1, dims=-1)


@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)


@torch.jit.script
def _maybe_expand_tensor(value: Union[float, torch.Tensor], size: Union[None, int], device: Union[None, torch.device]):
    if isinstance(value, (int, float)):
        value = torch.tensor(value, requires_grad=False, device=device, dtype=torch.float)
    assert isinstance(value, torch.Tensor)
    if size is not None:
        value = value.expand((size,))
    return value

@torch.jit.script
def quat_from_x_rot(angle_rad: Union[float, torch.Tensor], size: Union[None, int]=None, device: Union[None, torch.device]=None):
    angle_rad = _maybe_expand_tensor(angle_rad, size, device)
    y_angle_rad = torch.zeros_like(angle_rad)
    z_angle_rad = torch.zeros_like(angle_rad)
    return quat_from_euler_xyz(angle_rad, y_angle_rad, z_angle_rad)

@torch.jit.script
def quat_from_y_rot(angle_rad: Union[float, torch.Tensor], size: Union[None, int]=None, device: Union[None, torch.device]=None):
    angle_rad = _maybe_expand_tensor(angle_rad, size, device)
    x_angle_rad = torch.zeros_like(angle_rad)
    z_angle_rad = torch.zeros_like(angle_rad)
    return quat_from_euler_xyz(x_angle_rad, angle_rad, z_angle_rad)

@torch.jit.script
def quat_from_z_rot(angle_rad: Union[float, torch.Tensor], size: Union[None, int]=None, device: Union[None, torch.device]=None):
    angle_rad = _maybe_expand_tensor(angle_rad, size, device)
    x_angle_rad = torch.zeros_like(angle_rad)
    y_angle_rad = torch.zeros_like(angle_rad)
    return quat_from_euler_xyz(x_angle_rad, y_angle_rad, angle_rad)


@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch.jit.script
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

def apply_randomization(tensor: Union[torch.Tensor, float], params: Optional[Dict[str, Any]] = None, return_noise: bool = False) -> Union[torch.Tensor, float]:
    if params is None:
        return tensor

    if params["distribution"] == "gaussian":
        mu, var = params["range"]
        noise = torch.randn_like(tensor) if isinstance(tensor, torch.Tensor) else np.random.randn()
        noise_val = mu + var * noise
    elif params["distribution"] == "uniform":
        lower, upper = params["range"]
        noise = torch.rand_like(tensor) if isinstance(tensor, torch.Tensor) else np.random.rand()
        noise_val = lower + (upper - lower) * noise
    else:
        raise ValueError(f"Invalid randomization distribution: {params['distribution']}")

    if params["operation"] == "additive":
        result = tensor + noise_val
    elif params["operation"] == "scaling":
        result = tensor * noise_val
    else:
        raise ValueError(f"Invalid randomization operation: {params['operation']}")

    if return_noise:
        return result, noise
    else:
        return result

def symlog(x):
  return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
  return torch.sign(x) * torch.expm1(torch.abs(x))


@torch.jit.script
def logstd_to_std(logstd: torch.Tensor, minstd: float, maxstd: float):
  minstd = torch.tensor(minstd, device=logstd.device, dtype=torch.float32)
  maxstd = torch.tensor(maxstd, device=logstd.device, dtype=torch.float32)
  return torch.exp(torch.sigmoid(logstd) * (torch.log(maxstd) - torch.log(minstd)) + torch.log(minstd))


@torch.jit.script
def std_to_logstd(std: torch.Tensor, minstd: float, maxstd: float):
  minstd = torch.tensor(minstd, device=std.device, dtype=torch.float32)
  maxstd = torch.tensor(maxstd, device=std.device, dtype=torch.float32)
  return torch.logit((torch.log(std) - torch.log(minstd)) / (torch.log(maxstd) - torch.log(minstd)))


@torch.jit.script
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Input quaternions are assumed to be in (x, y, z, w) format.

    Args:
        quaternions: Quaternions as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    # Ensure unit quaternions for the conversion formula to be valid.
    q_norm = normalize(quaternions)
    
    qx = q_norm[..., 0]
    qy = q_norm[..., 1]
    qz = q_norm[..., 2]
    qw = q_norm[..., 3]

    # Precompute squares of quaternion components
    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz

    # Precompute products of quaternion components
    qxqy = qx * qy
    qxqz = qx * qz
    qxqw = qx * qw
    qyqz = qy * qz
    qyqw = qy * qw
    qzqw = qz * qw
    
    # Rotation matrix elements from (x, y, z, w) quaternion
    # R = [
    #     [1 - 2(qy^2 + qz^2),   2(qx*qy - qw*qz),       2(qx*qz + qw*qy)],
    #     [2(qx*qy + qw*qz),   1 - 2(qx^2 + qz^2),   2(qy*qz - qw*qx)],
    #     [2(qx*qz - qw*qy),   2(qy*qz + qw*qx),       1 - 2(qx^2 + qy^2)]
    # ]
    
    r00 = 1.0 - 2.0 * (qy2 + qz2)
    r01 = 2.0 * (qxqy - qzqw)
    r02 = 2.0 * (qxqz + qyqw)

    r10 = 2.0 * (qxqy + qzqw)
    r11 = 1.0 - 2.0 * (qx2 + qz2)
    r12 = 2.0 * (qyqz - qxqw)

    r20 = 2.0 * (qxqz - qyqw)
    r21 = 2.0 * (qyqz + qxqw)
    r22 = 1.0 - 2.0 * (qx2 + qy2)
    
    # Stack the elements into a matrix
    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    
    matrix = torch.stack([row0, row1, row2], dim=-2)

    return matrix


def _sigmoids(x: torch.Tensor, value_at_1: torch.Tensor, sigmoid: str):
  """Returns 1 when `x` == 0, between 0 and 1 otherwise.

  Args:
    x: Tensor of shape (...,).
    value_at_1: Tensor between 0 and 1 specifying the output when `x` == 1.
    sigmoid: String, choice of sigmoid type.

  Returns:
    Tensor of shape (...,).

  Raises:
    ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
      `quadratic` sigmoids which allow `value_at_1` == 0.
    ValueError: If `sigmoid` is of an unknown type.
  """
  if sigmoid in ('cosine', 'linear', 'quadratic'):
    val_in_range = (0 <= value_at_1) & (value_at_1 < 1)
    if not val_in_range.all():
      raise ValueError('`value_at_1` must be nonnegative and smaller than 1, '
                       'got {}.'.format(value_at_1))
  else:
    val_in_range = (0 < value_at_1) & (value_at_1 < 1)
    if not val_in_range.all():
      raise ValueError('`value_at_1` must be strictly between 0 and 1, '
                       'got {}.'.format(value_at_1))

  if sigmoid == 'gaussian':
    scale = torch.sqrt(-2 * torch.log(value_at_1))
    return torch.exp(-0.5 * (x*scale)**2)

  elif sigmoid == 'hyperbolic':
    scale = torch.arccosh(1/value_at_1)
    return 1 / torch.cosh(x*scale)

  elif sigmoid == 'long_tail':
    scale = torch.sqrt(1/value_at_1 - 1)
    return 1 / ((x*scale)**2 + 1)

  elif sigmoid == 'reciprocal':
    scale = 1 / value_at_1 - 1
    return 1 / (abs(x)*scale + 1)

  elif sigmoid == 'cosine':
    scale = torch.arccos(2*value_at_1 - 1) / torch.pi
    scaled_x = x*scale
    cos_pi_scaled_x = torch.cos(torch.pi*scaled_x)
    return torch.where(abs(scaled_x) < 1, (1 + cos_pi_scaled_x)/2, 0.0)

  elif sigmoid == 'linear':
    scale = 1-value_at_1
    scaled_x = x*scale
    return torch.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

  elif sigmoid == 'quadratic':
    scale = torch.sqrt(1-value_at_1)
    scaled_x = x*scale
    return torch.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

  elif sigmoid == 'tanh_squared':
    scale = torch.arctanh(torch.sqrt(1-value_at_1))
    return 1 - torch.tanh(x*scale)**2

  else:
    raise ValueError('Unknown sigmoid type {!r}.'.format(sigmoid))

def linear_or_qaussian_tolerance(val, target, linear_margin, gaussian_margin, gaussian: bool):
    linear_value = tolerance(
        val,
        lower=torch.where(target > 0, target, target - linear_margin),
        upper=torch.where(target > 0, target + linear_margin, target),
        margin=torch.abs(target),
        sigmoid='linear',
        value_at_margin=torch.zeros_like(target)
    )
    gaussian_value = tolerance(
        val,
        margin=torch.full_like(target, gaussian_margin),
        sigmoid='gaussian',
        value_at_margin=torch.full_like(target, 0.01)
    )
    return linear_value * ~gaussian + gaussian_value * gaussian

def tolerance(
        x: torch.Tensor,
        lower: Optional[torch.Tensor] = None,
        upper: Optional[torch.Tensor] = None,
        margin: Optional[torch.Tensor] = None,
        sigmoid: str = 'gaussian',
        value_at_margin: Optional[torch.Tensor] = None):
  """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.

  Args:
    x: A scalar or numpy array.
    lower: Lower bound for the target interval. Can be infinite if the interval
      is unbounded on the lower end. Can be equal to `upper` if the target value
      is exact.
    upper: Upper bound for the target interval. Can be infinite if the interval
      is unbounded on the upper end. Can be equal to `lower` if the target value
      is exact.
    margin: Controls how steeply the output decreases as
      `x` moves out-of-bounds.
      * If `margin == 0` then the output will be 0 for all values of `x`
        outside of `bounds`.
      * If `margin > 0` then the output will decrease sigmoidally with
        increasing distance from the nearest bound.
    sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
       'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
    value_at_margin: A float between 0 and 1 specifying the output value when
      the distance from `x` to the nearest bound is equal to `margin`. Ignored
      if `margin == 0`.

  Returns:
    A float or numpy array with values between 0.0 and 1.0.

  Raises:
    ValueError: If `bounds[0] > bounds[1]`.
    ValueError: If `margin` is negative.
  """

  if lower is None:
    lower = torch.zeros_like(x)
  if upper is None:
    upper = torch.zeros_like(x)
  if margin is None:
    margin = torch.zeros_like(x)
  if value_at_margin is None:
    value_at_margin = torch.full_like(x, 0.1)

  if (lower > upper).any():
    raise ValueError('Lower bound must be <= upper bound.')
  if (margin < 0).any():
    raise ValueError(f'`margin` must be non-negative. Got {margin}.')

  in_bounds = (lower <= x) & (x <= upper)

  margin_zero_value = torch.where(in_bounds, torch.ones_like(x), torch.zeros_like(x))

  d = torch.where(x < lower, lower - x, x - upper) / margin
  margin_nonzero_value = torch.where(
     in_bounds,
     torch.ones_like(x),
     _sigmoids(d, value_at_margin, sigmoid))

  value = torch.where(margin == 0, margin_zero_value, margin_nonzero_value)
  return value


  

  # in_bounds = np.logical_and(lower <= x, x <= upper)
  # if margin == 0:
  #   value = np.where(in_bounds, 1.0, 0.0)
  # else:
  #   d = np.where(x < lower, lower - x, x - upper) / margin
  #   value = np.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

  # return float(value) if np.isscalar(x) else value