from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.utils._pytree as pytree


def broadcast_right(y: torch.Tensor, x: torch.Tensor):
    "Broadcast y to the same shape as x by broadcasting right."
    num_expand = len(x.shape) - len(y.shape)
    y_expanded = y.clone()
    for _ in range(num_expand):
      y_expanded = y_expanded.unsqueeze(-1)
    y_expanded = y_expanded.expand_as(x).detach()
    return y_expanded


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, keepdim=False):
    mask_expanded = broadcast_right(mask, x)
    masked_x = x * mask_expanded
    count = mask_expanded.sum(dim=dim, keepdim=keepdim)
    sum_ = masked_x.sum(dim=dim, keepdim=keepdim)
    mean = sum_ / count.clamp(min=1)
    if not keepdim and dim is not None:
        mean = torch.where(count == 0, torch.zeros_like(mean), mean)
    return mean


def discount_values(rewards, dones, values, last_values, gamma, lam):
  advantages = torch.zeros_like(rewards)
  last_advantage = torch.zeros_like(advantages[-1, :])
  for t in reversed(range(rewards.shape[0])):
    next_nonterminal = 1.0 - dones[t, :].float()
    if t == rewards.shape[0] - 1:
      next_values = last_values
    else:
      next_values = values[t + 1, :]
    delta = (
      rewards[t, :] + gamma * next_nonterminal * next_values - values[t, :]
    )
    advantages[t, :] = last_advantage = (
      delta + gamma * lam * next_nonterminal * last_advantage
    )
  return advantages


def mirror_latent(latent):
  return pytree.tree_map(lambda x: -1. * x, latent)


def split_and_pad_trajectories(tensor, dones):
    """ Splits trajectories at done indices. Then concatenates them and padds with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example: 
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]    
            
    Assumes that the input has the following dimension order: [time, number of envs, aditional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1),trajectory_lengths_list)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)

    # Pad dimension 0 up to original tensor length
    if padded_trajectories.shape[0] < tensor.shape[0]:
        padding = torch.zeros(tensor.shape[0] - padded_trajectories.shape[0], 
                            *padded_trajectories.shape[1:], 
                            device=tensor.device)
        padded_trajectories = torch.cat([padded_trajectories, padding], dim=0)

    trajectory_masks = trajectory_lengths > torch.arange(0, padded_trajectories.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


@torch.jit.ignore
def unpad_trajectories(trajectories, masks):
    """ Does the inverse operation of  split_and_pad_trajectories()."""
    # Need to transpose before and after the masking to have proper reshaping
    obs_num_dims = len(trajectories.shape) - len(masks.shape)
    traj_transp = trajectories.transpose(1, 0)
    masks_transp = masks.transpose(1, 0)
    traj_indexed = traj_transp[masks_transp]
    traj_viewed = traj_indexed.view(-1, trajectories.shape[0], *trajectories.shape[-obs_num_dims:])
    traj_viewed_transp = traj_viewed.transpose(1, 0)
    return traj_viewed_transp


def get_mlp_cnn_keys(obs, project_dims={}) -> Tuple[
   Optional[List[str]], Optional[List[str]], Optional[List[str]], int]:
  # TODO: Add option to process 2D observations with CNN or MLP. Currently only supports MLP.
  mlp_keys, mlp_2d_reshape_keys, cnn_keys = [], [], []
  num_mlp_obs = 0
  for k, v in obs.items():
    if len(v.shape) == 1:
      mlp_keys.append(k)
      curr_mlp_obs = int(np.prod(v.shape))
    elif len(v.shape) == 2:
      mlp_keys.append(k)
      mlp_2d_reshape_keys.append(k)
      curr_mlp_obs = int(np.prod(v.shape))
    elif len(v.shape) == 3:
      cnn_keys.append(k)
      curr_mlp_obs = 0
    else:
      raise ValueError(f'Observation {k} has unexpected shape: {v.shape}')

    if k in project_dims:
      num_mlp_obs += project_dims[k]
    else:
      num_mlp_obs += curr_mlp_obs

  mlp_keys = mlp_keys if mlp_keys else None
  mlp_2d_reshape_keys = mlp_2d_reshape_keys if mlp_2d_reshape_keys else None
  cnn_keys = cnn_keys if cnn_keys else None
  return mlp_keys, mlp_2d_reshape_keys, cnn_keys, num_mlp_obs


@torch.jit.ignore
def reshape_output(tensor: torch.Tensor, output_size: List[int]) -> torch.Tensor:
  return tensor.reshape((*tensor.shape[:-1], *output_size))