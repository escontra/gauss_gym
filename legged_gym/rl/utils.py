import numpy as np
import torch

def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, keepdim=False):
    num_expand = len(x.shape) - len(mask.shape)
    mask_expanded = mask.clone()
    for _ in range(num_expand):
      mask_expanded = mask_expanded.unsqueeze(-1)
    mask_expanded = mask_expanded.expand_as(x).detach()
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


def surrogate_loss(
  old_actions_log_prob, actions_log_prob, advantages, e_clip=0.2
):
  ratio = torch.exp(actions_log_prob - old_actions_log_prob)
  surrogate = -advantages * ratio
  surrogate_clipped = -advantages * torch.clamp(
    ratio, 1.0 - e_clip, 1.0 + e_clip
  )
  surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
  return surrogate_loss


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


def get_mlp_cnn_keys(obs):
  # TODO: Add option to process 2D observations with CNN or MLP. Currently only supports MLP.
  mlp_keys, mlp_2d_reshape_keys, cnn_keys = [], [], []
  num_mlp_obs = 0
  for k, v in obs.items():
    if len(v.shape) == 1:
      mlp_keys.append(k)
      num_mlp_obs += np.prod(v.shape)
    elif len(v.shape) == 2:
      mlp_keys.append(k)
      mlp_2d_reshape_keys.append(k)
      num_mlp_obs += np.prod(v.shape)
    elif len(v.shape) == 3:
      cnn_keys.append(k)
    else:
      raise ValueError(f'Observation {k} has unexpected shape: {v.shape}')

  return mlp_keys, mlp_2d_reshape_keys, cnn_keys, num_mlp_obs