import os
import random
import itertools
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.utils._pytree as pytree
import torch.distributed as torch_distributed
from legged_gym.utils import config

from legged_gym import utils


def tree_cat(x, y, dim: int):
  return pytree.tree_map(lambda x, y: torch.cat([x, y], dim=dim), x, y)


@torch.jit.unused
def init_linear(
    layer: torch.nn.Linear,
    dist: str = 'trunc_normal',
    fan: str = 'in',
    scale: float = 1.0):
  fanin, fanout = torch.nn.init._calculate_fan_in_and_fan_out(layer.weight)
  fan_value = {
    'avg': (fanin + fanout) / 2, 'in': fanin, 'out': fanout, 'none': 1,
  }[fan]
  torch.nn.init.zeros_(layer.bias)
  if dist == 'trunc_normal':
    torch.nn.init.trunc_normal_(layer.weight, a=-2., b=2.)
    layer.weight.data *= 1.1368 * np.sqrt(1 / fan_value)
  elif dist == 'uniform':
    limit = np.sqrt(1 / fan)
    torch.nn.init.uniform_(layer.weight, a=-limit, b=limit)
  elif dist == 'normal':
    torch.nn.init.normal_(layer.weight, mean=0.0, std=1.0)
    layer.weight.data *= np.sqrt(1 / fan_value)
  elif dist == 'zeros':
    torch.nn.init.zeros_(layer.weight)
  elif dist == 'normed':
    torch.nn.init.uniform_(layer.weight, a=-1.0, b=1.0)
    layer.weight.data *= (1 / torch.linalg.norm(layer.weight.data.reshape((-1, layer.weight.data.shape[-1])), 2, 0))
  else:
    raise NotImplementedError(dist)

  layer.weight.data *= scale


def sync_grads_multi_gpu(param_list, multi_gpu_world_size: int):
  """
  param_list is a list of parameters from different networks.
  """
  # from RL-Games
  # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
  all_grads_list = []
  all_params_chain = itertools.chain(*param_list)

  all_params_list = list(all_params_chain)
  for param in all_params_list:
    if param.grad is not None:
      all_grads_list.append(param.grad.view(-1))
  all_grads = torch.cat(all_grads_list)
  # sum grads on each gpu
  torch_distributed.all_reduce(all_grads, op=torch_distributed.ReduceOp.SUM)
  offset = 0
  for param in all_params_list:
    if param.grad is not None:
      # copy data back from shared buffer
      param.grad.data.copy_(
        all_grads[offset : offset + param.numel()].view_as(param.grad.data) / multi_gpu_world_size
      )
      offset += param.numel()


def broadcast_scalar(scalar, src_rank: int, device):
  scalar_tensor = torch.tensor([scalar], device=device)
  torch_distributed.broadcast(scalar_tensor, src_rank)
  scalar = scalar_tensor.item()
  return scalar

def sync_state_dict(module, src_rank: int):
  """
  Syncs the state dict of a module to all GPUs.
  """
  state_dict = module.state_dict()
  model_params = [state_dict]
  torch_distributed.broadcast_object_list(model_params, src_rank)
  module.load_state_dict(model_params[0])


class SetpointScheduler:
  def __init__(self, warmup_steps: int, val_warmup, val_after):
    self.val_warmup = val_warmup
    self.val_after = val_after
    self.warmup_steps = warmup_steps

  def __call__(self, step: int):
    if step < self.warmup_steps:
      return self.val_warmup
    return self.val_after

def set_seed(seed):
  if seed == -1:
    seed = np.random.randint(0, 10000)
  utils.print("Setting RL seed: {}".format(seed))

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


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

@torch.jit.ignore
def flatten_obs(tensor: torch.Tensor, obs_size: List[int]) -> torch.Tensor:
  return tensor.reshape(*tensor.shape[:-len(obs_size)], -1)

@torch.jit.ignore
def unflatten_obs(tensor: torch.Tensor, obs_size: List[int]) -> torch.Tensor:
  return tensor.reshape(*tensor.shape[:-1], *obs_size)

@torch.jit.ignore
def flatten_batch(tensor: torch.Tensor, obs_size: List[int]) -> torch.Tensor:
  return tensor.reshape(-1, *obs_size)

@torch.jit.ignore
def unflatten_batch(tensor: torch.Tensor, batch_size: List[int]) -> torch.Tensor:
  return tensor.reshape(*batch_size, *tensor.shape[1:])


# currently hardcoded
WANDB_PROJECTS=("a1", "t1")


def get_wandb_path(load_run: str, wandb_project: str, multi_gpu: bool = False, multi_gpu_rank: int = 0) -> str:
    """
    Get the path to a wandb checkpoint.
    Saves in a temp folder in /tmp
    
    Args:
        load_run: Either a wandb run ID or a name prefixed with 'wandb_'
    
    Returns:
        Path to the downloaded checkpoint file
    """
    import wandb
    api = wandb.Api()
    run_id = None
    project = None
    
    # Handle run ID
    # if all(c.isalnum() or c == '-' for c in load_run):
        # run_id = load_run
    if load_run.startswith('wandb_id_'):
        run_id = load_run[9:]
    # Handle run name
    elif load_run.startswith('wandb_'):
        run_name = load_run[6:]  # Remove 'wandb_' prefix

        for project in WANDB_PROJECTS:
            runs = list(api.runs(project, filters={"display_name": run_name}))
            if runs:
                project = project
                break
        if not runs:
            raise ValueError(f"No wandb run found with name: {run_name}")
        if len(runs) > 1:
            print(f"Warning: Multiple runs found with name {run_name}, using the most recent one")
        run_id = runs[0].id
    else:
        return None
    
    print(f"Downloading checkpoint from wandb run ID: {run_id}")
    
    # Get the run and its checkpoints
    run = api.run(f"{project}/{run_id}")
    files = run.files()

    checkpoint_files = [f for f in files if 'model_' in f.name and f.name.endswith('.pth')]
    
    if not checkpoint_files:
        raise ValueError(f"No checkpoints found for wandb run {run_id}")
    
    # Get the latest checkpoint
    # If we upload numerical checkpoints, we can use the following:
    # latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.name.split('model_')[1].split('.')[0]))
    # print(f"Latest checkpoint: {latest_checkpoint.name}")
    # Currently, we assume that we only upload one checkpoint per run (and continually overwrite it)
    assert len(checkpoint_files) == 1, f"Expected 1 checkpoint file, got {len(checkpoint_files)}"
    latest_checkpoint = checkpoint_files[0]
    
    # Create target directory and download
    if multi_gpu:
        target_dir = os.path.join('/tmp/wandb_checkpoints/', run.name, f'rank_{multi_gpu_rank}')
    else:
        target_dir = os.path.join('/tmp/wandb_checkpoints/', run.name)
    os.makedirs(target_dir, exist_ok=True)
    
    target_path = os.path.join(target_dir, latest_checkpoint.name)
    print(f"Downloading checkpoint to {target_path}")
    latest_checkpoint.download(root=target_dir, replace=True)
    print("Download complete!")
    
    return target_path

def get_wandb_config(load_run: str, multi_gpu: bool = False, multi_gpu_rank: int = 0, config_name: str = "train_config.yaml") -> str:
    """
    Get the path to a wandb config YAML file.
    Saves in a temp folder in /tmp
    
    Args:
        load_run: Either a wandb run ID or a name prefixed with 'wandb_'
    
    Returns:
        Path to the downloaded config YAML file
    """
    import wandb
    import yaml
    api = wandb.Api()
    run_id = None
    project = None
    
    # Handle run ID
    if load_run.startswith('wandb_id_'):
        run_id = load_run[9:]
    # Handle run name
    elif load_run.startswith('wandb_'):
        run_name = load_run[6:]  # Remove 'wandb_' prefix
        for project in WANDB_PROJECTS:
            runs = list(api.runs(project, filters={"display_name": run_name}))
            if runs:
                project = project
                break
        if not runs:
            raise ValueError(f"No wandb run found with name: {run_name}")
        if len(runs) > 1:
            print(f"Warning: Multiple runs found with name {run_name}, using the most recent one")
        run_id = runs[0].id
    else:
        return None
    
    print(f"Downloading config from wandb run ID: {run_id}")
    
    # Get the run
    run = api.run(f"{project}/{run_id}")
    files = run.files()

    files = [f for f in files if config_name in f.name]

    assert len(files) == 1, f"Expected 1 config file, got {len(files)}"

    config_file = files[0]


    if multi_gpu:
        target_dir = os.path.join('/tmp/wandb_checkpoints/', run.name, f'rank_{multi_gpu_rank}')
    else:
        target_dir = os.path.join('/tmp/wandb_checkpoints/', run.name)
    os.makedirs(target_dir, exist_ok=True)
    
    target_path = os.path.join(target_dir, config_file.name)
    print(f"Downloading checkpoint to {target_path}")
    config_file.download(root=target_dir, replace=True)
    print("Download complete!")

    cfg = config.Config.load(target_path)

    return cfg