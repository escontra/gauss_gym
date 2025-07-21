from typing import Dict, Any, Tuple, Callable, Optional
import torch
import torch.utils._pytree as pytree
import torch.distributed as torch_distributed

from legged_gym.rl import experience_buffer, utils
from legged_gym.utils import voxel, agg, timer


_SYMMETRY_FN = Callable[[str, str], Callable[[torch.Tensor], torch.Tensor]]


def value_loss(
    batch: experience_buffer.MiniBatch,
    value_network: torch.nn.Module,
    value_obs_key: str,
    symmetry_augmentation: bool,
    gamma: float,
    lam: float,
    use_clipped_value_loss: bool,
    clip_param: float) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:

  if symmetry_augmentation:
    batch_value_obs = utils.tree_cat(
      batch.obs[value_obs_key],
      batch.obs_sym[value_obs_key],
      dim=1)
    batch_hidden_states = utils.tree_cat(
      batch.hidden_states[value_obs_key],
      batch.hidden_states_sym[value_obs_key],
      dim=1
    )
    batch_masks = batch.rl_values["masks"].repeat(1, 2)
    batch_time_outs = batch.rl_values["time_outs"].repeat(1, 2)
    batch_dones = batch.rl_values["dones"].repeat(1, 2)
    batch_last_values = batch.rl_values["last_value"].squeeze(-1).repeat(1, 2)
    batch_rewards = batch.rl_values["rewards"].repeat(1, 2)
    num_augs = 2
  else:
    batch_value_obs = batch.obs[value_obs_key]
    batch_hidden_states = batch.hidden_states[value_obs_key]
    batch_masks = batch.rl_values["masks"]
    batch_time_outs = batch.rl_values["time_outs"]
    batch_dones = batch.rl_values["dones"]
    batch_last_values = batch.rl_values["last_value"].squeeze(-1)
    batch_rewards = batch.rl_values["rewards"]
    num_augs = 1

  values, _, _ = value_network(
    batch_value_obs,
    masks=batch_masks,
    hidden_states=batch_hidden_states
  )
  # Compute returns and advantages.
  with torch.no_grad():
    rewards = batch_rewards.clone()
    value_preds = values['value'].pred().squeeze(-1)
    rewards[batch_time_outs] = value_preds[batch_time_outs]
    advantages = utils.discount_values(
      rewards,
      batch_dones | batch_time_outs,
      value_preds,
      batch_last_values,
      gamma,
      lam,
    )
    returns = value_preds + advantages

  # Value loss.
  if use_clipped_value_loss:
    value_clipped = batch.rl_values["old_values"].detach() + (values.pred() - batch.rl_values["old_values"].detach()).clamp(
        -clip_param, clip_param
    )
    value_losses = (values.pred() - returns.unsqueeze(-1)).pow(2)
    value_losses_clipped = (value_clipped - returns.unsqueeze(-1)).pow(2)
    value_loss = torch.max(value_losses, value_losses_clipped).mean()
  else:
    value_loss = torch.mean(values['value'].loss(returns.unsqueeze(-1)))
  metrics = {'value_loss': value_loss.item()}
  unpadded_batch_size = advantages.shape[1]
  unpadded_batch_size_orig = unpadded_batch_size // num_augs
  return value_loss, advantages[:, :unpadded_batch_size_orig], metrics


def surrogate_loss(
  old_actions_log_prob: torch.Tensor,
  actions_log_prob: torch.Tensor,
  advantages: torch.Tensor,
  e_clip: float = 0.2
) -> torch.Tensor:
  ratio = torch.exp(actions_log_prob - old_actions_log_prob)
  surrogate = -advantages * ratio
  surrogate_clipped = -advantages * torch.clamp(
    ratio, 1.0 - e_clip, 1.0 + e_clip
  )
  surrogate_loss = torch.max(surrogate, surrogate_clipped)
  return surrogate_loss


def actor_loss(
    advantages: torch.Tensor,
    batch: experience_buffer.MiniBatch,
    policy_network: torch.nn.Module,
    policy_obs_key: str,
    symmetry_loss: bool,
    symmetry_augmentation: bool,
    symmetry_fn: _SYMMETRY_FN,
    actor_loss_coefs: Dict[str, float],
    entropy_coefs: Dict[str, float],
    bound_coefs: Dict[str, float],
    symmetry_coefs: Dict[str, float],
    clip_param: float,
    multi_gpu: bool = False,
    multi_gpu_world_size: int = 1,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:

  if symmetry_augmentation:
    num_augs = 2
    batch_policy_obs = utils.tree_cat(
      batch.obs[policy_obs_key],
      batch.obs_sym[policy_obs_key],
      dim=1)
    batch_hidden_states = utils.tree_cat(
      batch.hidden_states[policy_obs_key],
      batch.hidden_states_sym[policy_obs_key],
      dim=1)
    batch_actions = utils.tree_cat(
      batch.rl_values["actions"],
      {k: symmetry_fn("actions", k)(v)
       for k, v in batch.rl_values["actions"].items()},
      dim=1)
    batch_masks = batch.rl_values["masks"].repeat(1, 2)
    batch_advantages = advantages.repeat(1, 2)
  else:
    num_augs = 1
    batch_policy_obs = batch.obs[policy_obs_key]
    batch_hidden_states = batch.hidden_states[policy_obs_key]
    batch_actions = batch.rl_values["actions"]
    batch_masks = batch.rl_values["masks"]
    batch_advantages = advantages

  # Actor loss.
  dists, _, _ = policy_network(
    batch_policy_obs,
    masks=batch_masks,
    hidden_states=batch_hidden_states
  )
  if symmetry_loss and not symmetry_augmentation:
    dists_sym, _, _ = policy_network(
      batch.obs_sym[policy_obs_key],
      masks=batch.rl_values["masks"],
      hidden_states=batch.hidden_states_sym[policy_obs_key]
    )

  kl_means = {}
  metrics = {}
  losses = []
  for name, dist in dists.items():
    actor_loss_coef = actor_loss_coefs[name]
    actions_log_prob = dist.logp(batch_actions[name].detach()).sum(dim=-1)
    unpadded_batch_size_orig = actions_log_prob.shape[1] // num_augs
    with torch.no_grad():
      # Compute old actions log prob for original batch.
      old_dist = batch.network_batch[policy_obs_key][name].repeat(num_augs, dim=1)
      old_actions_log_prob_batch = old_dist.logp(batch_actions[name].detach()).sum(dim=-1)
      old_actions_log_prob_batch = old_actions_log_prob_batch[:, :unpadded_batch_size_orig]
      old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(1, num_augs)

    actor_loss = surrogate_loss(
      old_actions_log_prob_batch, actions_log_prob, batch_advantages,
      e_clip=clip_param
    ).mean()
    losses.append(actor_loss_coef * actor_loss)
    metrics[f'actor_loss_{name}'] = actor_loss.item()

    # Compute KL for original batch.
    kl = torch.sum(dist.kl(old_dist), axis=-1)
    kl = kl[:, :unpadded_batch_size_orig] # TODO: this matters.
    klm = torch.mean(kl)
    # TODO -- kinda ugly to have mgpu sync here :(
    if multi_gpu:
      torch_distributed.all_reduce(klm, op=torch_distributed.ReduceOp.SUM)
      klm = klm / multi_gpu_world_size
    klm = klm.item()
    kl_means[name] = klm
    metrics[f'kl_mean_{name}'] = klm

    # Entropy loss. We only compute entropy for the original batch.
    entropy = dist.entropy().sum(dim=-1)
    entropy = entropy[:, :unpadded_batch_size_orig]
    metrics[f'entropy_{name}'] = entropy.mean().item()
    if name in entropy_coefs:
      losses.append(actor_loss_coef * entropy_coefs[name] * entropy.mean())
      metrics[f'entropy_loss_{name}'] = entropy.mean().item()

    # Bound loss.
    if name in bound_coefs:
      bound_loss = (
        torch.clip(dist.pred()[:, :unpadded_batch_size_orig] - 1.0, min=0.0).square().mean()
        + torch.clip(dist.pred()[:, :unpadded_batch_size_orig] + 1.0, max=0.0).square().mean()
      )
      losses.append(actor_loss_coef * bound_coefs[name] * bound_loss)
      metrics[f'bound_loss_{name}'] = bound_loss.item()

    if symmetry_loss and name in symmetry_coefs:
      # Symmetry loss.
      if symmetry_augmentation:
        dist_act_sym = symmetry_fn("actions", name)(dist.pred()[:, :unpadded_batch_size_orig])
        dist_sym_act = dist.pred()[:, unpadded_batch_size_orig:]
      else:
        dist_act_sym = symmetry_fn("actions", name)(dist.pred())
        dist_sym_act = dists_sym[name].pred()
      symmetry_loss = torch.nn.MSELoss()(dist_sym_act, dist_act_sym.detach())
      metrics[f'symmetry_loss_{name}'] = symmetry_loss.item()
      losses.append(actor_loss_coef * symmetry_coefs[name] * symmetry_loss.mean())

  total_loss = sum(losses)
  return total_loss, kl_means, metrics


def reconstruction_loss(
  batch: experience_buffer.MiniBatch,
  image_encoder_network: torch.nn.Module,
  image_encoder_obs_key: str,
  max_batch_size: int,
  symmetry_augmentation: bool,
  symmetry_fn: _SYMMETRY_FN,
):
  orig_batch_size = batch.rl_values["masks"].shape[1]
  if max_batch_size < orig_batch_size:
    sample_idxs = torch.randperm(orig_batch_size)[:max_batch_size]
  else:
    sample_idxs = torch.arange(orig_batch_size)
  batch_sampled = pytree.tree_map(lambda x: x[:, sample_idxs], batch.obs[image_encoder_obs_key])
  masks_sampled = batch.rl_values["masks"][:, sample_idxs]
  batch_sampled_hidden_states = pytree.tree_map(
    lambda x: x[:, sample_idxs],
    batch.hidden_states[image_encoder_obs_key])
  image_encoder_dists, _, _ = image_encoder_network(
    batch_sampled,
    masks=masks_sampled,
    hidden_states=batch_sampled_hidden_states,
    unpad=False
  )

  losses = []
  metrics = {}
  for name, dist in image_encoder_dists.items():
    obs_group, obs_name = name.split('/')
    recon_obs = pytree.tree_map(lambda x: x[:, sample_idxs], batch.obs[obs_group][obs_name])
    if 'ray_cast' in obs_name.lower():
      num_height_levels = dist.pred()[0].shape[-1]
      # Compute ground truth occupancy grid and centroid grid. Get mask for
      # unsaturated voxels.
      recon_occupancy_grid, recon_centroid_grid = voxel.heightmap_to_voxels(recon_obs, num_height_levels)
      unsaturated_mask = voxel.unsaturated_voxels_mask(recon_occupancy_grid)

      # Expand trajectory mask and combine with unsaturated mask.
      masks_sampled_expanded = utils.broadcast_right(masks_sampled, unsaturated_mask)
      masks_sampled_unsaturated = masks_sampled_expanded & unsaturated_mask

      # Compute masked occupancy loss.
      recon_loss_occupancy = dist.occupancy_grid_loss(recon_occupancy_grid.detach())
      recon_loss_occupancy = utils.masked_mean(recon_loss_occupancy, masks_sampled_unsaturated)
      metrics[f'image_encoder_recon_{obs_group}_{obs_name}_occupancy_loss'] = recon_loss_occupancy.item()

      # Compute masked centroid loss. Additionally mask out loss for
      # unoccupied voxels.
      recon_loss_centroid = dist.centroid_grid_loss(recon_centroid_grid.detach())
      # With ground truth occupancy grid mask.
      recon_loss_centroid = utils.masked_mean(recon_loss_centroid, masks_sampled_unsaturated & recon_occupancy_grid)
      # With predicted occupancy grid mask.
      # recon_loss_centroid = utils.masked_mean(recon_loss_centroid, masks_sampled_unsaturated & dist[0].pred())
      metrics[f'image_encoder_recon_{obs_group}_{obs_name}_centroid_loss'] = recon_loss_centroid.item()
      recon_loss = recon_loss_occupancy + recon_loss_centroid
      metrics[f'image_encoder_recon_{obs_group}_{obs_name}_loss'] = recon_loss.item()
    else:
      recon_loss = dist.loss(recon_obs.detach())
      recon_loss = utils.masked_mean(recon_loss, masks_sampled)
      metrics[f'image_encoder_recon_{obs_group}_{obs_name}_loss'] = recon_loss.item()
    losses.append(recon_loss)

  # if symmetry:
  #   batch_sampled_sym = pytree.tree_map(
  #     lambda x: x[:, sample_idxs],
  #     batch.obs_sym[image_encoder_obs_key],)
  #   batch_sampled_hidden_states_sym = pytree.tree_map(
  #     lambda x: x[:, sample_idxs],
  #     batch.hidden_states_sym[image_encoder_obs_key])
  #   image_encoder_dists_sym, _, _ = image_encoder_network(
  #     batch_sampled_sym,
  #     masks=masks_sampled,
  #     hidden_states=batch_sampled_hidden_states_sym,
  #     unpad=False
  #   )
  #   for name, dist in image_encoder_dists_sym.items():
  #     obs_group, obs_name = name.split('/')
  #     recon_loss = dist.loss(
  #       symmetry_fn(obs_group, obs_name)(
  #         pytree.tree_map(lambda x: x[:, sample_idxs], batch.obs_sym[obs_group][obs_name])
  #     ).detach())
  #     recon_loss = utils.masked_mean(recon_loss, masks_sampled)
  #     metrics[f'image_encoder_recon_{obs_group}_{obs_name}_loss_sym'] = recon_loss.item()
  #     losses.append(recon_loss)

  total_loss = sum(losses)
  return total_loss, metrics

@timer.section("learn_ppo")
def learn_ppo(
    buffer: experience_buffer.ExperienceBuffer,
    policy: torch.nn.Module,
    old_policy: torch.nn.Module,
    reward_normalizer: Optional[torch.nn.Module],
    value: torch.nn.Module,
    policy_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    policy_learning_rate: float,
    value_learning_rate: float,
    policy_obs_key: str,
    value_obs_key: str,
    symm_obs_key: str,
    symmetry_fn: _SYMMETRY_FN,
    last_privileged_obs: Dict[str, Any],
    last_value_hidden_states: Tuple[torch.Tensor, ...],
    it: int,
    algorithm_cfg: Dict[str, Any],
    multi_gpu: bool = False,
    multi_gpu_global_rank: int = 0,
    multi_gpu_world_size: int = 1,
  ):
  if it == 0:
    # Skip the first gradient update to initialize the observation normalizers.
    with torch.no_grad():
      policy.update_normalizer(buffer[policy_obs_key])
      value.update_normalizer(buffer[value_obs_key])
    return policy_learning_rate, value_learning_rate, {}

  learn_step_agg = agg.Agg()
  old_policy.load_state_dict(policy.state_dict())

  obs_groups = [policy_obs_key, value_obs_key]
  hidden_states_keys = [policy_obs_key, value_obs_key]
  obs_sym_groups = []
  if algorithm_cfg["symmetry_augmentation"]:
    obs_sym_groups.append(policy_obs_key)
    obs_sym_groups.append(value_obs_key)
  elif algorithm_cfg["symmetry_loss"]:
    obs_sym_groups.append(policy_obs_key)
  rl_normalizers = {}
  if algorithm_cfg["normalize_rewards"]:
    rl_normalizers["rewards"] = reward_normalizer

  for batch in buffer.reccurent_mini_batch_generator(
    algorithm_cfg["num_learning_epochs"],
    algorithm_cfg["num_mini_batches"],
    last_value_items=[value, last_privileged_obs, last_value_hidden_states],
    obs_groups=obs_groups,
    hidden_states_keys=hidden_states_keys,
    networks={
      policy_obs_key: old_policy,
    },
    networks_sym={},
    obs_sym_groups=obs_sym_groups,
    symm_key=symm_obs_key,
    symmetry_fn=symmetry_fn,
    symmetry_flip_latents=algorithm_cfg["symmetry_flip_latents"],
    dones_key="dones",
    rl_keys=[
      "values", "time_outs", "rewards", "dones", "actions"
    ],
    rl_normalizers=rl_normalizers,
  ):
    # Value loss.
    val_loss, advantages, metrics = value_loss(
      batch,
      value,
      value_obs_key,
      algorithm_cfg["symmetry_augmentation"],
      algorithm_cfg["gamma"],
      algorithm_cfg["lam"],
      algorithm_cfg["use_clipped_value_loss"],
      algorithm_cfg["clip_param"]
    )
    total_loss = algorithm_cfg["value_loss_coef"] * val_loss
    learn_step_agg.add(metrics)

    # Normalize advantages.
    if multi_gpu:
      advantages_mean, advantages_std = utils.broadcast_moments(advantages)
    else:
      advantages_mean, advantages_std = advantages.mean(), advantages.std()
    advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)

    # Actor loss.
    act_loss, kls, metrics = actor_loss(
      advantages,
      batch,
      policy,
      policy_obs_key,
      algorithm_cfg["symmetry_loss"],
      algorithm_cfg["symmetry_augmentation"],
      symmetry_fn,
      algorithm_cfg["actor_loss_coefs"],
      algorithm_cfg["entropy_coefs"],
      algorithm_cfg["bound_coefs"],
      algorithm_cfg["symmetry_coefs"],
      algorithm_cfg["clip_param"],
    )
    total_loss += act_loss
    learn_step_agg.add(metrics)

    # SGD.
    policy_optimizer.zero_grad()
    value_optimizer.zero_grad()
    total_loss.backward()
    all_params = list(policy.parameters()) + list(value.parameters())
    if multi_gpu:
      utils.sync_grads_multi_gpu([all_params], multi_gpu_world_size)
    torch.nn.utils.clip_grad_norm_(
      all_params,
      1.0)
    policy_optimizer.step()
    value_optimizer.step()

    # Learning rate scheduler.
    if algorithm_cfg["desired_kl"] > 0.0:
      if not multi_gpu or multi_gpu_global_rank == 0:
        if kls[algorithm_cfg["kl_key"]] > algorithm_cfg["desired_kl"] * 2.0:
          policy_learning_rate = max(1e-5, policy_learning_rate / 1.5)
        elif kls[algorithm_cfg["kl_key"]] < algorithm_cfg["desired_kl"] / 2.0 and kls[algorithm_cfg["kl_key"]] > 0.0:
          policy_learning_rate = min(1e-2, policy_learning_rate * 1.5)

      if multi_gpu:
        policy_learning_rate = utils.broadcast_scalar(
          policy_learning_rate,
          0,
          next(policy.parameters()).device
        )

      for param_group in policy_optimizer.param_groups:
        param_group["lr"] = policy_learning_rate

    policy_stats = {}
    for k, v in policy.stats().items():
      v = v.cpu().numpy()
      policy_stats[f'policy/{k}_mean'] = v.mean()
      policy_stats[f'policy/{k}_std'] = v.std()
      policy_stats[f'policy/{k}_min'] = v.min()
      policy_stats[f'policy/{k}_max'] = v.max()
    learn_step_agg.add(policy_stats)

  # Update the observation normalizers.
  with torch.no_grad():
    policy.update_normalizer(buffer[policy_obs_key], multi_gpu=multi_gpu)
    value.update_normalizer(buffer[value_obs_key], multi_gpu=multi_gpu)

  for i, param_group in enumerate(policy_optimizer.param_groups):
    assert isinstance(param_group["lr"], float)
    learn_step_agg.add({f"policy/param_group_{i}_lr": param_group["lr"]})
  for i, param_group in enumerate(value_optimizer.param_groups):
    assert isinstance(param_group["lr"], float)
    learn_step_agg.add({f"value/param_group_{i}_lr": param_group["lr"]})

  return policy_learning_rate, value_learning_rate, learn_step_agg.result()


@timer.section("learn_image_encoder")
def learn_image_encoder(
    buffer: experience_buffer.ExperienceBuffer,
    image_encoder: torch.nn.Module,
    image_encoder_optimizer: torch.optim.Optimizer,
    image_encoder_obs_key: str,
    value_obs_key: str,
    symm_key: str,
    symmetry_fn: _SYMMETRY_FN,
    num_learning_epochs: int,
    num_mini_batches: int,
    algorithm_cfg: Dict[str, Any],
    multi_gpu: bool = False,
    multi_gpu_global_rank: int = 0,
    multi_gpu_world_size: int = 1,
  ):
  learn_step_agg = agg.Agg()
  for batch in buffer.reccurent_mini_batch_generator(
    num_learning_epochs,
    num_mini_batches,
    last_value_items=None,
    obs_groups=[image_encoder_obs_key, value_obs_key],
    hidden_states_keys=[image_encoder_obs_key],
    networks={},
    networks_sym={},
    obs_sym_groups=[image_encoder_obs_key],
    symm_key=symm_key,
    symmetry_fn=symmetry_fn,
    symmetry_flip_latents=algorithm_cfg["symmetry_flip_latents"],
    dones_key="dones",
    rl_keys=[],
  ):
    # Reconstruction loss.
    recon_loss, metrics = reconstruction_loss(
      batch,
      image_encoder,
      image_encoder_obs_key,
      algorithm_cfg["image_encoder_batch_size"],
      algorithm_cfg["symmetry_augmentation"],
      symmetry_fn,
    )
    learn_step_agg.add(metrics)
    image_encoder_optimizer.zero_grad()
    recon_loss.backward()
    if multi_gpu:
      utils.sync_grads_multi_gpu([image_encoder.parameters()], multi_gpu_world_size)
    torch.nn.utils.clip_grad_norm_(
      image_encoder.parameters(),
      1.0)
    image_encoder_optimizer.step()

  for i, param_group in enumerate(image_encoder_optimizer.param_groups):
    assert isinstance(param_group["lr"], float)
    learn_step_agg.add({f"image_encoder/param_group_{i}_lr": param_group["lr"]})

  return learn_step_agg.result()
