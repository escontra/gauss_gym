from typing import Dict, Any, Tuple, Callable
import torch
import torch.utils._pytree as pytree

from legged_gym.rl import utils
from legged_gym.utils import voxel, agg, timer
from legged_gym.rl import experience_buffer


def value_loss(
    batch: experience_buffer.MiniBatch,
    value_network,
    value_obs_key,
    gamma,
    lam,
    use_clipped_value_loss,
    clip_param): 
  values, _, _ = value_network(
    batch.obs[value_obs_key],
    masks=batch.rl_values["masks"],
    hidden_states=batch.hidden_states[value_obs_key]
  )
  # Compute returns and advantages.
  with torch.no_grad():
    rewards = batch.rl_values["rewards"].clone()
    value_preds = values['value'].pred().squeeze(-1)
    rewards[batch.rl_values["time_outs"]] = value_preds[batch.rl_values["time_outs"]]
    advantages = utils.discount_values(
      rewards,
      batch.rl_values["dones"] | batch.rl_values["time_outs"],
      value_preds,
      batch.rl_values["last_value"].squeeze(-1),
      gamma,
      lam,
    )
    returns = value_preds + advantages

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
  return value_loss, advantages, metrics


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


def actor_loss(
    advantages,
    batch: experience_buffer.MiniBatch,
    policy_network,
    policy_obs_key,
    symmetry,
    symmetry_fn,
    actor_loss_coefs,
    entropy_coefs,
    bound_coefs,
    symmetry_coefs,
    clip_param,
):
  # Actor loss.
  dists, _, _ = policy_network(
    batch.obs[policy_obs_key],
    masks=batch.rl_values["masks"],
    hidden_states=batch.hidden_states[policy_obs_key]
  )
  if symmetry:
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
    actions_log_prob = dist.logp(batch.rl_values["actions"][name].detach()).sum(dim=-1)
    with torch.no_grad():
      old_actions_log_prob_batch = batch.network_batch[policy_obs_key][name].logp(
        batch.rl_values["actions"][name]).sum(dim=-1)
    actor_loss = surrogate_loss(
      old_actions_log_prob_batch, actions_log_prob, advantages,
      e_clip=clip_param
    )
    losses.append(actor_loss_coef * actor_loss)
    metrics[f'actor_loss_{name}'] = actor_loss.item()
    klm = torch.mean(torch.sum(dist.kl(batch.network_batch[policy_obs_key][name]), axis=-1)).item()
    kl_means[name] = klm
    metrics[f'kl_mean_{name}'] = klm

    # Entropy loss.
    entropy = dist.entropy().sum(dim=-1)
    metrics[f'entropy_{name}'] = entropy.mean().item()
    if name in entropy_coefs:
      losses.append(actor_loss_coef * entropy_coefs[name] * entropy.mean())
      metrics[f'entropy_loss_{name}'] = entropy.mean().item()

    # Bound loss.
    if name in bound_coefs:
      bound_loss = (
        torch.clip(dist.pred() - 1.0, min=0.0).square().mean()
        + torch.clip(dist.pred() + 1.0, max=0.0).square().mean()
      )
      losses.append(actor_loss_coef * bound_coefs[name] * bound_loss)
      metrics[f'bound_loss_{name}'] = bound_loss.item()


    if symmetry and name in symmetry_coefs:
      # Symmetry loss.
      dist_act_sym = symmetry_fn("actions", name)(dist.pred())
      symmetry_loss = torch.nn.MSELoss()(dists_sym[name].pred(), dist_act_sym.detach())
      # symmetry_loss = torch.mean(dists_sym[name].loss(dist_act_sym.detach()))
      metrics[f'symmetry_loss_{name}'] = symmetry_loss.item()

      losses.append(actor_loss_coef * symmetry_coefs[name] * symmetry_loss.mean())
    total_loss = sum(losses)
    return total_loss, kl_means, metrics


def reconstruction_loss(
  batch: experience_buffer.MiniBatch,
  image_encoder_network,
  image_encoder_obs_key,
  max_batch_size,
  symmetry,
  symmetry_fn,
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
    value: torch.nn.Module,
    policy_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    policy_learning_rate: float,
    value_learning_rate: float,
    policy_obs_key: str,
    value_obs_key: str,
    symm_obs_key: str,
    symmetry_fn: Callable[[str, str], Callable[[torch.Tensor], torch.Tensor]],
    last_privileged_obs: Dict[str, Any],
    last_value_hidden_states: Tuple[torch.Tensor, ...],
    it: int,
    ppo_cfg: Dict[str, Any],
  ):
  if it == 0:
    # Skip the first gradient update to initialize the observation normalizers.
    with torch.no_grad():
      policy.update_normalizer(buffer[policy_obs_key])
      value.update_normalizer(buffer[value_obs_key])
    return {}

  learn_step_agg = agg.Agg()
  old_policy.load_state_dict(policy.state_dict())

  obs_groups = [policy_obs_key, value_obs_key]
  hidden_states_keys = [policy_obs_key, value_obs_key]
  obs_sym_groups = [policy_obs_key]

  for batch in buffer.reccurent_mini_batch_generator(
    ppo_cfg["num_learning_epochs"],
    ppo_cfg["num_mini_batches"],
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
    symmetry_flip_latents=ppo_cfg["symmetry_flip_latents"],
    dones_key="dones",
    rl_keys=[
      "values", "time_outs", "rewards", "dones", "actions"
    ],
  ):
    # Value loss.
    val_loss, advantages, metrics = value_loss(
      batch,
      value,
      value_obs_key,
      ppo_cfg["gamma"],
      ppo_cfg["lam"],
      ppo_cfg["use_clipped_value_loss"],
      ppo_cfg["clip_param"]
    )
    total_loss = ppo_cfg["value_loss_coef"] * val_loss
    learn_step_agg.add(metrics)

    # Actor loss.
    act_loss, kls, metrics = actor_loss(
      advantages,
      batch,
      policy,
      policy_obs_key,
      ppo_cfg["symmetry"],
      symmetry_fn,
      ppo_cfg["actor_loss_coefs"],
      ppo_cfg["entropy_coefs"],
      ppo_cfg["bound_coefs"],
      ppo_cfg["symmetry_coefs"],
      ppo_cfg["clip_param"],
    )
    total_loss += act_loss
    learn_step_agg.add(metrics)

    # SGD.
    policy_optimizer.zero_grad()
    value_optimizer.zero_grad()
    total_loss.backward()
    all_params = list(policy.parameters()) + list(value.parameters())
    torch.nn.utils.clip_grad_norm_(
      all_params,
      1.0)
    policy_optimizer.step()
    value_optimizer.step()

    del batch

    # Learning rate scheduler.
    if ppo_cfg["desired_kl"] > 0.0:
      if kls[ppo_cfg["kl_key"]] > ppo_cfg["desired_kl"] * 2.0:
        policy_learning_rate = max(1e-5, policy_learning_rate / 1.5)
      elif kls[ppo_cfg["kl_key"]] < ppo_cfg["desired_kl"] / 2.0 and kls[ppo_cfg["kl_key"]] > 0.0:
        policy_learning_rate = min(1e-2, policy_learning_rate * 1.5)
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
    policy.update_normalizer(buffer[policy_obs_key])
    value.update_normalizer(buffer[value_obs_key])

  learning_rate_stats = {
    f"{policy_obs_key}_lr": policy_learning_rate,
    f"{value_obs_key}_lr": value_learning_rate,
  }

  return policy_learning_rate, value_learning_rate, {
      **learn_step_agg.result(),
      **learning_rate_stats,
  }
