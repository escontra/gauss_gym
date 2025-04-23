import os
import numpy as np
from typing import Any, Dict
import random
import time
import torch
import torch.utils._pytree as pytree
import pathlib

from legged_gym.rl import experience_buffer, recorder
from legged_gym.rl.env import vec_env
from legged_gym.rl.modules import models
from legged_gym.utils import agg, symmetry_groups, timer, when, space


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


class Runner:
  def __init__(self, env: vec_env.VecEnv, cfg: Dict[str, Any], device="cpu"):
    self.env = env
    self.device = device
    self.cfg = cfg
    self._set_seed()
    self.policy_key = self.cfg["policy"]["obs_key"]
    self.value_key = self.cfg["value"]["obs_key"]

    self.policy_learning_rate = self.cfg["policy"]["learning_rate"]
    self.policy: models.RecurrentModel = getattr(
      models, self.cfg["policy"]["class_name"])(
      self.env.action_space(),
      self.env.obs_space()[self.policy_key],
      **self.cfg["policy"]["params"]
    ).to(self.device)
    # For KL.
    self.old_policy = getattr(models, self.cfg["policy"]["class_name"])(
      self.env.action_space(),
      self.env.obs_space()[self.policy_key],
      **self.cfg["policy"]["params"]
    ).to(self.device)
    for param in self.old_policy.parameters():
      param.requires_grad = False

    self.value_learning_rate = self.cfg["value"]["learning_rate"]
    self.value: models.RecurrentModel = getattr(
      models, self.cfg["value"]["class_name"])(
      {'value': space.Space(np.float32, (1,), -np.inf, np.inf)},
      self.env.obs_space()[self.value_key],
      **self.cfg["value"]["params"]
    ).to(self.device)

    self.policy_optimizer = torch.optim.AdamW(
      self.policy.parameters(), lr=self.policy_learning_rate, weight_decay=self.cfg["policy"]["weight_decay"]
    )
    self.value_optimizer = torch.optim.AdamW(
      self.value.parameters(), lr=self.value_learning_rate, weight_decay=self.cfg["value"]["weight_decay"]
    )

    if self.cfg["algorithm"]["symmetry"]:
      assert "symmetries" in self.cfg, "Need `symmetries` in config when symmetry is enabled. Look at a1/config.yaml for an example."
      self.symmetry_groups = {}
      for group_name in self.cfg["symmetries"]:
        self.symmetry_groups[group_name] = {}
        for symmetry in self.cfg["symmetries"][group_name]["symmetries"]:
          symmetry_modifier = getattr(symmetry_groups, symmetry)
          self.symmetry_groups[group_name][symmetry_modifier.observation.name] = symmetry_modifier
      assert self.policy_key in self.symmetry_groups
      assert "actions" in self.symmetry_groups

  def state_dict_for_network(self, state_dict):
    new_state_dict = {}
    prefix = '_orig_mod.'
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

  def _set_seed(self):
    seed = self.cfg["seed"]
    if seed == -1:
      seed = np.random.randint(0, 10000)
    print("Setting RL seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

  def load(self, resume_root: pathlib.Path):
    if not self.cfg["runner"]["resume"]:
      return

    load_run = self.cfg["runner"]["load_run"]
    checkpoint = self.cfg["runner"]["checkpoint"]
    if (load_run == "-1") or (load_run == -1):
      resume_path = sorted(
        [item for item in resume_root.iterdir() if item.is_dir()],
        key=lambda path: path.stat().st_mtime,
      )[-1]
    else:
      resume_path = resume_root / load_run
    print(f"Loading checkpoint from: {resume_path}")
    print(f'\tNum checkpoints: {len(list((resume_path / "nn").glob("*.pth")))}')
    print(f'\tLoading checkpoint: {checkpoint}')
    if (checkpoint == "-1") or (checkpoint == -1):
      model_path = sorted(
        (resume_path / "nn").glob("*.pth"),
        key=lambda path: path.stat().st_mtime,
      )[-1]
    else:
      model_path = resume_path / "nn" / f"model_{checkpoint}.pth"
    print(f'\tLoading model weights from: {model_path}')
    model_dict = torch.load(
      model_path, map_location=self.device, weights_only=True
    )
    self.policy.load_state_dict(model_dict["policy"], strict=True)
    self.value.load_state_dict(model_dict["value"], strict=True)
    try:
      self.policy_optimizer.load_state_dict(model_dict["policy_optimizer"])
      self.value_optimizer.load_state_dict(model_dict["value_optimizer"])
    except Exception as e:
      print(f"Failed to load optimizer: {e}")

  def to_device(self, obs):
    return pytree.tree_map(lambda x: x.to(self.device), obs)

  def filter_nans(self, obs):
    if isinstance(obs, dict):
      for _, v in obs.items():
        num_nan_envs = v.isnan().any(dim=-1).sum()
    else:
      num_nan_envs = obs.isnan().any(dim=-1).sum()

    if num_nan_envs > 0:
      print(f"{num_nan_envs} NaN envs")
      obs = pytree.tree_map(lambda x: torch.nan_to_num(x, nan=0.0), obs)
    return obs

  def learn(self, num_learning_iterations, log_dir: pathlib.Path, init_at_random_ep_len=False):
    # Logger aggregators.
    self.step_agg = agg.Agg()
    self.episode_agg = agg.Agg()
    self.learn_agg = agg.Agg()
    self.action_agg = agg.Agg()
    self.should_log = when.Clock(self.cfg["runner"]["log_every"])
    self.should_save = when.Clock(self.cfg["runner"]["save_every"])
    self.recorder = recorder.Recorder(log_dir, self.cfg, self.env.deploy_config(), self.env.obs_space(), self.env.action_space())
    if self.cfg["runner"]["record_video"]:
      self.recorder.setup_recorder(self.env)

    # Initialize hidden states and set random episode length.
    obs_dict = self.to_device(self.env.reset())
    self.policy(obs_dict[self.policy_key], update_state=False)
    self.value(obs_dict[self.value_key], update_state=False)
    if init_at_random_ep_len:
        self.env.episode_length_buf = torch.randint_like(
          self.env.episode_length_buf,
          high=int(self.env.max_episode_length))


    # Replay buffer.
    self.buffer = experience_buffer.ExperienceBuffer(
      self.cfg["runner"]["num_steps_per_env"],
      self.env.num_envs,
      self.device,
    )
    self.buffer.add_buffer(self.policy_key, self.env.obs_space()[self.policy_key])
    self.buffer.add_buffer(self.value_key, self.env.obs_space()[self.value_key])
    self.buffer.add_buffer("actions", self.env.action_space())
    self.buffer.add_buffer("rewards", ())
    self.buffer.add_buffer("values", (1,))
    self.buffer.add_buffer("dones", (), dtype=bool)
    self.buffer.add_buffer("time_outs", (), dtype=bool)
    if self.policy.is_recurrent:
      self.buffer.add_hidden_state_buffers("policy_hidden_states", self.policy.get_hidden_states())
    if self.value.is_recurrent:
      self.buffer.add_hidden_state_buffers("value_hidden_states", self.value.get_hidden_states())

    for it in range(num_learning_iterations):
      start = time.time()
      for n in range(self.cfg["runner"]["num_steps_per_env"]):
        if self.cfg["runner"]["record_video"]:
          self.recorder.record_statistics(
            self.recorder.maybe_record(self.env),
            it * self.cfg["runner"]["num_steps_per_env"] * self.env.num_envs + n)
        with timer.section("buffer_add_obs"):
          self.buffer.update_data(self.policy_key, n, obs_dict[self.policy_key])
          self.buffer.update_data(self.value_key, n, obs_dict[self.value_key])
          if self.policy.is_recurrent:
            self.buffer.update_hidden_state_buffers(
              "policy_hidden_states", n, self.policy.get_hidden_states()
            )
          if self.value.is_recurrent:
            self.buffer.update_hidden_state_buffers(
              "value_hidden_states", n, self.value.get_hidden_states()
            )
        with timer.section("model_act"):
          with torch.no_grad():
            dists = self.policy(obs_dict[self.policy_key])
            value = self.value(obs_dict[self.value_key])
            actions = {k: dist.sample() for k, dist in dists.items()}
        with timer.section("env_step"):
          # Scale actions to the environment range.
          actions_env = {}
          for k, v in self.env.action_space().items():
            action = actions[k]
            low = torch.tensor(v.low, device=self.device)[None]
            high = torch.tensor(v.high, device=self.device)[None]
            needs_scaling = (torch.isfinite(low).all() and torch.isfinite(high).all()).item()
            if needs_scaling:
              scaled_action = (action + 1) / 2 * (high - low) + low
              scaled_action = torch.clamp(scaled_action, low, high)
              actions_env[k] = scaled_action
            else:
              actions_env[k] = action
            self.action_agg.add(
              {f'{dof_name}_{k}': actions_env[k][:, dof_idx].cpu().numpy()
               for dof_idx, dof_name in enumerate(self.env.dof_names)},
              agg='concat'
            )
          # Make sure all actions are processed
          assert len(actions_env) == len(actions), f"Missing actions in actions_env. Expected {len(actions)}, got {len(actions_env)}"

          obs_dict, rew, done, infos = self.env.step(actions_env)
          obs_dict, rew, done = self.to_device((obs_dict, rew, done))
        self.policy.reset(done)
        self.value.reset(done)
        with timer.section("buffer_update_data"):
          self.buffer.update_data("actions", n, actions)
          self.buffer.update_data("rewards", n, rew)
          self.buffer.update_data("dones", n, done)
          self.buffer.update_data(
            "time_outs", n, infos["time_outs"].to(self.device)
          )
        with timer.section("log_step"):
          bootstrapped_rew = torch.where(done, 0., rew)
          bootstrapped_rew = torch.where(infos["time_outs"], value['value'].pred().squeeze(-1), bootstrapped_rew)
          self.step_agg.add(infos["episode"])
          self.episode_agg.add(self.recorder.record_episode_statistics(
            done,
            {"reward": rew, "return": bootstrapped_rew},
            it,
            discount_factor_dict={"return": self.cfg["algorithm"]["gamma"]},
            write_record=n == (self.cfg["runner"]["num_steps_per_env"] - 1),
          ))

      # We skip the first gradient update to initialize the observation normalizers.
      learn_stats = self._learn(last_privileged_obs=obs_dict[self.value_key], is_first=it == 0)
      self.learn_agg.add(learn_stats)

      if self.should_log(it):
        with timer.section("logger_save"):
          step_stats = {f"step/{k}": v for k, v in self.step_agg.result().items()}
          learn_stats = {f"learn/{k}": v for k, v in self.learn_agg.result().items()}
          timer_stats = {f"timer/{k}": v for k, v in timer.stats().items()}
          episode_stats = {f"episode/{k}": v for k, v in self.episode_agg.result().items()}
          action_stats = {f"action/{k}": v for k, v in self.action_agg.result().items()}
          self.recorder.record_statistics(
            {
              **step_stats,
              **learn_stats,
              **timer_stats,
              **episode_stats,
              **action_stats,
            },
            it * self.cfg["runner"]["num_steps_per_env"] * self.env.num_envs
          )

      if self.should_save(it):
        with timer.section("model_save"):
          self.recorder.save( {
              "policy": self.policy.state_dict(),
              "value": self.value.state_dict(),
              "policy_optimizer": self.policy_optimizer.state_dict(),
              "value_optimizer": self.value_optimizer.state_dict(),
            },
            it + 1,
          )
      print(
        "epoch: {}/{} - {}s.".format(
          it + 1, num_learning_iterations, time.time() - start
        )
      )
      start = time.time()

  def reccurent_mini_batch_generator(self, last_value_obs):
      policy_obs = self.buffer[self.policy_key]
      value_obs = self.buffer[self.value_key]
      policy_obs_split = pytree.tree_map(
        lambda x: models.split_and_pad_trajectories(x, self.buffer["dones"]),
        policy_obs,
      )
      policy_obs = pytree.tree_map(
        lambda x: x[0].detach(), policy_obs_split, is_leaf=lambda x: isinstance(x, tuple)
      )
      traj_masks = pytree.tree_map(
        lambda x: x[1].detach(), policy_obs_split, is_leaf=lambda x: isinstance(x, tuple)
      )
      traj_masks = list(traj_masks.values())[0]
      last_was_done = torch.zeros_like(self.buffer["dones"], dtype=torch.bool)
      last_was_done[1:] = self.buffer["dones"][:-1]
      last_was_done[0] = True
      hid_a = [
        saved_hidden_states.permute(2, 0, 1, 3)[last_was_done.permute(1, 0)].transpose(1, 0)
        for saved_hidden_states in self.buffer["policy_hidden_states"]
      ]
      value_obs_split = pytree.tree_map(
        lambda x: models.split_and_pad_trajectories(x, self.buffer["dones"]),
        value_obs,
      )
      value_obs = pytree.tree_map(
        lambda x: x[0],
        value_obs_split,
        is_leaf=lambda x: isinstance(x, tuple),
      )
      hid_c = [
        saved_hidden_states.permute(2, 0, 1, 3)[last_was_done.permute(1, 0)].transpose(1, 0)
        for saved_hidden_states in self.buffer["value_hidden_states"]
      ]
      if self.cfg["algorithm"]["symmetry"]:
        # Symmetry-augmented observations.
        policy_obs_sym = {}
        for key, value in policy_obs.items():
          assert key in self.symmetry_groups[self.policy_key], f"{key} not in {self.symmetry_groups[self.policy_key]}"
          policy_obs_sym[key] = self.symmetry_groups[self.policy_key][key](self.env, value).detach()

      # Used for computing KL divergence and old log probs.
      self.old_policy.load_state_dict(self.policy.state_dict())

      mini_batch_size = self.env.num_envs // self.cfg["algorithm"]["num_mini_batches"]
      for _ in range(self.cfg["algorithm"]["num_learning_epochs"]):
          first_traj = 0
          with torch.no_grad():
            # Used for value bootstrap.
            last_value = self.value(last_value_obs, update_state=False)['value'].pred().detach()
          for i in range(self.cfg["algorithm"]["num_mini_batches"]):
              start = i * mini_batch_size
              stop = (i + 1) * mini_batch_size
              last_traj = first_traj + torch.sum(last_was_done[:, start:stop]).item()

              masks_batch = traj_masks[:, first_traj:last_traj]
              policy_obs_batch = pytree.tree_map(lambda x: x[:, first_traj:last_traj], policy_obs)
              value_obs_batch = pytree.tree_map(lambda x: x[:, first_traj:last_traj], value_obs)
              hid_a_batch = pytree.tree_map(lambda x: x[:, first_traj:last_traj], hid_a)
              hid_c_batch = pytree.tree_map(lambda x: x[:, first_traj:last_traj], hid_c)
              if self.cfg["algorithm"]["symmetry"]:
                symmetry_obs_batch = {
                  'policy_obs_sym': pytree.tree_map(lambda x: x[:, first_traj:last_traj], policy_obs_sym),
                }
              else:
                symmetry_obs_batch = {}

              dones_batch = self.buffer["dones"][:, start:stop]
              time_outs_batch = self.buffer["time_outs"][:, start:stop]
              rewards_batch = self.buffer["rewards"][:, start:stop]
              actions_batch = {k: self.buffer["actions"][k][:, start:stop] for k in self.buffer["actions"].keys()}
              last_value_batch = pytree.tree_map(lambda x: x[start:stop], last_value)
              old_values_batch = self.buffer["values"][:, start:stop]
              with torch.no_grad():
                old_dist_batch = self.old_policy(
                  policy_obs_batch, masks=masks_batch, hidden_states=hid_a_batch
                )
                if self.cfg["algorithm"]["symmetry"]:
                  symmetry_obs_batch['old_dists_sym'] = self.old_policy(
                    symmetry_obs_batch['policy_obs_sym'], masks=masks_batch, hidden_states=hid_a_batch
                  )

              yield {
                "policy_obs": policy_obs_batch,
                "value_obs": value_obs_batch,
                "actions": actions_batch,
                "last_value": last_value_batch,
                "masks": masks_batch,
                "hid_a": hid_a_batch,
                "hid_c": hid_c_batch,
                "dones": dones_batch,
                "time_outs": time_outs_batch,
                "rewards": rewards_batch,
                "old_values": old_values_batch,
                "old_dists": old_dist_batch,
                **symmetry_obs_batch,
              }
              
              first_traj = last_traj

  @timer.section("learn")
  def _learn(self, last_privileged_obs, is_first):
      learn_step_agg = agg.Agg()
      for n, batch in enumerate(self.reccurent_mini_batch_generator(last_privileged_obs)):
        values = self.value(
          batch["value_obs"], masks=batch["masks"], hidden_states=batch["hid_c"]
        )
        # Compute returns and advantages.
        with torch.no_grad():
          rewards = batch["rewards"].clone()
          value_preds = values['value'].pred().squeeze(-1)
          rewards[batch["time_outs"]] = value_preds[batch["time_outs"]]
          advantages = discount_values(
            rewards,
            batch["dones"] | batch["time_outs"],
            value_preds,
            batch["last_value"].squeeze(-1),
            self.cfg["algorithm"]["gamma"],
            self.cfg["algorithm"]["lam"],
          )
          returns = value_preds + advantages
          if self.cfg["algorithm"]["normalize_advantage_per_minibatch"]:
            advantage_stats = (advantages.mean(), advantages.std())
          elif n == 0:
            advantage_stats = (advantages.mean(), advantages.std())

          advantages = (advantages - advantage_stats[0]) / (advantage_stats[1] + 1e-8)

        # Value loss.
        if self.cfg["algorithm"]["use_clipped_value_loss"]:
          value_clipped = batch["old_values"].detach() + (values.pred() - batch["old_values"].detach()).clamp(
              -self.cfg["algorithm"]["clip_param"], self.cfg["algorithm"]["clip_param"]
          )
          value_losses = (values.pred() - returns.unsqueeze(-1)).pow(2)
          value_losses_clipped = (value_clipped - returns.unsqueeze(-1)).pow(2)
          value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
          value_loss = torch.mean(values['value'].loss(returns.unsqueeze(-1)))
        total_loss = self.cfg["algorithm"]["value_loss_coef"] * value_loss
        learn_step_agg.add({'value_loss': value_loss.item()})

        # Actor loss.
        # actor_loss_coef = 0.5 if self.cfg["algorithm"]["symmetry"] else 1.0
        actor_loss_coef = 1.0
        dists = self.policy(batch["policy_obs"], masks=batch["masks"], hidden_states=batch["hid_a"])
        if self.cfg["algorithm"]["symmetry"]:
          dists_sym = self.policy(batch["policy_obs_sym"], masks=batch["masks"], hidden_states=batch["hid_a"])
        kl_means = {}

        for name, dist in dists.items():
          actions_log_prob = dist.logp(batch["actions"][name].detach()).sum(dim=-1)
          with torch.no_grad():
            old_actions_log_prob_batch = batch["old_dists"][name].logp(batch["actions"][name]).sum(dim=-1)
          actor_loss = surrogate_loss(
            old_actions_log_prob_batch, actions_log_prob, advantages,
            e_clip=self.cfg["algorithm"]["clip_param"]
          )
          total_loss += actor_loss_coef * actor_loss
          learn_step_agg.add({f'actor_loss_{name}': actor_loss.item()})
          klm = torch.mean(torch.sum(dist.kl(batch["old_dists"][name]), axis=-1)).item()
          kl_means[name] = klm
          self.learn_agg.add({f'kl_mean_{name}': klm})

          # Entropy loss.
          entropy = dist.entropy().sum(dim=-1)
          if name in self.cfg["algorithm"]["entropy_keys"]:
            total_loss += actor_loss_coef * self.cfg["algorithm"]["entropy_coef"] * entropy.mean()
          learn_step_agg.add({f'entropy_{name}': entropy.mean().item()})

          # Bound loss.
          if name in self.cfg["algorithm"]["bound_loss_keys"]:
            bound_loss = (
              torch.clip(dist.pred() - 1.0, min=0.0).square().mean()
              + torch.clip(dist.pred() + 1.0, max=0.0).square().mean()
            )
            total_loss += actor_loss_coef * self.cfg["algorithm"]["bound_coef"] * bound_loss
            learn_step_agg.add({f'bound_loss_{name}': bound_loss.item()})


          if self.cfg["algorithm"]["symmetry"]:
            # Symmetry loss.
            dist_act_sym = self.symmetry_groups["actions"][name](self.env, dist.pred())
            symmetry_loss = torch.nn.MSELoss()(dists_sym[name].pred(), dist_act_sym.detach())
            # symmetry_loss = torch.mean(dists_sym[name].loss(dist_act_sym.detach()))
            if self.cfg["algorithm"]["symmetry_coef"] <= 0.0:
              symmetry_loss = symmetry_loss.detach()
            learn_step_agg.add({f'symmetry_loss_{name}': symmetry_loss.item()})
            total_loss += self.cfg["algorithm"]["symmetry_coef"] * symmetry_loss.mean()
            # TODO(alescontrela): Adding symmetric actor losses causes instability. Why?
            # # Actor loss with symmetric data.
            # actions_sym = self.symmetry_groups["actions"]["actions"](self.env, batch["actions"])
            # actions_log_prob_symm = dist_sym.logp(actions_sym.detach()).sum(dim=-1)
            # with torch.no_grad():
            #   old_actions_sym_log_prob_batch = batch["old_dists_sym"].logp(actions_sym).sum(dim=-1)
            # actor_loss_symm = surrogate_loss(
            #   old_actions_sym_log_prob_batch, actions_log_prob_symm, advantages,
            #   e_clip=self.cfg["algorithm"]["clip_param"]
            # )
            # learn_step_agg.add({'actor_loss_symm': actor_loss_symm.item()})
            # total_loss += actor_loss_coef * actor_loss_symm
            # kl_sym_mean = torch.mean(torch.sum(dist_sym.kl(batch["old_dists_sym"]), axis=-1))
            # self.learn_agg.add({'kl_sym_mean': kl_sym_mean.item()})
            # kl_mean = (kl_mean + kl_sym_mean) / 2.0

            # # Bound loss with symmetric data.
            # bound_loss_sym = (
            #   torch.clip(dist_sym.pred() - 1.0, min=0.0).square().mean()
            #   + torch.clip(dist_sym.pred() + 1.0, max=0.0).square().mean()
            # )
            # total_loss += actor_loss_coef * self.cfg["algorithm"]["bound_coef"] * bound_loss_sym
            # learn_step_agg.add({'bound_loss_sym': bound_loss_sym.item()})

            # # Entropy loss with symmetric data.
            # entropy_sym = dist_sym.entropy().sum(dim=-1)
            # total_loss += actor_loss_coef * self.cfg["algorithm"]["entropy_coef"] * entropy_sym.mean()
            # learn_step_agg.add({'entropy_sym': entropy_sym.mean().item()})


        # Skip the first gradient update to initialize the observation normalizers.
        if not is_first:
          self.policy_optimizer.zero_grad()
          self.value_optimizer.zero_grad()
          total_loss.backward()
          torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value.parameters()), 1.0)
          self.policy_optimizer.step()
          self.value_optimizer.step()

          if self.cfg["policy"]["desired_kl"] > 0.0:
            if kl_means[self.cfg["policy"]["kl_key"]] > self.cfg["policy"]["desired_kl"] * 2.0:
              self.policy_learning_rate = max(1e-5, self.policy_learning_rate / 1.5)
            elif kl_means[self.cfg["policy"]["kl_key"]] < self.cfg["policy"]["desired_kl"] / 2.0 and kl_means[self.cfg["policy"]["kl_key"]] > 0.0:
              self.policy_learning_rate = min(1e-2, self.policy_learning_rate * 1.5)
            for param_group in self.policy_optimizer.param_groups:
              param_group["lr"] = self.policy_learning_rate

        policy_stats = {}
        for k, v in self.policy.stats().items():
          policy_stats[f'policy/{k}_mean'] = v.mean()
          policy_stats[f'policy/{k}_std'] = v.std()
          policy_stats[f'policy/{k}_min'] = v.min()
          policy_stats[f'policy/{k}_max'] = v.max()
        learn_step_agg.add(policy_stats)

      # Update the observation normalizers.
      self.policy.update_normalizer(self.buffer[self.policy_key])
      self.value.update_normalizer(self.buffer[self.value_key])

      return {
          **learn_step_agg.result(),
          "policy_lr": self.policy_learning_rate,
          "value_lr": self.value_learning_rate,
      }

  def play(self):
    obs_dict = self.to_device(self.env.reset())

    policy = models.get_policy_jitted(self.policy)
    # Alternatively can use torch.compile. Not sure if this is available on all machines.
    # opt_policy = torch.compile(self.policy)

    inference_time, step = 0., 0
    step_agg = agg.Agg()
    while True:
      with torch.no_grad():
        start = time.time()
        act = policy(obs_dict[self.policy_key])
        # act = opt_policy(obs_dict_normalized[self.policy_key]).pred()
        inference_time += time.time() - start
        obs_dict, _, dones, infos = self.env.step(act)
        obs_dict = self.to_device(obs_dict)
        step_reward = infos["step_reward"]
        step_agg.add(pytree.tree_map(lambda x: x.cpu().numpy()[0], step_reward), agg='stack')
        done = dones[0].item()
        if done:
          import matplotlib.pyplot as plt
          results = step_agg.result()
          keys = list(results.keys())
          num_keys = len(keys)
          num_rows = 3
          num_cols = (num_keys + num_rows - 1) // num_rows  # Ceiling division

          fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 3), squeeze=False)
          axs_flat = axs.flatten()

          for i, key in enumerate(keys):
              print(key, results[key].shape)
              value = results[key]
              if value.ndim > 0: # Check if value is plottable (e.g., not scalar)
                  axs_flat[i].plot(np.squeeze(value)) # Assuming value is a torch tensor
                  axs_flat[i].set_title(key)
              else:
                   axs_flat[i].text(0.5, 0.5, f'{key}: {value.item():.4f}', horizontalalignment='center', verticalalignment='center')
                   axs_flat[i].set_title(key)
                   axs_flat[i].axis('off')


          # Hide any unused subplots
          for j in range(i + 1, num_rows * num_cols):
              axs_flat[j].axis('off')

          plt.tight_layout()
          plt.show() # Or plt.savefig("step_agg_plot.png")

          step_agg = agg.Agg()

      step += 1
      if step % 100 == 0:
        print(f"Average inference time: {inference_time / step}")
        print(f"\t Per env: {inference_time / step / self.env.num_envs}")
        inference_time, step = 0., 0


  def interrupt_handler(self, signal, frame):
    print("\nInterrupt received, waiting for video to finish...")
    self.interrupt = True
