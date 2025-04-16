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
from legged_gym.rl.modules import models, normalizers
from legged_gym.utils import agg, symmetry_groups, timer, when


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
    self.obs_group_sizes = {
      self.policy_key: self.env.obs_group_size_per_name(self.policy_key),
      self.value_key: self.env.obs_group_size_per_name(self.value_key),
    }
    self.obs_normalizers = {
      self.policy_key: normalizers.PyTreeNormalizer(
        self.obs_group_sizes[self.policy_key],
      ).to(self.device),
      self.value_key: normalizers.PyTreeNormalizer(
        self.obs_group_sizes[self.value_key],
      ).to(self.device),
    }

    self.policy_learning_rate = self.cfg["policy"]["learning_rate"]
    self.policy = getattr(models, self.cfg["policy"]["class_name"])(
      self.env.num_actions,
      self.obs_group_sizes[self.policy_key],
      **self.cfg["policy"]["params"]
    ).to(self.device)

    self.value_learning_rate = self.cfg["value"]["learning_rate"]
    self.value = getattr(models, self.cfg["value"]["class_name"])(
      1,
      self.obs_group_sizes[self.value_key],
      **self.cfg["value"]["params"]
    ).to(self.device)

    self.policy_optimizer = torch.optim.Adam(
      self.policy.parameters(), lr=self.policy_learning_rate
    )
    self.value_optimizer = torch.optim.Adam(
      self.value.parameters(), lr=self.value_learning_rate
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

    self.buffer = experience_buffer.ExperienceBuffer(
      self.cfg["runner"]["num_steps_per_env"],
      self.env.num_envs,
      self.device,
    )
    self.buffer.add_buffer(
      self.policy_key, self.obs_group_sizes[self.policy_key]
    )
    self.buffer.add_buffer(
      self.value_key, self.obs_group_sizes[self.value_key]
    )
    self.buffer.add_buffer("actions", (self.env.num_actions,))
    self.buffer.add_buffer("rewards", ())
    self.buffer.add_buffer("values", (1,))
    self.buffer.add_buffer("dones", (), dtype=bool)
    self.buffer.add_buffer("time_outs", (), dtype=bool)

    self.step_agg = agg.Agg()
    self.episode_agg = agg.Agg()
    self.learn_agg = agg.Agg()
    self.should_log = when.Clock(self.cfg["runner"]["log_every"])
    self.should_save = when.Clock(self.cfg["runner"]["save_every"])

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

    try:
      for k, v in self.obs_normalizers.items():
        v.load_state_dict(model_dict[f"obs_normalizer/{k}"])
    except Exception as e:
      print(f"Failed to load obs normalizer: {e}")

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

  def normalize_obs(self, obs_dict):
    obs_dict_normalized = pytree.tree_map(
      lambda normalizer, obs: normalizer.normalize(obs),
      {k: v for k, v in self.obs_normalizers.items() if k in obs_dict},
      obs_dict,
      is_leaf=lambda x: isinstance(x, normalizers.PyTreeNormalizer)
    )
    return obs_dict_normalized

  def learn(self, num_learning_iterations, log_dir: pathlib.Path, init_at_random_ep_len=False):
    self.recorder = recorder.Recorder(log_dir, self.cfg, self.env.deploy_config(), self.obs_group_sizes)
    obs_dict = self.to_device(self.env.reset())
    obs_dict_normalized = self.normalize_obs(obs_dict)
    if init_at_random_ep_len:
        self.env.episode_length_buf = torch.randint_like(
          self.env.episode_length_buf,
          high=int(self.env.max_episode_length))

    # Needed to initialize hidden states.
    self.policy(obs_dict_normalized[self.policy_key])
    self.value(obs_dict_normalized[self.value_key])

    if self.policy.is_recurrent:
      self.buffer.add_hidden_state_buffers("policy_hidden_states", self.policy.get_hidden_states())
    if self.value.is_recurrent:
      self.buffer.add_hidden_state_buffers("value_hidden_states", self.value.get_hidden_states())

    for it in range(num_learning_iterations):
      start = time.time()
      # within horizon_length, env.step() is called with same act
      for n in range(self.cfg["runner"]["num_steps_per_env"]):
        print('ACTING')
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
        import pprint
        pprint.pprint(pytree.tree_map(lambda x: x.shape, obs_dict_normalized))
        with timer.section("model_act"):
          with torch.no_grad():
            dist = self.policy(obs_dict_normalized[self.policy_key])
            value = self.value(obs_dict_normalized[self.value_key])
            act = dist.sample()
        with timer.section("env_step"):
          obs_dict, rew, done, infos = self.env.step(act)
          obs_dict, rew, done = self.to_device((obs_dict, rew, done))
          obs_dict_normalized = self.normalize_obs(obs_dict)
        self.policy.reset(done)
        self.value.reset(done)
        with timer.section("buffer_update_data"):
          self.buffer.update_data("actions", n, act)
          self.buffer.update_data("rewards", n, rew)
          self.buffer.update_data("dones", n, done)
          self.buffer.update_data(
            "time_outs", n, infos["time_outs"].to(self.device)
          )
        with timer.section("log_step"):
          bootstrapped_rew = torch.where(done, 0., rew)
          bootstrapped_rew = torch.where(infos["time_outs"], value.pred().squeeze(-1), bootstrapped_rew)
          self.step_agg.add(infos["episode"])
          self.episode_agg.add(self.recorder.record_episode_statistics(
            done,
            {"reward": rew, "return": bootstrapped_rew},
            it,
            discount_factor_dict={"return": self.cfg["algorithm"]["gamma"]},
            write_record=n == (self.cfg["runner"]["num_steps_per_env"] - 1),
          ))



      learn_stats = self._learn(last_privileged_obs=obs_dict_normalized[self.value_key])
      self.learn_agg.add(learn_stats)

      if self.should_log(it):
        with timer.section("logger_save"):
          step_stats = {f"step/{k}": v for k, v in self.step_agg.result().items()}
          learn_stats = {f"learn/{k}": v for k, v in self.learn_agg.result().items()}
          timer_stats = {f"timer/{k}": v for k, v in timer.stats().items()}
          episode_stats = {f"episode/{k}": v for k, v in self.episode_agg.result().items()}
          self.recorder.record_statistics(
            {
              **step_stats,
              **learn_stats,
              **timer_stats,
              **episode_stats},
            it * self.cfg["runner"]["num_steps_per_env"] * self.env.num_envs
          )


      if self.should_save(it):
        with timer.section("model_save"):
          self.recorder.save( {
              "policy": self.policy.state_dict(),
              "value": self.value.state_dict(),
              "policy_optimizer": self.policy_optimizer.state_dict(),
              "value_optimizer": self.value_optimizer.state_dict(),
              **{f"obs_normalizer/{k}": v.state_dict() for k, v in self.obs_normalizers.items()},
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
      import pprint
      policy_obs = self.buffer[self.policy_key]
      value_obs = self.buffer[self.value_key]
      print('POLICY OBS B4')
      pprint.pprint(pytree.tree_map(lambda x: x.shape, policy_obs))
      print('DONES')
      pprint.pprint(self.buffer["dones"].shape)
      policy_obs_split = pytree.tree_map(
        lambda x: models.split_and_pad_trajectories(x, self.buffer["dones"]),
        policy_obs,
      )
      print('POLICY OBS SPLIT')
      pprint.pprint(pytree.tree_map(lambda x: x.shape, policy_obs_split))
      policy_obs = pytree.tree_map(
        lambda x: x[0].detach(), policy_obs_split, is_leaf=lambda x: isinstance(x, tuple)
      )
      traj_masks = pytree.tree_map(
        lambda x: x[1].detach(), policy_obs_split, is_leaf=lambda x: isinstance(x, tuple)
      )
      traj_masks = list(traj_masks.values())[0].detach()
      print('ALL POLICY OBS')
      pprint.pprint(pytree.tree_map(lambda x: x.shape, policy_obs))
      print('ALL TRAJ MASKS')
      pprint.pprint(pytree.tree_map(lambda x: x.shape, traj_masks))
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
      normalized_obs = self.normalize_obs(
        {self.policy_key: policy_obs, self.value_key: value_obs}
      )
      policy_obs = normalized_obs[self.policy_key]
      value_obs = normalized_obs[self.value_key]
      with torch.no_grad():
        last_value = self.value(last_value_obs, update_state=False).pred()

      mini_batch_size = self.env.num_envs // self.cfg["algorithm"]["num_mini_batches"]
      for ep in range(self.cfg["algorithm"]["num_learning_epochs"]):
          first_traj = 0
          for i in range(self.cfg["algorithm"]["num_mini_batches"]):
              start = i * mini_batch_size
              stop = (i + 1) * mini_batch_size
              last_traj = first_traj + torch.sum(last_was_done[:, start:stop]).item()
              print(start, stop, first_traj, last_traj)

              masks_batch = traj_masks[:, first_traj:last_traj]
              policy_obs_batch = pytree.tree_map(lambda x: x[:, first_traj:last_traj], policy_obs)
              value_obs_batch = pytree.tree_map(lambda x: x[:, first_traj:last_traj], value_obs)
              hid_a_batch = pytree.tree_map(lambda x: x[:, first_traj:last_traj], hid_a)
              hid_c_batch = pytree.tree_map(lambda x: x[:, first_traj:last_traj], hid_c)
          

              dones_batch = self.buffer["dones"][:, start:stop]
              time_outs_batch = self.buffer["time_outs"][:, start:stop]
              rewards_batch = self.buffer["rewards"][:, start:stop]
              actions_batch = self.buffer["actions"][:, start:stop]
              last_value_batch = pytree.tree_map(lambda x: x[start:stop], last_value)
              old_values_batch = self.buffer["values"][:, start:stop]

              # print('actions')
              # print(self.buffer["actions"].shape)
              # print(actions_batch.shape)

              with torch.no_grad():
                # import pprint
                # print('policy_obs_batch')
                # pprint.pprint(pytree.tree_map(lambda x: x.shape, policy_obs_batch))
                # print('masks_batch')
                # pprint.pprint(masks_batch.shape)
                # print('hid_a_batch')
                # pprint.pprint(pytree.tree_map(lambda x: x.shape, hid_a_batch))

                old_dist_batch = self.policy(
                  policy_obs_batch, masks=masks_batch, hidden_states=hid_a_batch
                )
                old_actions_log_prob_batch = old_dist_batch.logp(actions_batch).sum(
                  dim=-1
                )

              yield {
                "policy_obs": policy_obs_batch,
                "value_obs": value_obs_batch,
                "actions": actions_batch,
                "old_actions_log_prob": old_actions_log_prob_batch,
                "last_value": last_value_batch,
                "masks": masks_batch,
                "hid_a": hid_a_batch,
                "hid_c": hid_c_batch,
                "dones": dones_batch,
                "time_outs": time_outs_batch,
                "rewards": rewards_batch,
                "old_values": old_values_batch,
                "old_dist": old_dist_batch,
              }
              
              first_traj = last_traj

  @timer.section("learn")
  def _learn(self, last_privileged_obs):
      import pprint
      pprint.pprint(pytree.tree_map(lambda x: x.shape, self.buffer.tensor_dict))
      learn_step_agg = agg.Agg()
      for n, batch in enumerate(self.reccurent_mini_batch_generator(last_privileged_obs)):
        values = self.value(
          batch["value_obs"], masks=batch["masks"], hidden_states=batch["hid_c"]
        )
        with torch.no_grad():
          rewards = batch["rewards"].clone()
          rewards[batch["time_outs"]] = values.pred().squeeze(-1)[batch["time_outs"]]
          advantages = discount_values(
            rewards,
            batch["dones"] | batch["time_outs"],
            values.pred().squeeze(-1),
            batch["last_value"].squeeze(-1),
            self.cfg["algorithm"]["gamma"],
            self.cfg["algorithm"]["lam"],
          )
          returns = values.pred().squeeze(-1) + advantages
          if self.cfg["algorithm"]["normalize_advantage_per_minibatch"]:
            advantage_stats = (advantages.mean(), advantages.std())
          elif n == 0:
            advantage_stats = (advantages.mean(), advantages.std())

          advantages = (advantages - advantage_stats[0]) / (advantage_stats[1] + 1e-8)

        if self.cfg["algorithm"]["use_clipped_value_loss"]:
          value_clipped = batch["old_values"].detach() + (values.pred() - batch["old_values"].detach()).clamp(
              -self.cfg["algorithm"]["clip_param"], self.cfg["algorithm"]["clip_param"]
          )
          value_losses = (values.pred() - returns.unsqueeze(-1)).pow(2)
          value_losses_clipped = (value_clipped - returns.unsqueeze(-1)).pow(2)
          value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
          value_loss = torch.mean(values.loss(returns.unsqueeze(-1)))

        dist = self.policy(batch["policy_obs"], masks=batch["masks"], hidden_states=batch["hid_a"])
        actions_log_prob = dist.logp(batch["actions"]).sum(dim=-1)
        actor_loss = surrogate_loss(
          batch["old_actions_log_prob"], actions_log_prob, advantages,
          e_clip=self.cfg["algorithm"]["clip_param"]
        )

        bound_loss = (
          torch.clip(dist.pred() - 1.0, min=0.0).square().mean()
          + torch.clip(dist.pred() + 1.0, max=0.0).square().mean()
        )

        entropy = dist.entropy().sum(dim=-1)

        loss = (
          self.cfg["algorithm"]["value_loss_coef"] * value_loss
          + actor_loss
          + self.cfg["algorithm"]["bound_coef"] * bound_loss
          + self.cfg["algorithm"]["entropy_coef"] * entropy.mean()
        )
        if self.cfg["algorithm"]["symmetry"]:
          policy_obs_sym = {}
          for key, value in batch["policy_obs"].items():
            assert key in self.symmetry_groups[self.policy_key], f"{key} not in {self.symmetry_groups[self.policy_key]}"
            policy_obs_sym[key] = self.symmetry_groups[self.policy_key][key](self.env, value).detach()
          policy_obs_sym_normalized = self.normalize_obs({self.policy_key: policy_obs_sym})[self.policy_key]

          act_with_sym_obs = self.policy(policy_obs_sym_normalized, masks=batch["masks"], hidden_states=batch["hid_a"]).pred()
          act_sym = self.symmetry_groups["actions"]["actions"](self.env, dist.pred())
          mse_loss = torch.nn.MSELoss()
          symmetry_loss = mse_loss(act_with_sym_obs, act_sym.detach())
          if self.cfg["algorithm"]["symmetry_coef"] <= 0.0:
            symmetry_loss = symmetry_loss.detach()
          learn_step_agg.add({'symmetry_loss': symmetry_loss.item()})
          loss += self.cfg["algorithm"]["symmetry_coef"] * symmetry_loss.mean()

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value.parameters()), 1.0)
        self.policy_optimizer.step()
        self.value_optimizer.step()

        kl = torch.sum(dist.kl(batch["old_dist"]), axis=-1)
        kl_mean = torch.mean(kl)
        if "desired_kl" in self.cfg["policy"]:
          if kl_mean > self.cfg["policy"]["desired_kl"] * 2.0:
            self.policy_learning_rate = max(1e-5, self.policy_learning_rate / 1.5)
          elif kl_mean < self.cfg["policy"]["desired_kl"] / 2.0 and kl_mean > 0.0:
            self.policy_learning_rate = min(1e-2, self.policy_learning_rate * 1.5)
          for param_group in self.policy_optimizer.param_groups:
            param_group["lr"] = self.policy_learning_rate

        policy_stats = {}
        for k, v in self.policy.stats().items():
          policy_stats[f'policy/{k}_mean'] = v.mean()
          policy_stats[f'policy/{k}_std'] = v.std()
          policy_stats[f'policy/{k}_min'] = v.min()
          policy_stats[f'policy/{k}_max'] = v.max()

        learn_step_agg.add({
          "value_loss": value_loss.item(),
          "actor_loss": actor_loss.item(),
          "bound_loss": bound_loss.item(),
          "entropy": entropy.mean().item(),
          **policy_stats
        })

      # Update the observation normalizersesolicy_obs)
      self.obs_normalizers[self.policy_key].update(self.buffer[self.policy_key])
      self.obs_normalizers[self.value_key].update(self.buffer[self.value_key])

      return {
          **learn_step_agg.result(),
          "kl_mean": kl_mean.item(),
          "policy_lr": self.policy_learning_rate,
          "value_lr": self.value_learning_rate,
      }


  @timer.section("learn")
  def _learn_old(self, last_privileged_obs):
      policy_obs = self.buffer[self.policy_key]
      value_obs = self.buffer[self.value_key]
      traj_masks, hid_a, hid_c = None, None, None

      if self.policy.is_recurrent:
        policy_obs_split = pytree.tree_map(
          lambda x: models.split_and_pad_trajectories(x, self.buffer["dones"]),
          policy_obs,
        )
        policy_obs = pytree.tree_map(
          lambda x: x[0], policy_obs_split, is_leaf=lambda x: isinstance(x, tuple)
        )
        traj_masks = pytree.tree_map(
          lambda x: x[1], policy_obs_split, is_leaf=lambda x: isinstance(x, tuple)
        )
        traj_masks = list(traj_masks.values())[0]
        last_was_done = torch.zeros_like(self.buffer["dones"], dtype=torch.bool)
        last_was_done[1:] = self.buffer["dones"][:-1]
        last_was_done[0] = True
        last_was_done = last_was_done.permute(1, 0)
        hid_a = self.buffer["policy_hidden_states"]
        hid_a = [
          saved_hidden_states.permute(2, 0, 1, 3)[last_was_done].transpose(1, 0)
          for saved_hidden_states in hid_a
        ]

      if self.value.is_recurrent:
        value_obs_split = pytree.tree_map(
          lambda x: models.split_and_pad_trajectories(x, self.buffer["dones"]),
          value_obs,
        )
        value_obs = pytree.tree_map(
          lambda x: x[0],
          value_obs_split,
          is_leaf=lambda x: isinstance(x, tuple),
        )
        if not self.policy.is_recurrent:
          traj_masks = pytree.tree_map(
            lambda x: x[1], value_obs_split, is_leaf=lambda x: isinstance(x, tuple)
          )
          traj_masks = list(traj_masks.values())[0]
          last_was_done = torch.zeros_like(self.buffer["dones"], dtype=torch.bool)
          last_was_done[1:] = self.buffer["dones"][:-1]
          last_was_done[0] = True
          last_was_done = last_was_done.permute(1, 0)

        hid_c = self.buffer["value_hidden_states"]
        hid_c = [
          saved_hidden_states.permute(2, 0, 1, 3)[last_was_done].transpose(1, 0)
          for saved_hidden_states in hid_c
        ]

      normalized_obs = self.normalize_obs(
        {self.policy_key: policy_obs, self.value_key: value_obs}
      )
      policy_obs_normalized = normalized_obs[self.policy_key]
      value_obs_normalized = normalized_obs[self.value_key]

      with torch.no_grad():
        old_dist = self.policy(
          policy_obs_normalized, masks=traj_masks, hidden_states=hid_a
        )
        old_actions_log_prob = old_dist.logp(self.buffer["actions"]).sum(
          dim=-1
        )

      learn_step_agg = agg.Agg()
      for n in range(self.cfg["algorithm"]["num_learning_epochs"]):
        values = self.value(
          value_obs_normalized, masks=traj_masks, hidden_states=hid_c
        )
        with torch.no_grad():
          last_values = self.value(last_privileged_obs, update_state=False)
          self.buffer["rewards"][self.buffer["time_outs"]] = values.pred().squeeze(-1)[
            self.buffer["time_outs"]
          ]
          advantages = discount_values(
            self.buffer["rewards"],
            self.buffer["dones"] | self.buffer["time_outs"],
            values.pred().squeeze(-1),
            last_values.pred().squeeze(-1),
            self.cfg["algorithm"]["gamma"],
            self.cfg["algorithm"]["lam"],
          )
          returns = values.pred().squeeze(-1) + advantages
          if self.cfg["algorithm"]["normalize_advantage_per_minibatch"]:
            advantage_stats = (advantages.mean(), advantages.std())
          elif n == 0:
            advantage_stats = (advantages.mean(), advantages.std())

          advantages = (advantages - advantage_stats[0]) / (advantage_stats[1] + 1e-8)

        if self.cfg["algorithm"]["use_clipped_value_loss"]:
          value_clipped = self.buffer["values"].detach() + (values.pred() - self.buffer["values"].detach()).clamp(
              -self.cfg["algorithm"]["clip_param"], self.cfg["algorithm"]["clip_param"]
          )
          value_losses = (values.pred() - returns.unsqueeze(-1)).pow(2)
          value_losses_clipped = (value_clipped - returns.unsqueeze(-1)).pow(2)
          value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
          value_loss = torch.mean(values.loss(returns.unsqueeze(-1)))

        dist = self.policy(policy_obs_normalized, masks=traj_masks, hidden_states=hid_a)
        actions_log_prob = dist.logp(self.buffer["actions"]).sum(dim=-1)
        actor_loss = surrogate_loss(
          old_actions_log_prob, actions_log_prob, advantages,
          e_clip=self.cfg["algorithm"]["clip_param"]
        )

        bound_loss = (
          torch.clip(dist.pred() - 1.0, min=0.0).square().mean()
          + torch.clip(dist.pred() + 1.0, max=0.0).square().mean()
        )

        entropy = dist.entropy().sum(dim=-1)

        loss = (
          self.cfg["algorithm"]["value_loss_coef"] * value_loss
          + actor_loss
          + self.cfg["algorithm"]["bound_coef"] * bound_loss
          + self.cfg["algorithm"]["entropy_coef"] * entropy.mean()
        )
        if self.cfg["algorithm"]["symmetry"]:
          policy_obs_sym = {}
          for key, value in policy_obs.items():
            assert key in self.symmetry_groups[self.policy_key], f"{key} not in {self.symmetry_groups[self.policy_key]}"
            policy_obs_sym[key] = self.symmetry_groups[self.policy_key][key](self.env, value).detach()
          policy_obs_sym_normalized = self.normalize_obs({self.policy_key: policy_obs_sym})[self.policy_key]

          act_with_sym_obs = self.policy(policy_obs_sym_normalized, masks=traj_masks, hidden_states=hid_a).pred()
          act_sym = self.symmetry_groups["actions"]["actions"](self.env, dist.pred())
          mse_loss = torch.nn.MSELoss()
          symmetry_loss = mse_loss(act_with_sym_obs, act_sym.detach())
          if self.cfg["algorithm"]["symmetry_coef"] <= 0.0:
            symmetry_loss = symmetry_loss.detach()
          learn_step_agg.add({'symmetry_loss': symmetry_loss.item()})
          loss += self.cfg["algorithm"]["symmetry_coef"] * symmetry_loss.mean()

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value.parameters()), 1.0)
        self.policy_optimizer.step()
        self.value_optimizer.step()

        kl = torch.sum(dist.kl(old_dist), axis=-1)
        kl_mean = torch.mean(kl)
        if "desired_kl" in self.cfg["policy"]:
          if kl_mean > self.cfg["policy"]["desired_kl"] * 2.0:
            self.policy_learning_rate = max(1e-5, self.policy_learning_rate / 1.5)
          elif kl_mean < self.cfg["policy"]["desired_kl"] / 2.0 and kl_mean > 0.0:
            self.policy_learning_rate = min(1e-2, self.policy_learning_rate * 1.5)
          for param_group in self.policy_optimizer.param_groups:
            param_group["lr"] = self.policy_learning_rate

        policy_stats = {}
        for k, v in self.policy.stats().items():
          policy_stats[f'policy/{k}_mean'] = v.mean()
          policy_stats[f'policy/{k}_std'] = v.std()
          policy_stats[f'policy/{k}_min'] = v.min()
          policy_stats[f'policy/{k}_max'] = v.max()

        learn_step_agg.add({
          "value_loss": value_loss.item(),
          "actor_loss": actor_loss.item(),
          "bound_loss": bound_loss.item(),
          "entropy": entropy.mean().item(),
          **policy_stats
        })

      # Update the observation normalizersesolicy_obs)
      self.obs_normalizers[self.policy_key].update(self.buffer[self.policy_key])
      self.obs_normalizers[self.value_key].update(self.buffer[self.value_key])

      return {
          **learn_step_agg.result(),
          "kl_mean": kl_mean.item(),
          "policy_lr": self.policy_learning_rate,
          "value_lr": self.value_learning_rate,
      }

  def play(self):
    obs_dict = self.to_device(self.env.reset())
    obs_dict_normalized = self.normalize_obs(obs_dict)

    policy = models.get_policy_jitted(self.policy, self.cfg["policy"]["params"])
    # Alternatively can use torch.compile. Not sure if this is available on all machines.
    # opt_policy = torch.compile(self.policy)

    inference_time, step = 0., 0
    while True:
      with torch.no_grad():
        start = time.time()
        act = policy(obs_dict_normalized[self.policy_key])
        # act = opt_policy(obs_dict_normalized[self.policy_key]).pred()
        inference_time += time.time() - start
        obs_dict, _, _, _ = self.env.step(act)
        obs_dict = self.to_device(obs_dict)
        obs_dict_normalized = self.normalize_obs(obs_dict)

      step += 1
      if step % 100 == 0:
        print(f"Average inference time: {inference_time / step}")
        print(f"\t Per env: {inference_time / step / self.env.num_envs}")
        inference_time, step = 0., 0


  def interrupt_handler(self, signal, frame):
    print("\nInterrupt received, waiting for video to finish...")
    self.interrupt = True
