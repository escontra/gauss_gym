import os
import numpy as np
from typing import Any, Dict, List
import random
import time
import torch
import torch.utils._pytree as pytree
import pathlib

from legged_gym.rl import experience_buffer, recorder
from legged_gym.rl.env import vec_env
from legged_gym.rl.modules import models
from legged_gym.utils import agg, symmetry_groups, timer, when, math


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
    self.test = True
    self.env = env
    self.device = device
    self.cfg = cfg
    self._set_seed()
    self.learning_rate = self.cfg["algorithm"]["learning_rate"]
    self.obs_group_sizes = {
      'teacher_observations': self.env.obs_group_size_per_name("teacher_observations"),
      'student_observations': self.env.obs_group_size_per_name("student_observations"),
    }
    self.model = getattr(models, self.cfg["runner"]["policy_class_name"])(
      self.env.num_actions,
      self.env.obs_group_size_per_name("student_observations"),
      self.env.obs_group_size_per_name("teacher_observations"),
      **self.cfg["policy"]
    ).to(self.device)
    self.optimizer = torch.optim.Adam(
      self.model.parameters(), lr=self.learning_rate
    )

    if self.cfg["algorithm"]["symmetry"]:
      assert "symmetries" in self.cfg, "Need `symmetries` in config when symmetry is enabled. Look at a1/config.yaml for an example."
      self.symmetry_groups = {}
      for group_name in self.cfg["symmetries"]:
        self.symmetry_groups[group_name] = {}
        for symmetry in self.cfg["symmetries"][group_name]["symmetries"]:
          symmetry_modifier = getattr(symmetry_groups, symmetry)
          self.symmetry_groups[group_name][symmetry_modifier.observation.name] = symmetry_modifier
      assert "student_observations" in self.symmetry_groups
      assert "actions" in self.symmetry_groups

    self.buffer = experience_buffer.ExperienceBuffer(
      self.cfg["runner"]["num_steps_per_env"],
      self.env.num_envs,
      self.device,
    )
    self.buffer.add_buffer(
      "obses", self.env.obs_group_size_per_name("student_observations")
    )
    self.buffer.add_buffer(
      "privileged_obses",
      self.env.obs_group_size_per_name("teacher_observations"),
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
    self.model.load_state_dict(model_dict["model"], strict=False)
    try:
      self.optimizer.load_state_dict(model_dict["optimizer"])
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
    self.recorder = recorder.Recorder(log_dir, self.cfg, self.env.deploy_config(), self.obs_group_sizes)
    obs, privileged_obs = self.env.reset()
    obs = self.to_device(obs)
    privileged_obs = self.to_device(privileged_obs)

    if init_at_random_ep_len:
        self.env.episode_length_buf = torch.randint_like(
          self.env.episode_length_buf,
          high=int(self.env.max_episode_length))

    # Needed to initialize hidden states.
    self.model.act(obs)
    self.model.est_value(privileged_obs)

    if self.model.is_recurrent:
      self.buffer.add_hidden_state_buffers(self.model.get_hidden_states())

    for it in range(num_learning_iterations):
      start = time.time()
      # within horizon_length, env.step() is called with same act
      for n in range(self.cfg["runner"]["num_steps_per_env"]):
        with timer.section("buffer_add_obs"):
          self.buffer.update_data("obses", n, obs)
          self.buffer.update_data("privileged_obses", n, privileged_obs)
          if self.model.is_recurrent:
            self.buffer.update_hidden_state_buffers(
              n, self.model.get_hidden_states()
            )
        with timer.section("model_act"):
          with torch.no_grad():
            dist = self.model.act(obs)
            value = self.model.est_value(privileged_obs)
            act = dist.sample()
        with timer.section("env_step"):
          obs, privileged_obs, rew, done, infos = self.env.step(act)
          obs, privileged_obs, rew, done = self.to_device((obs, privileged_obs, rew, done))
        self.model.reset(done)
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



      learn_stats = self._learn(last_privileged_obs=privileged_obs)
      self.learn_agg.add(learn_stats)

      if self.should_log(it):
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
        self.recorder.save(
          {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
          },
          it + 1,
        )
      print(
        "epoch: {}/{} - {}s.".format(
          it + 1, num_learning_iterations, time.time() - start
        )
      )
      start = time.time()

  @timer.section("learn")
  def _learn(self, last_privileged_obs):
      all_obses = self.buffer["obses"]
      all_privileged_obses = self.buffer["privileged_obses"]
      if self.model.is_recurrent:
        obses_split = pytree.tree_map(
          lambda x: models.split_and_pad_trajectories(x, self.buffer["dones"]),
          all_obses,
        )
        all_obses = pytree.tree_map(
          lambda x: x[0], obses_split, is_leaf=lambda x: isinstance(x, tuple)
        )
        traj_masks = pytree.tree_map(
          lambda x: x[1], obses_split, is_leaf=lambda x: isinstance(x, tuple)
        )
        traj_masks = list(traj_masks.values())[0]

        privileged_obses_split = pytree.tree_map(
          lambda x: models.split_and_pad_trajectories(x, self.buffer["dones"]),
          all_privileged_obses,
        )
        all_privileged_obses = pytree.tree_map(
          lambda x: x[0],
          privileged_obses_split,
          is_leaf=lambda x: isinstance(x, tuple),
        )

        last_was_done = torch.zeros_like(self.buffer["dones"], dtype=torch.bool)
        last_was_done[1:] = self.buffer["dones"][:-1]
        last_was_done[0] = True
        last_was_done = last_was_done.permute(1, 0)
        hid_a = self.buffer["hid_a"]
        hid_c = self.buffer["hid_c"]
        hid_a = [
          saved_hidden_states.permute(2, 0, 1, 3)[last_was_done].transpose(1, 0)
          for saved_hidden_states in hid_a
        ]
        hid_c = [
          saved_hidden_states.permute(2, 0, 1, 3)[last_was_done].transpose(1, 0)
          for saved_hidden_states in hid_c
        ]
      else:
        traj_masks = None
        hid_a = None
        hid_c = None

      with torch.no_grad():
        old_dist = self.model.act(
          all_obses, masks=traj_masks, hidden_states=hid_a
        )
        old_actions_log_prob = old_dist.logp(self.buffer["actions"]).sum(
          dim=-1
        )

      mean_value_loss = 0
      mean_actor_loss = 0
      mean_bound_loss = 0
      mean_entropy = 0
      mean_symmetry_loss = 0
      for n in range(self.cfg["algorithm"]["num_learning_epochs"]):
        values = self.model.est_value(
          all_privileged_obses, masks=traj_masks, hidden_states=hid_c
        )
        with torch.no_grad():
          last_values = self.model.est_value(last_privileged_obs, upd_state=False)
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

        dist = self.model.act(all_obses, masks=traj_masks, hidden_states=hid_a)
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
          all_obses_sym = {}
          for key, value in all_obses.items():
            assert key in self.symmetry_groups["student_observations"], f"{key} not in {self.symmetry_groups['student_observations']}"
            all_obses_sym[key] = self.symmetry_groups["student_observations"][key](self.env, value).detach()

          act_with_sym_obs = self.model.act(all_obses_sym, masks=traj_masks, hidden_states=hid_a).pred()
          act_sym = self.symmetry_groups["actions"]["actions"](self.env, dist.pred())
          mse_loss = torch.nn.MSELoss()
          symmetry_loss = mse_loss(act_with_sym_obs, act_sym.detach())
          if self.cfg["algorithm"]["symmetry_coef"] <= 0.0:
            symmetry_loss = symmetry_loss.detach()
          mean_symmetry_loss += symmetry_loss.item()
          loss += self.cfg["algorithm"]["symmetry_coef"] * symmetry_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        with torch.no_grad():
          kl = torch.sum(dist.kl(old_dist), axis=-1)
          kl_mean = torch.mean(kl)
          if kl_mean > self.cfg["algorithm"]["desired_kl"] * 2.0:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
          elif kl_mean < self.cfg["algorithm"]["desired_kl"] / 2.0 and kl_mean > 0.0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

        mean_value_loss += value_loss.item()
        mean_actor_loss += actor_loss.item()
        mean_bound_loss += bound_loss.item()
        mean_entropy += entropy.mean()
      mean_value_loss /= self.cfg["algorithm"]["num_learning_epochs"]
      mean_actor_loss /= self.cfg["algorithm"]["num_learning_epochs"]
      mean_bound_loss /= self.cfg["algorithm"]["num_learning_epochs"]
      mean_entropy /= self.cfg["algorithm"]["num_learning_epochs"]
      mean_symmetry_loss /= self.cfg["algorithm"]["num_learning_epochs"]
      return {
          "value_loss": mean_value_loss,
          "actor_loss": mean_actor_loss,
          "bound_loss": mean_bound_loss,
          "symmetry_loss": mean_symmetry_loss,
          "entropy": mean_entropy.item(),
          "kl_mean": kl_mean.item(),
          "lr": self.learning_rate,
      }

  def play(self):
    obs, _ = self.env.reset()
    obs = self.to_device(obs)

    memory_module = self.model.memory_a
    actor = self.model.actor
    actor_head = self.model.actor_head.mean_net
    mlp_keys = self.model.mlp_keys_a
    cnn_keys = self.model.cnn_keys_a
    cnn_model = self.model.cnn_a

    @torch.jit.script
    def policy(observations: Dict[str, torch.Tensor], mlp_keys: List[str], cnn_keys: List[str], symlog_inputs: bool) -> torch.Tensor:
        if symlog_inputs:
          features = torch.cat([math.symlog(observations[k]) for k in mlp_keys], dim=-1)
        else:
          features = torch.cat([observations[k] for k in mlp_keys], dim=-1)
        if cnn_keys:
          cnn_features = []
          for k in cnn_keys:
            cnn_obs = observations[k]
            if cnn_obs.shape[-1] in [1, 3]:
              cnn_obs = permute_cnn_obs(cnn_obs)
            if cnn_obs.dtype == torch.uint8:
              cnn_obs = cnn_obs.float() / 255.0

            orig_batch_size = cnn_obs.shape[0]
            cnn_obs = cnn_obs.reshape(-1, cnn_obs.shape[-3], cnn_obs.shape[-2], cnn_obs.shape[-1])  # Shape: [M*N*L*O, C, H, W]
            cnn_feat = cnn_model(cnn_obs)
            cnn_feat = cnn_feat.reshape(orig_batch_size, cnn_feat.shape[-1])
            cnn_features.append(cnn_feat)

          cnn_features = torch.cat(cnn_features, dim=-1)
          features = torch.cat([features, cnn_features], dim=-1)
        input_a = memory_module(features, None, None)
        dist = actor_head(actor(input_a.squeeze(0)))
        return dist
        # return dist.pred()


    while True:
      with torch.no_grad():
        obs = self.filter_nans(obs)
        act = policy(obs, mlp_keys, cnn_keys, symlog_inputs=self.cfg["policy"]["symlog_inputs"])
        obs, _, _, _, _ = self.env.step(act)
        obs = self.to_device(obs)

  def interrupt_handler(self, signal, frame):
    print("\nInterrupt received, waiting for video to finish...")
    self.interrupt = True


def permute_cnn_obs(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 5:
        return x.permute(0, 1, 4, 2, 3)  # [B, T, H, W, C] -> [B, T, C, H, W]
    elif x.dim() == 4:
        return x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    else:
        return x  # or raise an error if that's not expected
