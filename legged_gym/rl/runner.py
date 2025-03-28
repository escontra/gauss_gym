import os
import numpy as np
from typing import Dict, List
import random
import time
import torch
import pickle
import wandb
import torch.nn.functional as F
import torch.utils._pytree as pytree
import pathlib
from torch.utils.tensorboard import SummaryWriter

from legged_gym.rl.env import vec_env
from legged_gym.rl.modules import models
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
from legged_gym.utils.helpers import class_to_dict

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


class Recorder:
  def __init__(self, log_dir: pathlib.Path, env_cfg, cfg: LeggedRobotCfgPPO, obs_group_sizes):
    self.env_cfg = env_cfg
    self.cfg = cfg
    self.log_dir = log_dir
    self.obs_group_sizes = obs_group_sizes
    self.initialized = False

  def maybe_init(self):
    if self.initialized:
      return
    print(f"Recording to: {self.log_dir}")
    self.log_dir.mkdir(parents=True, exist_ok=True)
    self.model_dir = self.log_dir / "nn"
    self.model_dir.mkdir(parents=True, exist_ok=True)
    self.writer = SummaryWriter(self.log_dir / "summaries")
    if self.cfg.runner.use_wandb:
      wandb.init(
        project=self.cfg.task_name,
        dir=self.log_dir,
        name=self.log_dir.name,
        notes=self.cfg.description,
        config=self.cfg,
      )

    self.episode_statistics = {}
    self.last_episode = {}
    self.last_episode["steps"] = []
    self.episode_steps = None

    with open(self.log_dir / "train_config.pkl", "wb") as file:
      pickle.dump(self.cfg, file)

    with open(self.log_dir / "train_config_dict.pkl", "wb") as file:
      pickle.dump(class_to_dict(self.cfg), file)

    with open(self.log_dir / "env_config.pkl", "wb") as file:
      pickle.dump(self.env_cfg, file)

    with open(self.log_dir / "env_config_dict.pkl", "wb") as file:
      pickle.dump(class_to_dict(self.env_cfg), file)

    with open(self.log_dir / "obs_group_sizes.pkl", "wb") as file:
      pickle.dump(self.obs_group_sizes, file)
    self.initialized = True

  def record_episode_statistics(self, done, ep_info, it, discount_factor_dict={}, write_record=False):
    self.maybe_init()
    if self.episode_steps is None:
      self.episode_steps = torch.zeros_like(done, dtype=int)
    else:
      self.episode_steps += 1

    for key, value in ep_info.items():
      if self.episode_statistics.get(key) is None:
        self.episode_statistics[key] = torch.zeros_like(value)
      discount_factor = discount_factor_dict.get(key, 1.0)
      discount_factor = discount_factor ** self.episode_steps
      self.episode_statistics[key] += value * discount_factor
      if self.last_episode.get(key) is None:
        self.last_episode[key] = []
      for done_value in self.episode_statistics[key][done]:
        self.last_episode[key].append(done_value.item())
      self.episode_statistics[key][done] = 0

    for val in self.episode_steps[done]:
      self.last_episode["steps"].append(val.item())
    self.episode_steps[done] = 0

    if write_record:
      for key in self.last_episode.keys():
        path = ("" if key == "steps" or key == "reward" else "episode/") + key
        value = self._mean(self.last_episode[key])
        self.writer.add_scalar(path, value, it)
        if self.cfg.runner.use_wandb:
          wandb.log({path: value}, step=it)
        self.last_episode[key].clear()

  def record_statistics(self, statistics, it):
    self.maybe_init()
    for key, value in statistics.items():
      self.writer.add_scalar(key, float(value), it)
      if self.cfg.runner.use_wandb:
        wandb.log({key: float(value)}, step=it)

  def save(self, model_dict, it):
    self.maybe_init()
    path = self.model_dir / f"model_{it}.pth"
    print("Saving model to {}".format(path))
    torch.save(model_dict, path)

  def _mean(self, data):
    if len(data) == 0:
      return 0.0
    else:
      return sum(data) / len(data)


class ExperienceBuffer:
  def __init__(self, horizon_length, num_envs, device):
    self.tensor_dict = {}
    self.horizon_length = horizon_length
    self.num_envs = num_envs
    self.device = device

  def add_buffer(self, name, shape, dtype=None):
    if isinstance(shape, tuple):
      self.tensor_dict[name] = torch.zeros(
        self.horizon_length,
        self.num_envs,
        *shape,
        dtype=dtype,
        device=self.device,
      )
    elif isinstance(shape, dict):
      self.tensor_dict[name] = {}
      for obs_name, obs_group in shape.items():
        self.tensor_dict[name][obs_name] = torch.zeros(
          self.horizon_length,
          self.num_envs,
          *obs_group[0],
          dtype=obs_group[1],
          device=self.device,
        )

  def add_hidden_state_buffers(self, hidden_states):
    if hidden_states is None or hidden_states == (None, None):
      return
    # make a tuple out of GRU hidden state sto match the LSTM format
    hid_a = (
      hidden_states[0]
      if isinstance(hidden_states[0], tuple)
      else (hidden_states[0],)
    )
    hid_c = (
      hidden_states[1]
      if isinstance(hidden_states[1], tuple)
      else (hidden_states[1],)
    )

    self.tensor_dict["hid_a"] = [
      torch.zeros(self.horizon_length, *hid_a[i].shape, device=self.device)
      for i in range(len(hid_a))
    ]
    self.tensor_dict["hid_c"] = [
      torch.zeros(self.horizon_length, *hid_c[i].shape, device=self.device)
      for i in range(len(hid_c))
    ]

  def update_hidden_state_buffers(self, idx, hidden_states):
    if hidden_states is None or hidden_states == (None, None):
      return

    # make a tuple out of GRU hidden state sto match the LSTM format
    hid_a = (
      hidden_states[0]
      if isinstance(hidden_states[0], tuple)
      else (hidden_states[0],)
    )
    hid_c = (
      hidden_states[1]
      if isinstance(hidden_states[1], tuple)
      else (hidden_states[1],)
    )

    for i in range(len(hid_a)):
      self.tensor_dict["hid_a"][i][idx].copy_(hid_a[i].clone().detach())
      self.tensor_dict["hid_c"][i][idx].copy_(hid_c[i].clone().detach())

  def update_data(self, name, idx, data):
    if isinstance(data, dict):
      for k, v in data.items():
        self.tensor_dict[name][k][idx].copy_(v)
    else:
      self.tensor_dict[name][idx].copy_(data)

  def __len__(self):
    return len(self.tensor_dict)

  def __getitem__(self, buf_name):
    return self.tensor_dict[buf_name]

  def keys(self):
    return self.tensor_dict.keys()


class Runner:
  def __init__(self, env: vec_env.VecEnv, train_cfg: LeggedRobotCfgPPO, log_dir: pathlib.Path, device="cpu"):
    self.test = True
    self.env = env
    self.device = device
    self.log_dir = log_dir
    self.cfg = self.env.cfg
    self.train_cfg = train_cfg
    self._set_seed()
    self.learning_rate = self.train_cfg.algorithm.learning_rate
    self.obs_group_sizes = {
      'teacher_observations': self.env.obs_group_size_per_name("teacher_observations"),
      'student_observations': self.env.obs_group_size_per_name("student_observations"),
    }
    self.model = getattr(models, self.train_cfg.runner.policy_class_name)(
      self.env.num_actions,
      self.env.obs_group_size_per_name("student_observations"),
      self.env.obs_group_size_per_name("teacher_observations"),
      self.train_cfg.policy.init_noise_std,
      self.train_cfg.policy.mu_activation,
    ).to(self.device)
    self.optimizer = torch.optim.Adam(
      self.model.parameters(), lr=self.learning_rate
    )

    self.buffer = ExperienceBuffer(
      self.train_cfg.runner.num_steps_per_env,
      self.env.num_envs,
      self.device,
    )
    self.buffer.add_buffer("actions", (self.env.num_actions,))
    self.buffer.add_buffer(
      "obses", self.env.obs_group_size_per_name("student_observations")
    )
    self.buffer.add_buffer(
      "privileged_obses",
      self.env.obs_group_size_per_name("teacher_observations"),
    )
    self.buffer.add_buffer("rewards", ())
    self.buffer.add_buffer("dones", (), dtype=bool)
    self.buffer.add_buffer("time_outs", (), dtype=bool)

  def _set_seed(self):
    if self.train_cfg.seed == -1:
      self.train_cfg.seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(self.train_cfg.seed))

    random.seed(self.train_cfg.seed)
    np.random.seed(self.train_cfg.seed)
    torch.manual_seed(self.train_cfg.seed)
    os.environ["PYTHONHASHSEED"] = str(self.train_cfg.seed)
    torch.cuda.manual_seed(self.train_cfg.seed)
    torch.cuda.manual_seed_all(self.train_cfg.seed)

  def load(self, resume_root: pathlib.Path):
    if not self.train_cfg.runner.resume:
      return

    load_run = self.train_cfg.runner.load_run
    checkpoint = self.train_cfg.runner.checkpoint
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

  def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    self.recorder = Recorder(self.log_dir, self.cfg, self.train_cfg, self.obs_group_sizes)
    obs, privileged_obs = self.env.reset()
    obs = self.to_device(obs)
    privileged_obs = self.to_device(privileged_obs)

    if hasattr(self.train_cfg.algorithm, "clip_min_std"):
      # clip_min_std = torch.tensor(self.train_cfg["algorithm"]["clip_min_std"], device=self.device) if isinstance(self.train_cfg["algorithm"]["clip_min_std"], (tuple, list)) else self.train_cfg["algorithm"]["clip_min_std"]
      clip_min_std = torch.tensor(
        self.train_cfg.algorithm.clip_min_std, device=self.device
      )
    else:
      clip_min_std = None

    # Needed to initialize hidden states.
    self.model.act(obs)
    self.model.est_value(privileged_obs)

    if self.model.is_recurrent:
      self.buffer.add_hidden_state_buffers(self.model.get_hidden_states())

    for it in range(num_learning_iterations):
      start = time.time()
      # within horizon_length, env.step() is called with same act
      for n in range(self.train_cfg.runner.num_steps_per_env):
        self.buffer.update_data("obses", n, obs)
        self.buffer.update_data("privileged_obses", n, privileged_obs)
        if self.model.is_recurrent:
          self.buffer.update_hidden_state_buffers(
            n, self.model.get_hidden_states()
          )
        with torch.no_grad():
          dist = self.model.act(obs)
          value = self.model.est_value(privileged_obs)
          act = dist.sample()
        obs, privileged_obs, rew, done, infos = self.env.step(act)
        obs, privileged_obs, rew, done = (
          self.to_device(obs),
          self.to_device(privileged_obs),
          rew.to(self.device),
          done.to(self.device),
        )
        self.model.reset(done)
        self.buffer.update_data("actions", n, act)
        self.buffer.update_data("rewards", n, rew)
        self.buffer.update_data("dones", n, done)
        self.buffer.update_data(
          "time_outs", n, infos["time_outs"].to(self.device)
        )
        bootstrapped_rew = torch.where(done, 0., rew)
        bootstrapped_rew = torch.where(infos["time_outs"], value, bootstrapped_rew)
        ep_info = {
          "reward": rew,
          "return": bootstrapped_rew,
        }
        self.recorder.record_statistics(
          {f"episode/{k}": v for k, v in infos["episode"].items()},
          it,
        )
        self.recorder.record_episode_statistics(
          done,
          ep_info,
          it,
          discount_factor_dict={"return": self.train_cfg.algorithm.gamma},
          write_record=n == (self.train_cfg.runner.num_steps_per_env - 1),
        )

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
        old_actions_log_prob = old_dist.log_prob(self.buffer["actions"]).sum(
          dim=-1
        )

      mean_value_loss = 0
      mean_actor_loss = 0
      mean_bound_loss = 0
      mean_entropy = 0
      for n in range(self.train_cfg.algorithm.num_learning_epochs):
        values = self.model.est_value(
          all_privileged_obses, masks=traj_masks, hidden_states=hid_c
        )
        with torch.no_grad():
          last_values = self.model.est_value(privileged_obs, upd_state=False)
          self.buffer["rewards"][self.buffer["time_outs"]] = values[
            self.buffer["time_outs"]
          ]
          advantages = discount_values(
            self.buffer["rewards"],
            self.buffer["dones"] | self.buffer["time_outs"],
            values,
            last_values,
            self.train_cfg.algorithm.gamma,
            self.train_cfg.algorithm.lam,
          )
          returns = values + advantages
          advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
          )
        value_loss = F.mse_loss(values, returns)

        dist = self.model.act(all_obses, masks=traj_masks, hidden_states=hid_a)
        actions_log_prob = dist.log_prob(self.buffer["actions"]).sum(dim=-1)
        actor_loss = surrogate_loss(
          old_actions_log_prob, actions_log_prob, advantages
        )

        bound_loss = (
          torch.clip(dist.loc - 1.0, min=0.0).square().mean()
          + torch.clip(dist.loc + 1.0, max=0.0).square().mean()
        )

        entropy = dist.entropy().sum(dim=-1)

        loss = (
          value_loss
          + actor_loss
          + self.train_cfg.algorithm.bound_coef * bound_loss
          + self.train_cfg.algorithm.entropy_coef * entropy.mean()
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        with torch.no_grad():
          kl = torch.sum(
            torch.log(dist.scale / old_dist.scale)
            + 0.5
            * (
              torch.square(old_dist.scale)
              + torch.square(dist.loc - old_dist.loc)
            )
            / torch.square(dist.scale)
            - 0.5,
            axis=-1,
          )
          kl_mean = torch.mean(kl)
          if kl_mean > self.train_cfg.algorithm.desired_kl * 2:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
          elif kl_mean < self.train_cfg.algorithm.desired_kl / 2:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

        mean_value_loss += value_loss.item()
        mean_actor_loss += actor_loss.item()
        mean_bound_loss += bound_loss.item()
        mean_entropy += entropy.mean()
      mean_value_loss /= self.train_cfg.algorithm.num_learning_epochs
      mean_actor_loss /= self.train_cfg.algorithm.num_learning_epochs
      mean_bound_loss /= self.train_cfg.algorithm.num_learning_epochs
      mean_entropy /= self.train_cfg.algorithm.num_learning_epochs
      self.recorder.record_statistics(
        {
          "value_loss": mean_value_loss,
          "actor_loss": mean_actor_loss,
          "bound_loss": mean_bound_loss,
          "entropy": mean_entropy,
          "kl_mean": kl_mean,
          "lr": self.learning_rate,
          # "curriculum/mean_lin_vel_level": self.env.mean_lin_vel_level,
          # "curriculum/mean_ang_vel_level": self.env.mean_ang_vel_level,
          # "curriculum/max_lin_vel_level": self.env.max_lin_vel_level,
          # "curriculum/max_ang_vel_level": self.env.max_ang_vel_level,
        },
        it,
      )

      if clip_min_std is not None:
        self.model.clip_std(min=clip_min_std)

      if (it + 1) % self.train_cfg.runner.save_interval == 0:
        self.recorder.save(
          {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            # "curriculum": self.env.curriculum_prob,
          },
          it + 1,
        )
      print(
        "epoch: {}/{} - {}s.".format(
          it + 1, num_learning_iterations, time.time() - start
        )
      )
      start = time.time()

  def play(self):
    obs, _ = self.env.reset()
    obs = self.to_device(obs)

    memory_module = self.model.memory_a
    actor = self.model.actor
    mlp_keys = self.model.mlp_keys_a
    cnn_keys = self.model.cnn_keys_a
    cnn_model = self.model.cnn_a

    @torch.jit.script
    def policy(observations: Dict[str, torch.Tensor], mlp_keys: List[str], cnn_keys: List[str]) -> torch.Tensor:
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
        actions = actor(input_a.squeeze(0))
        return actions


    while True:
      with torch.no_grad():
        obs = self.filter_nans(obs)
        act = policy(obs, mlp_keys, cnn_keys)
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
