import os
import numpy as np
from typing import Any, Dict
import random
import time
import torch
import torch.utils._pytree as pytree
import pathlib
import copy

from legged_gym.rl import experience_buffer, loss, recorder
from legged_gym.rl.env import vec_env
from legged_gym.rl.modules import models
from legged_gym.utils import agg, observation_groups, symmetry_groups, timer, when, space

torch.backends.cuda.matmul.allow_tf32 = True


class Runner:
  def __init__(self, env: vec_env.VecEnv, cfg: Dict[str, Any], device="cpu"):
    self.env = env
    self.device = device
    self.cfg = cfg
    self._set_seed()
    self.image_encoder_key = self.cfg["image_encoder"]["obs_key"]
    self.image_encoder_obs_space = self.env.obs_space()[self.image_encoder_key]
    self.policy_key = self.cfg["policy"]["obs_key"]
    self.policy_obs_space = self.env.obs_space()[self.policy_key]
    self.value_key = self.cfg["value"]["obs_key"]
    self.value_obs_space = self.env.obs_space()[self.value_key]

    # Symmetry-augmented observations to compute during the environment step, as opposed to
    # computing from the replay buffer.
    self.symm_key = "symms"
    self.symm_obs_space = {}

    # Image encoder.
    if len(self.env.obs_space()[self.image_encoder_key].keys()) > 0:
      self.image_encoder_learning_rate = self.cfg["image_encoder"]["learning_rate"]
      reconstruct_space = None
      reconstruct_head = None
      if self.cfg["image_encoder"]["reconstruct_observations"] is not None:
        reconstruct_space = {}
        reconstruct_head = {}
        for key in self.cfg["image_encoder"]["reconstruct_observations"]:
          obs_group, obs_name = key.split('/')
          obs_name = getattr(observation_groups, obs_name).name
          reconstruct_space[f'{obs_group}/{obs_name}'] = self.env.obs_space()[obs_group][obs_name]
          reconstruct_head[f'{obs_group}/{obs_name}'] = {
            'output_type': 'mse',
            'outscale': 1.0
          }
      self.image_encoder: models.RecurrentModel = getattr(
        models, self.cfg["image_encoder"]["class_name"])(
        reconstruct_space,
        self.image_encoder_obs_space,
        head=reconstruct_head,
        **self.cfg["image_encoder"]["params"]
      ).to(self.device)
    else:
      self.image_encoder = None

    self.use_image_encoder_features = True
    if self.use_image_encoder_features:
      image_encoder_space = space.Space(np.float32, (self.image_encoder.rnn_state_size,), -np.inf, np.inf)
      self.policy_obs_space[self.image_encoder_key] = image_encoder_space
      self.value_obs_space[self.image_encoder_key] = image_encoder_space
      self.symm_obs_space[self.image_encoder_key] = image_encoder_space

    self.policy_learning_rate = self.cfg["policy"]["learning_rate"]
    self.policy: models.RecurrentModel = getattr(
      models, self.cfg["policy"]["class_name"])(
      self.env.action_space(),
      self.policy_obs_space,
      **self.cfg["policy"]["params"]
    ).to(self.device)
    # For KL.
    self.old_policy: models.RecurrentModel = copy.deepcopy(self.policy)
    for param in self.old_policy.parameters():
      param.requires_grad = False

    # Value.
    self.value_learning_rate = self.cfg["value"]["learning_rate"]
    self.value: models.RecurrentModel = getattr(
      models, self.cfg["value"]["class_name"])(
      {'value': space.Space(np.float32, (1,), -np.inf, np.inf)},
      self.value_obs_space,
      **self.cfg["value"]["params"]
    ).to(self.device)

    self.global_num_updates = 0

    # Optimizers.
    self.train_image_encoder = when.Every(self.cfg["image_encoder"]["train_every"])
    self.image_encoder_optimizer = torch.optim.Adam(
      self.image_encoder.parameters(), lr=self.image_encoder_learning_rate
    )
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

  def _get_symmetry_fn(self, obs_group, obs_name):
    assert obs_group in self.symmetry_groups, f"{obs_group} not in {self.symmetry_groups}"
    symmetry_fns = self.symmetry_groups[obs_group]
    assert obs_name in symmetry_fns, f"{obs_name} not in {symmetry_fns}"
    symmetry_fn = symmetry_fns[obs_name]
    return lambda val: symmetry_fn(self.env, val)

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
    self.image_encoder.load_state_dict(model_dict["image_encoder"], strict=True)
    self.policy.load_state_dict(model_dict["policy"], strict=True)
    self.value.load_state_dict(model_dict["value"], strict=True)
    try:
      self.image_encoder_optimizer.load_state_dict(model_dict["image_encoder_optimizer"])
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
    self.completion_agg = agg.Agg()
    self.episode_agg = agg.Agg()
    self.learn_agg = agg.Agg()
    self.action_agg = agg.Agg()
    self.should_log = when.Clock(self.cfg["runner"]["log_every"])
    self.should_save = when.Clock(self.cfg["runner"]["save_every"])
    self.recorder = recorder.Recorder(
      log_dir, self.cfg, self.env.deploy_config(),
      {'obs_space': self.env.obs_space(),
       'action_space': self.env.action_space()})
    if self.cfg["runner"]["record_video"]:
      self.recorder.setup_recorder(self.env)

    self.image_encoder.train()
    self.policy.train()
    self.value.train()
    self.image_encoder.flatten_parameters()
    self.policy.flatten_parameters()
    self.old_policy.flatten_parameters()
    self.value.flatten_parameters()

    # Initialize hidden states and set random episode length.
    obs_dict = self.to_device(self.env.reset())
    symms = {}
    image_encoder_hidden_states = self.image_encoder.reset(torch.zeros(self.env.num_envs, dtype=torch.bool), None)
    image_encoder_hidden_states_sym = self.image_encoder.reset(torch.zeros(self.env.num_envs, dtype=torch.bool), None)
    policy_hidden_states = self.policy.reset(torch.zeros(self.env.num_envs, dtype=torch.bool), None)
    value_hidden_states = self.value.reset(torch.zeros(self.env.num_envs, dtype=torch.bool), None)
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

    self.buffer.add_buffer(self.image_encoder_key, self.image_encoder_obs_space)
    self.buffer.add_buffer(self.policy_key, self.policy_obs_space)
    self.buffer.add_buffer(self.value_key, self.value_obs_space)
    self.buffer.add_buffer(self.symm_key, self.symm_obs_space)
    self.buffer.add_buffer("actions", self.env.action_space())
    self.buffer.add_buffer("rewards", ())
    self.buffer.add_buffer("values", (1,))
    self.buffer.add_buffer("dones", (), dtype=bool)
    self.buffer.add_buffer("time_outs", (), dtype=bool)

    if self.image_encoder.is_recurrent:
      self.buffer.add_hidden_state_buffers(f"{self.image_encoder_key}_hidden_states", (image_encoder_hidden_states,))
    if self.policy.is_recurrent:
      self.buffer.add_hidden_state_buffers(f"{self.policy_key}_hidden_states", (policy_hidden_states,))
    if self.value.is_recurrent:
      self.buffer.add_hidden_state_buffers(f"{self.value_key}_hidden_states", (value_hidden_states,))

    for it in range(num_learning_iterations):
      start = time.time()
      for n in range(self.cfg["runner"]["num_steps_per_env"]):
        if self.cfg["runner"]["record_video"]:
          potential_images = {k: obs_dict[self.policy_key][k] for k in self.policy.cnn_keys}
          potential_images.update({k: obs_dict[self.image_encoder_key][k] for k in self.image_encoder.cnn_keys})
          self.recorder.record_statistics(
            self.recorder.maybe_record(self.env, image_features=potential_images),
            it * self.cfg["runner"]["num_steps_per_env"] * self.env.num_envs + n)
        with timer.section("buffer_add_obs"):
          if self.image_encoder.is_recurrent:
            self.buffer.update_hidden_state_buffers(
              f"{self.image_encoder_key}_hidden_states", n, (image_encoder_hidden_states,)
            )
          if self.policy.is_recurrent:
            self.buffer.update_hidden_state_buffers(
              f"{self.policy_key}_hidden_states", n, (policy_hidden_states,)
            )
          if self.value.is_recurrent:
            self.buffer.update_hidden_state_buffers(
              f"{self.value_key}_hidden_states", n, (value_hidden_states,)
            )
          with torch.no_grad():
            _, image_encoder_rnn_state, image_encoder_hidden_states = self.image_encoder(
              obs_dict[self.image_encoder_key],
              image_encoder_hidden_states
            )
            if self.use_image_encoder_features:
              obs_dict[self.policy_key][self.image_encoder_key] = image_encoder_rnn_state
              obs_dict[self.value_key][self.image_encoder_key] = image_encoder_rnn_state
              _, image_encoder_rnn_state_sym, image_encoder_hidden_states_sym = self.image_encoder(
                {k: self._get_symmetry_fn(self.image_encoder_key, k)(v) for k, v in obs_dict[self.image_encoder_key].items()},
                image_encoder_hidden_states_sym
              )
              symms[self.image_encoder_key] = image_encoder_rnn_state_sym

          self.buffer.update_data(self.image_encoder_key, n, obs_dict[self.image_encoder_key])
          self.buffer.update_data(self.policy_key, n, obs_dict[self.policy_key])
          self.buffer.update_data(self.value_key, n, obs_dict[self.value_key])
          self.buffer.update_data(self.symm_key, n, symms)
        with timer.section("model_act"):
          with torch.no_grad():
            dists, _, policy_hidden_states = self.policy(
              obs_dict[self.policy_key],
              policy_hidden_states
            )
            value, _, value_hidden_states = self.value(
              obs_dict[self.value_key],
              value_hidden_states
            )
            actions = {k: dist.sample() for k, dist in dists.items()}
        with timer.section("env_step"):
          # Scale actions to the environment range.
          actions_scaled = self.scale_actions(actions)
          for k, v in actions_scaled.items():
            self.action_agg.add(
              {f'{dof_name}_{k}': v[:, dof_idx].cpu().numpy()
               for dof_idx, dof_name in enumerate(self.env.dof_names)},
              agg='concat'
            )

          obs_dict, rew, done, infos = self.env.step(actions_scaled)
          obs_dict, rew, done = self.to_device((obs_dict, rew, done))

        image_encoder_hidden_states = self.image_encoder.reset(done, image_encoder_hidden_states)
        image_encoder_hidden_states_sym = self.image_encoder.reset(done, image_encoder_hidden_states_sym)
        policy_hidden_states = self.policy.reset(done, policy_hidden_states)
        value_hidden_states = self.value.reset(done, value_hidden_states)
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
          self.completion_agg.add(infos["completion_counter"])
          self.episode_agg.add(self.recorder.record_episode_statistics(
            done,
            {"reward": rew, "return": bootstrapped_rew},
            it,
            discount_factor_dict={"return": self.cfg["algorithm"]["gamma"]},
            write_record=n == (self.cfg["runner"]["num_steps_per_env"] - 1),
          ))

      with torch.no_grad():
        _, last_image_encoder_rnn_state, _ = self.image_encoder(
          obs_dict[self.image_encoder_key],
          image_encoder_hidden_states
        )
      last_privileged_obs = {**obs_dict[self.value_key], self.image_encoder_key: last_image_encoder_rnn_state}
      # We skip the first gradient update to initialize the observation normalizers.
      learn_stats = self._learn(last_privileged_obs, last_value_hidden_states=value_hidden_states, is_first=it == 0)
      self.learn_agg.add(learn_stats)

      if self.should_log(it):
        with timer.section("logger_save"):
          step_stats = {f"step/{k}": v for k, v in self.step_agg.result().items()}
          learn_stats = {f"learn/{k}": v for k, v in self.learn_agg.result().items()}
          timer_stats = {f"timer/{k}": v for k, v in timer.stats().items()}
          episode_stats = {f"episode/{k}": v for k, v in self.episode_agg.result().items()}
          action_stats = {f"action/{k}": v for k, v in self.action_agg.result().items()}
          completion_stats = {f"completion/{k}": v for k, v in self.completion_agg.result().items()}
          self.recorder.record_statistics(
            {
              **step_stats,
              **learn_stats,
              **timer_stats,
              **episode_stats,
              **action_stats,
              **completion_stats,
            },
            it * self.cfg["runner"]["num_steps_per_env"] * self.env.num_envs
          )

      if self.should_save(it):
        with timer.section("model_save"):
          self.recorder.save( {
              "policy": self.policy.state_dict(),
              "value": self.value.state_dict(),
              "image_encoder": self.image_encoder.state_dict(),
              "policy_optimizer": self.policy_optimizer.state_dict(),
              "value_optimizer": self.value_optimizer.state_dict(),
              "image_encoder_optimizer": self.image_encoder_optimizer.state_dict(),
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
  def _learn(self, last_privileged_obs, last_value_hidden_states, is_first):
      learn_step_agg = agg.Agg()
      if is_first:
        # Skip the first gradient update to initialize the observation normalizers.
        with torch.no_grad():
          self.policy.update_normalizer(self.buffer[self.policy_key])
          self.value.update_normalizer(self.buffer[self.value_key])
        return {}

      self.old_policy.load_state_dict(self.policy.state_dict())
      for batch in self.buffer.reccurent_mini_batch_generator(
        self.cfg["algorithm"]["num_learning_epochs"],
        self.cfg["algorithm"]["num_mini_batches"],
        self.old_policy,
        self.value,
        last_privileged_obs,
        last_value_hidden_states,
        self.image_encoder_key,
        self.policy_key,
        self.value_key,
        "dones",
        self.cfg["algorithm"]["symmetry"],
        self._get_symmetry_fn,
        self.symm_key
      ):
        # Value loss.
        value_loss, advantages, metrics = loss.value_loss(
          batch,
          self.value,
          self.value_key,
          self.cfg["algorithm"]["gamma"],
          self.cfg["algorithm"]["lam"],
          self.cfg["algorithm"]["use_clipped_value_loss"],
          self.cfg["algorithm"]["clip_param"]
        )
        total_loss = self.cfg["algorithm"]["value_loss_coef"] * value_loss
        learn_step_agg.add(metrics)

        # Actor loss.
        actor_loss, kls, metrics = loss.actor_loss(
          advantages,
          batch,
          self.policy,
          self.policy_key,
          self.cfg["algorithm"]["symmetry"],
          self._get_symmetry_fn,
          self.cfg["algorithm"]["actor_loss_coefs"],
          self.cfg["algorithm"]["entropy_coefs"],
          self.cfg["algorithm"]["bound_coefs"],
          self.cfg["algorithm"]["symmetry_coefs"],
          self.cfg["algorithm"]["clip_param"],
          self.cfg["algorithm"]["symmetry_flip_latents"]
        )
        total_loss += actor_loss
        learn_step_agg.add(metrics)

        # Reconstruction loss.
        if self.train_image_encoder(self.global_num_updates):
          recon_loss, metrics = loss.reconstruction_loss(
            batch,
            self.image_encoder,
            self.image_encoder_key,
            self.cfg["image_encoder"]["max_batch_size"],
            self.cfg["algorithm"]["symmetry"],
            self._get_symmetry_fn,
            self.cfg["algorithm"]["symmetry_flip_latents"]
          )
          total_loss += recon_loss
          learn_step_agg.add(metrics)

        # SGD.
        self.image_encoder_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
          list(self.policy.parameters()) + list(self.value.parameters()) + list(self.image_encoder.parameters()),
          1.0)
        self.image_encoder_optimizer.step()
        self.policy_optimizer.step()
        self.value_optimizer.step()

        del batch

        # Learning rate scheduler.
        if self.cfg["policy"]["desired_kl"] > 0.0:
          if kls[self.cfg["policy"]["kl_key"]] > self.cfg["policy"]["desired_kl"] * 2.0:
            self.policy_learning_rate = max(1e-5, self.policy_learning_rate / 1.5)
          elif kls[self.cfg["policy"]["kl_key"]] < self.cfg["policy"]["desired_kl"] / 2.0 and kls[self.cfg["policy"]["kl_key"]] > 0.0:
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
        self.global_num_updates += 1

      # Update the observation normalizers.
      with torch.no_grad():
        self.policy.update_normalizer(self.buffer[self.policy_key])
        self.value.update_normalizer(self.buffer[self.value_key])

      return {
          **learn_step_agg.result(),
          "policy_lr": self.policy_learning_rate,
          "value_lr": self.value_learning_rate,
      }

  def scale_actions(self, actions):
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
    return actions_env

  def play(self):
    obs_dict = self.to_device(self.env.reset())
    policy_hidden_states = self.policy.reset(torch.zeros(self.env.num_envs, dtype=torch.bool), None)

    inference_time, step = 0., 0
    while True:
      with torch.no_grad():
        start = time.time()
        dists, _, policy_hidden_states = self.policy(
          obs_dict[self.policy_key],
          policy_hidden_states
        )
        actions = {k: dist.pred() for k, dist in dists.items()}
        actions_scaled = self.scale_actions(actions)
        inference_time += time.time() - start
        obs_dict, _, _, _ = self.env.step(actions_scaled)
        obs_dict = self.to_device(obs_dict)

      step += 1
      if step % 100 == 0:
        print(f"Average inference time: {inference_time / step}")
        print(f"\t Per env: {inference_time / step / self.env.num_envs}")
        inference_time, step = 0., 0
