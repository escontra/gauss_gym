from typing import Any, Dict
import time
import torch
import torch.utils._pytree as pytree
import pathlib
import copy
import numpy as np

from legged_gym.rl import experience_buffer, loss, recorder
from legged_gym.rl.modules import normalizers
import legged_gym.rl.utils as rl_utils
from legged_gym.rl.env import vec_env
from legged_gym.rl.modules import models
from legged_gym import utils
from legged_gym.utils import agg, observation_groups, symmetry_groups, timer, when, space, visualization, wrappers, math
from legged_gym.utils import wandb as wandb_utils

torch.backends.cuda.matmul.allow_tf32 = True


class Runner:
  def __init__(self, env: vec_env.VecEnv, cfg: Dict[str, Any], device="cpu"):
    self.env = env
    self.device = device
    self.multi_gpu = cfg["multi_gpu"]
    self.multi_gpu_global_rank = cfg["multi_gpu_global_rank"]
    self.multi_gpu_local_rank = cfg["multi_gpu_local_rank"]
    self.multi_gpu_world_size = cfg["multi_gpu_world_size"]

    if self.multi_gpu:
      assert self.device == 'cuda:' + str(self.multi_gpu_local_rank) # check it was set correctly

    self.rank_zero = not self.multi_gpu or self.multi_gpu_global_rank == 0

    self.cfg = cfg
    rl_utils.set_seed(self.cfg["seed"])

    # Image encoder.
    self.image_encoder_enabled = self.cfg["algorithm"]["train_image_encoder"]
    self.image_encoder= None
    if self.image_encoder_enabled:
      self.image_encoder_key = self.cfg["image_encoder"]["obs_key"]
      self.image_encoder_obs_space = self.env.obs_space()[self.image_encoder_key]
      if len(self.env.obs_space()[self.image_encoder_key].keys()) > 0:
        reconstruct_space = None
        reconstruct_head = None
        if self.cfg["image_encoder"]["reconstruct_observations"] is not None:
          reconstruct_space = {}
          reconstruct_head = {}
          for key in self.cfg["image_encoder"]["reconstruct_observations"]:
            obs_group, obs_name = key.split('/')
            obs_name = getattr(observation_groups, obs_name).name
            if 'ray_cast' in obs_name.lower():
              reconstruct_head[f'{obs_group}/{obs_name}'] = {
                'class_name': 'VoxelGridDecoderHead',
                'params': {}
              }
              reconstruct_space[f'{obs_group}/{obs_name}'] = space.Space(
                torch.float32,
                shape=(*self.env.obs_space()[obs_group][obs_name].shape,
                      self.cfg["image_encoder"]["voxel_height_levels"]))
            else:
              reconstruct_space[f'{obs_group}/{obs_name}'] = self.env.obs_space()[obs_group][obs_name]
              reconstruct_head[f'{obs_group}/{obs_name}'] = {
                'class_name': 'MSEHead',
                'params': {'outscale': 1.0}
              }
        self.image_encoder: models.RecurrentModel = getattr(
          models, self.cfg["image_encoder"]["class_name"])(
          reconstruct_space,
          self.image_encoder_obs_space,
          head=reconstruct_head,
          **self.cfg["image_encoder"]["params"]
        ).to(self.device)

        if self.multi_gpu:
          rl_utils.sync_state_dict(self.image_encoder, 0)

        # Image encoder wrapper computes latents during the environment step.
        self.env = wrappers.ImageEncoderWrapper(self.env, self.image_encoder_key, self.image_encoder)
      if self.image_encoder is None:
        raise ValueError(
          "Could not initialize image encoder, either because no image inputs "
          "were found or because no reconstruction observations were specified."
        )

    if self.image_encoder is None:
      for obs_group in self.env.obs_groups:
        assert observation_groups.IMAGE_ENCODER_LATENT not in obs_group.observations, (
          f"IMAGE_ENCODER_LATENT must not be in {obs_group.name}.observations"
          f"when 'train_image_encoder' is False."
        )

    # Policy and value.
    self.policy_key = self.cfg["policy"]["obs_key"]
    self.policy_obs_space = self.env.obs_space()[self.policy_key]
    self.value_key = self.cfg["value"]["obs_key"]
    self.value_obs_space = self.env.obs_space()[self.value_key]

    # Reward normalizer.
    self.reward_normalizer = None
    if self.cfg["algorithm"]["normalize_rewards"]:
      self.reward_normalizer = normalizers.RewardNormalizer(
        self.cfg["algorithm"]["gamma"],
        num_envs=self.env.num_envs
      ).to(self.device)

    policy_project_dims, value_project_dims = {}, {}
    if self.image_encoder_enabled and self.image_encoder_key in self.policy_obs_space:
      if self.cfg["policy"]["project_image_encoder_latent"]:
        policy_project_dims[self.image_encoder_key] = self.cfg["policy"]["project_image_encoder_latent_dim"]
    if self.image_encoder_enabled and self.image_encoder_key in self.value_obs_space:
      if self.cfg["value"]["project_image_encoder_latent"]:
        value_project_dims[self.image_encoder_key] = self.cfg["value"]["project_image_encoder_latent_dim"]

    self.policy_learning_rate = self.cfg["policy"]["learning_rate"]
    self.policy: models.RecurrentModel = getattr(
      models, self.cfg["policy"]["class_name"])(
      self.env.action_space(),
      self.policy_obs_space,
      project_dims=policy_project_dims,
      **self.cfg["policy"]["params"]
    ).to(self.device)

    # Sync policy to all GPUs.
    if self.multi_gpu:
      rl_utils.sync_state_dict(self.policy, 0)
    # For KL.
    self.old_policy: models.RecurrentModel = copy.deepcopy(self.policy)
    for param in self.old_policy.parameters():
      param.requires_grad = False

    # Value.
    self.value_learning_rate = self.cfg["value"]["learning_rate"]
    self.value: models.RecurrentModel = getattr(
      models, self.cfg["value"]["class_name"])(
      {'value': space.Space(torch.float32, (1,), -torch.inf, torch.inf)},
      self.value_obs_space,
      project_dims=value_project_dims,
      **self.cfg["value"]["params"]
    ).to(self.device)

    # Sync value to all GPUs.
    if self.multi_gpu:
      rl_utils.sync_state_dict(self.value, 0)

    # Optimizers.
    if self.image_encoder_enabled:
      self.train_ratio_scheduler = rl_utils.SetpointScheduler(
        warmup_steps=self.cfg["image_encoder"]["train_ratio_scheduler"]["warmup_steps"],
        val_warmup=self.cfg["image_encoder"]["train_ratio_scheduler"]["train_ratio_warmup"],
        val_after=self.cfg["image_encoder"]["train_ratio_scheduler"]["train_ratio_after"]
      )
      self.image_encoder_learning_rate_scheduler = rl_utils.SetpointScheduler(
        warmup_steps=self.cfg["image_encoder"]["learning_rate_scheduler"]["warmup_steps"],
        val_warmup=self.cfg["image_encoder"]["learning_rate_scheduler"]["learning_rate_warmup"],
        val_after=self.cfg["image_encoder"]["learning_rate_scheduler"]["learning_rate_after"]
      )
      image_encoder_parameters = list(self.image_encoder.recurrent_model.parameters())
      if not self.cfg["image_encoder"]["freeze_encoder"]:
        image_encoder_parameters.extend(list(self.image_encoder.image_feature_model.parameters()))
      self.image_encoder_optimizer = torch.optim.Adam(
        image_encoder_parameters, lr=self.image_encoder_learning_rate_scheduler(0)
      )

    self.policy_optimizer = torch.optim.AdamW(
      self.policy.parameters(), lr=self.policy_learning_rate
    )
    self.value_optimizer = torch.optim.AdamW(
      self.value.parameters(), lr=self.value_learning_rate
    )

    self._flatten_parameters()
    self._set_eval_mode()

    if self.cfg["algorithm"]["symmetry_loss"] or self.cfg["algorithm"]["symmetry_augmentation"]:
      assert "symmetries" in self.cfg, "Need `symmetries` in config when symmetry is enabled. Look at a1/config.yaml for an example."
      self.symmetry_lookup = symmetry_groups.symmetry_dict_from_config(self.cfg["symmetries"])
      assert self.policy_key in self.symmetry_lookup
      assert "actions" in self.symmetry_lookup

  def _get_symmetry_fn(self, obs_group, obs_name):
    assert obs_group in self.symmetry_lookup, f"{obs_group} not in {self.symmetry_lookup}"
    symmetry_fns = self.symmetry_lookup[obs_group]
    assert obs_name in symmetry_fns, f"{obs_name} not in {symmetry_fns}"
    symmetry_fn = symmetry_fns[obs_name]
    return lambda val: symmetry_fn(self.env, val)

  def load(self, resume_root: pathlib.Path) -> pathlib.Path:
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

    if load_run.startswith('wandb_'):
      model_path = wandb_utils.get_wandb_path(load_run, self.multi_gpu, self.multi_gpu_global_rank)
    else:
      utils.print(f"Loading checkpoint from: {resume_path}", color='blue')
      utils.print(f'\tNum checkpoints: {len(list((resume_path / "nn").glob("*.pth")))}', color='blue')
      utils.print(f'\tLoading checkpoint: {checkpoint}', color='blue')
      if (checkpoint == "-1") or (checkpoint == -1):
        model_path = sorted(
          (resume_path / "nn").glob("*.pth"),
          key=lambda path: path.stat().st_mtime,
        )[-1]
      else:
        model_path = resume_path / "nn" / f"model_{checkpoint:06d}.pth"
    utils.print(f'\tLoading model weights from: {model_path}', color='blue')
    model_dict = torch.load(
      model_path, map_location=self.device, weights_only=True
    )
    if self.image_encoder_enabled:
      self.image_encoder.load_state_dict(model_dict["image_encoder"], strict=True)
    self.policy.load_state_dict(model_dict["policy"], strict=True)
    self.value.load_state_dict(model_dict["value"], strict=True)
    try:
      if self.image_encoder_enabled:
        self.image_encoder_optimizer.load_state_dict(model_dict["image_encoder_optimizer"])
      self.policy_optimizer.load_state_dict(model_dict["policy_optimizer"])
      self.value_optimizer.load_state_dict(model_dict["value_optimizer"])
    except Exception as e:
      utils.print(f"Failed to load optimizer: {e}", color='red')
    return resume_path

  def to_device(self, obs):
    return pytree.tree_map(lambda x: x.to(self.device), obs)

  def _flatten_parameters(self):
    if self.image_encoder_enabled:
      self.image_encoder.flatten_parameters()
    self.policy.flatten_parameters()
    self.value.flatten_parameters()
    self.old_policy.flatten_parameters()

  def _set_train_mode(self):
    if self.image_encoder_enabled:
      self.image_encoder.train()
    self.policy.train()
    self.value.train()

  def _set_eval_mode(self):
    if self.image_encoder_enabled:
      self.image_encoder.eval()
    self.policy.eval()
    self.value.eval()

  @property
  def spaces_dict(self):
    spaces_dict = {
      'policy_obs_space': self.policy_obs_space,
      'value_obs_space': self.value_obs_space,
      'action_space': self.env.action_space(),
    }
    if self.image_encoder_enabled:
      spaces_dict['image_encoder_obs_space'] = self.image_encoder_obs_space
    return spaces_dict


  def learn(self, num_learning_iterations, log_dir: pathlib.Path, init_at_random_ep_len=False):
    # Logger aggregators.
    self.step_agg = agg.Agg()
    self.completion_agg = agg.Agg()
    self.episode_agg = agg.Agg()
    self.learn_agg = agg.Agg()
    self.action_agg = agg.Agg()
    self.should_log = when.Clock(self.cfg["runner"]["log_every"])
    self.should_save = when.Clock(self.cfg["runner"]["save_every"])
    if self.image_encoder_enabled:
      self.env.init_image_encoder_replay_buffer(self.cfg["image_encoder"]["num_steps_per_env"])
    self.recorder = recorder.Recorder(log_dir, self.cfg, self.env.deploy_config(), self.spaces_dict, rank_zero=self.rank_zero)
    if self.cfg["runner"]["record_video"]:
      self.recorder.setup_recorder(self.env)

    self._set_eval_mode()

    # Initialize hidden states and set random episode length.
    obs_dict = self.to_device(self.env.reset())
    policy_hidden_states = self.policy.reset(torch.zeros(self.env.num_envs, dtype=torch.bool), None)
    value_hidden_states = self.value.reset(torch.zeros(self.env.num_envs, dtype=torch.bool), None)
    if init_at_random_ep_len:
        self.env.episode_length_buf = torch.randint_like(
          self.env.episode_length_buf,
          high=int(self.env.max_episode_length))

    # Replay buffer.
    buffer = experience_buffer.ExperienceBuffer(
      self.cfg["runner"]["num_steps_per_env"],
      self.env.num_envs,
      self.device,
    )

    buffer.add_buffer(self.policy_key, self.policy_obs_space)
    buffer.add_buffer(self.value_key, self.value_obs_space)
    buffer.add_buffer("actions", self.env.action_space())
    buffer.add_buffer("rewards", ())
    buffer.add_buffer("values", (1,))
    buffer.add_buffer("dones", (), dtype=bool)
    buffer.add_buffer("time_outs", (), dtype=bool)

    if self.policy.is_recurrent:
      buffer.add_buffer(f"{self.policy_key}_hidden_states", (policy_hidden_states,), is_hidden_state=True)
    if self.value.is_recurrent:
      buffer.add_buffer(f"{self.value_key}_hidden_states", (value_hidden_states,), is_hidden_state=True)

    policy_cnn_keys = self.policy.cnn_keys or []
    potential_images = {k: obs_dict[self.policy_key][k] for k in policy_cnn_keys}
    if self.image_encoder_enabled:
      image_encoder_cnn_keys = self.image_encoder.cnn_keys or []
      potential_images.update({k: obs_dict[self.image_encoder_key][k] for k in image_encoder_cnn_keys})

    # TODO: Choose a correct value here.
    running_train_ratio = normalizers.RunningEMA(alpha=0.9)
    prev_train_ratio_error = 0.0
    curr_num_updates = self.cfg["image_encoder"]["init_num_updates"]
    for it in range(num_learning_iterations):
      start = time.time()
      num_enc_updates = 0
      num_pol_updates = self.cfg["algorithm"]["num_learning_epochs"] * self.cfg["algorithm"]["num_mini_batches"]
      for n in range(self.cfg["runner"]["num_steps_per_env"]):
        if self.cfg["runner"]["record_video"]:
          for k in policy_cnn_keys:
            potential_images[k] = obs_dict[self.policy_key][k]
          if self.image_encoder_enabled: # and image_encoder_obs_available:
            for k in image_encoder_cnn_keys:
              if k in obs_dict[self.image_encoder_key]:
                potential_images[k] = obs_dict[self.image_encoder_key][k]
          self.recorder.record_statistics(
            self.recorder.maybe_record(self.env, image_features=potential_images),
            it * self.cfg["runner"]["num_steps_per_env"] * self.env.num_envs + n)
        with timer.section("buffer_add_obs"):
          if self.policy.is_recurrent:
            buffer.update_data(
              f"{self.policy_key}_hidden_states", n, (policy_hidden_states,), is_hidden_state=True
            )
          if self.value.is_recurrent:
            buffer.update_data(
              f"{self.value_key}_hidden_states", n, (value_hidden_states,), is_hidden_state=True
            )
          buffer.update_data(self.policy_key, n, obs_dict[self.policy_key])
          buffer.update_data(self.value_key, n, obs_dict[self.value_key])
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
          # Log action distributions.
          for k, v in actions.items():
            self.action_agg.add(
              {f'{dof_name}_{k}': v[:, dof_idx].cpu().numpy()
               for dof_idx, dof_name in enumerate(self.env.dof_names)},
              agg='concat'
            )

          obs_dict, rew, done, infos = self.env.step(actions)
          if self.reward_normalizer is not None:
            self.reward_normalizer.update(rew, done)
            if self.multi_gpu:
              rl_utils.sync_state_dict(self.reward_normalizer, 0)

        if self.image_encoder_enabled:
          image_encoder_buffer = infos.pop(f'{self.image_encoder_key}_buffer', None)
          if image_encoder_buffer:
            for param_group in self.image_encoder_optimizer.param_groups:
              param_group['lr'] = self.image_encoder_learning_rate_scheduler(it)
            num_learning_epochs, num_mini_batches = math.nearest_factors(curr_num_updates)
            self._set_train_mode()
            image_encoder_metrics = loss.learn_image_encoder(
              image_encoder_buffer,
              self.image_encoder,
              self.image_encoder_optimizer,
              self.image_encoder_key,
              self.value_key,
              None,
              self._get_symmetry_fn,
              num_learning_epochs=num_learning_epochs,
              num_mini_batches=num_mini_batches,
              algorithm_cfg=self.cfg["algorithm"],
              multi_gpu=self.multi_gpu,
              multi_gpu_global_rank=self.multi_gpu_global_rank,
              multi_gpu_world_size=self.multi_gpu_world_size,
            )
            self.learn_agg.add(image_encoder_metrics)
            self._set_eval_mode()
            num_enc_updates += num_learning_epochs * num_mini_batches

        policy_hidden_states = self.policy.reset(done, policy_hidden_states)
        value_hidden_states = self.value.reset(done, value_hidden_states)
        with timer.section("buffer_update_data"):
          buffer.update_data("actions", n, actions)
          buffer.update_data("rewards", n, rew)
          buffer.update_data("dones", n, done)
          buffer.update_data(
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

      self._set_train_mode()
      self.policy_learning_rate, self.value_learning_rate, learn_stats = loss.learn_ppo(
        buffer,
        self.policy,
        self.old_policy,
        self.reward_normalizer,
        self.value,
        self.policy_optimizer,
        self.value_optimizer,
        self.policy_learning_rate,
        self.value_learning_rate,
        self.policy_key,
        self.value_key,
        None,
        self._get_symmetry_fn,
        obs_dict[self.value_key],
        value_hidden_states,
        it,
        self.cfg["algorithm"],
        multi_gpu=self.multi_gpu,
        multi_gpu_global_rank=self.multi_gpu_global_rank,
        multi_gpu_world_size=self.multi_gpu_world_size,
      )
      self.learn_agg.add(learn_stats)
      self._set_eval_mode()

      if self.image_encoder_enabled:
        if not self.multi_gpu or self.multi_gpu_global_rank == 0:
          running_train_ratio.update(num_enc_updates / num_pol_updates)
          desired_train_ratio = self.train_ratio_scheduler(it)
          error = desired_train_ratio - running_train_ratio.ema
          derivative = error - prev_train_ratio_error
          # Proportional gain — tune this!
          delta = 5.0 * error + 0.4 * derivative
          curr_num_updates = np.clip(
            curr_num_updates + delta,
            self.cfg["image_encoder"]["num_updates_range"][0],
            self.cfg["image_encoder"]["num_updates_range"][1]
          )
          curr_num_updates = max(
            self.cfg["image_encoder"]["num_updates_range"][0],
            curr_num_updates
          )
          curr_num_updates = min(
            self.cfg["image_encoder"]["num_updates_range"][1],
            curr_num_updates)
          self.learn_agg.add({'train_ratio': running_train_ratio.ema})
          self.learn_agg.add({'curr_num_updates': curr_num_updates})
          prev_train_ratio_error = error

        if self.multi_gpu:
          curr_num_updates = rl_utils.broadcast_scalar(
            float(curr_num_updates),
            0,
            self.device
          )

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
          to_save = {
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
          }
          if self.image_encoder_enabled:
            to_save["image_encoder"] = self.image_encoder.state_dict()
            to_save["image_encoder_optimizer"] = self.image_encoder_optimizer.state_dict()
          self.recorder.save(to_save,
            it + 1,
          )
      utils.print(
        "epoch: {}/{} - {}s.".format(
          it + 1, num_learning_iterations, time.time() - start
        ), color='green'
      )
      start = time.time()

  def play(self):
    obs_dict = self.to_device(self.env.reset())
    policy_hidden_states = self.policy.reset(torch.zeros(self.env.num_envs, dtype=torch.bool), None)
    self._set_eval_mode()
    inference_time, step = 0., 0
    while True:
      with torch.no_grad():
        start = time.time()
        dists, _, policy_hidden_states = self.policy(
          obs_dict[self.policy_key],
          policy_hidden_states
        )
        actions = {k: dist.pred() for k, dist in dists.items()}
        inference_time += time.time() - start
        obs_dict, _, done, _ = self.env.step(actions)
        obs_dict, done = self.to_device((obs_dict, done))
        policy_hidden_states = self.policy.reset(done, policy_hidden_states)

      step += 1
      if step % 100 == 0:
        utils.print(f"Average inference time: {inference_time / step}", color='green')
        utils.print(f"\t Per env: {inference_time / step / self.env.num_envs}", color='green')
        inference_time, step = 0., 0
