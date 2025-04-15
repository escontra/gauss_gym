import os
import numpy as np
from typing import Dict, List
import random
import torch
import pickle
import torch.utils._pytree as pytree
import pathlib

from legged_gym.rl.env import vec_env
from legged_gym.rl.modules import models

class MuJoCoRunner:
  def __init__(self, env: vec_env.VecEnv, cfg: Dict, log_dir: pathlib.Path, device="cpu"):
    self.test = True
    self.env = env
    self.device = device
    self.log_dir = log_dir
    self.cfg = cfg
    self._set_seed()
    self.obs_group_sizes = pickle.load(open(log_dir / "obs_group_sizes.pkl", "rb"))
    self.model = getattr(models, self.cfg["runner"]["policy_class_name"])(
      self.env.num_actions,
      self.obs_group_sizes["student_observations"],
      self.obs_group_sizes["teacher_observations"],
      self.cfg["policy"]["init_noise_std"],
      layer_activation=self.cfg["policy"]["layer_activation"],
      mu_activation=self.cfg["policy"]["mu_activation"],
    ).to(self.device)

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

  def act(self, obs):
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


    with torch.no_grad():
      obs = self.filter_nans(obs)
      act = policy(obs, mlp_keys, cnn_keys)
    
    return act

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
