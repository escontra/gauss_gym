import os
import numpy as np
from typing import Dict
import random
import torch
import pickle
import torch.utils._pytree as pytree
import pathlib

from legged_gym.rl.env import vec_env
from legged_gym.rl.modules import models
from legged_gym.rl.modules import normalizers_notensordict as normalizers

class MuJoCoRunner:
  def __init__(self, env: vec_env.VecEnv, cfg: Dict, device="cpu"):
    self.device = device
    self.cfg = cfg
    self._set_seed()
    self.policy_key = self.cfg["policy"]["obs_key"]
    self.env = env

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

    # Create policy and observation normalizer.
    obs_group_sizes = pickle.load(open(resume_path / "obs_group_sizes.pkl", "rb"))
    self.policy = getattr(models, self.cfg["policy"]["class_name"])(
      self.env.num_actions,
      obs_group_sizes[self.policy_key],
      **self.cfg["policy"]["params"]
    ).to(self.device)
    print('NORMALIZE STATE DICT')
    print(model_dict[f"obs_normalizer/{self.policy_key}"])
    self.observation_normalizer = normalizers.PyTreeNormalizer(
      obs_group_sizes[self.policy_key],
    ).to(self.device)
    print('NEW NORMALIZER STATE DICT')
    print(self.observation_normalizer.state_dict())

    # Load policy and observation normalizer.
    self.policy.load_state_dict(model_dict["policy"], strict=False)
    self.observation_normalizer.load_state_dict(model_dict[f"obs_normalizer/{self.policy_key}"], strict=False)
    print('FINAL NORMALIZER STATE DICT')
    print(self.observation_normalizer.state_dict())

  def to_device(self, obs):
    return pytree.tree_map(lambda x: x.to(self.device), obs)

  def act(self, obs):
    obs = self.to_device(obs)
    obs = self.observation_normalizer.normalize(obs)
    policy = models.get_policy_jitted(self.policy, self.cfg["policy"]["params"])
    with torch.no_grad():
      act = policy(obs)
    
    return act

  def interrupt_handler(self, signal, frame):
    print("\nInterrupt received, waiting for video to finish...")
    self.interrupt = True
