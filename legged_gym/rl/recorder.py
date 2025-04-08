import pathlib
import pickle
import wandb
import torch
from typing import Dict, Any
from torch.utils.tensorboard import SummaryWriter

from legged_gym.utils import config, timer

class Recorder:
  def __init__(self, log_dir: pathlib.Path, cfg: Dict[str, Any], deploy_cfg: Dict[str, Any], obs_group_sizes):
    self.cfg = cfg
    self.deploy_cfg = deploy_cfg
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
    if self.cfg["runner"]["use_wandb"]:
      wandb.init(
        project=self.cfg["task"],
        dir=self.log_dir,
        name=self.log_dir.name,
        notes=self.cfg["runner"]["description"],
        config=dict(self.cfg),
      )

    self.episode_statistics = {}
    self.last_episode = {}
    self.last_episode["steps"] = []
    self.episode_steps = None

    config.Config(self.cfg).save(self.log_dir / "config.yaml")
    config.Config({'deploy': self.deploy_cfg}).save(self.log_dir / "deploy_config.yaml")

    with open(self.log_dir / "obs_group_sizes.pkl", "wb") as file:
      pickle.dump(self.obs_group_sizes, file)
    self.initialized = True

  @timer.section("record_episode_statistics")
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

    episode_stats = {}
    for key in self.last_episode.keys():
      episode_stats[key] = self._mean(self.last_episode[key])
      # path = ("" if key == "steps" or key == "reward" else "episode/") + key
      # value = self._mean(self.last_episode[key])
      # self.writer.add_scalar(path, value, it)
      # if self.cfg["runner"]["use_wandb"]:
      #   wandb.log({path: value}, step=it)
      self.last_episode[key].clear()
    return episode_stats

    if write_record:
      for key in self.last_episode.keys():
        path = ("" if key == "steps" or key == "reward" else "episode/") + key
        value = self._mean(self.last_episode[key])
        self.writer.add_scalar(path, value, it)
        if self.cfg["runner"]["use_wandb"]:
          wandb.log({path: value}, step=it)
        self.last_episode[key].clear()

  @timer.section("record_statistics")
  def record_statistics(self, statistics, it):
    self.maybe_init()
    for key, value in statistics.items():
      if isinstance(value, str):
        self.writer.add_text(key, value, it)
        if self.cfg["runner"]["use_wandb"]:
          wandb.log({key: value}, step=it)
      elif isinstance(value, (float, int)):
        self.writer.add_scalar(key, float(value), it)
        if self.cfg["runner"]["use_wandb"]:
          wandb.log({key: float(value)}, step=it)
      else:
        raise ValueError(f"Unsupported type for {key}: {type(value)}")

  @timer.section("save")
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

