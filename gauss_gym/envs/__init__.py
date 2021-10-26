import pathlib

import gauss_gym
from gauss_gym.envs.base.legged_robot import LeggedRobot
from gauss_gym.envs.anymal_c.anymal import Anymal
from gauss_gym.envs.t1 import t1  # noqa: F401
from gauss_gym.utils import config

from gauss_gym.utils.task_registry import task_registry

task_registry.register(
  'a1',
  LeggedRobot,
  config.from_yaml(pathlib.Path(gauss_gym.GAUSS_GYM_ENVS_DIR) / 'a1' / 'config.yaml'),
)
task_registry.register(
  'a1_vision',
  LeggedRobot,
  config.from_yaml(
    pathlib.Path(gauss_gym.GAUSS_GYM_ENVS_DIR) / 'a1' / 'config_vision.yaml'
  ),
)
task_registry.register(
  'go1',
  LeggedRobot,
  config.from_yaml(pathlib.Path(gauss_gym.GAUSS_GYM_ENVS_DIR) / 'go1' / 'config.yaml'),
)
task_registry.register(
  'go1_vision',
  LeggedRobot,
  config.from_yaml(
    pathlib.Path(gauss_gym.GAUSS_GYM_ENVS_DIR) / 'go1' / 'config_vision.yaml'
  ),
)
task_registry.register(
  't1',
  t1.T1,
  config.from_yaml(pathlib.Path(gauss_gym.GAUSS_GYM_ENVS_DIR) / 't1' / 'config.yaml'),
)
task_registry.register(
  't1_vision',
  t1.T1,
  config.from_yaml(
    pathlib.Path(gauss_gym.GAUSS_GYM_ENVS_DIR) / 't1' / 'config_vision.yaml'
  ),
)
task_registry.register(
  'anymal_c',
  Anymal,
  config.from_yaml(
    pathlib.Path(gauss_gym.GAUSS_GYM_ENVS_DIR) / 'anymal_c' / 'config.yaml'
  ),
)
task_registry.register(
  'anymal_c_vision',
  Anymal,
  config.from_yaml(
    pathlib.Path(gauss_gym.GAUSS_GYM_ENVS_DIR) / 'anymal_c' / 'config_vision.yaml'
  ),
)
