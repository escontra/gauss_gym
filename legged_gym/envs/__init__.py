import pathlib

import legged_gym
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.anymal_c.anymal import Anymal
from legged_gym.envs.t1.t1 import T1
from legged_gym.utils import config

from legged_gym.utils.task_registry import task_registry

task_registry.register("a1", LeggedRobot, config.from_yaml(pathlib.Path(legged_gym.GAUSS_GYM_ENVS_DIR) / "a1" / "config.yaml"))
task_registry.register("t1", T1, config.from_yaml(pathlib.Path(legged_gym.GAUSS_GYM_ENVS_DIR) / "t1" / "config.yaml"))
task_registry.register("anymal_c", Anymal, config.from_yaml(pathlib.Path(legged_gym.GAUSS_GYM_ENVS_DIR) / "anymal_c" / "config.yaml"))
