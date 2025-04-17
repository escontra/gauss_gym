# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time

from legged_gym.rl.env import vec_env
from legged_gym.rl.runner import Runner

import legged_gym

import pathlib
from legged_gym.utils import config

class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.cfgs = {}
    
    def register(self, name: str, task_class: vec_env.VecEnv, cfg: config.Config):
        self.task_classes[name] = task_class
        self.cfgs[name] = cfg

    def get_task_class(self, name: str) -> vec_env.VecEnv:
        return self.task_classes[name]
    
    def get_cfgs(self, name: str) -> config.Config:
        return self.cfgs[name]
    
    def make_alg_runner(self, env, cfg: config.Config) -> Runner:
        """ Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            cfg (Config): Config file.

        Returns:
            Runner: The created algorithm
        """
        new_run_name = [f'{cfg["task"]}']
        if cfg["runner"]["run_name"] != "":
            new_run_name += [cfg["runner"]["run_name"]]
        new_run_name += [time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())]
        new_run_name = '_'.join(new_run_name)

        if cfg["logdir"]=="default":
            log_root = pathlib.Path(legged_gym.GAUSS_GYM_ROOT_DIR) / 'logs'
            log_dir = pathlib.Path(log_root) / new_run_name
        elif cfg["logdir"] is None:
            log_dir = None
        else:
            log_dir = pathlib.Path(log_root) / new_run_name
        
        runner = eval(cfg["runner"]["class_name"])(env, cfg, log_dir, device=cfg["rl_device"])
        
        #save resume path before creating a new log_dir
        if cfg["runner"]["resume"]:
            runner.load(log_root)
        return runner

# make global task registry
task_registry = TaskRegistry()