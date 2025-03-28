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

import os
import warnings
import pathlib
import pickle

import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.helpers import  get_args



def play(args):
    log_root = pathlib.Path(os.path.join(LEGGED_GYM_ROOT_DIR, 'logs'))
    load_run_path = None
    if args.load_run is not None:
      load_run_path = log_root / args.load_run
    else:
      load_run_path = sorted(
        [item for item in log_root.iterdir() if item.is_dir()],
        key=lambda path: path.stat().st_mtime,
      )[-1]

    print(f'Loading run from: {load_run_path}...')

    with open(load_run_path / 'train_config.pkl', 'rb') as f:
      train_cfg = pickle.load(f)

    with open(load_run_path / 'env_config.pkl', 'rb') as f:
      env_cfg = pickle.load(f)

    if args.task is not None:
      if args.task != env_cfg.task_name:
         warnings.warn(f'Args and cfg task name mismatch: {args.task} != {env_cfg.task_name}. Overriding...')
      env_cfg.task_name = args.task

    print(f'\tTask name: {env_cfg.task_name}')

    env_cfg.env.num_envs = 50
    # Disable domain randomization.
    for attr_name in dir(env_cfg.domain_rand):
       if isinstance(getattr(env_cfg.domain_rand, attr_name), bool):
          setattr(env_cfg.domain_rand, attr_name, False)

    env, _ = task_registry.make_env(name=env_cfg.task_name, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=env_cfg.task_name, args=args, train_cfg=train_cfg)
    ppo_runner.play()

if __name__ == '__main__':
    args = get_args()
    play(args)
