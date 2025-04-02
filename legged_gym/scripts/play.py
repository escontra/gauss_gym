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
from legged_gym.utils import flags, config
from legged_gym.utils import helpers


def main(argv = None):
    log_root = pathlib.Path(os.path.join(LEGGED_GYM_ROOT_DIR, 'logs'))
    load_run_path = None
    parsed, other = flags.Flags(load_run='', checkpoint=-1).parse_known(argv)
    if parsed.load_run != '':
      load_run_path = log_root / parsed.load_run
    else:
      load_run_path = sorted(
        [item for item in log_root.iterdir() if item.is_dir()],
        key=lambda path: path.stat().st_mtime,
      )[-1]

    print(f'Loading run from: {load_run_path}...')
    cfg = config.Config.load(load_run_path / 'config.yaml')
    cfg = cfg.update({'runner.load_run': load_run_path.name})
    cfg = cfg.update({'runner.checkpoint': parsed.checkpoint})
    cfg = cfg.update({'runner.resume': True})
    cfg = cfg.update({'headless': False})

    cfg = cfg.update({'env.num_envs': 50})
    # Disable domain randomization.
    for attr_name in dict(cfg.domain_rand):
      attr = getattr(cfg.domain_rand, attr_name)
      if hasattr(attr, 'apply'):
          print(f'{attr_name} = {attr.apply}')
          cfg = cfg.update({f'domain_rand.{attr_name}.apply': False})
    cfg = flags.Flags(cfg).parse(other)
    print(cfg)
    cfg = dict(cfg)
    task_class = task_registry.get_task_class(cfg["task"])
    helpers.set_seed(cfg["seed"])
    env = task_class(cfg=cfg)
    ppo_runner = task_registry.make_alg_runner(env=env, cfg=cfg)
    ppo_runner.play()

if __name__ == '__main__':
    main()
