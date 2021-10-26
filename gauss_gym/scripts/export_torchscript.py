import types
import os
import isaacgym  # noqa: F401
import pathlib
import torch

import gauss_gym
from gauss_gym.envs import *  # noqa: F403
from gauss_gym.utils import flags, config
from gauss_gym.rl import deployment_runner  # noqa: F401
from gauss_gym.utils.task_registry import task_registry
from gauss_gym.utils import helpers

from gauss_gym.rl.runner import Runner


def main(argv=None):
  log_root = pathlib.Path(os.path.join(gauss_gym.GAUSS_GYM_ROOT_DIR, 'logs'))
  load_run_path = None
  parsed, other = flags.Flags({'runner': {'load_run': ''}}).parse_known(argv)
  if parsed.runner.load_run != '':
    load_run_path = log_root / parsed.runner.load_run
  else:
    load_run_path = sorted(
      [item for item in log_root.iterdir() if item.is_dir()],
      key=lambda path: path.stat().st_mtime,
    )[-1]

  print(f'Loading run from: {load_run_path}...')
  cfg = config.Config.load(load_run_path / 'train_config.yaml')
  cfg = cfg.update({'runner.load_run': load_run_path.name})
  cfg = cfg.update({'runner.resume': True})
  cfg = cfg.update({'headless': True})
  cfg = cfg.update({'env.num_envs': 1})
  cfg = flags.Flags(cfg).parse(other)
  cfg = types.MappingProxyType(dict(cfg))

  task_class = task_registry.get_task_class(cfg['task'])
  helpers.set_seed(cfg['seed'])
  env = task_class(cfg=cfg)

  if cfg['logdir'] == 'default':
    log_root = pathlib.Path(gauss_gym.GAUSS_GYM_ROOT_DIR) / 'logs'
  elif cfg['logdir'] != '':
    log_root = pathlib.Path(cfg['logdir'])
  else:
    raise ValueError("Must specify logdir as 'default' or a path.")

  runner: Runner = eval(cfg['runner']['class_name'])(env, cfg, device=cfg['rl_device'])
  resume_path = runner.load(log_root)
  print(f'Resume path: {resume_path}')

  print('Exporting policy to ONNX...')
  model = runner.value
  model.eval()

  scripted_model = torch.jit.script(model)
  init_hidden_states = scripted_model.reset(torch.zeros(1), None)

  # Prepare dummy inputs
  print('Preparing dummy inputs...')
  policy_obs_space = runner.value_obs_space
  policy_dummy_obs = {
    k: torch.ones(v.shape)[None].to(runner.device) for k, v in policy_obs_space.items()
  }
  print('POLICY DUMMY OBS:')
  for k, v in policy_dummy_obs.items():
    print(f'\t{k}: {v.shape}')
  ordered_obs_keys = list(policy_dummy_obs.keys())
  dummy_obs_tuple = tuple(policy_dummy_obs[k] for k in ordered_obs_keys)

  init_hidden_states = model.reset(torch.zeros(1), None)
  ordered_hidden_keys = [f'hidden_{i}' for i in range(len(init_hidden_states))]

  print('INIT HIDDEN STATES:')
  for k, v in zip(ordered_hidden_keys, init_hidden_states):
    print(f'\t{k}: {v.shape}')

  dummy_input_tuple = dummy_obs_tuple + init_hidden_states
  print(f'Dummy input tuple: {dummy_input_tuple}')
  input_names = ordered_obs_keys + ordered_hidden_keys
  print(f'Input names: {input_names}')
  dynamic_axes = {k: {0: 'batch_size'} for k in ordered_obs_keys}
  dynamic_axes.update({k: {1: 'batch_size'} for k in ordered_hidden_keys})

  action_names = [f'out_{k}' for k in env.action_space().keys()]
  out_hidden_names = [f'out_{k}' for k in ordered_hidden_keys]
  output_names = action_names + out_hidden_names
  print(f'Output names: {output_names}')
  dynamic_axes.update({k: {0: 'batch_size'} for k in action_names})
  dynamic_axes.update({k: {1: 'batch_size'} for k in out_hidden_names})

  dists, rnn, new_hidden_states = scripted_model(policy_dummy_obs, init_hidden_states)
  print('Dists:')
  for k, v in dists.items():
    print(f'\t{k}: {v}')


if __name__ == '__main__':
  main()
  import sys

  sys.exit(0)
