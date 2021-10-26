import os
import wandb

from gauss_gym.utils import config
from gauss_gym import utils


def get_run(load_run: str):
  """Get run for wandb run ID or name.

  load_run: a string with the following format options:
  - wandb_id_<run_id>
  - wandb_id_<run_id>:<project>
  - wandb_id_<run_id>:<project>:<entity>
  - wandb_<run_name>
  - wandb_<run_name>:<project>
  - wandb_<run_name>:<project>:<entity>
  """
  api = wandb.Api()
  entity = wandb.Api().default_entity
  run_id = None
  run_name = None
  project = None

  is_id = load_run.startswith('wandb_id_')
  run_str = load_run[9:] if is_id else load_run[6:]
  run_items = run_str.split(':')

  if is_id:
    run_id = run_items[0]
  else:
    run_name = run_items[0]

  if len(run_items) >= 2:
    project = run_items[1]

  if len(run_items) == 3:
    entity = run_items[2]

  if project is None:
    projects = [p.name for p in api.projects(entity=entity)]
  else:
    projects = [project]
  for _project in projects:
    if is_id:
      runs = list(api.runs(f'{entity}/{_project}', filters={'name': {'$in': [run_id]}}))
    else:
      runs = list(api.runs(f'{entity}/{_project}', filters={'display_name': run_name}))
      if len(runs) > 1:
        raise ValueError(
          f'Multiple runs found with name: {run_name}.'
          f'Try using the run ID instead: wandb_id_<run_id>'
        )

    if len(runs) == 0:
      continue
    else:
      utils.print(f'Found run: {runs[0]}', color='blue')
      return runs[0]

  raise ValueError(
    f'No wandb run found with entity: {entity}, project: {project}, '
    f'run ID: {run_id}, run name: {run_name}'
  )


def get_wandb_path(
  load_run: str, multi_gpu: bool = False, multi_gpu_rank: int = 0
) -> str:
  """
  Get the path to a wandb checkpoint.
  Saves in a temp folder in /tmp

  Args:
      load_run: See get_run for format options.

  Returns:
      Path to the downloaded checkpoint file
  """
  run = get_run(load_run)
  files = run.files()

  checkpoint_files = [
    f for f in files if 'model_' in f.name and f.name.endswith('.pth')
  ]

  if not checkpoint_files:
    raise ValueError(f'No checkpoints found for wandb run {run}')

  # Get the latest checkpoint
  # If we upload numerical checkpoints, we can use the following:
  # latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.name.split('model_')[1].split('.')[0]))
  # print(f"Latest checkpoint: {latest_checkpoint.name}")
  # Currently, we assume that we only upload one checkpoint per run (and continually overwrite it)
  assert len(checkpoint_files) == 1, (
    f'Expected 1 checkpoint file, got {len(checkpoint_files)}'
  )
  latest_checkpoint = checkpoint_files[0]

  # Create target directory and download
  if multi_gpu:
    target_dir = os.path.join(
      '/tmp/wandb_checkpoints/', run.name, f'rank_{multi_gpu_rank}'
    )
  else:
    target_dir = os.path.join('/tmp/wandb_checkpoints/', run.name)
  os.makedirs(target_dir, exist_ok=True)

  target_path = os.path.join(target_dir, latest_checkpoint.name)
  utils.print(f'Downloading wandb checkpoint to {target_path}', color='blue')
  latest_checkpoint.download(root=target_dir, replace=True)
  utils.print('Download complete!', color='blue')
  return target_path


def get_wandb_config(
  load_run: str,
  multi_gpu: bool = False,
  multi_gpu_rank: int = 0,
  config_name: str = 'train_config.yaml',
) -> str:
  """
  Get the path to a wandb config YAML file.
  Saves in a temp folder in /tmp

  Args:
      load_run: Either a wandb run ID or a name prefixed with 'wandb_'

  Returns:
      Path to the downloaded config YAML file
  """
  run = get_run(load_run)
  files = run.files()

  files = [f for f in files if config_name in f.name]

  assert len(files) == 1, f'Expected 1 config file, got {len(files)}'

  config_file = files[0]

  if multi_gpu:
    target_dir = os.path.join(
      '/tmp/wandb_checkpoints/', run.name, f'rank_{multi_gpu_rank}'
    )
  else:
    target_dir = os.path.join('/tmp/wandb_checkpoints/', run.name)
  os.makedirs(target_dir, exist_ok=True)

  target_path = os.path.join(target_dir, config_file.name)
  utils.print(f'Downloading wandb config to {target_path}', color='blue')
  config_file.download(root=target_dir, replace=True)
  utils.print('Download complete!', color='blue')

  cfg = config.Config.load(target_path)

  return cfg
