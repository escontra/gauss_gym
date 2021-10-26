"""Clean up wandb runs and files."""

import wandb
import tqdm
import collections

from gauss_gym import utils
from gauss_gym.utils import flags


def get_files_for_folder_path(run, folder_path):
  return [file for file in run.files() if file.name.startswith(folder_path)]


def get_files_for_extension(run, extension):
  return [file for file in run.files() if file.name.endswith(extension)]


def delete_files(files, leave_num_files: int, dry_run: bool = False):
  # Determine number of files to leave for each step.
  file_dict = collections.defaultdict(list)
  for file in files:
    log_step = int(file.name.split('_')[-2])
    file_dict[log_step].append(file)

  sorted_file_dict = dict(sorted(file_dict.items(), key=lambda x: x[0]))

  if len(sorted_file_dict) <= leave_num_files:
    return
  else:
    remove_keys = list(sorted(sorted_file_dict.keys()))[:-leave_num_files]
    files = []
    for k in remove_keys:
      files.extend(sorted_file_dict[k])

  utils.print(
    f'\t\tDeleting {len(files):,} files: {[file.name for file in files[:5]]}',
    color='blue',
  )

  if dry_run:
    return

  for file in tqdm.tqdm(files):
    try:
      file.server_supports_delete_file_with_project_id = False
      file.delete()
    except Exception as e:
      utils.print(
        f'\t\t\tError deleting file: {file.name}, {str(e)}',
        bold=True,
        color='red',
      )

  utils.print(f'\t\tDeleted {len(files):,} files', color='green')


def main(argv=None):
  parsed, other = flags.Flags(
    {
      'dry_run': False,
      'history_threshold': 3,
      'entity': wandb.Api().default_entity,
      'project': '',
      'leave_num_files': 2,
    }
  ).parse_known(argv)
  args = flags.Flags(parsed).parse(other)
  utils.print(f'Running wandb_cleanup with args:\n{args}', color='blue')

  api = wandb.Api(timeout=29)

  image_extensions = ['png', 'jpg', 'jpeg', 'bmp', 'gif']
  media_extensions = ['mp4', 'mp3', 'wav', 'avi', 'mov']
  folder_paths_to_delete = [
    'media',
  ]
  extensions_to_delete = image_extensions + media_extensions

  if args.project != '':
    projects = [api.project(args.project, entity=args.entity)]
  else:
    projects = api.projects(args.entity)

  # Iterate over all projects
  for project in projects:
    utils.print(f'Processing project: {project.name}', color='cyan')

    # Iterate over all runs in the project
    for run in reversed(api.runs(f'{args.entity}/{project.name}')):
      utils.print(f'\tProcessing run: {run.name} ({run.id})', color='yellow')

      # Get history of the run to check it its an empty run. Delete if so.
      try:
        history = run.history(samples=args.history_threshold, pandas=False)
      except Exception as e:
        utils.print(
          f'\t\tError getting history for run: {run.id}, {str(e)}',
          bold=True,
          color='red',
        )
        continue

      if len(history) < args.history_threshold:
        utils.print(
          f'\t\tDeleting run because it only has {len(history)} rows of history: {run.id} [< {args.history_threshold}]',
          color='blue',
        )
        if not args.dry_run:
          run.delete()
        utils.print(f'\t\tDeleted run {run.name} ({run.id})', color='green')
        continue

      for ext in extensions_to_delete:
        files = get_files_for_extension(run, ext)
        delete_files(files, args.leave_num_files, args.dry_run)

      for folder_path in folder_paths_to_delete:
        files = get_files_for_folder_path(run, folder_path)
        delete_files(files, args.leave_num_files, args.dry_run)


if __name__ == '__main__':
  main()
