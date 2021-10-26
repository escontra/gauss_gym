import time
from typing import Optional, Union, List
import huggingface_hub
from huggingface_hub import list_repo_tree, errors, HfApi
from huggingface_hub.hf_api import RepoFile, RepoFolder
from collections import defaultdict

from gauss_gym import utils
from gauss_gym.utils import path


def walk(repo_id, repo_type='dataset'):
  items = list(list_repo_tree(repo_id, repo_type=repo_type, recursive=True))

  # Group by directory
  dir_contents = defaultdict(lambda: {'dirs': set(), 'files': []})

  for item in items:
    if isinstance(item, RepoFolder):
      parent = '/'.join(item.path.split('/')[:-1]) if '/' in item.path else ''
      dir_name = item.path.split('/')[-1]
      dir_contents[parent]['dirs'].add(dir_name)
    elif isinstance(item, RepoFile):
      parent = '/'.join(item.path.split('/')[:-1]) if '/' in item.path else ''
      file_name = item.path.split('/')[-1]
      dir_contents[parent]['files'].append(file_name)

  # Yield in os.walk format
  for root in sorted(dir_contents.keys()):
    dirs = sorted(list(dir_contents[root]['dirs']))
    files = sorted(dir_contents[root]['files'])
    yield root, dirs, files


def snapshot_download(
  repo_id: str,
  repo_type: str = 'dataset',
  allow_patterns: Optional[Union[List[str], str]] = None,
  wait_for_rank_zero_download: bool = False,
  wait_for_rank_zero_timeout: float = 500.0,
):
  if wait_for_rank_zero_download:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    files = list(
      huggingface_hub.utils.filter_repo_objects(
        items=files, allow_patterns=allow_patterns
      )
    )
    download_path = None
    start_time = time.time()
    while download_path is None:
      try:
        incomplete_path = huggingface_hub.snapshot_download(
          repo_id=repo_id,
          repo_type=repo_type,
          local_files_only=True,
          allow_patterns=allow_patterns,
        )
        missing = [
          f for f in files if not (path.LocalPath(incomplete_path) / f).exists()
        ]
        if len(missing) != 0:
          raise errors.LocalEntryNotFoundError(f'Missing {len(missing)} files')
        else:
          download_path = incomplete_path
      except errors.LocalEntryNotFoundError as e:
        utils.print(f'Waiting for {repo_id} to become available: {e}', color='yellow')
        time.sleep(1)
      if time.time() - start_time > wait_for_rank_zero_timeout:
        raise RuntimeError(f'Timed out waiting for {repo_id} to become available')
  else:
    download_path = huggingface_hub.snapshot_download(
      repo_id=repo_id, repo_type=repo_type, allow_patterns=allow_patterns
    )
  return download_path
