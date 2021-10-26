from typing import Generic, TypeVar
import shutil
import subprocess
import os
import time
import random

P = TypeVar('P')
T = TypeVar('T')


class RetryWrapper(Generic[P, T]):
  def __init__(
    self, fn, max_retries: int = 30, delay_s: float = 10, backoff_s: float = 60
  ) -> None:
    """Wrap function to retry if it fails, with randomized backoff between retries.

    Ideally we'd define this as a higher-order function, but it doesn't play nicely with multiprocessing due to pickling.

    Parameters
    ----------
    fn : Callable[P, T]
        The function to wrap.
    max_retries : int
        Maximum number of retries.
    delay_s : float, optional
        The backoff will be sampled from `uniform(delay_s, delay_s + backoff_s)`.
    backoff_s : float, optional.
        See `delay_s`.
    """
    self._fn = fn
    self._max_retries = max_retries
    self._delay_s = delay_s
    self._backoff_s = backoff_s

  def __call__(self, *args, **kwargs) -> T:
    if bool(int(os.getenv('DISABLE_RETRY_WRAPPER', '0'))):
      return self._fn(*args, **kwargs)

    delay_s = self._delay_s
    for i in range(self._max_retries + 1):
      try:
        return self._fn(*args, **kwargs)
      except Exception as e:
        if i == self._max_retries:
          raise e
        delay_s_randomized = delay_s + random.uniform(0, self._backoff_s)
        print(
          f'Caught exception {e}. Retrying {i + 1}/{self._max_retries} after {delay_s_randomized} seconds.'
        )
        time.sleep(delay_s_randomized)
        # Cap the delay at 5 minutes.
        delay_s = min(300.0, delay_s * 2)
    raise RuntimeError('Unreachable code')


def walk(s3_path: str):
  """Walk an S3 directory tree using s5cmd, yielding (root, dirs, files) tuples.

  Similar to os.walk, yields tuples of (root_path, list_of_dirs, list_of_files).

  Parameters
  ----------
  s3_path : str
      S3 path to walk (e.g., 's3://bucket/prefix/')

  Yields
  ------
  tuple
      (root, dirs, files) where root is the directory path, dirs is a list of
      subdirectories, and files is a list of files in that directory.
  """
  assert shutil.which('s5cmd') is not None, 's5cmd not found'

  # Ensure s3_path ends with /
  if not s3_path.endswith('/'):
    s3_path = s3_path + '/'

  # Use s5cmd ls to list all objects recursively
  try:
    result = subprocess.run(
      ['s5cmd', 'ls', s3_path + '*'], capture_output=True, text=True, check=True
    )
  except subprocess.CalledProcessError as e:
    print(f'Failed to list {s3_path}: {e}')
    return

  from collections import defaultdict

  # Parse s5cmd output
  # Format: date time size path
  dir_contents = defaultdict(lambda: {'dirs': set(), 'files': []})

  for line in result.stdout.strip().split('\n'):
    if not line:
      continue

    parts = line.split()
    if len(parts) < 4:
      continue

    # Last part is the path
    full_path = parts[-1]

    # Remove the s3_path prefix to get relative path
    if not full_path.startswith(s3_path):
      continue

    rel_path = full_path[len(s3_path) :]

    if full_path.endswith('/'):
      # It's a directory
      rel_path = rel_path.rstrip('/')
      if '/' in rel_path:
        parent = '/'.join(rel_path.split('/')[:-1])
        dir_name = rel_path.split('/')[-1]
      else:
        parent = ''
        dir_name = rel_path
      dir_contents[parent]['dirs'].add(dir_name)
    else:
      # It's a file
      if '/' in rel_path:
        parent = '/'.join(rel_path.split('/')[:-1])
        file_name = rel_path.split('/')[-1]
      else:
        parent = ''
        file_name = rel_path
      dir_contents[parent]['files'].append(file_name)

  # Yield in os.walk format
  for root in sorted(dir_contents.keys()):
    dirs = sorted(list(dir_contents[root]['dirs']))
    files = sorted(dir_contents[root]['files'])
    yield root, dirs, files


def find_files_by_pattern(s3_path: str, pattern: str):
  """Find all files matching a specific pattern in S3.

  Much faster than walk() for large S3 trees when you only care about
  specific files.

  Parameters
  ----------
  s3_path : str
      S3 path to search (e.g., 's3://bucket/prefix/')
  pattern : str
      Filename pattern to search for (e.g., 'mesh.ply' or '*.json')

  Returns
  -------
  list
      List of full S3 paths matching the pattern
  """
  assert shutil.which('s5cmd') is not None, 's5cmd not found'

  # Ensure s3_path ends with /
  if not s3_path.endswith('/'):
    s3_path = s3_path + '/'

  # Search for the specific file pattern
  search_pattern = f'{s3_path}**/{pattern}'
  print(search_pattern)

  try:
    result = subprocess.run(
      ['s5cmd', 'ls', search_pattern], capture_output=True, text=True, check=True
    )
  except subprocess.CalledProcessError:
    return []
  print(result.stdout)

  paths = []
  for line in result.stdout.strip().split('\n'):
    if not line:
      continue

    parts = line.split()
    if len(parts) < 4:
      continue

    # Last part is the path
    full_path = parts[-1]
    if full_path.startswith(s3_path) and not full_path.endswith('/'):
      paths.append(full_path)

  return paths


def list_directory(s3_path: str):
  """List immediate contents of a specific S3 directory (non-recursive).

  Parameters
  ----------
  s3_path : str
      S3 directory path to list (e.g., 's3://bucket/prefix/dir/')

  Returns
  -------
  tuple
      (subdirs, files) where subdirs is a list of subdirectory names and
      files is a list of filenames in the directory
  """
  assert shutil.which('s5cmd') is not None, 's5cmd not found'

  # Ensure s3_path ends with /
  if not s3_path.endswith('/'):
    s3_path = s3_path + '/'

  try:
    result = subprocess.run(
      ['s5cmd', 'ls', s3_path], capture_output=True, text=True, check=True
    )
  except subprocess.CalledProcessError:
    return [], []

  subdirs = []
  files = []

  for line in result.stdout.strip().split('\n'):
    if not line:
      continue

    parts = line.split()
    if len(parts) < 2:
      continue

    # s5cmd ls output format can be:
    # 1. "DIR  directory_name/" for directories
    # 2. "date time size file_path" for files (4+ parts)

    if parts[0] == 'DIR':
      # Directory format: "DIR  name/"
      dir_name = parts[1].rstrip('/')
      subdirs.append(dir_name)
    elif len(parts) >= 4:
      # File format: "date time size path"
      full_path = parts[-1]
      if not full_path.startswith(s3_path):
        continue

      # Get the name after the directory path
      name = full_path[len(s3_path) :]

      # Skip if it contains additional slashes (deeper nesting)
      if '/' in name:
        continue

      files.append(name)

  return subdirs, files


def s5cmd_cp(
  src: str,
  dst: str,
  max_retries: int = 3,
  num_parts: int = 5,
  part_size_mb: int = 50,
) -> None:
  """Execute s5cmd cp in a subprocess shell.

  This is useful for large files, which s5cmd can handle concurrently. Increase `num_parts` and `part_size_mb` for large files.

  See `s5cmd cp -h` for semantics of specifying `src` and `dst`.
  For instance, if `src` is an S3 "folder", it must end with a wildcard "/*" to be interpreted as a "folder".

  Parameters
  ----------
  max_retries : int
      Number of retries.
  num_parts : int
      Number of parts to split each file into and copy concurrently.
  part_size_mb : int
      Size of each part in MB.
  """
  assert shutil.which('s5cmd') is not None, 's5cmd not found'
  flags = ['-s', '-u', '-p', str(num_parts), '-c', str(part_size_mb)]
  try:
    RetryWrapper(subprocess.check_call, max_retries=max_retries)(
      ['s5cmd', 'cp'] + flags + [src, dst]
    )
  except subprocess.CalledProcessError as e:
    print(f'Failed to cp {src} to {dst}: {e}')
