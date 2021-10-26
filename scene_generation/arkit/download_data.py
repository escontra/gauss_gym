from __future__ import annotations

import argparse
import subprocess
import pandas as pd
import math
import os

from typing import Generic, TypeVar
import shutil
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


def s5cmd_ls(
  path: str,
  max_retries: int = 3,
) -> list[str]:
  """Execute s5cmd ls in a subprocess shell and return the results.

  This is useful for listing S3 objects/directories.

  See `s5cmd ls -h` for semantics of specifying `path`.

  Parameters
  ----------
  path : str
      S3 path to list (e.g., 's3://bucket/prefix/' or 's3://bucket/prefix/*').
  max_retries : int
      Number of retries.

  Returns
  -------
  list[str]
      List of S3 paths/objects found.
  """
  assert shutil.which('s5cmd') is not None, 's5cmd not found'
  try:
    result = RetryWrapper(subprocess.check_output, max_retries=max_retries)(
      ['s5cmd', 'ls', path], text=True
    )
    # Parse s5cmd ls output (format: "date time size path" or "DIR path")
    lines = [line.strip() for line in result.split('\n') if line.strip()]
    # Extract the path (last column after splitting by whitespace)
    return [line.split()[-1] for line in lines if line]
  except subprocess.CalledProcessError as e:
    print(f'Failed to ls {path}: {e}')
    return []


def s3_path_exists(
  path: str,
  max_retries: int = 3,
) -> bool:
  """Check if an S3 path exists.

  Parameters
  ----------
  path : str
      S3 path to check (e.g., 's3://bucket/prefix/').
  max_retries : int
      Number of retries.

  Returns
  -------
  bool
      True if the path exists, False otherwise.
  """
  assert shutil.which('s5cmd') is not None, 's5cmd not found'
  try:
    RetryWrapper(subprocess.check_output, max_retries=max_retries)(
      ['s5cmd', 'ls', path],
      text=True,
      stderr=subprocess.DEVNULL,  # Suppress error output
    )
    return True
  except subprocess.CalledProcessError:
    return False


ARkitscense_url = (
  'https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1'
)
TRAINING = 'Training'
VALIDATION = 'Validation'
HIGRES_DEPTH_ASSET_NAME = 'highres_depth'
POINT_CLOUDS_FOLDER = 'laser_scanner_point_clouds'

default_raw_dataset_assets = [
  'mov',
  'annotation',
  'mesh',
  'confidence',
  'highres_depth',
  'lowres_depth',
  'lowres_wide.traj',
  'lowres_wide',
  'lowres_wide_intrinsics',
  'ultrawide',
  'ultrawide_intrinsics',
  'vga_wide',
  'vga_wide_intrinsics',
]

missing_3dod_assets_video_ids = [
  '47334522',
  '47334523',
  '42897421',
  '45261582',
  '47333152',
  '47333155',
  '48458535',
  '48018733',
  '47429677',
  '48458541',
  '42897848',
  '47895482',
  '47333960',
  '47430089',
  '42899148',
  '42897612',
  '42899153',
  '42446164',
  '48018149',
  '47332198',
  '47334515',
  '45663223',
  '45663226',
  '45663227',
]


def raw_files(video_id, assets, metadata):
  file_names = []
  for asset in assets:
    if HIGRES_DEPTH_ASSET_NAME == asset:
      in_upsampling = metadata.loc[
        metadata['video_id'] == float(video_id), ['is_in_upsampling']
      ].iat[0, 0]
      if not in_upsampling:
        print(
          f'Skipping asset {asset} for video_id {video_id} - Video not in upsampling dataset'
        )
        continue  # highres_depth asset only available for video ids from upsampling dataset

    if asset in [
      'confidence',
      'highres_depth',
      'lowres_depth',
      'lowres_wide',
      'lowres_wide_intrinsics',
      'ultrawide',
      'ultrawide_intrinsics',
      'vga_wide',
      'vga_wide_intrinsics',
    ]:
      file_names.append(asset + '.zip')
    elif asset == 'mov':
      file_names.append(f'{video_id}.mov')
    elif asset == 'mesh':
      if video_id not in missing_3dod_assets_video_ids:
        file_names.append(f'{video_id}_3dod_mesh.ply')
    elif asset == 'annotation':
      if video_id not in missing_3dod_assets_video_ids:
        file_names.append(f'{video_id}_3dod_annotation.json')
    elif asset == 'lowres_wide.traj':
      if video_id not in missing_3dod_assets_video_ids:
        file_names.append('lowres_wide.traj')
    else:
      raise Exception(f'No asset = {asset} in raw dataset')
  return file_names


def download_file(url, file_name, dst):
  os.makedirs(dst, exist_ok=True)
  filepath = os.path.join(dst, file_name)

  if not os.path.isfile(filepath):
    command = f'curl {url} -o {file_name}.tmp --fail'
    print(f'Downloading file {filepath}')
    try:
      subprocess.check_call(command, shell=True, cwd=dst)
    except Exception as error:
      print(f'Error downloading {url}, error: {error}')
      return False
    os.rename(filepath + '.tmp', filepath)
  else:
    print(f'WARNING: skipping download of existing file: {filepath}')
  return True


def unzip_file(file_name, dst, keep_zip=True):
  filepath = os.path.join(dst, file_name)
  print(f'Unzipping zip file {filepath}')
  command = f'unzip -oq {filepath} -d {dst}'
  try:
    subprocess.check_call(command, shell=True)
  except Exception as error:
    print(f'Error unzipping {filepath}, error: {error}')
    return False
  if not keep_zip:
    os.remove(filepath)
  return True


def download_laser_scanner_point_clouds_for_video(video_id, metadata, download_dir):
  video_metadata = metadata.loc[metadata['video_id'] == float(video_id)]
  visit_id = video_metadata['visit_id'].iat[0]
  has_laser_scanner_point_clouds = video_metadata['has_laser_scanner_point_clouds'].iat[
    0
  ]

  if not has_laser_scanner_point_clouds:
    print(f'Warning: Laser scanner point clouds for video {video_id} are not available')
    return

  if math.isnan(visit_id) or not visit_id.is_integer():
    print(
      f'Warning: Downloading laser scanner point clouds for video {video_id} failed - Bad visit id {visit_id}'
    )
    return

  visit_id = int(visit_id)  # Expecting an 8 digit integer
  laser_scanner_point_clouds_ids = laser_scanner_point_clouds_for_visit_id(
    visit_id, download_dir
  )

  for point_cloud_id in laser_scanner_point_clouds_ids:
    download_laser_scanner_point_clouds(point_cloud_id, visit_id, download_dir)


def laser_scanner_point_clouds_for_visit_id(visit_id, download_dir):
  point_cloud_to_visit_id_mapping_filename = 'laser_scanner_point_clouds_mapping.csv'
  if not os.path.exists(point_cloud_to_visit_id_mapping_filename):
    point_cloud_to_visit_id_mapping_url = f'{ARkitscense_url}/raw/laser_scanner_point_clouds/{point_cloud_to_visit_id_mapping_filename}'
    if not download_file(
      point_cloud_to_visit_id_mapping_url,
      point_cloud_to_visit_id_mapping_filename,
      download_dir,
    ):
      print(
        f'Error downloading point cloud for visit_id {visit_id} at location '
        f'{point_cloud_to_visit_id_mapping_url}'
      )
      return []

  point_cloud_to_visit_id_mapping_filepath = os.path.join(
    download_dir, point_cloud_to_visit_id_mapping_filename
  )
  point_cloud_to_visit_id_mapping = pd.read_csv(
    point_cloud_to_visit_id_mapping_filepath
  )
  point_cloud_ids = point_cloud_to_visit_id_mapping.loc[
    point_cloud_to_visit_id_mapping['visit_id'] == visit_id,
    ['laser_scanner_point_clouds_id'],
  ]
  point_cloud_ids_list = [scan_id[0] for scan_id in point_cloud_ids.values]

  return point_cloud_ids_list


def download_laser_scanner_point_clouds(
  laser_scanner_point_cloud_id, visit_id, download_dir
):
  laser_scanner_point_clouds_folder_path = os.path.join(
    download_dir, POINT_CLOUDS_FOLDER, str(visit_id)
  )
  os.makedirs(laser_scanner_point_clouds_folder_path, exist_ok=True)

  for extension in ['.ply', '_pose.txt']:
    filename = f'{laser_scanner_point_cloud_id}{extension}'
    filepath = os.path.join(laser_scanner_point_clouds_folder_path, filename)
    if os.path.exists(filepath):
      return
    file_url = f'{ARkitscense_url}/raw/laser_scanner_point_clouds/{visit_id}/{filename}'
    download_file(file_url, filename, laser_scanner_point_clouds_folder_path)


def get_metadata(dataset, download_dir):
  filename = 'metadata.csv'
  url = (
    f'{ARkitscense_url}/threedod/{filename}'
    if '3dod' == dataset
    else f'{ARkitscense_url}/{dataset}/{filename}'
  )
  dst_folder = os.path.join(download_dir, dataset)
  dst_file = os.path.join(dst_folder, filename)

  if not download_file(url, filename, dst_folder):
    return

  metadata = pd.read_csv(dst_file)
  return metadata


def download_data(
  dataset,
  video_ids,
  dataset_splits,
  download_dir,
  keep_zip,
  raw_dataset_assets,
  should_download_laser_scanner_point_cloud,
):
  metadata = get_metadata(dataset, download_dir)
  if None is metadata:
    print(f'Error retrieving metadata for dataset {dataset}')
    return

  # already_downloaded = {}
  # for split in set(dataset_splits):
  #   s3_path = f's3://far-falcon-assets/ARKitScenes/{dataset}/{split}/'
  #   if not s3_path_exists(s3_path, max_retries=0):
  #     already_downloaded[split] = []
  #     continue

  #   paths = s5cmd_ls(s3_path)
  #   paths = [p.rstrip('/') for p in paths]
  #   already_downloaded[split] = paths

  download_dir = os.path.abspath(download_dir)
  for video_id in sorted(set(video_ids)):
    split = dataset_splits[video_ids.index(video_id)]
    dst_dir = os.path.join(download_dir, dataset, split)

    # if video_id in already_downloaded[split]:
    #   print(f'Video {split}/{video_id} already downloaded, skipping...')
    #   continue
    # else:
    #   print(f'Video {split}/{video_id} not downloaded, downloading...')

    if dataset == 'raw':
      url_prefix = ''
      file_names = []
      if not raw_dataset_assets:
        print(f'Warning: No raw assets given for video id {video_id}')
      else:
        dst_dir = os.path.join(dst_dir, str(video_id))
        url_prefix = f'{ARkitscense_url}/raw/{split}/{video_id}' + '/{}'
        file_names = raw_files(video_id, raw_dataset_assets, metadata)
    elif dataset == '3dod':
      url_prefix = f'{ARkitscense_url}/threedod/{split}' + '/{}'
      file_names = [
        f'{video_id}.zip',
      ]
    elif dataset == 'upsampling':
      url_prefix = f'{ARkitscense_url}/upsampling/{split}' + '/{}'
      file_names = [
        f'{video_id}.zip',
      ]
    else:
      raise Exception(f'No such dataset = {dataset}')

    if should_download_laser_scanner_point_cloud and dataset == 'raw':
      # Point clouds only available for the raw dataset
      download_laser_scanner_point_clouds_for_video(video_id, metadata, download_dir)

    for file_name in file_names:
      dst_path = os.path.join(dst_dir, file_name)
      url = url_prefix.format(file_name)

      if not file_name.endswith('.zip') or not os.path.isdir(dst_path[: -len('.zip')]):
        download_file(url, dst_path, dst_dir)
      else:
        print(f'WARNING: skipping download of existing zip file: {dst_path}')
      if file_name.endswith('.zip') and os.path.isfile(dst_path):
        unzip_file(file_name, dst_dir, keep_zip)
    if video_id not in missing_3dod_assets_video_ids:
      s3_path = f's3://far-falcon-assets/ARKitScenes/{dataset}/{split}/'
      if dataset == 'raw':
        final_dst_dir = dst_dir
      elif dataset == '3dod':
        final_dst_dir = os.path.join(dst_dir, video_id)
      s5cmd_cp(final_dst_dir, s3_path)
      print(f'COPIED {final_dst_dir} to {s3_path}')
      time.sleep(1)
      shutil.rmtree(final_dst_dir)

  if dataset == 'upsampling' and VALIDATION in dataset_splits:
    val_attributes_file = 'val_attributes.csv'
    url = f'{ARkitscense_url}/upsampling/{VALIDATION}/{val_attributes_file}'
    dst_file = os.path.join(download_dir, dataset, VALIDATION)
    download_file(url, val_attributes_file, dst_file)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('dataset', choices=['3dod', 'upsampling', 'raw'])

  parser.add_argument(
    '--split',
    choices=['Training', 'Validation'],
  )

  parser.add_argument('--video_id', nargs='*')

  parser.add_argument(
    '--video_id_csv',
  )

  parser.add_argument(
    '--download_dir',
    default='data',
  )

  parser.add_argument('--keep_zip', action='store_true')

  parser.add_argument('--download_laser_scanner_point_cloud', action='store_true')

  parser.add_argument(
    '--raw_dataset_assets', nargs='+', choices=default_raw_dataset_assets
  )

  args = parser.parse_args()
  assert args.video_id is not None or args.video_id_csv is not None, (
    'video_id or video_id_csv must be specified'
  )
  assert args.video_id is None or args.video_id_csv is None, (
    'only video_id or video_id_csv must be specified'
  )
  assert args.video_id is None or args.split is not None, (
    'given video_id the split argument must be specified'
  )

  if args.video_id is not None:
    video_ids_ = args.video_id
    splits_ = splits = [
      args.split,
    ] * len(video_ids_)
  elif args.video_id_csv is not None:
    df = pd.read_csv(args.video_id_csv)
    if args.split is not None:
      df = df[df['fold'] == args.split]
    video_ids_ = df['video_id'].to_list()
    video_ids_ = list(map(str, video_ids_))  # Expecting video id to be a string
    splits_ = df['fold'].to_list()
  else:
    raise Exception('No video ids specified')

  download_data(
    args.dataset,
    video_ids_,
    splits_,
    args.download_dir,
    args.keep_zip,
    default_raw_dataset_assets,
    #   args.raw_dataset_assets,
    args.download_laser_scanner_point_cloud,
  )
