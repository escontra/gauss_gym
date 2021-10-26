from typing import List
import argparse

from huggingface_hub import snapshot_download, whoami
import pathlib
import sys
import os
import shutil
import tarfile
import re

directory = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(directory.parent))
__package__ = directory.name

from .utils import s5cmd_cp, s3_directory_exists, download_from_s3  # noqa: E402

print(whoami())

os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

_DEFAULT_TOPICS = [
  'hdr_front',
  'hdr_left',
  'hdr_right',
  'livox_points_undistorted',
  'hesai_points_undistorted',
  'anymal_state_odometry',
  'tf',
  'dlio_map_odometry',
  'depth_camera_front_upper',
  'depth_camera_left',
  'depth_camera_rear_upper',
  'depth_camera_right',
]


# The script is configured to download the data required for:
# -- dynamic_points_filtering_using_images.py
# -- generate_elevation_maps.py
# You can change the mission and set the dataset_folder to your desired location.


def move_dataset(cache, dataset_folder, allow_patterns=['*']):
  print(f'Start moving from {cache} to {dataset_folder} !')

  def convert_glob_patterns_to_regex(glob_patterns):
    regex_parts = []
    for pat in glob_patterns:
      # Escape regex special characters except for * and ?
      pat = re.escape(pat)
      # Convert escaped glob wildcards to regex equivalents
      pat = pat.replace(r'\*', '.*').replace(r'\?', '.')
      # Make sure it matches full paths
      regex_parts.append(f'.*{pat}$')

    # Join with |
    combined = '|'.join(regex_parts)
    return re.compile(combined)

  pattern = convert_glob_patterns_to_regex(allow_patterns)
  files = [f for f in pathlib.Path(cache).rglob('*') if pattern.match(str(f))]
  tar_files = [f for f in files if f.suffix == '.tar']

  for source_path in tar_files:
    dest_path = dataset_folder / source_path.relative_to(cache)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
      with tarfile.open(source_path, 'r') as tar:
        tar.extractall(path=dest_path.parent)
    except tarfile.ReadError as e:
      print(f"Error opening or extracting tar file '{source_path}': {e}")
    except Exception as e:
      print(f'An unexpected error occurred while processing {source_path}: {e}')

  other_files = [f for f in files if not f.suffix == '.tar' and f.is_file()]
  for source_path in other_files:
    dest_path = dataset_folder / source_path.relative_to(cache)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dest_path)

  print(f'Moved data from {cache} to {dataset_folder} !')


def download_mission(
  mission: str,
  dataset_folder: pathlib.Path,
  topics: List[str] = _DEFAULT_TOPICS,
  s3_path: str = None,
  overwrite_s3: bool = False,
):
  if s3_path is not None and not overwrite_s3:
    assert s3_path.startswith('s3://')
    directory_exists = s3_directory_exists(
      s3_path[5:], os.path.join('grand_tour', mission)
    )
    if directory_exists:
      print(f'Downloading {s3_path} to {dataset_folder / mission}')
      download_from_s3(
        s3_path=os.path.join(s3_path, 'grand_tour', mission) + '/',
        local_path=dataset_folder / mission,
        recursive=True,
      )
      return

  allow_patterns = [f'{mission}/*.yaml', '*/.zgroup']
  allow_patterns += [f'{mission}/*{topic}*' for topic in topics]
  hugging_face_data_cache_path = snapshot_download(
    repo_id='leggedrobotics/grand_tour_dataset',
    allow_patterns=allow_patterns,
    repo_type='dataset',
  )
  move_dataset(
    hugging_face_data_cache_path, dataset_folder, allow_patterns=allow_patterns
  )
  if s3_path is not None:
    src = str(dataset_folder / mission).rstrip('/')
    dst = os.path.join(s3_path, 'grand_tour') + '/'
    print(f'Copying {src} to {dst}')
    s5cmd_cp(src, dst)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Download Grand Tour dataset missions',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )

  parser.add_argument(
    '--mission',
    type=str,
    default='2024-11-04-10-57-34',
    help="Mission name to download (e.g., '2024-11-04-10-57-34')",
  )

  parser.add_argument(
    '--dataset-folder',
    type=str,
    default='~/grand_tour_dataset',
    help='Dataset folder path where data will be downloaded',
  )

  parser.add_argument(
    '--topics',
    nargs='*',
    default=_DEFAULT_TOPICS,
    help='List of topics to download (space-separated)',
  )

  parser.add_argument(
    '--s3-path', type=str, default=None, help='S3 path to copy the data to.'
  )

  args = parser.parse_args()

  mission = args.mission
  dataset_folder = pathlib.Path(args.dataset_folder).expanduser()
  topics = args.topics

  print(f'Downloading mission: {mission}')
  print(f'Dataset folder: {dataset_folder}')
  print(f'Topics: {topics}')

  dataset_folder.mkdir(parents=True, exist_ok=True)
  download_mission(mission, dataset_folder, topics, args.s3_path)
