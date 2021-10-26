#!/usr/bin/env python3
"""
Upload slices from S3 grand_tour dataset to HuggingFace.

This script:
1. Lists all missions from S3 (e.g., mission1_nerfstudio, mission2_nerfstudio, ...)
2. For each mission:
   - Downloads slices data to a temporary directory
   - Filters out: hidden files, nerfstudio_models/, mesh_filled.ply, pcd.ply
   - Uploads to HF under <mission>/splats/ path (with slice_ prefix removed)
   - Cleans up downloaded data to save disk space

Path transformation:
  S3: <mission>_nerfstudio/slices/slice_*/*
  HF: <mission>/splats/*/* (slices renamed to splats, slice_ prefix removed)
"""

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

# Configuration
BUCKET = 's3://far-falcon-assets/grand_tour'
REPO_ID = 'leggedrobotics/grand_tour_dataset'
TOKEN = os.getenv('HF_TOKEN')

print(f'TOKEN: {TOKEN}')

# Files to ignore (matching copy_data.sh filters)
IGNORE_PATTERNS = [
  r'(^|/)\.',  # Hidden files
  r'/nerfstudio_models/',  # nerfstudio_models directory
  r'mesh_filled\.ply$',  # mesh_filled.ply files
  r'pcd\.ply$',  # pcd.ply files
]

# Required files that must exist in each slice for it to be included
SLICE_REQUIRED_FILES = [
  'splatfacto/splat.ply',
  'splatfacto/config.yml',
  'splatfacto/dataparser_transforms.json',
  'mesh.ply',
  'mesh_filled_decimated.ply',
  'transforms.json',
]


def should_ignore_file(path: str) -> bool:
  """Check if a file should be ignored based on ignore patterns."""
  for pattern in IGNORE_PATTERNS:
    if re.search(pattern, path):
      return True
  return False


def validate_slices(files: List[str]) -> set:
  """
  Validate which slices have all required files.

  Args:
    files: List of relative file paths like "slice_XXX/file.ext"

  Returns:
    Set of valid slice names (e.g., {"slice_XXX", "slice_YYY", ...})
  """
  from collections import defaultdict

  # Group files by slice
  slice_files = defaultdict(list)
  for file_path in files:
    # Extract slice name (first directory component)
    parts = file_path.split('/')
    if len(parts) >= 2 and parts[0].startswith('slice_'):
      slice_name = parts[0]
      # Store the path relative to the slice
      relative_path = '/'.join(parts[1:])
      slice_files[slice_name].append(relative_path)

  # Check which slices have all required files
  valid_slices = set()
  for slice_name, files_in_slice in slice_files.items():
    has_all_required = all(
      any(required_file in f for f in files_in_slice)
      for required_file in SLICE_REQUIRED_FILES
    )

    if has_all_required:
      valid_slices.add(slice_name)
    else:
      missing = [
        req for req in SLICE_REQUIRED_FILES if not any(req in f for f in files_in_slice)
      ]
      print(f'  Skipping {slice_name}: missing {missing}')

  return valid_slices


def list_missions() -> List[str]:
  """List all mission names from S3 bucket."""
  print('Listing missions from S3...')

  cmd = ['s5cmd', 'ls', f'{BUCKET}/']
  result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())

  if result.returncode != 0:
    print('Error running s5cmd:')
    print(f'STDERR: {result.stderr}')
    raise RuntimeError(f's5cmd failed with return code {result.returncode}')

  if not result.stdout.strip():
    print('WARNING: No output from s5cmd ls command')
    print('This might be due to:')
    print('  1. No data in the S3 bucket')
    print('  2. Missing AWS credentials in the environment')
    print('  3. Insufficient permissions')
    return []

  missions = []
  for line in result.stdout.strip().split('\n'):
    if not line:
      continue
    # Extract directory name from s5cmd ls output
    # Format: "                                  DIR  directory_name/"
    parts = line.split()
    if len(parts) >= 2 and parts[0] == 'DIR':
      # Directory name is the second part (after DIR)
      mission_dir = parts[1].rstrip('/')
      # Extract mission name (remove _nerfstudio suffix)
      if mission_dir.endswith('_nerfstudio'):
        mission = mission_dir[: -len('_nerfstudio')]
        missions.append(mission)

  print(f'Found {len(missions)} missions')
  return missions


def download_mission_slices(
  mission: str, temp_dir: Path, dry_run: bool = False
) -> Path:
  """
  Download slices for a mission from S3 to temp directory.
  Returns the path containing downloaded data.
  """
  print(f'\n{"=" * 60}')
  print(f'Processing mission: {mission}')
  print(f'{"=" * 60}')

  # Create mission directory structure
  # Local structure: mission/splats/ (renamed from slices)
  mission_dir = temp_dir / mission / 'splats'
  if not dry_run:
    mission_dir.mkdir(parents=True, exist_ok=True)
  else:
    print(f'[DRY RUN] Would create directory: {mission_dir}')

  # List all files in mission slices
  s3_prefix = f'{BUCKET}/{mission}_nerfstudio/slices/'
  print(f'Listing files from {s3_prefix}...')

  cmd = ['s5cmd', 'ls', f'{s3_prefix}*']
  result = subprocess.run(cmd, capture_output=True, text=True, check=True)

  # First pass: collect all files and apply ignore patterns
  all_files = []
  for line in result.stdout.strip().split('\n'):
    if not line:
      continue

    parts = line.split()
    if len(parts) < 4:
      continue

    # s5cmd ls output format for files:
    # YYYY/MM/DD HH:MM:SS filesize relative/path/to/file
    # parts[0]: date, parts[1]: time, parts[2]: size, parts[3+]: relative path
    # Note: path might contain spaces, so join everything from index 3 onwards
    relative_path = ' '.join(parts[3:])

    # When listing with wildcard like "slices/*", the output is relative to the prefix
    # e.g., "slice_XXX/file.ext" rather than full path
    # relative_path is like: slice_0/file.ext or slice_0/subdir/file.ext

    # Apply ignore patterns
    if should_ignore_file(relative_path):
      continue

    all_files.append(relative_path)

  print(f'Found {len(all_files)} files after ignore filtering')

  # Validate slices - only include slices with all required files
  valid_slices = validate_slices(all_files)
  print(f'Found {len(valid_slices)} valid slices (with all required files)')

  # Second pass: prepare download list for valid slices only
  download_list = []
  for relative_path in all_files:
    # Extract slice name (e.g., "slice_1148c086")
    slice_name = relative_path.split('/')[0]

    # Only include files from valid slices
    if slice_name not in valid_slices:
      continue

    # Transform path: remove "slice_" prefix from directory name
    # relative_path is like: slice_1148c086/mesh.ply
    # We want: 1148c086/mesh.ply
    path_parts = relative_path.split('/', 1)
    if len(path_parts) == 2 and path_parts[0].startswith('slice_'):
      # Remove "slice_" prefix
      transformed_slice_name = path_parts[0][6:]  # Remove "slice_" (6 characters)
      transformed_relative_path = f'{transformed_slice_name}/{path_parts[1]}'
    else:
      # Fallback: use original path
      transformed_relative_path = relative_path

    # Full S3 path for download
    full_s3_path = f'{s3_prefix}{relative_path}'

    # Prepare destination path: temp_dir/mission/splats/XXX/file.ext
    dest_path = mission_dir / transformed_relative_path
    download_list.append((full_s3_path, dest_path))

  print(f'Total files to download: {len(download_list)}')

  # Download files
  if not download_list:
    print('No files to download for this mission.')
    return temp_dir / mission

  if dry_run:
    print(f'[DRY RUN] Would download {len(download_list)} files')
    print('[DRY RUN] Sample files (first 10):')
    for s3_path, dest_path in download_list[:10]:
      print(f'  {s3_path} -> {dest_path}')
    if len(download_list) > 10:
      print(f'  ... and {len(download_list) - 10} more files')
    return temp_dir / mission

  # Create download commands file for s5cmd
  with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
    cmdfile_path = f.name
    for s3_path, dest_path in download_list:
      # Ensure parent directory exists
      dest_path.parent.mkdir(parents=True, exist_ok=True)
      f.write(f'cp {s3_path} {dest_path}\n')

  try:
    print(f'Downloading {len(download_list)} files...')
    subprocess.run(['s5cmd', 'run', cmdfile_path], check=True)
    print('Download complete!')
  finally:
    os.unlink(cmdfile_path)

  return temp_dir / mission


def upload_to_huggingface(mission: str, mission_path: Path, dry_run: bool = False):
  """Upload mission data to HuggingFace dataset."""
  print(f'\nUploading {mission} to HuggingFace...')

  if dry_run:
    print(f'[DRY RUN] Would upload from: {mission_path}')
    print(f'[DRY RUN] Would upload to: {REPO_ID}/{mission}/')
    print('[DRY RUN] Skipping actual upload')
    return

  # Import HfApi only when actually needed (not in dry-run)
  from huggingface_hub import HfApi

  api = HfApi(token=TOKEN)

  # Ensure repository exists
  #   try:
  #     api.create_repo(
  #       repo_id=REPO_ID,
  #       repo_type='dataset',
  #       exist_ok=True,
  #       token=TOKEN,
  #     )
  #   except Exception as e:
  #     print(f'Repository already exists or error: {e}')

  # Upload the mission folder
  # mission_path points to: temp_dir/mission/
  # This will be uploaded to HF as: mission/splats/...
  try:
    api.upload_folder(
      folder_path=str(mission_path),
      repo_id=REPO_ID,
      repo_type='dataset',
      path_in_repo=mission,  # Upload to <mission>/ in the repo
      commit_message=f'Upload splat/mesh slices for mission {mission}',
      commit_description=f'Upload splat/mesh slices for mission {mission}',
      token=TOKEN,
    )
    print(f'Successfully uploaded {mission} to HuggingFace!')
  except Exception as e:
    raise RuntimeError(f'Error uploading {mission} to HuggingFace: {e}')


def cleanup_mission_data(mission_path: Path, dry_run: bool = False):
  """Delete downloaded mission data to free up disk space."""
  if dry_run:
    print(f'[DRY RUN] Would clean up: {mission_path}')
    return

  print(f'\nCleaning up {mission_path}...')
  if mission_path.exists():
    shutil.rmtree(mission_path)
    print(f'Cleaned up {mission_path}')


def main():
  """Main execution flow."""
  parser = argparse.ArgumentParser(
    description='Upload slices from S3 grand_tour dataset to HuggingFace',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Dry run mode - show what would be done without downloading/uploading
  python upload_slices_to_hf.py --dry-run

  # Normal mode - actually download and upload
  python upload_slices_to_hf.py
        """,
  )
  parser.add_argument(
    '--dry-run',
    action='store_true',
    help='Dry run mode: show what would be done without actually downloading/uploading',
  )
  args = parser.parse_args()

  if args.dry_run:
    print('=' * 60)
    print('DRY RUN MODE - No actual downloads or uploads will occur')
    print('=' * 60)

  if not TOKEN and not args.dry_run:
    raise ValueError('HF_TOKEN environment variable not set')

  # List all missions
  missions = list_missions()

  if not missions:
    print('No missions found in S3 bucket.')
    return

  # Process each mission one at a time
  for i, mission in enumerate(missions, 1):
    print(f'\n\n{"#" * 60}')
    print(f'# Processing mission {i}/{len(missions)}: {mission}')
    print(f'{"#" * 60}')

    # Create temporary directory for this mission
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_dir_path = Path(temp_dir)

      try:
        # Download mission slices
        mission_path = download_mission_slices(
          mission, temp_dir_path, dry_run=args.dry_run
        )

        # Upload to HuggingFace
        upload_to_huggingface(mission, mission_path, dry_run=args.dry_run)

        print(f'\n✓ Successfully processed {mission}')
      except Exception as e:
        print(f'\n✗ Error processing {mission}: {e}')
        import traceback

        traceback.print_exc()
        # Continue with next mission
        continue

    # temp_dir is automatically cleaned up when exiting the context manager
    if args.dry_run:
      print('[DRY RUN] Temporary directory would be cleaned up automatically')
    else:
      print(f'Temporary data for {mission} cleaned up automatically')

  print(f'\n\n{"=" * 60}')
  print('All missions processed!')
  print(f'{"=" * 60}')


if __name__ == '__main__':
  main()
