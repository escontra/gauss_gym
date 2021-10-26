#!/usr/bin/env python3
"""
Upload ARKit scenes from S3 to HuggingFace.

This script:
1. Lists all video IDs from S3 Training and Validation directories
2. For each video ID:
   - Validates that all required files exist in S3
   - Downloads only the required files to a temporary directory
   - Uploads to HF under training/<video_id>/ or validation/<video_id>/
   - Cleans up downloaded data to save disk space

Required files per video:
  - splatfacto/splat.ply
  - splatfacto/config.yml
  - splatfacto/dataparser_transforms.json
  - <video_id>_3dod_mesh.ply
  - <video_id>_3dod_mesh_filled_decimated.ply
  - lowres_wide.traj

Path transformation:
  S3: ARKitScenes/raw/Training/<video_id>/*
  HF: training/<video_id>/*

  S3: ARKitScenes/raw/Validation/<video_id>/*
  HF: validation/<video_id>/*
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

from huggingface_hub import HfApi, CommitOperationAdd


# Configuration
BUCKET = 's3://far-falcon-assets/ARKitScenes/raw'
REPO_ID = 'escontra/gauss_gym_arkit'
TOKEN = os.getenv('HF_TOKEN')

print(f'TOKEN: {TOKEN}')


def get_required_files_for_video(video_id: str) -> List[str]:
  """
  Get list of required files for a video ID.

  Args:
    video_id: The video ID (e.g., "40753679")

  Returns:
    List of required file paths relative to video directory
  """
  return [
    'splatfacto/splat.ply',
    'splatfacto/config.yml',
    'splatfacto/dataparser_transforms.json',
    f'{video_id}_3dod_mesh.ply',
    f'{video_id}_3dod_mesh_filled_decimated.ply',
    'lowres_wide.traj',
  ]


def list_videos(split: str) -> List[str]:
  """
  List all video IDs from S3 bucket for a given split.

  Args:
    split: Either "Training" or "Validation"

  Returns:
    List of video IDs
  """
  print(f'Listing {split} videos from S3...')

  cmd = ['s5cmd', 'ls', f'{BUCKET}/{split}/']
  result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())

  if result.returncode != 0:
    print('Error running s5cmd:')
    print(f'STDERR: {result.stderr}')
    raise RuntimeError(f's5cmd failed with return code {result.returncode}')

  if not result.stdout.strip():
    print(f'WARNING: No output from s5cmd ls command for {split}')
    print('This might be due to:')
    print('  1. No data in the S3 bucket')
    print('  2. Missing AWS credentials in the environment')
    print('  3. Insufficient permissions')
    return []

  videos = []
  for line in result.stdout.strip().split('\n'):
    if not line:
      continue
    # Extract directory name from s5cmd ls output
    # Format: "                                  DIR  directory_name/"
    parts = line.split()
    if len(parts) >= 2 and parts[0] == 'DIR':
      # Directory name is the second part (after DIR)
      video_id = parts[1].rstrip('/')
      videos.append(video_id)

  print(f'Found {len(videos)} videos in {split}')
  return videos


def download_video_batch(
  video_ids: List[str], split: str, temp_dir: Path, dry_run: bool = False
) -> List[Tuple[str, bool]]:
  """
  Download a batch of videos in parallel using s5cmd, then validate locally.

  Args:
    video_ids: List of video IDs to download
    split: Either "Training" or "Validation"
    temp_dir: Temporary directory for downloads
    dry_run: If True, only simulate the download

  Returns:
    List of tuples (video_id, success)
  """
  split_lower = split.lower()

  # Prepare download list for all videos in batch (no pre-validation)
  all_downloads = []

  print(f'\nPreparing batch download for {len(video_ids)} videos...')

  for video_id in video_ids:
    video_dir = temp_dir / split_lower / video_id

    if not dry_run:
      video_dir.mkdir(parents=True, exist_ok=True)

    # Get required files for this video
    required_files = get_required_files_for_video(video_id)
    s3_prefix = f'{BUCKET}/{split}/{video_id}/'

    # Add files to download list (no S3 validation - we'll check locally after download)
    for required_file in required_files:
      full_s3_path = f'{s3_prefix}{required_file}'
      dest_path = video_dir / required_file
      all_downloads.append((full_s3_path, dest_path))

  print(f'  Total files to download: {len(all_downloads)}')

  if dry_run:
    print(
      f'[DRY RUN] Would download {len(all_downloads)} files for {len(video_ids)} videos'
    )
    return [(vid, True) for vid in video_ids]

  # Create download commands file for s5cmd (batch download)
  with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
    cmdfile_path = f.name
    for s3_path, dest_path in all_downloads:
      # Ensure parent directory exists
      dest_path.parent.mkdir(parents=True, exist_ok=True)
      f.write(f'cp {s3_path} {dest_path}\n')

  try:
    print(f'  Downloading batch of {len(all_downloads)} files in parallel...')
    # Run s5cmd without check=True so it doesn't fail if some files are missing
    result = subprocess.run(
      ['s5cmd', 'run', cmdfile_path], capture_output=True, text=True
    )
    print('  ✓ Batch download complete!')

    if result.returncode != 0 and result.stderr:
      print('  Note: Some files may have failed to download (will validate locally)')
  finally:
    os.unlink(cmdfile_path)

  # Validate locally which videos have all required files
  results = []
  for video_id in video_ids:
    video_dir = temp_dir / split_lower / video_id
    required_files = get_required_files_for_video(video_id)

    # Check if all required files exist locally
    all_files_exist = True
    for required_file in required_files:
      file_path = video_dir / required_file
      if not file_path.exists():
        print(f'  ✗ {video_id}: missing {required_file}')
        all_files_exist = False
        break

    results.append((video_id, all_files_exist))

  valid_count = sum(1 for _, valid in results if valid)
  print(f'  Valid videos after download: {valid_count}/{len(video_ids)}')

  return results


def upload_batch_to_huggingface(
  videos_to_upload: List[Tuple[str, Path]], split: str, dry_run: bool = False
) -> Tuple[bool, str]:
  """
  Upload a batch of videos to HuggingFace dataset in a single commit.

  Args:
    videos_to_upload: List of (video_id, video_dir) tuples
    split: Either "training" or "validation" (lowercase)
    dry_run: If True, only simulate the upload

  Returns:
    Tuple of (success, error_message)
  """
  if dry_run:
    print(f'  [DRY RUN] Would upload {len(videos_to_upload)} videos in a single commit')
    return True, ''

  # Import HF modules only when actually needed (not in dry-run)

  api = HfApi(token=TOKEN)

  try:
    # Collect all file operations for all videos
    operations = []
    total_files = 0

    for video_id, video_dir in videos_to_upload:
      # Walk through all files in the video directory
      for file_path in video_dir.rglob('*'):
        if file_path.is_file():
          # Get relative path from temp_dir to include split prefix
          # The video_dir structure is: temp_dir/training/video_id/ or temp_dir/validation/video_id/
          # We want path_in_repo like: training/40753679/splatfacto/splat.ply
          rel_path = file_path.relative_to(video_dir.parent.parent)

          # Create operation with path in repo like: training/40753679/splatfacto/splat.ply
          operations.append(
            CommitOperationAdd(
              path_in_repo=str(rel_path), path_or_fileobj=str(file_path)
            )
          )
          total_files += 1

    print(
      f'  Creating single commit with {total_files} files from {len(videos_to_upload)} videos...'
    )

    # Single commit with all files from all videos in the batch
    api.create_commit(
      repo_id=REPO_ID,
      repo_type='dataset',
      operations=operations,
      commit_message=f'Upload batch of {len(videos_to_upload)} {split} videos ({total_files} files)',
    )

    return True, ''
  except Exception as e:
    return False, str(e)


def ensure_hf_repo_exists():
  """Ensure the HuggingFace repository exists."""

  api = HfApi(token=TOKEN)

  try:
    print(f'Ensuring repository {REPO_ID} exists...')
    api.create_repo(
      repo_id=REPO_ID,
      repo_type='dataset',
      exist_ok=True,
      token=TOKEN,
    )
    print(f'✓ Repository {REPO_ID} is ready\n')
  except Exception as e:
    print(f'Note: Repository check/creation returned: {e}\n')


def main():
  """Main execution flow."""
  parser = argparse.ArgumentParser(
    description='Upload ARKit scenes from S3 to HuggingFace',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Dry run mode - show what would be done without downloading/uploading
  python upload_to_hf.py --dry-run

  # Normal mode - download and upload in batches (batch: 10)
  python upload_to_hf.py

  # Process only Training or Validation split
  python upload_to_hf.py --split Training
  python upload_to_hf.py --split Validation

  # Use larger batch size for maximum speed (more disk usage)
  python upload_to_hf.py --batch-size 20

  # Test with a few videos first
  python upload_to_hf.py --split Training --limit 5

Note:
  - batch-size controls how many videos are downloaded and uploaded together (higher = faster but more disk)
  - All videos in a batch are uploaded in a single commit to avoid rate limits
        """,
  )
  parser.add_argument(
    '--dry-run',
    action='store_true',
    help='Dry run mode: show what would be done without actually downloading/uploading',
  )
  parser.add_argument(
    '--split',
    choices=['Training', 'Validation'],
    help='Process only a specific split (Training or Validation). If not specified, processes both.',
  )
  parser.add_argument(
    '--limit',
    type=int,
    help='Limit the number of videos to process (useful for testing)',
  )
  parser.add_argument(
    '--batch-size',
    type=int,
    default=10,
    help='Number of videos to download and upload together per batch (default: 10)',
  )
  args = parser.parse_args()

  if args.dry_run:
    print('=' * 60)
    print('DRY RUN MODE - No actual downloads or uploads will occur')
    print('=' * 60)

  if not TOKEN and not args.dry_run:
    raise ValueError('HF_TOKEN environment variable not set')

  # Determine which splits to process
  splits_to_process = [args.split] if args.split else ['Training', 'Validation']

  # Ensure HF repository exists (once at the start)
  if not args.dry_run:
    ensure_hf_repo_exists()

  # Process each split
  for split in splits_to_process:
    print(f'\n\n{"#" * 60}')
    print(f'# Processing {split} split')
    print(f'{"#" * 60}')

    # List all videos in this split
    videos = list_videos(split)

    if not videos:
      print(f'No videos found in {split} split.')
      continue

    # Apply limit if specified
    if args.limit:
      videos = videos[: args.limit]
      print(f'Limited to first {len(videos)} videos')

    split_lower = split.lower()
    successful_count = 0
    failed_count = 0

    # Process videos in batches
    batch_size = args.batch_size
    num_batches = (len(videos) + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
      start_idx = batch_num * batch_size
      end_idx = min(start_idx + batch_size, len(videos))
      batch_videos = videos[start_idx:end_idx]

      print(f'\n\n{"#" * 60}')
      print(
        f'# Batch {batch_num + 1}/{num_batches}: Processing videos {start_idx + 1}-{end_idx}'
      )
      print(f'{"#" * 60}')

      # Create a temporary directory for this batch
      with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        try:
          # Download entire batch in parallel
          batch_results = download_video_batch(
            batch_videos, split, temp_dir_path, dry_run=args.dry_run
          )

          # Prepare list of videos to upload
          videos_to_upload = []
          for video_id, download_success in batch_results:
            if not download_success:
              print(f'  ✗ Skipped {video_id} (missing required files)')
              failed_count += 1
              continue

            video_dir = temp_dir_path / split_lower / video_id
            if video_dir.exists() or args.dry_run:
              videos_to_upload.append((video_id, video_dir))
            else:
              print(f'  ✗ Skipped {video_id} (directory not found)')
              failed_count += 1

          # Upload entire batch in a single commit
          if videos_to_upload:
            print(f'\n  Uploading {len(videos_to_upload)} videos in a single commit...')

            success, error_msg = upload_batch_to_huggingface(
              videos_to_upload, split_lower, dry_run=args.dry_run
            )

            if success:
              successful_count += len(videos_to_upload)
              print(
                f'  ✓ Successfully uploaded batch of {len(videos_to_upload)} videos'
              )
              for vid, _ in videos_to_upload:
                print(f'    - {vid}')
            else:
              failed_count += len(videos_to_upload)
              print(f'  ✗ Failed to upload batch: {error_msg}')

        except Exception as e:
          print(f'✗ Error processing batch {batch_num + 1}: {e}')
          import traceback

          traceback.print_exc()
          failed_count += len(batch_videos)

      # temp_dir is automatically cleaned up when exiting the context manager
      if not args.dry_run:
        print(f'\n  Cleaned up batch {batch_num + 1} temporary files')

    # Summary for this split
    print(f'\n{"=" * 60}')
    print(f'{split} split summary:')
    print(f'  Successfully uploaded: {successful_count}')
    print(f'  Failed/Skipped: {failed_count}')
    print(f'  Total: {len(videos)}')
    print(f'{"=" * 60}')

  print(f'\n\n{"=" * 60}')
  print('All splits processed!')
  print(f'{"=" * 60}')


if __name__ == '__main__':
  main()
