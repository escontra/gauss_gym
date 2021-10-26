import argparse
import os
import shutil

import pathlib
import sys
import yaml

directory = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(directory.parent))
__package__ = directory.name

from . import (  # noqa: E402
  download_data,
  generate_masks,
  utils,
  nerfstudio_convert,
  nerfstudio_slice,
  generate_meshes,
)

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
    default=download_data._DEFAULT_TOPICS,
    help='List of topics to download (space-separated)',
  )

  parser.add_argument(
    '--s3-path', type=str, default=None, help='S3 path to copy the data to.'
  )

  parser.add_argument(
    '--image-tags',
    nargs='*',
    default=['hdr_front', 'hdr_left', 'hdr_right'],
    help='List of image tags to process (space-separated)',
  )

  parser.add_argument(
    '--batch-size',
    type=int,
    default=16,
    help='Batch size for processing images (larger = faster but more memory)',
  )
  parser.add_argument(
    '--download-only',
    action='store_true',
    help='Only download the data, do not process it.',
  )
  parser.add_argument(
    '--skip-masks', action='store_true', help='Skip generating masks.'
  )
  parser.add_argument(
    '--skip-download', action='store_true', help='Skip generating masks.'
  )

  args = parser.parse_args()

  mission = args.mission
  dataset_folder = pathlib.Path(args.dataset_folder).expanduser()
  topics = args.topics

  print(f'Downloading mission: {mission}')
  print(f'Dataset folder: {dataset_folder}')
  print(f'Topics: {topics}')

  # Download data to local directory.
  if not args.skip_download:
    download_data.download_mission(mission, dataset_folder, topics, args.s3_path)
    try:
      import zarr

      zarr.open_group(store=dataset_folder / mission / 'data', mode='r')
    except Exception as e:
      print(f'Error opening zarr group: {e}')
      if args.s3_path is not None:
        assert args.s3_path.startswith('s3://')
        # utils.s3_delete_prefix(args.s3_path[5:], os.path.join('grand_tour', mission) + '/')
        # utils.s5cmd_delete_prefix(
        #     args.s3_path[5:],
        #     os.path.join('grand_tour', mission) + '/'
        # )
        download_data.download_mission(
          mission, dataset_folder, topics, args.s3_path, overwrite_s3=True
        )
        try:
          zarr.open_group(store=dataset_folder / mission / 'data', mode='r')
        except Exception as e:
          print(f'Error opening zarr group: {e}, even after re-downloading.')
          # utils.s5cmd_delete_prefix(
          #     args.s3_path[5:],
          #     os.path.join('grand_tour', mission) + '/'
          # )
          raise ValueError('Still cannot open zarr group, even after re-downloading.')
      else:
        raise ValueError('S3 path is not set, cannot delete masks from s3.')

  if args.download_only:
    exit(0)

  if not args.skip_masks:
    # Generate masks.
    generate_masks.generate_masks_batch(
      dataset_folder / mission,
      args.image_tags,
      batch_size=args.batch_size,
      s3_path=args.s3_path,
    )

  # Load config.
  with open(pathlib.Path(__file__).parent / 'grand_tour_release.yaml', 'r') as f:
    config = yaml.safe_load(f)

  # Convert to Nerfstudio format.
  run_converter = True
  if args.s3_path is not None:
    assert args.s3_path.startswith('s3://')
    directory_exists = utils.s3_directory_exists(
      args.s3_path[5:], os.path.join('grand_tour', f'{mission}_nerfstudio') + '/'
    )
    if directory_exists:
      print(f'Nerfstudio data already exists for {mission}, skipping...')
      utils.download_from_s3(
        os.path.join(args.s3_path, 'grand_tour', f'{mission}_nerfstudio') + '/',
        local_path=dataset_folder / f'{mission}_nerfstudio',
        recursive=True,
      )
      run_converter = False

  if run_converter:
    ns_converter = nerfstudio_convert.NerfstudioConverter(
      config=config,
      mission_folder=dataset_folder / mission,
      output_folder=dataset_folder / f'{mission}_nerfstudio',
    )
    ns_converter.run()
    # Copy to S3.
    utils.s5cmd_cp(
      str(dataset_folder / f'{mission}_nerfstudio').rstrip('/'),
      os.path.join(args.s3_path, 'grand_tour') + '/',
    )

  # Create slices.
  ns_slicer = nerfstudio_slice.NerfStudioSlicer(
    config=config,
    mission_folder=dataset_folder / mission,
    output_folder=dataset_folder / f'{mission}_nerfstudio',
  )
  estimated_num_slices = len(ns_slicer.slice_dict)
  print(f'Estimated number of slices: {estimated_num_slices}')
  run_slicer = True
  slice_path = dataset_folder / f'{mission}_nerfstudio' / 'slices'
  if slice_path.exists():
    num_slices = len(list(slice_path.iterdir()))
    print(f'Number of slices: {num_slices}')
    if num_slices != estimated_num_slices:
      print(
        'Number of slices does not match estimated number, deleting and regenerating...'
      )
      shutil.rmtree(slice_path)
      if args.s3_path is not None:
        assert args.s3_path.startswith('s3://')
        # utils.s5cmd_delete_prefix(
        #     args.s3_path[5:],
        #     os.path.join('grand_tour', f'{mission}_nerfstudio' / "slices") + '/'
        # )
    else:
      run_slicer = False
  if run_slicer:
    ns_slicer.run()
    # Copy to S3.
    if args.s3_path is not None:
      assert args.s3_path.startswith('s3://')
      utils.s5cmd_cp(
        str(slice_path).rstrip('/'),
        os.path.join(args.s3_path, 'grand_tour', f'{mission}_nerfstudio') + '/',
      )

  # Generate meshes.
  mesher = generate_meshes.MeshGenerator(
    config=config,
    mission_folder=dataset_folder / mission,
    output_folder=dataset_folder / f'{mission}_nerfstudio',
  )
  mesher.run()

  if args.s3_path is not None:
    utils.s5cmd_cp(
      str(slice_path).rstrip('/'),
      os.path.join(args.s3_path, 'grand_tour', f'{mission}_nerfstudio') + '/',
    )
