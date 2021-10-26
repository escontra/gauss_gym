import yaml
import pathlib
import sys
import json
import cv2
import numpy as np
from PIL import Image
import zarr
from tqdm import tqdm
from scipy.ndimage import grey_erosion
import argparse
import viser.transforms as vtf

directory = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(directory.parent))
__package__ = directory.name

from .utils import (  # noqa: E402
  attrs_to_se3,
  FastGetClosestOdomToBaseTf,
  pq_to_se3,
  ros_to_gl_transform,
  undistort_image,
  s5cmd_cp,
)


class NerfstudioConverter:
  def __init__(self, config, mission_folder, output_folder):
    self.config = config
    self.mission_root: zarr.Group = zarr.open_group(
      store=mission_folder / 'data', mode='r'
    )
    self.mission_folder = mission_folder

    self.tf_lookup = FastGetClosestOdomToBaseTf('dlio_map_odometry', self.mission_root)

    self.output_folder = output_folder
    self.output_folder.mkdir(parents=True, exist_ok=True)
    self.frames_json_file = self.output_folder / 'transforms.json'
    self.images_folder = self.output_folder / 'rgb'
    self.images_folder.mkdir(parents=True, exist_ok=True)

    self.image_counters = {key['tag']: 0 for key in self.config['cameras']}
    self.image_last_stored = {key['tag']: 0 for key in self.config['cameras']}

  def run(self):
    base_to_box_base = pq_to_se3(
      self.mission_root['tf'].attrs['tf']['box_base']['translation'],
      self.mission_root['tf'].attrs['tf']['box_base']['rotation'],
    )
    frames_data = {'camera_model': 'OPENCV', 'frames': []}

    for camera in self.config['cameras']:
      camera_tag = camera['tag']
      data = self.mission_root[camera_tag]
      timestamps = data['timestamp'][:]
      seqs = data['sequence_id'][:]
      # last_t = None
      last_pos = None
      last_rot = None
      for i in tqdm(range(timestamps.shape[0]), desc=f'Processing {camera_tag}'):
        timestamp = timestamps[i]

        # if last_t is not None and timestamp - last_t < 1 / camera['hz'] + 0.001:
        #   continue
        # last_t = timestamp

        odom_to_base__t_camera = self.tf_lookup(timestamp)
        transform = vtf.SE3.from_matrix(odom_to_base__t_camera)
        curr_pos = transform.translation()
        curr_rot = transform.rotation()

        if last_pos is not None:
          distance_not_exceeded = (
            np.linalg.norm(last_pos - curr_pos) < camera['distance_threshold']
          )
          rot_relative = curr_rot.inverse() @ last_rot
          tangent = rot_relative.log()
          angle = np.linalg.norm(tangent)
          angle_not_exceeded = angle < camera['rot_threshold']
          if distance_not_exceeded and angle_not_exceeded:
            continue

        last_pos = curr_pos
        last_rot = curr_rot

        # if (
        #   last_pos is not None
        #   and np.linalg.norm(last_pos - odom_to_base__t_camera[:2, 3])
        #   < camera['distance_threshold']
        # ):
        #   continue
        # last_pos = odom_to_base__t_camera[:2, 3]

        cam_to_box_base = attrs_to_se3(data.attrs)

        odom_to_cam__t_camera = (
          odom_to_base__t_camera @ base_to_box_base @ np.linalg.inv(cam_to_box_base)
        )

        image_filename = f'{camera_tag}_{seqs[i]:05d}.png'
        image_path = self.images_folder / image_filename

        cv_image = cv2.imread(
          self.mission_folder / 'images' / camera_tag / f'{i:06d}.jpeg'
        )

        if camera['blur_threshold'] > 0:
          blur = cv2.Laplacian(cv_image, cv2.CV_64F).var()
          if blur < camera['blur_threshold']:
            tqdm.write(f'Warning: Image too blurry (blur value: {blur}). Skipping.')
            continue

        self.image_counters[camera_tag] += 1
        if self.image_counters[camera_tag] >= camera.get('max_images', float('inf')):
          tqdm.write(f'Skipping image {i} for camera {camera_tag} due to max limit.')
          break

        cv_image, undist_meta = undistort_image(cv_image, camera_tag, self.mission_root)

        cv2.imwrite(str(image_path), cv_image)

        # Convert to OpenGL convention
        odom_to_cam__t_camera_gl = ros_to_gl_transform(odom_to_cam__t_camera)

        timestamp = timestamps[i]
        secs = int(timestamp)
        nsecs = int((timestamp - secs) * 1e9)

        K = undist_meta['new_camera_matrix']
        D = undist_meta['D_new']

        frame_data = {
          'file_path': f'./rgb/{image_filename}',
          'transform_matrix': odom_to_cam__t_camera_gl.tolist(),
          'camera_frame_id': int(seqs[i]),
          'fl_x': str(K[0, 0]),
          'fl_y': str(K[1, 1]),
          'cx': str(K[0, 2]),
          'cy': str(K[1, 2]),
          'w': str(data.attrs['camera_info']['width']),
          'h': str(data.attrs['camera_info']['height']),
          'k1': str(D[0]),
          'k2': str(D[1]),
          'p1': str(D[2]),
          'p2': str(D[3]),
          'timestamp': str(secs) + '_' + str(nsecs),
        }

        invalid_mask = undist_meta['invalid_mask']
        invalid_mask = (invalid_mask * 255).astype(np.uint8)
        invalid_mask = grey_erosion(
          invalid_mask,
          size=(
            self.config['undistort_mask_erosion_tolerance'],
            self.config['undistort_mask_erosion_tolerance'],
          ),
        )
        invalid_mask = (invalid_mask / 255).astype(bool)

        mask = Image.open(
          self.mission_folder / 'images' / (camera_tag + '_mask') / f'{i:06d}.png'
        )
        mask = np.array(mask).astype(bool)
        mask = (mask * 255).astype(np.uint8)
        mask = grey_erosion(
          mask,
          size=(
            self.config['human_mask_erosion_tolerance'],
            self.config['human_mask_erosion_tolerance'],
          ),
        )
        mask = mask[..., None].repeat(3, axis=-1)
        mask, _ = undistort_image(mask, camera_tag, self.mission_root)
        mask = mask[..., 0]
        mask[invalid_mask] = 0
        pathlib.Path(str(image_path).replace('rgb', 'mask')).parent.mkdir(
          parents=True, exist_ok=True
        )
        mask_file_path = str(image_path).replace('rgb', 'mask').replace('.jpeg', '.png')
        Image.fromarray(mask, mode='L').save(mask_file_path)
        frame_data['mask_path'] = f'./mask/{image_filename}'

        frames_data['frames'].append(frame_data)

    if not self.frames_json_file.exists():
      with open(self.frames_json_file, 'w') as f:
        json.dump(frames_data, f, indent=2)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Convert Grand Tour dataset to Nerfstudio format',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )

  parser.add_argument(
    '--mission',
    type=str,
    default='2024-11-04-10-57-34',
    help="Mission name to convert (e.g., '2024-11-04-10-57-34')",
  )

  parser.add_argument(
    '--dataset-folder',
    type=str,
    default='~/grand_tour_dataset',
    help='Dataset folder path containing missions',
  )

  parser.add_argument(
    '--output-folder',
    type=str,
    default=None,
    help='Output folder for Nerfstudio format (default: {dataset_folder}/{mission}_nerfstudio)',
  )

  parser.add_argument(
    '--s3-path', type=str, default=None, help='S3 path to copy the data to.'
  )

  args = parser.parse_args()

  # Setup paths
  mission = args.mission
  dataset_folder = pathlib.Path(args.dataset_folder).expanduser()
  mission_folder = dataset_folder / mission

  if args.output_folder:
    output_folder = pathlib.Path(args.output_folder).expanduser()
  else:
    output_folder = dataset_folder / f'{mission}_nerfstudio'

  config_file = pathlib.Path(__file__).parent / 'grand_tour_release.yaml'

  print(f'Converting mission: {mission}')
  print(f'Dataset folder: {dataset_folder}')
  print(f'Mission folder: {mission_folder}')
  print(f'Output folder: {output_folder}')

  # Load config
  with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

  # Create output folder
  output_folder.mkdir(exist_ok=True, parents=True)

  # Run conversion
  print('Starting Nerfstudio conversion...')
  converter = NerfstudioConverter(
    config=config,
    mission_folder=mission_folder,
    output_folder=output_folder,
  )
  converter.run()
  print('Nerfstudio conversion completed!')

  if args.s3_path is not None:
    src = str(output_folder)
    dst = args.s3_path
    if not dst.endswith('/'):
      dst = dst + '/'
    s5cmd_cp(src, dst)
