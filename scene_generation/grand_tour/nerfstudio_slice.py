from typing import Optional
import yaml
import pathlib
import sys
import json
import cv2
import numpy as np
import torch
from PIL import Image
import zarr
from tqdm import tqdm
import viser.transforms as vtf
import open3d as o3d
from scipy.ndimage import grey_dilation, grey_erosion
import matplotlib.pyplot as plt
from collections import defaultdict
import copy
import zlib
import argparse

directory = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(directory.parent))
__package__ = directory.name

from .utils import (  # noqa: E402
  attrs_to_se3,
  FastGetClosestOdomToBaseTf,
  pq_to_se3,
  project_lidar_to_camera,
  undistort_image,
  s5cmd_cp,
  split_by_distance_gap,
  crop_pointcloud,
)


def generate_pointcloud(
  mission_root,
  get_closest_tf,
  lidars,
  cameras,
  mission_folder,
  voxel_size,
  max_lidar_projection_distance,
  human_mask_erosion_tolerance,
  undistort_mask_erosion_tolerance,
  slice_dict=None,
  viz_pcd=False,
  viz_overlay=False,
):
  pcd = o3d.geometry.PointCloud()

  base_to_box_base = pq_to_se3(
    mission_root['tf'].attrs['tf']['box_base']['translation'],
    mission_root['tf'].attrs['tf']['box_base']['rotation'],
  )

  lidar_tags = [lidar['tag'] for lidar in lidars]
  image_tags = [camera['tag'] for camera in cameras]

  # prefetch image timestamps
  image_timestamps = {}
  for image_tag in image_tags:
    image_timestamps[image_tag] = mission_root[image_tag]['timestamp'][:]

  for lidar_tag in tqdm(lidar_tags):
    lidar = None
    for _lidar in lidars:
      if _lidar['tag'] == lidar_tag:
        lidar = _lidar

    # prefetch lidar data
    lidar_timestamps = mission_root[lidar_tag]['timestamp'][:]
    if slice_dict is not None:
      subset_lidar_timestamps = slice_dict[lidar_tag]
    else:
      subset_lidar_timestamps = lidar_timestamps

    lidar_points_pre = mission_root[lidar_tag]['points'][:, :]
    valid_points = mission_root[lidar_tag]['valid'][:, 0]

    bar = tqdm(
      range(0, len(subset_lidar_timestamps)),
      desc=lidar_tag,
      mininterval=None,
      maxinterval=None,
      miniters=None,
      leave=False,
    )
    for lidar_id in range(0, mission_root[lidar_tag]['sequence_id'].shape[0]):
      lidar_timestamp = lidar_timestamps[lidar_id]
      if lidar_timestamp not in subset_lidar_timestamps:
        continue
      bar.update(1)
      lidar_points = lidar_points_pre[lidar_id, : valid_points[lidar_id]]
      lidar_colors = np.zeros_like(lidar_points)
      lidar_points_mask = np.zeros(lidar_points.shape[0], dtype=bool)

      odom_to_base__tlidar = get_closest_tf(lidar_timestamp)
      lidar_to_box_base = attrs_to_se3(mission_root[lidar_tag].attrs)

      image_idx_lookup = {}
      for image_tag in image_tags:
        # find closest images based on timestamp
        idx = np.argmin(np.abs(image_timestamps[image_tag] - lidar_timestamp))
        image_idx_lookup[image_tag] = idx

      for image_tag, idx in image_idx_lookup.items():
        image_timestamp = image_timestamps[image_tag][idx]

        camera = None
        for c in cameras:
          if c['tag'] == image_tag:
            camera = c

        odom_to_base__tcam = get_closest_tf(image_timestamp)

        # compute relate pointcloud motion
        t1_t2_motion = odom_to_base__tcam @ np.linalg.inv(odom_to_base__tlidar)
        cam_to_box_base = attrs_to_se3(mission_root[image_tag].attrs)

        # get translation from lidar_timestamp to image timestamp
        box_lidar_image = np.linalg.inv(
          lidar_to_box_base @ np.linalg.inv(cam_to_box_base)
        )
        lidar_to_camera = t1_t2_motion @ box_lidar_image

        cv_image = cv2.imread(
          str(mission_folder / 'images' / image_tag / f'{idx:06d}.jpeg')
        )
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        if camera['blur_threshold'] > 0:
          blur = cv2.Laplacian(cv_image, cv2.CV_32F).var()
          if blur < camera['blur_threshold']:
            tqdm.write(f'Warning: Image too blurry (blur value: {blur}). Skipping.')
            continue

        # rgb_image_np = np.array(cv_image)
        rgb_image, helper = undistort_image(cv_image, image_tag, mission_root)
        rgb_image_np = np.array(rgb_image)

        invalid_mask = helper['invalid_mask']
        invalid_mask = (invalid_mask * 255).astype(np.uint8)
        invalid_mask = grey_erosion(
          invalid_mask,
          size=(undistort_mask_erosion_tolerance, undistort_mask_erosion_tolerance),
        )
        invalid_mask = (invalid_mask / 255).astype(bool)

        # 0 == dynamic, 1 == not dynamic
        mask_image = Image.open(
          mission_folder / 'images' / (image_tag + '_mask') / f'{idx:06d}.png'
        )
        mask_image = grey_erosion(
          mask_image, size=(human_mask_erosion_tolerance, human_mask_erosion_tolerance)
        )
        # mask_image = np.array(mask_image).astype(bool)
        mask_image = (np.array(mask_image)[..., None].repeat(3, axis=-1) * 255).astype(
          np.uint8
        )
        mask_image, _ = undistort_image(mask_image, image_tag, mission_root)
        mask_image_eroded = (mask_image[..., 0] / 255).astype(bool)
        # mask_image_eroded = grey_erosion(mask_image, size=(erosion_tolerance, erosion_tolerance))
        if lidar['scan_section'] == 'above':
          mask_image_eroded[mask_image_eroded.shape[0] // 2 :] = False
        elif lidar['scan_section'] == 'below':
          mask_image_eroded[: mask_image_eroded.shape[0] // 2] = False
        mask_image_eroded[invalid_mask] = False

        K = helper['new_camera_matrix']
        D = helper['D_new']
        distortion_model = 'pinhole'

        # K = mission_root[image_tag].attrs["camera_info"]["K"]
        # D = mission_root[image_tag].attrs["camera_info"]["D"]
        W = mission_root[image_tag].attrs['camera_info']['width']
        H = mission_root[image_tag].attrs['camera_info']['height']
        # distortion_model = mission_root[image_tag].attrs["camera_info"]["distortion_model"]

        # with Timer('project_lidar_to_camera'):
        depth_image, mapping_image = project_lidar_to_camera(
          lidar_points.copy(),
          K,
          lidar_to_camera.copy(),
          W,
          H,
          D,
          distortion_model=distortion_model,
        )

        valid_rays = np.unique(mapping_image[mask_image_eroded])
        valid_rays = valid_rays[valid_rays >= 0]  # Remove -1 values
        lidar_points_mask[valid_rays] = True

        ray_indices = mapping_image[mask_image_eroded]

        rgb_image = torch.from_numpy(rgb_image_np).to('cuda:0')
        mask_image = torch.from_numpy(mask_image_eroded).to('cuda:0')
        valid_colors = rgb_image[mask_image] / 255.0

        lidar_colors = torch.from_numpy(lidar_colors).to('cuda:0')
        ray_indices = torch.from_numpy(ray_indices).to('cuda:0')
        lidar_colors[ray_indices] = valid_colors
        lidar_colors = lidar_colors.cpu().numpy()

        if viz_overlay:
          rgb_image = rgb_image_np

          # Normalize depth image for visualization - max range 10m
          depth_normalized = (depth_image.clip(0, 10.0) / 10.0 * 255).astype(np.uint8)

          # Dilate the depth image to increase pixel width to 3
          depth_normalized = grey_dilation(depth_normalized, size=(3, 3))

          # rgb_image H,W,3
          alpha = 0

          cmap = plt.get_cmap('turbo').reversed()
          color_depth = cmap(depth_normalized)  # H,W,4

          # Set alpha to 0 where depth is 0
          color_depth[..., 3] = np.where(depth_normalized == 0, 0, color_depth[..., 3])

          # Convert color_depth from float [0,1] to uint8 [0,255] and remove alpha channel
          color_depth_rgb = (color_depth[..., :3] * 255).astype(np.uint8)

          # Use alpha channel for blending: where alpha==0, keep rgb_image pixel
          alpha_mask = color_depth[..., 3][..., None]
          overlay = (alpha * rgb_image + (1 - alpha) * color_depth_rgb).astype(np.uint8)
          overlay = np.where(alpha_mask == 0, rgb_image, overlay)

          # Convert overlay to BGR for cv2 if needed
          overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
          overlay_bgr[~mask_image_eroded] = (0, 0, 255)
          cv2.imwrite(f'depth_overlay_{lidar_id}_{image_tag}.png', overlay_bgr)

      # Add lidar points to pointcloud.
      box_base_to_lidar = np.linalg.inv(lidar_to_box_base)
      odom_to_lidar = odom_to_base__tlidar @ base_to_box_base @ box_base_to_lidar
      odom_to_lidar_vtf = vtf.SE3.from_matrix(odom_to_lidar)
      # To OpenGL coordinate system.
      cv_to_gl = np.eye(4)
      cv_to_gl[1:3, 1:3] = np.array([[-1, 0], [0, -1]])
      odom_to_lidar_vtf = vtf.SE3.from_matrix(cv_to_gl).multiply(odom_to_lidar_vtf)
      lidar_points_mask = lidar_points_mask * (
        np.linalg.norm(lidar_points, axis=1) <= max_lidar_projection_distance
      )
      lidar_points_transformed = odom_to_lidar_vtf.apply(
        lidar_points[lidar_points_mask]
      )
      pcd_keep = o3d.geometry.PointCloud()
      pcd_keep.points = o3d.utility.Vector3dVector(lidar_points_transformed)
      pcd_keep.colors = o3d.utility.Vector3dVector(lidar_colors[lidar_points_mask])
      pcd_keep.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
      )
      try:
        pcd_keep.orient_normals_towards_camera_location(odom_to_lidar_vtf.translation())
        pcd += pcd_keep
      except Exception as e:
        tqdm.write(f'Error orienting normals for lidar_id {lidar_id}: {e}.')

      if bar.n % 50 == 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd = pcd.remove_duplicated_points()

        if viz_pcd:
          pcd_removed = o3d.geometry.PointCloud()
          lidar_points_removed = odom_to_lidar_vtf.apply(
            lidar_points[~lidar_points_mask]
          )
          pcd_removed.points = o3d.utility.Vector3dVector(lidar_points_removed)
          pcd_removed.paint_uniform_color([1, 0, 0])  # red

          pcd_keep.paint_uniform_color([0, 1, 0])  # green

          vis = o3d.visualization.Visualizer()
          vis.create_window(window_name='Lidar Points', width=920, height=580)

          # Add original point cloud first (smaller, semi-transparent green)
          vis.add_geometry(pcd_removed)
          vis.add_geometry(pcd_keep)
          vis.add_geometry(pcd)
          vis.run()
          vis.destroy_window()

  pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
  pcd = pcd.remove_duplicated_points()
  return pcd


class NerfStudioSlicer:
  def __init__(self, config, mission_folder: pathlib.Path, output_folder: pathlib.Path):
    self.config = config
    self.mission_root: zarr.Group = zarr.open_group(
      store=mission_folder / 'data', mode='r'
    )
    self.mission_folder = mission_folder

    self.tf_lookup = FastGetClosestOdomToBaseTf('dlio_map_odometry', self.mission_root)
    assert output_folder.exists(), f'Root folder {output_folder} does not exist'

    self.frames_json_file = output_folder / 'transforms.json'
    assert self.frames_json_file.exists(), (
      f'Frames json file {self.frames_json_file} does not exist'
    )
    with self.frames_json_file.open('r') as f:
      self.transforms_data = json.load(f)

    self.output_folder = output_folder / 'slices'
    self.output_folder.mkdir(parents=True, exist_ok=True)

    self.timestamps_to_transforms = defaultdict(dict)
    for frame in self.transforms_data['frames']:
      secs, nsecs = frame['timestamp'].split('_')
      timestamp = int(secs) + int(nsecs) / 1e9
      camera_tag = '_'.join(frame['file_path'].split('/')[-1].split('_')[:-1])
      self.timestamps_to_transforms[timestamp][camera_tag] = frame

    self.anchor_sensor = self.config['lidars'][0]
    self.slice_dict = self.compute_slices()

  def get_pointcloud_for_slice(self, slice_idx):
    pcd = generate_pointcloud(
      self.mission_root,
      self.tf_lookup,
      self.config['lidars'],
      self.config['cameras'],
      self.mission_folder,
      self.config['pcl_voxel_size'],
      self.config['max_lidar_projection_distance'],
      self.config['human_mask_erosion_tolerance'],
      self.config['undistort_mask_erosion_tolerance'],
      slice_dict=self.slice_dict[slice_idx],
    )
    slice_folder = self.output_folder / str(slice_idx)
    slice_folder.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(slice_folder / 'pcd.ply'), pcd)
    return pcd

  def slice_pointcloud(self, global_pcd, slice_idx, transforms_data):
    tqdm.write(f'Number of points in the pointcloud: {len(global_pcd.points)}')
    pcd = crop_pointcloud(global_pcd, transforms_data, self.config)
    slice_folder = self.output_folder / str(slice_idx)
    slice_folder.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(slice_folder / 'pcd.ply'), pcd)
    return pcd

  def get_transforms_for_slice(self, slice_idx):
    curr_transform = copy.deepcopy(self.transforms_data)
    curr_transform['ply_file_path'] = './pcd.ply'
    frames = []
    for sensor, timestamps in self.slice_dict[slice_idx].items():
      is_camera = False
      for c in self.config['cameras']:
        if c['tag'] == sensor:
          is_camera = True
          break
      if not is_camera:
        continue

      for t in timestamps:
        frame = copy.deepcopy(self.timestamps_to_transforms[t][sensor])
        frame['file_path'] = '../.' + frame['file_path']
        frame['mask_path'] = '../.' + frame['mask_path']
        frames.append(frame)
    curr_transform['frames'] = frames
    with open(self.output_folder / str(slice_idx) / 'transforms.json', 'w') as f:
      json.dump(curr_transform, f, indent=2)
    return curr_transform

  def get_anchor_timestamps_and_lookup(
    self,
    distance_threshold: Optional[float] = None,
    rot_threshold: Optional[float] = None,
  ):
    anchor_timestamps = self.mission_root[self.anchor_sensor['tag']]['timestamp'][:]
    anchor_timestamps_filtered = []
    last_pos = None
    last_rot = None
    for timestamp in anchor_timestamps:
      odom_to_base__t_lidar = self.tf_lookup(timestamp)
      transform = vtf.SE3.from_matrix(odom_to_base__t_lidar)
      position = transform.translation()
      rotation = transform.rotation()
      if last_pos is not None:
        if distance_threshold is not None:
          distance_not_exceeded = (
            np.linalg.norm(last_pos - position) < distance_threshold
          )
        else:
          distance_not_exceeded = False
        if rot_threshold is not None:
          rot_relative = rotation.inverse() @ last_rot
          tangent = rot_relative.log()
          angle = np.linalg.norm(tangent)
          angle_not_exceeded = angle < self.config['global_lidar_rot_threshold']
        else:
          angle_not_exceeded = False
        if distance_not_exceeded and angle_not_exceeded:
          continue
      anchor_timestamps_filtered.append(timestamp)
      last_pos = position
      last_rot = rotation
    anchor_timestamps = anchor_timestamps_filtered

    # Nearest timestamp dict (Simplify).
    all_sensors = self.config['lidars'] + self.config['cameras']
    nearest_timestamp_dict = {
      t.item(): {s['tag']: [] for s in all_sensors[1:]} for t in anchor_timestamps
    }
    for sensor in all_sensors[1:]:
      sensor_timestamps = self.mission_root[sensor['tag']]['timestamp'][:]
      for anchor_t in anchor_timestamps:
        # Find the nearest timestamp in the reference sensor's timestamps
        sensor_t = min(sensor_timestamps, key=lambda x: abs(x - anchor_t))
        if sensor_t in nearest_timestamp_dict[anchor_t][sensor['tag']]:
          continue
        if sensor in self.config['cameras']:
          if sensor_t in self.timestamps_to_transforms:
            transforms = self.timestamps_to_transforms[sensor_t]
            if sensor['tag'] in transforms:
              nearest_timestamp_dict[anchor_t][sensor['tag']].append(sensor_t.item())
        else:
          nearest_timestamp_dict[anchor_t][sensor['tag']].append(sensor_t.item())
    return anchor_timestamps, nearest_timestamp_dict

  def run(self):
    anchor_timestamps, nearest_timestamp_dict = self.get_anchor_timestamps_and_lookup(
      distance_threshold=self.config['global_lidar_distance_threshold'],
      rot_threshold=self.config['global_lidar_rot_threshold'],
    )

    # Lidar for global pointcloud.
    global_dict = defaultdict(list)
    global_dict[self.anchor_sensor['tag']] = anchor_timestamps
    for timestamp in global_dict[self.anchor_sensor['tag']]:
      for s, ts in nearest_timestamp_dict[timestamp].items():
        global_dict[s].extend(ts)

    global_pcd = generate_pointcloud(
      self.mission_root,
      self.tf_lookup,
      self.config['lidars'],
      self.config['cameras'],
      self.mission_folder,
      self.config['pcl_voxel_size'],
      self.config['max_lidar_projection_distance'],
      self.config['human_mask_erosion_tolerance'],
      self.config['undistort_mask_erosion_tolerance'],
      slice_dict=global_dict,
    )
    global_pcd, _ = global_pcd.remove_statistical_outlier(
      nb_neighbors=20, std_ratio=1.5
    )
    o3d.io.write_point_cloud(str(self.output_folder / 'global_pcd.ply'), global_pcd)

    for slice_idx in self.slice_dict:
      slice_folder = self.output_folder / str(slice_idx)
      slice_folder.mkdir(parents=True, exist_ok=True)
      # self.get_pointcloud_for_slice(slice_idx)
      transforms_data = self.get_transforms_for_slice(slice_idx)
      self.slice_pointcloud(global_pcd, slice_idx, transforms_data)

  def find_nearest_slice(self, anchor_timestamps, slice_dict, target_slice_idx):
    """Find the nearest slice to merge with based on centroid distance"""

    slice_timestamps = slice_dict[target_slice_idx][self.anchor_sensor['tag']]
    # Find indices in anchor_timestamps that correspond to slice_timestamps
    slice_indices = []
    for timestamp in slice_timestamps:
      # Find the closest timestamp in anchor_timestamps
      idx = np.argmin(np.abs(anchor_timestamps - timestamp))
      slice_indices.append(idx)

    prev_idx = min(slice_indices) - 1
    max_idx = max(slice_indices) + 1
    valid_timestamps = []
    if prev_idx >= 0:
      valid_timestamps.append(anchor_timestamps[prev_idx])
    if max_idx < len(anchor_timestamps):
      valid_timestamps.append(anchor_timestamps[max_idx])

    slice_indices = {k: None for k in valid_timestamps}
    for slice_idx in slice_dict.keys():
      if slice_idx == target_slice_idx:
        continue
      for timestamp in valid_timestamps:
        if timestamp in slice_dict[slice_idx][self.anchor_sensor['tag']]:
          slice_indices[timestamp] = slice_idx

    # Get the key for the minimum value in slice_indices
    slice_indices = {k: v for k, v in slice_indices.items() if v is not None}
    slice_indices = list(zip(slice_indices.keys(), slice_indices.values()))
    min_key = min(
      slice_indices, key=lambda x: len(slice_dict[x[1]][self.anchor_sensor['tag']])
    )[1]
    return min_key

  def merge_slices(self, slice_dict, source_slice_idx, target_slice_idx):
    """Merge source slice into target slice"""
    source_slice = slice_dict[source_slice_idx]
    target_slice = slice_dict[target_slice_idx]

    # Merge all sensor data
    for sensor_tag, timestamps in source_slice.items():
      target_slice[sensor_tag].extend(timestamps)

    # Remove the source slice
    del slice_dict[source_slice_idx]

  def merge_small_slices(
    self, anchor_timestamps, slice_dict, min_slice_size, cell_centers, cell_size
  ):
    """Merge slices that are too small"""
    small_slices = [
      idx
      for idx, data in slice_dict.items()
      if 0 < len(data[self.anchor_sensor['tag']]) < min_slice_size
    ]

    for small_idx in small_slices:
      # Find nearest slice by average position
      best_merge_idx = self.find_nearest_slice(anchor_timestamps, slice_dict, small_idx)
      if best_merge_idx is not None:
        # Merge small slice into best_merge_idx
        self.merge_slices(slice_dict, small_idx, best_merge_idx)
      tqdm.write(f'Merged slice {small_idx} into {best_merge_idx}')
      # self.visualize_slices(slice_dict, cell_centers, cell_size)

  def merge_small_trajectories(
    self, anchor_timestamps, slice_dict, min_slice_size, cell_centers, cell_size
  ):
    """Merge small trajectories into larger trajectories"""
    for k, _ in slice_dict.items():
      ts = slice_dict[k][self.anchor_sensor['tag']]
      distances = np.stack(
        [vtf.SE3.from_matrix(self.tf_lookup(s)).translation() for s in ts], axis=0
      )
      ts_splits = split_by_distance_gap(
        distances, ts, max_distance=1.0, merge_groups=True
      )

      ts_final = []
      for ts_split in ts_splits:
        if len(ts_split) < min_slice_size:
          min_ts = min(ts_split)
          max_ts = max(ts_split)
          min_slice_key1, min_slice_key2 = None, None
          min_dist1, min_dist2 = None, None
          for slice_key, _ in slice_dict.items():
            if slice_key == k:
              continue
            slice_ts = np.array(slice_dict[slice_key][self.anchor_sensor['tag']])
            if len(slice_ts) == 0:
              continue
            d1 = np.abs(min_ts - slice_ts).min().item()
            d2 = np.abs(max_ts - slice_ts).min().item()
            if min_dist1 is None or d1 < min_dist1:
              min_dist1 = d1
              min_slice_key1 = slice_key
            if min_dist2 is None or d2 < min_dist2:
              min_dist2 = d2
              min_slice_key2 = slice_key
          assert min_slice_key1 is not None and min_slice_key2 is not None
          num_ts1 = len(slice_dict[min_slice_key1][self.anchor_sensor['tag']])
          num_ts2 = len(slice_dict[min_slice_key2][self.anchor_sensor['tag']])
          if num_ts1 < num_ts2:
            min_slice_key = min_slice_key1
          else:
            min_slice_key = min_slice_key2
          slice_dict[min_slice_key][self.anchor_sensor['tag']].extend(ts_split)
        else:
          ts_final.extend(ts_split)
      slice_dict[k][self.anchor_sensor['tag']] = ts_final

  def ensure_contiguous(
    self, anchor_timestamps, slice_dict, min_slice_size, cell_centers, cell_size
  ):
    """Any uncontiguous trajectory will be merged into the nearest contiguous trajectory."""
    all_keys = []
    not_contiguous_keys = []
    for k, _ in slice_dict.items():
      ts = slice_dict[k][self.anchor_sensor['tag']]
      if len(ts) < min_slice_size:
        continue
      all_keys.append(k)
      distances = np.stack(
        [vtf.SE3.from_matrix(self.tf_lookup(s)).translation() for s in ts], axis=0
      )
      ts_splits = split_by_distance_gap(
        distances, ts, max_distance=1.0, merge_groups=True
      )
      if len(ts_splits) > 1:
        not_contiguous_keys.append(k)

    for key in not_contiguous_keys:
      ts = slice_dict[key][self.anchor_sensor['tag']]
      distances = np.stack(
        [vtf.SE3.from_matrix(self.tf_lookup(s)).translation() for s in ts], axis=0
      )
      ts_splits = split_by_distance_gap(
        distances, ts, max_distance=1.0, merge_groups=True
      )

      suff_ts_splits = []
      suff_ts_split_idxs = []
      for split_idx, ts_split in enumerate(ts_splits):
        if len(ts_split) >= min_slice_size:
          suff_ts_split_idxs.append(split_idx)
          suff_ts_splits.append(ts_split)
        # if largest_ts_split_idx is None or len(ts_split) > len(largest_ts_split):
        #   if len(ts_split) > min_slice_size:
        #     largest_ts_split = ts_split
        #     largest_ts_split_idx = split_idx
        # if largest_ts_split_idx is not None:
      ts_splits = [
        ts_splits[i] for i in range(len(ts_splits)) if i not in suff_ts_split_idxs
      ]

      # largest_ts_split = []
      # largest_ts_split_idx = None
      # for split_idx, ts_split in enumerate(ts_splits):
      #   if largest_ts_split_idx is None or len(ts_split) > len(largest_ts_split):
      #     if len(ts_split) > min_slice_size:
      #       largest_ts_split = ts_split
      #       largest_ts_split_idx = split_idx
      # if largest_ts_split_idx is not None:
      #   ts_splits = [
      #     ts_splits[i] for i in range(len(ts_splits)) if i != largest_ts_split_idx
      #   ]

      for ts_split in ts_splits:
        min_ts = min(ts_split)
        max_ts = max(ts_split)
        min_slice_key1, min_slice_key2 = None, None
        min_dist1, min_dist2 = None, None
        for slice_key, _ in slice_dict.items():
          if slice_key in not_contiguous_keys:
            continue
          slice_ts = np.array(slice_dict[slice_key][self.anchor_sensor['tag']])
          if len(slice_ts) == 0:
            continue
          d1 = np.abs(min_ts - slice_ts).min().item()
          d2 = np.abs(max_ts - slice_ts).min().item()
          if min_dist1 is None or d1 < min_dist1:
            min_dist1 = d1
            min_slice_key1 = slice_key
          if min_dist2 is None or d2 < min_dist2:
            min_dist2 = d2
            min_slice_key2 = slice_key
        assert min_slice_key1 is not None and min_slice_key2 is not None
        if min_dist1 < min_dist2:
          min_slice_key = min_slice_key1
        else:
          min_slice_key = min_slice_key2
        slice_dict[min_slice_key][self.anchor_sensor['tag']].extend(ts_split)

      slice_dict[key][self.anchor_sensor['tag']] = []
      for its_split_idx, its_split in enumerate(suff_ts_splits):
        slice_dict[f'{key}_{its_split_idx}'][self.anchor_sensor['tag']] = its_split
        cell_centers[f'{key}_{its_split_idx}'] = cell_centers[key]

    return cell_centers
    # slice_dict[key][self.anchor_sensor['tag']] = largest_ts_split

  def remove_empty_slices(self, slice_dict):
    to_delete = []
    for k, v in slice_dict.items():
      if len(v[self.anchor_sensor['tag']]) == 0:
        to_delete.append(k)
    for k in to_delete:
      del slice_dict[k]
    return slice_dict

  def slice_by_area(
    self, anchor_timestamps, nearest_timestamp_dict, cell_size, min_slice_size
  ):  # cell_size in meters
    slice_dict = defaultdict(lambda: defaultdict(list))
    cell_centers = {}

    for timestamp in anchor_timestamps:
      # Get 3D position
      odom_to_base__t_lidar = self.tf_lookup(timestamp)
      position = vtf.SE3.from_matrix(odom_to_base__t_lidar).translation()

      grid_cell = (
        int((position[0] + cell_size / 2) // cell_size),
        int((position[1] + cell_size / 2) // cell_size),
        int((position[2] + cell_size / 2) // cell_size),
      )

      # for grid_cell in overlapping_cells:
      coord_str = f'{grid_cell[0]},{grid_cell[1]},{grid_cell[2]}'
      hash_val = zlib.crc32(coord_str.encode()) & 0xFFFFFFFF
      current_slice_idx = f'slice_{hash_val:08x}'
      cell_centers[current_slice_idx] = (
        np.array(grid_cell) * cell_size
      )  # + cell_size / 2

      # Add timestamp to appropriate slice
      slice_dict[current_slice_idx][self.anchor_sensor['tag']].append(timestamp)

    tqdm.write(f'Before merging: {len(slice_dict)} slices')
    self.visualize_slices(slice_dict, cell_centers, cell_size)
    self.merge_small_trajectories(
      anchor_timestamps,
      slice_dict,
      min_slice_size=min_slice_size,
      cell_centers=cell_centers,
      cell_size=cell_size,
    )
    slice_dict = self.remove_empty_slices(slice_dict)
    tqdm.write(f'After merge small trajectories: {len(slice_dict)} slices')
    self.visualize_slices(slice_dict, cell_centers, cell_size)
    for _ in range(5):
      cell_centers = self.ensure_contiguous(
        anchor_timestamps,
        slice_dict,
        min_slice_size=min_slice_size,
        cell_centers=cell_centers,
        cell_size=cell_size,
      )
      slice_dict = self.remove_empty_slices(slice_dict)
    # self.merge_small_slices(
    #   anchor_timestamps,
    #   slice_dict,
    #   min_slice_size=min_slice_size,
    #   cell_centers=cell_centers,
    #   cell_size=cell_size,
    # )

    # Add synchronized sensor data
    for current_slice_idx in slice_dict:
      for timestamp in slice_dict[current_slice_idx][self.anchor_sensor['tag']]:
        for s, ts in nearest_timestamp_dict[timestamp].items():
          slice_dict[current_slice_idx][s].extend(ts)

    for slice_idx, slice_data in slice_dict.items():
      for sensor_tag, timestamps in slice_data.items():
        slice_dict[slice_idx][sensor_tag] = sorted(list(set(timestamps)))
    tqdm.write(f'After contiguous: {len(slice_dict)} slices')
    self.visualize_slices(slice_dict, cell_centers, cell_size)
    return slice_dict

  def visualize_slices(
    self,
    slice_dict,
    cell_centers,
    cell_size,
    show_trajectory=True,
    show_centroids=True,
    show_cell=True,
  ):
    """Visualize slice positions using Open3D with different colors for each slice"""

    # Generate distinct colors for each slice
    num_slices = len(slice_dict)
    colors = plt.cm.tab20(np.linspace(0, 1, num_slices))[:, :3]  # RGB only

    geometries = []

    # Create point clouds for each slice
    for i, (slice_idx, slice_data) in enumerate(slice_dict.items()):
      positions = []

      # Get all positions for this slice
      for timestamp in slice_data[self.anchor_sensor['tag']]:
        # for timestamp in slice_data[self.config['cameras'][0]['tag']]:
        odom_to_base__t_lidar = self.tf_lookup(timestamp)
        position = vtf.SE3.from_matrix(odom_to_base__t_lidar).translation()
        positions.append(position)

      if len(positions) == 0:
        continue

      positions = np.array(positions)

      # Create point cloud for trajectory points
      if show_trajectory:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        pcd.paint_uniform_color(colors[i % len(colors)])
        geometries.append(pcd)

      # Create sphere for centroid
      if show_centroids:
        centroid = np.mean(positions, axis=0)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        sphere.translate(centroid)
        sphere.paint_uniform_color(colors[i % len(colors)])
        geometries.append(sphere)

        # Add text label (if possible)
        tqdm.write(f'Slice {slice_idx}: {len(positions)} poses, centroid at {centroid}')

      if show_cell:
        cell_center = cell_centers[slice_idx]
        # Create rectangular prism for cell
        box = o3d.geometry.TriangleMesh.create_box(
          width=cell_size, height=cell_size, depth=cell_size
        )
        # Center the box at cell_center
        box.translate(
          cell_center - np.array([cell_size / 2, cell_size / 2, cell_size / 2])
        )
        # Make it wireframe and semi-transparent
        box.paint_uniform_color(colors[i % len(colors)])
        # Convert to wireframe
        box_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(box)
        box_wireframe.paint_uniform_color(colors[i % len(colors)])
        geometries.append(box_wireframe)

    # Create coordinate frame at origin
    tqdm.write(
      f'Visualizing {num_slices} slices with {sum(len(data[self.anchor_sensor["tag"]]) for data in slice_dict.values())} total poses'
    )
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    geometries.append(coord_frame)

    o3d.visualization.draw_geometries(
      geometries, window_name='Trajectory Slices Visualization', width=1200, height=800
    )

  def compute_slices(self):
    anchor_timestamps, nearest_timestamp_dict = self.get_anchor_timestamps_and_lookup(
      distance_threshold=self.config['lidar_distance_threshold'],
      rot_threshold=self.config['lidar_rot_threshold'],
    )

    # Slice poses by area and merge small slices.
    slice_dict = self.slice_by_area(
      anchor_timestamps,
      nearest_timestamp_dict,
      self.config['gsplat_slice_distance'],
      self.config['min_slice_size'],
    )
    for slice_idx, slice_data in slice_dict.items():
      tqdm.write(slice_idx)
      for sensor_tag, timestamps in slice_data.items():
        tqdm.write(f'\t{sensor_tag}: {len(timestamps)}')

    return slice_dict


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
    '--s3-path', type=str, default=None, help='S3 path to copy the data to.'
  )

  args = parser.parse_args()

  # Setup paths
  mission = args.mission
  dataset_folder = pathlib.Path(args.dataset_folder).expanduser()
  mission_folder = dataset_folder / mission
  output_folder = dataset_folder / f'{mission}_nerfstudio'

  print(f'Converting mission: {mission}')
  print(f'Dataset folder: {dataset_folder}')
  print(f'Mission folder: {mission_folder}')
  print(f'Output folder: {output_folder}')

  with open(pathlib.Path(__file__).parent / 'grand_tour_release.yaml', 'r') as f:
    config = yaml.safe_load(f)

  slicer = NerfStudioSlicer(
    config=config,
    mission_folder=mission_folder,
    output_folder=output_folder,
  )
  slicer.run()

  if args.s3_path is not None:
    src = str(output_folder)
    dst = args.s3_path
    if not dst.endswith('/'):
      dst = dst + '/'
    s5cmd_cp(src, dst)
