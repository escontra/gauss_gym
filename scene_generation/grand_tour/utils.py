from typing import Callable, Generic, ParamSpec, TypeVar
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import interp1d
import cv2
import open3d as o3d
from tqdm import tqdm

import os
import shutil
import subprocess
import random
import time
from pathlib import Path
import viser.transforms as vtf


P = ParamSpec('P')
T = TypeVar('T')


def crop_pointcloud(pcd, transforms_data, config):
  cropped_pcd = o3d.geometry.PointCloud()
  for i, frame in enumerate(tqdm(transforms_data['frames'], leave=False)):
    pose = vtf.SE3.from_matrix(np.array(frame['transform_matrix']))
    bbox = rectangular_prism(
      pose.translation(),
      config['mesh_bbox_width'],
      config['mesh_bbox_height'],
      'z',
      (1, 0, 0),
      '+',
    )
    cropped_pcd += pcd.crop(bbox)
    if i % 100 == 0:
      # Many duplicated points are introduced by cropping.
      cropped_pcd = cropped_pcd.remove_duplicated_points()
  cropped_pcd = cropped_pcd.remove_duplicated_points()
  return cropped_pcd


def ros_to_gl_transform(transform_ros):
  cv_to_gl = np.eye(4)
  cv_to_gl[1:3, 1:3] = np.array([[-1, 0], [0, -1]])
  transform_gl = cv_to_gl @ transform_ros @ np.linalg.inv(cv_to_gl)
  return transform_gl


def gl_to_ros_transform(transform_gl):
  cv_to_gl = np.eye(4)
  cv_to_gl[1:3, 1:3] = np.array([[-1, 0], [0, -1]])
  transform_ros = np.linalg.inv(cv_to_gl) @ transform_gl @ cv_to_gl
  return transform_ros


def pq_to_se3(p, q):
  se3 = np.eye(4, dtype=np.float32)
  try:
    se3[:3, :3] = R.from_quat([q['x'], q['y'], q['z'], q['w']]).as_matrix()
    se3[:3, 3] = [p['x'], p['y'], p['z']]
  except Exception:
    se3[:3, :3] = R.from_quat(q).as_matrix()
    se3[:3, 3] = p
  return se3


def attrs_to_se3(attrs):
  return pq_to_se3(attrs['transform']['translation'], attrs['transform']['rotation'])


class FastGetClosestOdomToBaseTf:
  def __init__(self, odom_key, mission_root):
    self.odom_key = odom_key
    self.mission_root = mission_root
    odom = mission_root[odom_key]
    self.odom = mission_root[odom_key]
    self.timestamps = odom['timestamp'][:]
    self.pose_pos = odom['pose_pos'][:]
    self.pose_orien = odom['pose_orien'][:]

  def __call__(
    self, timestamp: float, interpolate_leg_odometry: bool = False
  ) -> np.ndarray:
    assert interpolate_leg_odometry is False, 'Interpolation not implemented yet'
    idx = np.argmin(np.abs(self.timestamps - timestamp))

    # Handle boundary cases
    if idx == 0 or idx == len(self.timestamps) - 1 or self.timestamps[idx] == timestamp:
      tqdm.write(
        f'Requested timestamp {timestamp} is at border of available times or exact match.'
      )
      p = self.pose_pos[idx]
      q = self.pose_orien[idx]
      odom_to_base = pq_to_se3(p, q)
    else:
      # Normal case: determine which two points to interpolate between
      if timestamp <= self.timestamps[idx]:
        # Interpolate between previous and current
        idx1, idx2 = idx - 1, idx
      else:
        # Interpolate between current and next
        idx1, idx2 = idx, idx + 1

      # Get the two poses for interpolation
      t1, t2 = self.timestamps[idx1], self.timestamps[idx2]
      pos1, pos2 = self.pose_pos[idx1], self.pose_pos[idx2]
      quat1, quat2 = self.pose_orien[idx1], self.pose_orien[idx2]

      # Create timestamps array for interpolation
      timestamps = np.array([t1, t2])
      target_time = np.array([timestamp])

      # Linear interpolation for position
      positions = np.array([pos1, pos2])
      translation_interpolator = interp1d(
        timestamps,
        positions,
        kind='linear',
        axis=0,
        bounds_error=False,
        fill_value=(pos1, pos2),
      )
      interpolated_position = translation_interpolator(target_time)[0]

      # SLERP interpolation for rotation
      rotations = R.from_quat([quat1, quat2])
      slerp_interpolator = Slerp(timestamps, rotations)
      interpolated_rotation = slerp_interpolator(target_time)
      interpolated_quat = interpolated_rotation.as_quat()[0]
      odom_to_base = pq_to_se3(interpolated_position, interpolated_quat)

    if self.odom_key == 'dlio_map_odometry':
      # This odometry topic is from: dlio_world_to_hesai frame
      dlio_world_to_hesai = odom_to_base
      odom_to_box_base = dlio_world_to_hesai @ attrs_to_se3(
        self.mission_root['hesai_points_undistorted'].attrs
      )
      base_to_box_base = pq_to_se3(
        self.mission_root['tf'].attrs['tf']['box_base']['translation'],
        self.mission_root['tf'].attrs['tf']['box_base']['rotation'],
      )
      odom_to_base = odom_to_box_base @ np.linalg.inv(base_to_box_base)

    return odom_to_base


def undistort_image(image, image_tag, mission_root):
  K = np.array(mission_root[image_tag].attrs['camera_info']['K']).reshape((3, 3))
  D = np.array(mission_root[image_tag].attrs['camera_info']['D'])
  h, w = image.shape[:2]

  helper = {}

  # Fill in auxiliary data for undistortion
  if not hasattr(helper, 'new_camera_info'):
    if (
      mission_root[image_tag].attrs['camera_info']['distortion_model'] == 'equidistant'
    ):
      helper['new_camera_matrix'] = (
        cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
          K, D, (w, h), np.eye(3), balance=1.0, fov_scale=1.0
        )
      )
      helper['D_new'] = [0, 0, 0, 0]
      helper['map1'], helper['map2'] = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), helper['new_camera_matrix'], (w, h), cv2.CV_16SC2
      )
    else:
      helper['new_camera_matrix'], _ = cv2.getOptimalNewCameraMatrix(
        K, D, (w, h), 1, (w, h)
      )
      helper['D_new'] = [0, 0, 0, 0, 0]
      helper['map1'], helper['map2'] = cv2.initUndistortRectifyMap(
        K, D, None, helper['new_camera_matrix'], (w, h), cv2.CV_16SC2
      )
    helper['invalid_mask'] = (
      cv2.remap(
        np.ones(image.shape[:2], dtype=np.uint8),
        helper['map1'],
        helper['map2'],
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
      )
      == 0
    )

  undistorted_image = cv2.remap(
    image,
    helper['map1'],
    helper['map2'],
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
  )
  return undistorted_image, helper


def project_lidar_to_camera(
  lidar_points,
  K,
  lidar_to_camera_transform,
  image_width,
  image_height,
  D=None,
  distortion_model='pinhole',
):
  """Project LiDAR points onto camera image plane with distortion correction (OpenCV)"""
  lidar_points_homo = np.hstack([lidar_points, np.ones((lidar_points.shape[0], 1))])
  camera_points_homo = (lidar_to_camera_transform @ lidar_points_homo.T).T
  camera_points = camera_points_homo[:, :3]

  # Project to image plane
  if distortion_model == 'pinhole':
    image_points = (K @ camera_points.T).T
    image_points[:, 0] /= image_points[:, 2]
    image_points[:, 1] /= image_points[:, 2]

  elif distortion_model == 'radtan':
    # OpenCV expects points in shape (N, 1, 3)
    objectPoints = camera_points.reshape(-1, 1, 3).astype(np.float32)
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)
    image_points, _ = cv2.projectPoints(objectPoints, rvec, tvec, K, D)
    image_points = image_points.reshape(-1, 2)

  elif distortion_model == 'equidistant':
    # Undistorted (pinhole) model
    objectPoints = camera_points.reshape(-1, 1, 3).astype(np.float32)
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)

    image_points, _ = cv2.fisheye.projectPoints(
      objectPoints,
      rvec,
      tvec,
      np.asarray(K, dtype=np.float64).reshape(3, 3),
      np.asarray(D, dtype=np.float64).reshape(-1, 1) if D is not None else None,
    )
    image_points = image_points.reshape(-1, 2)

  else:
    raise ValueError(f'Unsupported distortion model: {distortion_model}')

  # Filter points within image bounds
  valid_pixels = (
    (image_points[:, 0] >= 0)
    & (image_points[:, 0] < image_width)
    & (image_points[:, 1] >= 0)
    & (image_points[:, 1] < image_height)
    & (camera_points[:, 2] > 0)
  )

  mapping_idx = np.arange(len(image_points))[valid_pixels]

  valid_image_points = image_points[valid_pixels]
  valid_depths = camera_points[valid_pixels, 2]

  # Create depth image
  depth_image = np.full((image_height, image_width), -1, dtype=np.float32)

  mapping_image = np.full((image_height, image_width), -1, dtype=np.int32)

  if len(valid_image_points) > 0:
    pixel_coords = valid_image_points[:, :2].astype(int)
    for i, (x, y) in enumerate(pixel_coords):
      if 0 <= x < image_width and 0 <= y < image_height:
        # Use closest depth if multiple points project to same pixel
        if depth_image[y, x] == -1 or valid_depths[i] < depth_image[y, x]:
          depth_image[y, x] = valid_depths[i]
          mapping_image[y, x] = mapping_idx[i]

  return depth_image, mapping_image


def rectangular_prism(
  init_position, width, height, up_axis, color, slice_direction=None
):
  w, h = width / 2, height / 2
  min_bound = np.array([-w] * 3)
  max_bound = np.array([w] * 3)
  idx = {'x': 0, 'y': 1, 'z': 2}[up_axis]
  min_bound[idx] = -h
  max_bound[idx] = h
  if slice_direction is not None:
    if slice_direction == '+':
      min_bound[idx] = 0.0
    elif slice_direction == '-':
      max_bound[idx] = 0.0
    else:
      raise ValueError(f'Invalid slice direction: {slice_direction}')

  min_bound = init_position + min_bound
  max_bound = init_position + max_bound

  bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
  bbox.color = color
  return bbox


def remove_small_clusters(pcd, eps, min_points, min_cluster_size):
  """Remove small disconnected clusters of points"""

  # Perform DBSCAN clustering to find connected components
  with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
    labels = np.array(
      pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True)
    )

  # Count points in each cluster
  max_label = labels.max()
  tqdm.write(f'Point cloud has {max_label + 1} clusters')

  cluster_sizes = {}
  for i in range(max_label + 1):
    cluster_sizes[i] = np.sum(labels == i)
    tqdm.write(f'Cluster {i}: {cluster_sizes[i]} points')

  # Keep only large clusters
  large_cluster_indices = []
  for cluster_id, size in cluster_sizes.items():
    if size >= min_cluster_size:
      cluster_indices = np.where(labels == cluster_id)[0]
      large_cluster_indices.extend(cluster_indices)

  # Also keep noise points (label = -1) if you want
  noise_indices = np.where(labels == -1)[0]
  tqdm.write(f'Noise points: {len(noise_indices)}')

  # Create filtered point cloud
  filtered_pcd = pcd.select_by_index(large_cluster_indices)
  removed_pcd = pcd.select_by_index(large_cluster_indices, invert=True)

  tqdm.write(f'Kept {len(filtered_pcd.points)} points from large clusters')
  tqdm.write(f'Removed {len(removed_pcd.points)} points from small clusters')

  return filtered_pcd, removed_pcd, labels


def remove_outliers_comprehensive(
  pcd, statistical_nb_neighbors=10, statistical_std_ratio=0.05
):
  pcd_stat, _ = pcd.remove_statistical_outlier(
    nb_neighbors=statistical_nb_neighbors, std_ratio=statistical_std_ratio
  )
  return pcd_stat


class RetryWrapper(Generic[P, T]):
  def __init__(
    self,
    fn: Callable[P, T],
    max_retries: int = 30,
    delay_s: float = 10,
    backoff_s: float = 60,
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

  def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
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


def download_from_s3(s3_path, local_path, recursive=False):
  Path(local_path).parent.mkdir(parents=True, exist_ok=True)

  # Try s5cmd first
  if shutil.which('s5cmd'):
    try:
      cmd = [
        's5cmd',
        'cp',
        '--concurrency',
        '5',
        '--part-size',
        '50',
        '--if-size-differ',
        '--if-source-newer',
      ]
      s5cmd_s3_path = s3_path
      if recursive:
        s5cmd_s3_path = s3_path.rstrip('/') + '/*'

      cmd.extend([s5cmd_s3_path, local_path])

      subprocess.run(cmd, check=True, capture_output=True, text=True)
      print(f'Successfully downloaded {s3_path} using s5cmd')
      return  # Exit if s5cmd succeeds
    except subprocess.CalledProcessError as e:
      print(
        f's5cmd failed with exit code {e.returncode}: {e.stderr}. Falling back to AWS CLI'
      )
    except Exception as e:
      print(f'Unexpected s5cmd error: {str(e)}. Falling back to AWS CLI')
  else:
    print('s5cmd not found, falling back to AWS CLI')

  # Fallback to AWS CLI
  try:
    print(f'Downloading {s3_path} using AWS CLI')
    cmd = ['aws', 's3', 'cp']
    if recursive:
      cmd.append('--recursive')
    cmd.extend([s3_path, local_path])

    subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f'Successfully downloaded {s3_path} using AWS CLI')
  except subprocess.CalledProcessError as e:
    print(f'AWS CLI failed with exit code {e.returncode}: {e.stderr}')
    raise
  except Exception as e:
    print(f'Unexpected AWS CLI error: {str(e)}')
    raise


def s3_directory_exists(bucket_name, prefix):
  # Ensure prefix ends with "/" to match "folder"
  import boto3

  s3 = boto3.client('s3')

  if not prefix.endswith('/'):
    prefix += '/'

  resp = s3.list_objects_v2(
    Bucket=bucket_name,
    Prefix=prefix,
    MaxKeys=1,  # only need to know if at least one object exists
  )
  s3.close()
  return 'Contents' in resp


def s3_delete_prefix(bucket_name, prefix):
  import boto3

  s3 = boto3.client('s3')
  paginator = s3.get_paginator('list_objects_v2')

  for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
    if 'Contents' in page:
      # Build list of objects to delete
      objects = [{'Key': obj['Key']} for obj in page['Contents']]

      # Batch delete (max 1000 at a time)
      s3.delete_objects(Bucket=bucket_name, Delete={'Objects': objects})
      print(f'Deleted {len(objects)} objects from {prefix}')
  s3.close()


def s5cmd_delete_prefix(
  bucket_name: str,
  prefix: str,
  max_retries: int = 3,
  num_workers: int = 50,
  dry_run: bool = False,
) -> None:
  """Execute s5cmd rm to delete all objects with a given prefix.

  This is useful for bulk deletions, which s5cmd can handle much faster than boto3.

  Parameters
  ----------
  bucket_name : str
      S3 bucket name.
  prefix : str
      Prefix to delete (e.g., "data/models/").
  max_retries : int
      Number of retries.
  num_workers : int
      Number of concurrent workers for s5cmd.
  dry_run : bool
      If True, show what would be deleted without actually deleting.
  """
  assert shutil.which('s5cmd') is not None, 's5cmd not found'

  # Construct the S3 path with wildcard
  s3_path = f's3://{bucket_name}/{prefix.rstrip("/")}/*'

  # Build command flags
  flags = ['--numworkers', str(num_workers)]
  if dry_run:
    flags.append('--dry-run')

  try:
    RetryWrapper(subprocess.check_call, max_retries=max_retries)(
      ['s5cmd'] + flags + ['rm', s3_path]
    )
    if not dry_run:
      print(
        f'Successfully deleted objects with prefix {prefix} from bucket {bucket_name}'
      )
    else:
      print(f'Dry run completed for prefix {prefix} in bucket {bucket_name}')
  except subprocess.CalledProcessError as e:
    print(f'Failed to delete prefix {prefix} from bucket {bucket_name}: {e}')


def merge_close_groups(groups, timestamps, distances, max_distance=1.0):
  """
  Merge groups where any point in one group is within max_distance of any point in another.

  Args:
      groups: List of lists of timestamps (output from split_by_distance_gap)
      timestamps: [T] array of all timestamps (for mapping)
      distances: [T, 3] array of 3D positions
      max_distance: maximum distance threshold for merging

  Returns:
      List of merged groups (list of lists of timestamps)
  """
  if len(groups) <= 1:
    return groups

  timestamps = np.array(timestamps)
  distances = np.array(distances)
  n_groups = len(groups)

  # Create mapping from timestamp to index
  ts_to_idx = {ts: idx for idx, ts in enumerate(timestamps)}

  # Build adjacency matrix: groups[i] and groups[j] are connected if close
  connected = np.zeros((n_groups, n_groups), dtype=bool)

  for i in range(n_groups):
    for j in range(i + 1, n_groups):
      # Get indices for both groups
      indices_i = [ts_to_idx[ts] for ts in groups[i]]
      indices_j = [ts_to_idx[ts] for ts in groups[j]]

      # Get positions for both groups
      pos_i = distances[indices_i]  # [N_i, 3]
      pos_j = distances[indices_j]  # [N_j, 3]

      # Compute pairwise distances between all points in groups i and j
      # Broadcasting: pos_i[:, None, :] is [N_i, 1, 3], pos_j[None, :, :] is [1, N_j, 3]
      diffs = pos_i[:, None, :] - pos_j[None, :, :]  # [N_i, N_j, 3]
      pairwise_dists = np.linalg.norm(diffs, axis=2)  # [N_i, N_j]

      # Check if any pair is within threshold
      if np.any(pairwise_dists <= max_distance):
        connected[i, j] = True
        connected[j, i] = True

  # Find connected components using Union-Find
  parent = list(range(n_groups))

  def find(x):
    if parent[x] != x:
      parent[x] = find(parent[x])  # Path compression
    return parent[x]

  def union(x, y):
    root_x, root_y = find(x), find(y)
    if root_x != root_y:
      parent[root_x] = root_y

  # Union connected groups
  for i in range(n_groups):
    for j in range(i + 1, n_groups):
      if connected[i, j]:
        union(i, j)

  # Group indices by their root
  components = {}
  for i in range(n_groups):
    root = find(i)
    if root not in components:
      components[root] = []
    components[root].append(i)

  # Merge groups in same component
  merged_groups = []
  for component in components.values():
    merged_timestamps = []
    for group_idx in component:
      merged_timestamps.extend(groups[group_idx])
    merged_groups.append(sorted(merged_timestamps))

  return merged_groups


def split_by_distance_gap(distances, timestamps, max_distance=1.0, merge_groups=False):
  """
  Split data into groups whenever the spatial distance between consecutive points exceeds threshold.

  Args:
      distances: [T, 3] array of 3D positions
      timestamps: [T] array of timestamps (used for sorting)
      max_distance: maximum allowed Euclidean distance between consecutive points

  Returns:
      List of lists, where each sublist contains indices of spatially continuous data
  """
  distances = np.array(distances)
  timestamps = np.array(timestamps)

  # Sort by timestamps to ensure temporal order
  sorted_indices = np.argsort(timestamps)
  sorted_timestamps = timestamps[sorted_indices]
  sorted_distances = distances[sorted_indices]

  # Compute Euclidean distances between consecutive positions
  # diff gives us [pos[i+1] - pos[i]] for each i
  position_diffs = np.diff(sorted_distances, axis=0)  # [T-1, 3]
  euclidean_dists = np.linalg.norm(position_diffs, axis=1)  # [T-1]

  # Find where distances exceed threshold
  split_points = np.where(euclidean_dists > max_distance)[0] + 1

  # Split timestamps into groups
  groups = np.split(sorted_timestamps, split_points)
  groups = [group.tolist() for group in groups]
  if merge_groups:
    groups = merge_close_groups(
      groups, timestamps, distances, max_distance=max_distance
    )
  return groups


def split_by_time_gap(timestamps, max_gap_seconds=1.0):
  """
  Split timestamps into groups whenever the gap exceeds max_gap_seconds.

  Args:
      timestamps: list or array of Unix timestamps
      max_gap_seconds: maximum allowed gap in seconds

  Returns:
      List of lists, where each sublist contains indices of consecutive timestamps
  """
  timestamps = np.array(timestamps)

  # Sort timestamps to ensure they're in order
  sorted_indices = np.argsort(timestamps)
  sorted_timestamps = timestamps[sorted_indices]

  # Compute differences between consecutive timestamps
  time_diffs = np.diff(sorted_timestamps)

  # Find where gaps exceed threshold
  split_points = np.where(time_diffs > max_gap_seconds)[0] + 1

  # Split indices into groups
  groups = np.split(sorted_timestamps, split_points)

  return [group.tolist() for group in groups]
