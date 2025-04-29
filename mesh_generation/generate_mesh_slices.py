from typing import Optional
import os
from absl import app
import pickle
import time
import open3d as o3d
import open3d.core as o3c
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path
import viser.transforms as vtf
import copy
import cv2

from ml_collections import config_flags
from configs.base import get_config
from PIL import Image

from rendering import batch_gs_renderer

# _CONFIG = config_flags.DEFINE_config_dict("config", get_config())
_CONFIG = config_flags.DEFINE_config_file("config")


def _distance(p1, p2, up_axis):
  axis_indices = {
    "x": [1, 2],  # yz plane
    "y": [0, 2],  # xz plane
    "z": [0, 1],  # xy plane
  }

  if up_axis not in axis_indices:
    raise ValueError(f"Invalid up_axis: {up_axis}")

  idx = axis_indices[up_axis]
  return np.linalg.norm(p1[..., idx] - p2[..., idx], axis=-1)


def load_transforms_ns(json_path: Path) -> dict:
  print(f"Opening: {json_path}")
  with json_path.open("r") as f:
    transforms = json.load(f)
  return transforms


def get_voxel_block_grid(
  device: o3d.core.Device, integrate_color: bool, voxel_size: float
) -> o3d.t.geometry.VoxelBlockGrid:
  if integrate_color:
    return o3d.t.geometry.VoxelBlockGrid(
      attr_names=("tsdf", "weight", "color"),
      attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
      attr_channels=((1), (1), (3)),
      voxel_size=voxel_size,
      block_resolution=18,
      block_count=100000,
      device=device,
    )
  else:
    return o3d.t.geometry.VoxelBlockGrid(
      attr_names=("tsdf", "weight"),
      attr_dtypes=(o3c.float32, o3c.float32),
      attr_channels=((1), (1)),
      voxel_size=voxel_size,
      block_resolution=18,
      block_count=100000,
      device=device,
    )


def rectangular_prism(init_pose, width, height, up_axis, color, slice_direction: Optional[str] = None):
  w, h = width / 2, height / 2
  min_bound = np.array([-w] * 3)
  max_bound = np.array([w] * 3)
  idx = {'x': 0, 'y': 1, 'z': 2}[up_axis]
  min_bound[idx] = -h
  max_bound[idx] = h
  if slice_direction is not None:
    if slice_direction == '+':
      min_bound[idx] = 0.
    elif slice_direction == '-':
      max_bound[idx] = 0.
    else:
      raise ValueError(f"Invalid slice direction: {slice_direction}")

  min_bound = init_pose.translation() + min_bound
  max_bound = init_pose.translation() + max_bound

  bbox = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=min_bound, max_bound=max_bound
  )
  bbox.color = color
  return bbox

def load_arkit_poses(traj_file: Path):
  # Check if the trajectory file exists
  if traj_file.exists():
      # Read the trajectory file
      # Format: timestamp rotation(axis-angle, 3 values) translation(3 values)
      traj_data = []
      with open(traj_file, 'r') as f:
          for line in f:
              values = line.strip().split()
              if len(values) >= 7:  # Ensure we have all required values
                  timestamp = float(values[0])
                  rotation = np.array([float(values[1]), float(values[2]), float(values[3])])
                  matrix, _ = cv2.Rodrigues(rotation)
                  quaternion = vtf.SO3.from_matrix(matrix).as_quaternion_xyzw()
                  translation = np.array([float(values[4]), float(values[5]), float(values[6])])
                  traj_data.append([timestamp, *translation, *quaternion])
      
      # Convert to numpy array
      traj_array = np.array(traj_data)
      return traj_array
  else:
    raise ValueError(f"Trajectory file not found: {traj_file}")


def load_arkit_data(
    traj_path: Path,
    depth_path: Path,
    color_path: Path,
    intrinsics_path: Path,
) -> tuple[list[Path], list[Path], list[Path]]:

  traj_poses = load_arkit_poses(traj_path)
  depth_files = list(depth_path.glob('*.png'))
  color_files = list(color_path.glob('*.png'))
  intrinsics_files = list(intrinsics_path.glob('*.pincam'))
  prefix = depth_files[0].name.split('_')[0]
  timestap_strings = [f'{prefix}_{np.round(timestamp, 4):.3f}' for timestamp in traj_poses[:, 0]]
  transforms = []
  for i, timestamp_string in enumerate(timestap_strings):
    depth_file = depth_path / (timestamp_string + '.png')
    color_file = color_path / (timestamp_string + '.png')
    intrinsics_file = intrinsics_path / (timestamp_string + '.pincam')
    assert depth_file.exists(), depth_file
    assert color_file.exists(), color_file
    assert intrinsics_file.exists(), intrinsics_file
    with open(intrinsics_file, 'r') as f:
      line = f.readline().strip()
      values = line.split()
      intrinsics = {
        'width': int(values[0]),
        'height': int(values[1]),
        'fl_x': float(values[2]),
        'fl_y': float(values[3]),
        'cx': float(values[4]),
        'cy': float(values[5])
      }
    transform_matrix = vtf.SE3.from_rotation_and_translation(
      translation=traj_poses[i, 1:4],
      rotation=vtf.SO3.from_quaternion_xyzw(traj_poses[i, 4:])
    ).as_matrix()

    # Transforms from: https://docs.nerf.studio/_modules/nerfstudio/data/dataparsers/arkitscenes_dataparser.html#ARKitScenes
    transform_matrix = np.linalg.inv(transform_matrix)
    transform_matrix[0:3, 1:3] *= -1
    transform_matrix = transform_matrix[np.array([1, 0, 2, 3]), :]
    transform_matrix[2, :] *= -1

    transforms.append({
      'depth_file_path': depth_file,
      'file_path': color_file,
      'transform_matrix': transform_matrix,
      **intrinsics,
    })
  print(f'Loaded {len(depth_files)} depth files, {len(color_files)} color files, and {len(intrinsics_files)} intrinsics files')

  return transforms

# From: https://github.com/nerfstudio-project/nerfstudio/blob/5003d0e2711d9231908d81bc0e0b7823f96889b0/nerfstudio/cameras/lie_groups.py#L25
def exp_map_SO3xR3(tangent_vector):
    """Compute the exponential map of the direct product group `SO(3) x R^3`.

    This can be used for learning pose deltas on SE(3), and is generally faster than `exp_map_SE3`.

    Args:
        tangent_vector: Tangent vector; length-3 translations, followed by an `so(3)` tangent vector.
    Returns:
        [R|t] transformation matrices.
    """
    import torch
    # code for SO3 map grabbed from pytorch3d and stripped down to bare-bones
    log_rot = tangent_vector[:, 3:]
    nrms = (log_rot * log_rot).sum(1)
    rot_angles = torch.clamp(nrms, 1e-4).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = torch.zeros((log_rot.shape[0], 3, 3), dtype=log_rot.dtype, device=log_rot.device)
    skews[:, 0, 1] = -log_rot[:, 2]
    skews[:, 0, 2] = log_rot[:, 1]
    skews[:, 1, 0] = log_rot[:, 2]
    skews[:, 1, 2] = -log_rot[:, 0]
    skews[:, 2, 0] = -log_rot[:, 1]
    skews[:, 2, 1] = log_rot[:, 0]
    skews_square = torch.bmm(skews, skews)

    ret = torch.zeros(tangent_vector.shape[0], 3, 4, dtype=tangent_vector.dtype, device=tangent_vector.device)
    ret[:, :3, :3] = (
        fac1[:, None, None] * skews
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    # Compute the translation
    ret[:, :3, 3] = tangent_vector[:, :3]
    return ret

def load_ns_optimized_traj(splatfacto_dir: Path, transforms: list[dict]):
  ckpt_dir = splatfacto_dir / 'nerfstudio_models'
  ckpt = list(ckpt_dir.glob('*.ckpt'))[-1]
  print(f'Loading NS optimized trajectory from {ckpt}')
  import torch
  pose_adjustment = torch.load(ckpt)['pipeline']['_model.camera_optimizer.pose_adjustment']
  pose_adjustment = exp_map_SO3xR3(pose_adjustment).cpu().numpy()
  dataparser_transform = np.array(json.load(open(splatfacto_dir / 'dataparser_transforms.json'))['transform'])
  dataparser_transform = vtf.SE3.from_rotation_and_translation(
    rotation=vtf.SO3.from_matrix(dataparser_transform[:, :3]),
    translation=dataparser_transform[:, 3:].squeeze()
  )
  # renderer = batch_gs_renderer.BatchPLYRenderer(
  #   ply_path=splatfacto_dir / 'splat.ply',
  #   device='cuda'
  # )
  new_transforms = []
  for i, transform in enumerate(transforms):

    # render_pose = vtf.SE3.from_matrix(transform['transform_matrix'])
    # render_pose = vtf.SE3.from_rotation_and_translation(
    #   translation=render_pose.translation(),
    #   rotation=render_pose.rotation() @ vtf.SO3.from_x_radians(-np.pi),
    # )
    # if i % 25 == 0:
    #   render_pose = torch.tensor(render_pose.as_matrix(), device='cuda', dtype=torch.float32)[None]
    #   render = renderer.batch_render(
    #     render_pose, focal=500, h=512, w=512, out_device='cuda'
    #   )
    #   print(render.shape)
    #   import matplotlib.pyplot as plt
    #   plt.imshow(render[0].cpu().numpy())
    #   plt.show()

    adj = torch.tensor(vtf.SE3.from_rotation_and_translation(
      rotation=vtf.SO3.from_matrix(pose_adjustment[i, :, :3]),
      translation=pose_adjustment[i, :, 3:].squeeze()
    ).as_matrix(), device='cuda', dtype=torch.float32)[None]
    transform_matrix = vtf.SE3.from_matrix(transform['transform_matrix']).as_matrix()
    c2ws = torch.tensor(transform_matrix, device='cuda', dtype=torch.float32)[None]
    # c2ws = torch.bmm(renderer.dataparser_transform.repeat(c2ws.shape[0], 1, 1), c2ws)
    # c2ws[:, :3, 3] *= renderer.dataparser_scale
    c2ws = torch.cat(
            [
                # Apply rotation to directions in world coordinates, without touching the origin.
                # Equivalent to: directions -> correction[:3,:3] @ directions
                torch.bmm(adj[..., :3, :3], c2ws[..., :3, :3]),
                # Apply translation in world coordinate, independently of rotation.
                # Equivalent to: origins -> origins + correction[:3,3]
                c2ws[..., :3, 3:] + adj[..., :3, 3:],
            ],
            dim=-1,
    ).cpu().numpy()[0]
    transform['transform_matrix'] = vtf.SE3.from_rotation_and_translation(
      rotation=vtf.SO3.from_matrix(c2ws[:3, :3]),
      translation=c2ws[:3, 3:].squeeze()
    ).as_matrix()
    new_transforms.append(transform)

  return new_transforms


def process_transforms_arkit(
  arkit_dir: Path,
  vbg: o3d.t.geometry.VoxelBlockGrid,
  integrate_color: bool,
  depth_scale: float,
  depth_max: float,
  bbox_slice_size: float,
  up_axis: str,
  slice_direction: str,
  device: o3d.core.Device,
):

  frames_path = None
  for file in arkit_dir.iterdir():
    if file.name.endswith("frames"):
      frames_path = file


  # Path to the trajectory file
  traj_path = frames_path / "lowres_wide.traj"
  depth_path = frames_path / 'lowres_depth'
  color_path = frames_path / 'lowres_wide'
  intrinsics_path = frames_path / 'lowres_wide_intrinsics'
  transforms = load_arkit_data(
    traj_path, depth_path, color_path, intrinsics_path
  )

  splatfacto_dir = arkit_dir / "splatfacto"
  transforms = load_ns_optimized_traj(splatfacto_dir, transforms)

  start = time.time()
  # Bounding boxes used to extract slices from input pointcloud.
  bounding_boxes = []
  # Bounding boxes used to check if slices have enough points.
  point_checking_bounding_boxes = []
  # Camera positions and orientations.
  camera_positions = []
  camera_orientations = []
  for frame in tqdm(transforms):
    depth_path = frame["depth_file_path"]
    if "right" in str(depth_path):
      continue  # TODO: Use both
    if depth_path.suffix == '.png':
      depth = np.array(Image.open(depth_path))  # Reads as 16-bit integer
      depth = depth.astype(float) / 1000.0  # Convert millimeters to meters
    elif depth_path.suffix in ('.npy', '.npz'):
      depth = np.load(depth_path)
    H, W = depth.shape
    depth = o3d.t.geometry.Image(
      o3d.core.Tensor(depth, dtype=o3d.core.Dtype.Float32)
    ).to(device)
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
      width=W,
      height=H,
      fx=frame["fl_x"],
      fy=frame["fl_y"],
      cx=frame["cx"],
      cy=frame["cy"],
    )
    intrinsic = o3d.core.Tensor(
      intrinsic.intrinsic_matrix, o3d.core.Dtype.Float64
    )
    extrinsic = np.array(frame["transform_matrix"])
    extrinsic = vtf.SE3.from_matrix(extrinsic)
    extrinsic = vtf.SE3.from_rotation_and_translation(
      translation=extrinsic.translation(),
      rotation=extrinsic.rotation() @ vtf.SO3.from_x_radians(-np.pi),
    ).inverse()
    extrinsic = o3d.core.Tensor(extrinsic.as_matrix(), dtype=o3d.core.Dtype.Float64)
    frustum_block_coords = vbg.compute_unique_block_coordinates(
      depth, intrinsic, extrinsic, depth_scale, depth_max
    )

    if integrate_color:
      color_path = frame["file_path"]
      color = o3d.t.io.read_image(str(color_path)).as_tensor().numpy()
      color = color.astype(np.float32) / 255.0
      color = o3d.t.geometry.Image(color).to(device)
      vbg.integrate(
        frustum_block_coords,
        depth,
        color,
        intrinsic,
        intrinsic,
        extrinsic,
        depth_scale,
        depth_max,
      )
    else:
      vbg.integrate(
        frustum_block_coords,
        depth,
        intrinsic,
        extrinsic,
        depth_scale,
        depth_max,
      )

    pose = vtf.SE3.from_matrix(np.array(frame["transform_matrix"]))
    camera_positions.append(pose.translation())
    camera_orientations.append(pose.rotation().as_quaternion_xyzw())

    # Bounding boxes used to extract slices from input pointcloud.
    bounding_boxes.append(rectangular_prism(pose,bbox_slice_size, 4., up_axis, (1.0, 0.0, 0.0), slice_direction))
    point_checking_bounding_boxes.append(rectangular_prism(pose, 0.5, 4., up_axis, (0.0, 0.0, 1.0), slice_direction))

  print(
    "Finished integrating {} frames in {} seconds".format(
      len(transforms), time.time() - start
    )
  )

  pcd = vbg.extract_point_cloud().cpu().to_legacy()
  return (
    pcd,
    bounding_boxes,
    point_checking_bounding_boxes,
    camera_positions,
    camera_orientations,
  )

def preprocess_transforms_ns(
  json_path: Path,
  output_dir: Path,
):
  json_path = Path(json_path)
  parent_dir = json_path.parents[0]
  save_dir = parent_dir / output_dir
  if not Path.exists(save_dir):
    Path.mkdir(save_dir)
  transforms = load_transforms_ns(json_path)
  transforms = [frame for frame in transforms["frames"]]

  for frame in transforms:
    frame["depth_file_path"] = str(json_path.parents[0] / frame["depth_file_path"])
    frame["file_path"] = str(json_path.parents[0] / frame["file_path"])

    

  return transforms, save_dir


def process_transforms_ns(
  json_path: Path,
  transforms: dict,
  vbg: o3d.t.geometry.VoxelBlockGrid,
  integrate_color: bool,
  depth_scale: float,
  depth_max: float,
  bbox_slice_size: float,
  up_axis: str,
  device: o3d.core.Device,
):
  start = time.time()
  # Bounding boxes used to extract slices from input pointcloud.
  bounding_boxes = []
  # Bounding boxes used to check if slices have enough points.
  point_checking_bounding_boxes = []
  # Camera positions and orientations.
  camera_positions = []
  camera_orientations = []
  for frame in tqdm(transforms["frames"]):
    depth_path = json_path.parents[0] / frame["depth_file_path"]
    if "right" in str(depth_path):
      continue  # TODO: Use both
    if depth_path.suffix == '.png':
      depth = np.array(Image.open(depth_path))  # Reads as 16-bit integer
      depth = depth.astype(float) / 1000.0  # Convert millimeters to meters
    elif depth_path.suffix in ('.npy', '.npz'):
      depth = np.load(depth_path)
    H, W = depth.shape
    depth = o3d.t.geometry.Image(
      o3d.core.Tensor(depth, dtype=o3d.core.Dtype.Float32)
    ).to(device)
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
      width=W,
      height=H,
      fx=frame["fl_x"],
      fy=frame["fl_y"],
      cx=frame["cx"],
      cy=frame["cy"],
    )
    intrinsic = o3d.core.Tensor(
      intrinsic.intrinsic_matrix, o3d.core.Dtype.Float64
    )
    extrinsic = np.array(frame["transform_matrix"])
    extrinsic = vtf.SE3.from_matrix(extrinsic)
    extrinsic = vtf.SE3.from_rotation_and_translation(
      translation=extrinsic.translation(),
      rotation=extrinsic.rotation() @ vtf.SO3.from_x_radians(-np.pi),
    ).inverse()
    extrinsic = extrinsic.as_matrix()
    extrinsic = o3d.core.Tensor(extrinsic, dtype=o3d.core.Dtype.Float64)
    color_path = json_path.parents[0] / frame["file_path"]
    color = o3d.t.io.read_image(str(color_path)).as_tensor().numpy()
    color = color.astype(np.float32) / 255.0
    color = o3d.t.geometry.Image(color).to(device)
    frustum_block_coords = vbg.compute_unique_block_coordinates(
      depth, intrinsic, extrinsic, depth_scale, depth_max
    )

    if integrate_color:
      vbg.integrate(
        frustum_block_coords,
        depth,
        color,
        intrinsic,
        intrinsic,
        extrinsic,
        depth_scale,
        depth_max,
      )
    else:
      vbg.integrate(
        frustum_block_coords,
        depth,
        intrinsic,
        extrinsic,
        depth_scale,
        depth_max,
      )
    dt = time.time() - start

    pose = vtf.SE3.from_matrix(np.array(frame["transform_matrix"]))
    camera_positions.append(pose.translation())
    camera_orientations.append(pose.rotation().as_quaternion_xyzw())

    # Bounding boxes used to extract slices from input pointcloud.
    bounding_boxes.append(rectangular_prism(pose,bbox_slice_size, 3., up_axis, (1.0, 0.0, 0.0)))
    point_checking_bounding_boxes.append(rectangular_prism(pose, 0.5, 3., up_axis, (0.0, 0.0, 1.0)))

  print(
    "Finished integrating {} frames in {} seconds".format(
      len(transforms["frames"]), dt
    )
  )

  # THIS IS WHAT WE USE
  pcd = vbg.extract_point_cloud().cpu().to_legacy()
  return (
    pcd,
    bounding_boxes,
    point_checking_bounding_boxes,
    camera_positions,
    camera_orientations,
  )


def visualize_geometries(geometries: list[o3d.geometry.Geometry]):
  vis = o3d.visualization.Visualizer()
  vis.create_window()
  for geometry in geometries:
    vis.add_geometry(geometry)
  vis.run()


def enough_points(pcd, bbox, num_points):
  # Check if the bounding box has enough points.
  if len(pcd.crop(bbox).points) < num_points:
    return False
  return True


def march_idx_along_distance(positions, idx, distance, increment, up_axis):
  new_idx = idx
  # dist_sum = 0.0
  # while dist_sum < distance:
  while _distance(positions[idx], positions[new_idx], up_axis) < distance:
    # dist_sum += _distance(positions[idx], positions[new_idx], up_axis)
    new_idx += increment
    if new_idx == -1:
      return 0
    if new_idx == len(positions):
      return len(positions) - 1
  return new_idx


def get_mesh_at_frac(
  pcd,
  bounding_boxes,
  point_checking_bounding_boxes,
  camera_positions,
  camera_orientations,
  frac,
  buffer_distance,
  slice_distance,
  min_poses_per_segment,
  voxel_size,
  poisson_depth,
  density_threshold,
  decimation_factor,
  sharpen_mesh,
  sharpen_iterations,
  sharpen_strength,
  to_ig_euler_xyz,
  up_axis,
  visualize,
):
  num_cameras_after_buffer = 1 + march_idx_along_distance(
    camera_positions, len(camera_positions) - 1, buffer_distance, -1, up_axis
  )

  # Indices along which to slice pointcloud.
  bbox_start_idx = min(
    int(num_cameras_after_buffer * frac), num_cameras_after_buffer - 1
  )
  while not enough_points(pcd, point_checking_bounding_boxes[bbox_start_idx], 100):
    bbox_start_idx += 1

  start_idx = end_idx = march_idx_along_distance(
    camera_positions, bbox_start_idx, buffer_distance, 1, up_axis
  )
  bbox_end_idx = march_idx_along_distance(
    camera_positions, start_idx, slice_distance + buffer_distance, 1, up_axis
  )
  end_idx = march_idx_along_distance(
    camera_positions, bbox_end_idx, buffer_distance, -1, up_axis
  )

  if (end_idx - start_idx) < min_poses_per_segment:
    raise ValueError(
      f"Segment {frac} only has {1 + (end_idx - start_idx)} poses."
    )

  # Slice pointcloud.
  curr_pcd = o3d.geometry.PointCloud()
  for box_idx in tqdm(range(bbox_start_idx, bbox_end_idx), leave=False):
    if not enough_points(pcd, point_checking_bounding_boxes[box_idx], 100):
      raise ValueError(
        f"Segment {frac} has insufficient points in slice for box {box_idx}"
      )
    curr_pcd += pcd.crop(bounding_boxes[box_idx])

  curr_pcd = curr_pcd.voxel_down_sample(voxel_size)
  mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    curr_pcd, depth=poisson_depth
  )
  vertices_to_remove = densities < np.quantile(densities, density_threshold)
  mesh.remove_vertices_by_mask(vertices_to_remove)
  triangle_clusters, cluster_n_triangles, cluster_area = (
    mesh.cluster_connected_triangles()
  )
  largest_cluster_idx = np.argmax(cluster_n_triangles)
  triangles_to_remove = triangle_clusters != largest_cluster_idx
  mesh.remove_triangles_by_mask(triangles_to_remove)

  mesh.remove_degenerate_triangles()
  mesh.remove_duplicated_triangles()
  mesh.remove_duplicated_vertices()
  mesh.remove_non_manifold_edges()
  if decimation_factor > 1:
    mesh = mesh.simplify_quadric_decimation(
      len(mesh.triangles) // decimation_factor
    )
  if sharpen_mesh:
    mesh = mesh.filter_sharpen(
      number_of_iterations=sharpen_iterations, strength=sharpen_strength
    )
  mesh = mesh.compute_vertex_normals()

  curr_camera_positions = camera_positions[start_idx:end_idx]
  curr_camera_orientations = camera_orientations[start_idx:end_idx]
  curr_camera_orientations = vtf.SO3.from_quaternion_xyzw(
    np.array(curr_camera_orientations)
  ).as_matrix()

  mean_vertex = np.mean(np.array(mesh.vertices), axis=0, keepdims=True)
  mesh.translate((-1 * mean_vertex[0]).tolist())
  curr_camera_positions -= mean_vertex

  # To IsaacGym coordinate frame.
  mesh, curr_camera_positions, curr_camera_orientations = (
    rotate_mesh_and_cameras(
      mesh,
      curr_camera_positions,
      curr_camera_orientations,
      vtf.SO3.from_x_radians(to_ig_euler_xyz[0]),
    )
  )
  mesh, curr_camera_positions, curr_camera_orientations = (
    rotate_mesh_and_cameras(
      mesh,
      curr_camera_positions,
      curr_camera_orientations,
      vtf.SO3.from_y_radians(to_ig_euler_xyz[1]),
    )
  )
  mesh, curr_camera_positions, curr_camera_orientations = (
    rotate_mesh_and_cameras(
      mesh,
      curr_camera_positions,
      curr_camera_orientations,
      vtf.SO3.from_z_radians(to_ig_euler_xyz[2]),
    )
  )

  curr_camera_orientations = vtf.SO3.from_matrix(
    curr_camera_orientations
  ).as_quaternion_xyzw()

  triangles = np.array(mesh.triangles)
  vertices = np.array(mesh.vertices)
  curr_vals = dict(
    vertices=vertices,
    triangles=triangles,
    cam_trans=curr_camera_positions,
    cam_quat=curr_camera_orientations,
    offset=mean_vertex,
  )

  if visualize:
    camera_point_cloud = o3d.geometry.PointCloud()
    camera_point_cloud.points = o3d.utility.Vector3dVector(
      np.array(curr_camera_positions)
    )
    colors = np.zeros_like(
      np.array(curr_camera_positions)
    )  # All black for example
    colors[:, 1] = 1  # Set green color
    camera_point_cloud.colors = o3d.utility.Vector3dVector(colors)
    visualize_geometries([camera_point_cloud, mesh])

  return curr_vals


def rotate_mesh_and_cameras(
  mesh, camera_positions, camera_orientations, rotation
):
  mesh = copy.deepcopy(mesh)
  mesh.rotate(rotation.as_matrix())
  camera_positions = np.dot(camera_positions, rotation.as_matrix().T)
  camera_orientations = rotation.as_matrix() @ camera_orientations
  return mesh, camera_positions, camera_orientations


def estimate_num_slices(
  camera_positions, up_axis, slice_distance, buffer_distance, slice_overlap
):
  total_distance = 0.0
  for i in range(1, len(camera_positions)):
    total_distance += _distance(
      camera_positions[i], camera_positions[i - 1], up_axis
    )
  total_slice_distance = slice_distance + (2 * buffer_distance) - slice_overlap
  return max(1, int(total_distance / total_slice_distance))


def main(_):
  config = _CONFIG.value


  device = o3d.core.Device("CUDA:0")
  vbg = get_voxel_block_grid(device, config.integrate_color, config.voxel_size)

  if config.format == "ns":
    json_path = Path(config.json_path)
    parent_dir = json_path.parents[0]
    save_dir = parent_dir / config.output_dir
    if not Path.exists(save_dir):
      Path.mkdir(save_dir)
    transforms = load_transforms_ns(json_path)
    (
      pcd,
      bounding_boxes,
      point_checking_bounding_boxes,
      camera_positions,
      camera_orientations,
    ) = process_transforms_ns(
      json_path,
      transforms,
      vbg,
      config.integrate_color,
      config.depth_scale,
      config.depth_max,
      config.bbox_slice_size,
      config.up_axis,
      device,
    )
  elif config.format == "arkit":
    arkit_dir = Path(os.path.expandvars(config.arkit_path))
    # parent_dir = arkit_dir.parents[0]
    save_dir = arkit_dir / config.output_dir
    if not Path.exists(save_dir):
      Path.mkdir(save_dir)
    (
      pcd,
      bounding_boxes,
      point_checking_bounding_boxes,
      camera_positions,
      camera_orientations,
    ) = process_transforms_arkit(
      arkit_dir,
      vbg,
      config.integrate_color,
      config.depth_scale,
      config.depth_max,
      config.bbox_slice_size,
      config.up_axis,
      config.slice_direction,
      device,
    )

  if config.visualize:
    camera_point_cloud = o3d.geometry.PointCloud()
    camera_point_cloud.points = o3d.utility.Vector3dVector(
      np.array(camera_positions)
    )
    colors = np.zeros_like(np.array(camera_positions))
    colors[:, 1] = 1
    camera_point_cloud.colors = o3d.utility.Vector3dVector(colors)
    visualize_geometries(
      [
        pcd,
        camera_point_cloud,
        point_checking_bounding_boxes[0],
        bounding_boxes[0],
      ]
    )

  num_slices = estimate_num_slices(
    camera_positions,
    config.up_axis,
    config.slice_distance,
    config.buffer_distance,
    config.slice_overlap,
  )

  for i in tqdm(range(0, num_slices)):
    progress = float(i) / num_slices

    try:
      curr_vals = get_mesh_at_frac(
        pcd,
        bounding_boxes,
        point_checking_bounding_boxes,
        camera_positions,
        camera_orientations,
        progress,
        config.buffer_distance,
        config.slice_distance,
        config.min_poses_per_segment,
        config.voxel_size,
        config.poisson_depth,
        config.density_threshold,
        config.decimation_factor,
        config.sharpen_mesh,
        config.sharpen_iterations,
        config.sharpen_strength,
        config.to_ig_euler_xyz,
        config.up_axis,
        config.visualize,
      )

      print(
        f"Saving mesh with {len(curr_vals['vertices'])} triangles and {len(curr_vals['triangles'])} vertices"
      )
      with open(
        str(save_dir / f"slice_{str(i).zfill(4)}.pkl"),
        "wb",
      ) as f:
        pickle.dump(curr_vals, f)
    except ValueError as e:
      print(e)
      continue


if __name__ == "__main__":
  app.run(main)
