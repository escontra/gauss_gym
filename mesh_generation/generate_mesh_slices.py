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

from ml_collections import config_flags
from configs.base import get_config

_CONFIG = config_flags.DEFINE_config_dict("config", get_config())


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


def process_transforms(
  json_path: Path,
  transforms: dict,
  vbg: o3d.t.geometry.VoxelBlockGrid,
  integrate_color: bool,
  depth_scale: float,
  depth_max: float,
  bbox_slice_size: float,
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
    min_bound = pose.translation() + np.array(
      [-bbox_slice_size / 2, -3, -bbox_slice_size / 2]
    )
    max_bound = pose.translation() + np.array(
      [bbox_slice_size / 2, 3, bbox_slice_size / 2]
    )
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    bounding_box.color = (1.0, 0.0, 0.0)
    bounding_boxes.append(bounding_box)

    min_bound = pose.translation() + np.array([-0.25, -3, -0.25])
    max_bound = pose.translation() + np.array([0.25, 3, 0.25])
    p_bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    p_bounding_box.color = (0.0, 0.0, 1.0)
    point_checking_bounding_boxes.append(p_bounding_box)

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
  while np.linalg.norm(positions[idx] - positions[new_idx]) < distance:
    if new_idx == 0 or new_idx == len(positions) - 1:
      break
    new_idx += increment
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
  sharpen_strength,
  to_ig_euler_xyz,
  up_axis,
  visualize,
):
  num_cameras_after_buffer = 1 + march_idx_along_distance(
    camera_positions, len(camera_positions) - 1, buffer_distance, -1, up_axis
  )

  # Indices along which to slice pointcloud.
  start_idx = bbox_start_idx = end_idx = min(
    int(num_cameras_after_buffer * frac), num_cameras_after_buffer - 1
  )
  bbox_start_idx = march_idx_along_distance(
    camera_positions, start_idx, buffer_distance, -1, up_axis
  )
  end_idx = march_idx_along_distance(
    camera_positions, start_idx, slice_distance, 1, up_axis
  )
  bbox_end_idx = march_idx_along_distance(
    camera_positions, end_idx, buffer_distance, 1, up_axis
  )

  if (end_idx - start_idx) < min_poses_per_segment:
    raise ValueError(
      f"Segment {frac} only has {1 + (end_idx - start_idx)} poses."
    )

  # Slice pointcloud.
  curr_pcd = o3d.geometry.PointCloud()
  for box_idx in range(bbox_start_idx, bbox_end_idx):
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
  mesh = mesh.simplify_quadric_decimation(
    len(mesh.triangles) // decimation_factor
  )
  if sharpen_mesh:
    mesh = mesh.filter_sharpen(
      number_of_iterations=1, strength=sharpen_strength
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
  num_slices = 1 + int(np.ceil(total_distance / total_slice_distance))
  return num_slices


def main(_):
  config = _CONFIG.value

  json_path = Path(config.json_path)
  parent_dir = json_path.parents[0]
  if not Path.exists(parent_dir / config.output_dir):
    Path.mkdir(parent_dir / config.output_dir)
  transforms = load_transforms_ns(json_path)

  device = o3d.core.Device("CUDA:0")
  vbg = get_voxel_block_grid(device, config.integrate_color, config.voxel_size)

  (
    pcd,
    bounding_boxes,
    point_checking_bounding_boxes,
    camera_positions,
    camera_orientations,
  ) = process_transforms(
    json_path,
    transforms,
    vbg,
    config.integrate_color,
    config.depth_scale,
    config.depth_max,
    config.bbox_slice_size,
    device,
  )

  if config.visualize:
    # Render camera poses, pointcloud, and bounding boxes.
    camera_point_cloud = o3d.geometry.PointCloud()
    camera_point_cloud.points = o3d.utility.Vector3dVector(
      np.array(camera_positions)
    )
    colors = np.zeros_like(np.array(camera_positions))  # All black for example
    colors[:, 1] = 1  # Set green color
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
        config.sharpen_strength,
        config.to_ig_euler_xyz,
        config.up_axis,
        config.visualize,
      )

      print(
        f"Saving mesh with {len(curr_vals['vertices'])} triangles and {len(curr_vals['triangles'])} vertices"
      )
      with open(
        str(parent_dir / config.output_dir / f"slice_{str(i).zfill(4)}.pkl"),
        "wb",
      ) as f:
        pickle.dump(curr_vals, f)
    except ValueError as e:
      print(e)
      continue


if __name__ == "__main__":
  app.run(main)
