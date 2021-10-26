import trimesh
from typing import List
from yourdfpy import URDF
from scipy.spatial import ConvexHull
from shapely.geometry import LineString, Polygon
from collections import defaultdict
import numpy as np


def get_mesh_for_links(urdf_path: str, link_names: List[str]) -> trimesh.Trimesh:
  robot = URDF.load(
    urdf_path,
    build_collision_scene_graph=True,
    load_collision_meshes=True,
    force_collision_mesh=False,
  )

  meshes = []
  for link_name in link_names:
    link = robot.link_map[link_name]
    tmp_scene = trimesh.Scene(base_frame=link_name)
    geometries = link.collisions
    robot._add_geometries_to_scene(
      tmp_scene,
      geometries,
      link_name,
      load_geometry=None,
      force_mesh=False,
      force_single_geometry=False,
      skip_materials=True,
    )
    meshes.append(tmp_scene.to_mesh())  # Single combined mesh
  return meshes


def mesh_sampler_grid(mesh, num_x_points, num_y_points, scan_direction='z'):
  if scan_direction == 'z':
    points_2d = mesh.vertices[:, :2]
  elif scan_direction == 'x':
    points_2d = mesh.vertices[:, 1:]
  elif scan_direction == 'y':
    points_2d = mesh.vertices[:, [0, 2]]
  else:
    raise ValueError(
      f"Scan direction must be one of 'z', 'x', or 'y', got: {scan_direction}"
    )

  hull = ConvexHull(points_2d)
  footprint_polygon = Polygon(points_2d[hull.vertices])

  min_x, min_y, max_x, max_y = footprint_polygon.bounds
  sampled_points = []

  # N uniformly spaced x-coordinates
  x_cell_width = (max_x - min_x) / (num_x_points * 2)
  x_coords = np.linspace(min_x + x_cell_width, max_x - x_cell_width, num_x_points)
  for x in x_coords:
    # Create a tall vertical line to intersect with the polygon
    vertical_line = LineString([(x, min_y - 1), (x, max_y + 1)])
    intersection = footprint_polygon.intersection(vertical_line)

    # The intersection can be empty if the line is outside the polygon
    if intersection.is_empty:
      continue

    # Get the y-bounds of the vertical slice
    slice_min_y, slice_max_y = intersection.bounds[1], intersection.bounds[3]

    # Create M uniformly spaced y-points within that slice
    y_cell_width = (slice_max_y - slice_min_y) / (num_y_points * 2)
    y_coords = np.linspace(
      slice_min_y + y_cell_width, slice_max_y - y_cell_width, num_y_points
    )

    sampled_points.extend((x, y) for y in y_coords)
  return np.array(sampled_points)


def compute_mesh_ray_points(
  mesh,
  resolution_x=10,
  resolution_y=10,
  buffer=0.1,
  scan_direction='+z',
  visualize: bool = True,
):
  """
  Enhanced visualization showing mesh, rays, and intersection points.

  Args:
      mesh: trimesh.Trimesh object
      resolution_x: number of rays in first grid direction
      resolution_y: number of rays in second grid direction
      buffer: extra distance beyond mesh bounds for ray origins
      scan_direction: direction to cast rays, one of:
          '+x', '-x', '+y', '-y', '+z', '-z'
      visualize: whether to show 3D visualization

  Returns:
      intersection_points: array of intersection points
      scene: trimesh.Scene object (if visualize=True)
  """

  scan_direction = scan_direction.lower()
  # Validate scan direction
  valid_directions = ['+x', '-x', '+y', '-y', '+z', '-z']
  if scan_direction not in valid_directions:
    raise ValueError(
      f'scan_direction must be one of {valid_directions}, got: {scan_direction}'
    )

  # Get mesh bounds
  bounds = mesh.bounds
  x_min, y_min, z_min = bounds[0]
  x_max, y_max, z_max = bounds[1]

  # Determine ray configuration based on scan direction
  if scan_direction in ['+z', '-z']:
    # Rays parallel to Z-axis, grid in XY plane
    coord1 = np.linspace(x_min, x_max, resolution_x)
    coord2 = np.linspace(y_min, y_max, resolution_y)
    C1, C2 = np.meshgrid(coord1, coord2)

    if scan_direction == '+z':
      # Rays shoot upward from below mesh
      ray_origins = np.column_stack(
        [C1.flatten(), C2.flatten(), np.full(C1.size, z_min - buffer)]
      )
      ray_directions = np.tile([0, 0, 1], (len(ray_origins), 1))
    else:  # '-z'
      # Rays shoot downward from above mesh
      ray_origins = np.column_stack(
        [C1.flatten(), C2.flatten(), np.full(C1.size, z_max + buffer)]
      )
      ray_directions = np.tile([0, 0, -1], (len(ray_origins), 1))

  elif scan_direction in ['+y', '-y']:
    # Rays parallel to Y-axis, grid in XZ plane
    coord1 = np.linspace(x_min, x_max, resolution_x)
    coord2 = np.linspace(z_min, z_max, resolution_y)
    C1, C2 = np.meshgrid(coord1, coord2)

    if scan_direction == '+y':
      # Rays shoot in +Y direction from -Y side
      ray_origins = np.column_stack(
        [C1.flatten(), np.full(C1.size, y_min - buffer), C2.flatten()]
      )
      ray_directions = np.tile([0, 1, 0], (len(ray_origins), 1))
    else:  # '-y'
      # Rays shoot in -Y direction from +Y side
      ray_origins = np.column_stack(
        [C1.flatten(), np.full(C1.size, y_max + buffer), C2.flatten()]
      )
      ray_directions = np.tile([0, -1, 0], (len(ray_origins), 1))

  else:  # '+x' or '-x'
    # Rays parallel to X-axis, grid in YZ plane
    coord1 = np.linspace(y_min, y_max, resolution_x)
    coord2 = np.linspace(z_min, z_max, resolution_y)
    C1, C2 = np.meshgrid(coord1, coord2)

    if scan_direction == '+x':
      # Rays shoot in +X direction from -X side
      ray_origins = np.column_stack(
        [np.full(C1.size, x_min - buffer), C1.flatten(), C2.flatten()]
      )
      ray_directions = np.tile([1, 0, 0], (len(ray_origins), 1))
    else:  # '-x'
      # Rays shoot in -X direction from +X side
      ray_origins = np.column_stack(
        [np.full(C1.size, x_max + buffer), C1.flatten(), C2.flatten()]
      )
      ray_directions = np.tile([-1, 0, 0], (len(ray_origins), 1))

  # Find all intersections
  locations, index_ray, _ = mesh.ray.intersects_location(
    ray_origins=ray_origins, ray_directions=ray_directions
  )

  # Group intersections by ray
  ray_intersections = defaultdict(list)
  for i, ray_idx in enumerate(index_ray):
    ray_intersections[ray_idx].append(locations[i])

  # Create scene
  if visualize:
    scene = trimesh.Scene()
    # Add mesh with semi-transparency
    mesh_copy = mesh.copy()
    mesh_copy.visual.face_colors = [100, 100, 200, 150]  # Semi-transparent blue
    scene.add_geometry(mesh_copy)

  # Add rays and intersection points
  intersection_points = []

  for ray_idx in range(len(ray_origins)):
    ray_origin = ray_origins[ray_idx]
    ray_direction = ray_directions[ray_idx]

    # Calculate ray endpoint based on scan direction
    if scan_direction in ['+z', '-z']:
      max_distance = (z_max - z_min + 2 * buffer) * 1.5
    elif scan_direction in ['+y', '-y']:
      max_distance = (y_max - y_min + 2 * buffer) * 1.5
    else:  # '+x' or '-x'
      max_distance = (x_max - x_min + 2 * buffer) * 1.5

    ray_end = ray_origin + ray_direction * max_distance

    if ray_idx in ray_intersections:
      intersections = ray_intersections[ray_idx]

      # Color ray based on number of intersections
      if visualize:
        if len(intersections) >= 2:
          color = [0, 255, 0, 200]  # Green for thickness computation
        else:
          color = [255, 255, 0, 200]  # Yellow for single intersection

      # Use first intersection as ray endpoint for visualization
      ray_end = intersections[0]
      intersection_points.extend(intersections)

      # Add small spheres at intersection points
      if visualize:
        for intersection in intersections:
          sphere = trimesh.creation.icosphere(radius=0.002, subdivisions=1)
          sphere.apply_translation(intersection)
          sphere.visual.face_colors = [255, 0, 0, 255]  # Red spheres
          scene.add_geometry(sphere)
    else:
      color = [128, 128, 128, 100]  # Gray for no intersection

    if visualize:
      # Create and add ray line
      ray_line = trimesh.load_path([ray_origin, ray_end])
      ray_line.colors = [color]
      scene.add_geometry(ray_line)

  intersection_points = np.array(intersection_points)
  if visualize:
    # Add coordinate frame
    coordinate_frame = trimesh.creation.axis(
      origin_size=0.01,
      axis_radius=0.003,
      axis_length=max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.15,
    )
    scene.add_geometry(coordinate_frame)

    # Add grid plane at ray origin level for reference
    grid_thickness = 0.001
    if scan_direction in ['+z', '-z']:
      # Grid in XY plane
      grid_plane = trimesh.creation.box(
        extents=[x_max - x_min, y_max - y_min, grid_thickness]
      )
      if scan_direction == '+z':
        grid_z = z_min - buffer
      else:
        grid_z = z_max + buffer
      grid_plane.apply_translation([(x_max + x_min) / 2, (y_max + y_min) / 2, grid_z])
    elif scan_direction in ['+y', '-y']:
      # Grid in XZ plane
      grid_plane = trimesh.creation.box(
        extents=[x_max - x_min, grid_thickness, z_max - z_min]
      )
      if scan_direction == '+y':
        grid_y = y_min - buffer
      else:
        grid_y = y_max + buffer
      grid_plane.apply_translation([(x_max + x_min) / 2, grid_y, (z_max + z_min) / 2])
    else:  # '+x' or '-x'
      # Grid in YZ plane
      grid_plane = trimesh.creation.box(
        extents=[grid_thickness, y_max - y_min, z_max - z_min]
      )
      if scan_direction == '+x':
        grid_x = x_min - buffer
      else:
        grid_x = x_max + buffer
      grid_plane.apply_translation([grid_x, (y_max + y_min) / 2, (z_max + z_min) / 2])

    grid_plane.visual.face_colors = [255, 255, 255, 50]  # Semi-transparent white
    scene.add_geometry(grid_plane)

    print('Visualization includes:')
    print(f'  - Mesh with {len(mesh.faces)} faces')
    print(f'  - {len(ray_origins)} rays in {scan_direction} direction')
    print(f'  - {len(intersection_points)} intersection points')
    print(
      '  - Ray colors: Green=thickness rays, Yellow=single intersection, Gray=no intersection'
    )
    scene.show()
    return intersection_points, scene
  else:
    return intersection_points, None
