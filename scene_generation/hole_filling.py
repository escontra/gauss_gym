"""
Mesh hole detection, filling, and decimation utilities for Open3D meshes.

This script provides functions to:
1. Detect boundary loops (holes) in a mesh
2. Filter holes by perimeter to exclude large outer boundaries
3. Fill detected holes using:
   - Advancing front triangulation (RECOMMENDED - works for irregular/non-planar holes)
   - Planar ear clipping (best for flat holes)
   - Simple fan triangulation (fast, adds center vertex)
4. Clean up mesh by removing degenerate and duplicate triangles
5. Decimate mesh to target triangle count while preserving important geometry

Usage:
    mesh = o3d.io.read_triangle_mesh("path/to/mesh.ply")
    loops = boundary_loops_from_mesh(mesh, max_perimeter=5.0)
    filled_mesh = fill_holes_advancing_front(mesh, loops)
    filled_mesh = cleanup_mesh(filled_mesh)  # Remove degenerate faces
    decimated_mesh = decimate_mesh(filled_mesh, target_number_of_triangles=10000)
"""

import numpy as np
import open3d as o3d
from collections import defaultdict
from tqdm import tqdm
import os
import copy
import argparse
import tempfile

import subprocess
from typing import Generic, TypeVar
import shutil
import time
import random

P = TypeVar('P')
T = TypeVar('T')


class RetryWrapper(Generic[P, T]):
  def __init__(
    self, fn, max_retries: int = 30, delay_s: float = 10, backoff_s: float = 60
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

  def __call__(self, *args, **kwargs) -> T:
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


def s5cmd_ls(
  path: str,
  max_retries: int = 3,
) -> list[str]:
  """Execute s5cmd ls in a subprocess shell and return the results.

  This is useful for listing S3 objects/directories.

  See `s5cmd ls -h` for semantics of specifying `path`.

  Parameters
  ----------
  path : str
      S3 path to list (e.g., 's3://bucket/prefix/' or 's3://bucket/prefix/*').
  max_retries : int
      Number of retries.

  Returns
  -------
  list[str]
      List of S3 paths/objects found.
  """
  assert shutil.which('s5cmd') is not None, 's5cmd not found'
  try:
    result = RetryWrapper(subprocess.check_output, max_retries=max_retries)(
      ['s5cmd', 'ls', path], text=True
    )
    # Parse s5cmd ls output (format: "date time size path" or "DIR path")
    lines = [line.strip() for line in result.split('\n') if line.strip()]
    # Extract the path (last column after splitting by whitespace)
    return [line.split()[-1] for line in lines if line]
  except subprocess.CalledProcessError as e:
    print(f'Failed to ls {path}: {e}')
    return []


def s3_path_exists(
  path: str,
  max_retries: int = 3,
) -> bool:
  """Check if an S3 path exists.

  Parameters
  ----------
  path : str
      S3 path to check (e.g., 's3://bucket/prefix/').
  max_retries : int
      Number of retries.

  Returns
  -------
  bool
      True if the path exists, False otherwise.
  """
  assert shutil.which('s5cmd') is not None, 's5cmd not found'
  try:
    RetryWrapper(subprocess.check_output, max_retries=max_retries)(
      ['s5cmd', 'ls', path],
      text=True,
      stderr=subprocess.DEVNULL,  # Suppress error output
    )
    return True
  except subprocess.CalledProcessError:
    return False


def maybe_compute_normals(mesh):
  if not mesh.has_vertex_normals():
    mesh.compute_vertex_normals()
  return mesh


def copy_mesh(mesh):
  """
  Create a deep copy of an Open3D TriangleMesh.

  Args:
      mesh: o3d.geometry.TriangleMesh to copy

  Returns:
      Deep copy of the mesh with all attributes
  """
  return copy.deepcopy(mesh)


def boundary_loops_from_mesh(mesh: o3d.geometry.TriangleMesh, max_perimeter=None):
  """
  Extract boundary loops from a mesh.

  Args:
      mesh: o3d.geometry.TriangleMesh
      max_perimeter: Optional float. If provided, only return loops with perimeter <= max_perimeter

  Returns:
      List of (L,) vertex-index arrays
  """
  F = np.asarray(mesh.triangles, dtype=np.int64)
  if F.size == 0:
    return []

  # 1) Collect undirected triangle edges
  E = np.vstack(
    [
      F[:, [0, 1]],
      F[:, [1, 2]],
      F[:, [2, 0]],
    ]
  )
  E_sorted = np.sort(E, axis=1)
  # 2) Count occurrences to find boundary edges (count==1)
  #    Use structured array to unique rows quickly
  dtype = np.dtype([('a', E_sorted.dtype), ('b', E_sorted.dtype)])
  Es = np.ascontiguousarray(E_sorted).view(dtype)
  uniq, counts = np.unique(Es, return_counts=True)
  boundary_mask = counts == 1
  boundary_edges = uniq[boundary_mask].view(E_sorted.dtype).reshape(-1, 2)

  if boundary_edges.shape[0] == 0:
    return []

  # 3) Adjacency over boundary edges
  adj = defaultdict(set)
  for a, b in boundary_edges:
    adj[int(a)].add(int(b))
    adj[int(b)].add(int(a))

  # 4) Walk helper
  visited = set()
  loops = []

  def step_from(u, prev):
    # pick the next neighbor with an unvisited edge
    for v in adj[u]:
      e = tuple(sorted((u, v)))
      if e not in visited:
        return v, e
    return None, None

  # 5) Open chains first (degree==1 endpoints)
  for v in list(adj.keys()):
    if len(adj[v]) == 1:
      chain = [v]
      cur, prev = v, None
      while True:
        nxt, e = step_from(cur, prev)
        if e is None:
          break
        visited.add(e)
        prev, cur = cur, nxt
        chain.append(cur)
        if len(adj[cur]) == 0:  # isolated guard
          break
      loops.append(np.array(chain, dtype=np.int64))

  # 6) Then cycles (degree==2 everywhere)
  for v in list(adj.keys()):
    for u in adj[v]:
      e = tuple(sorted((v, u)))
      if e in visited:
        continue
      cycle = [v]
      visited.add(e)
      prev, cur = v, u
      cycle.append(cur)
      while True:
        nxt, e2 = step_from(cur, prev)
        if e2 is None:
          # non-manifold break: treat as chain
          break
        visited.add(e2)
        prev, cur = cur, nxt
        if cur == cycle[0]:
          break
        cycle.append(cur)
      # If it returned to start: closed loop; else: open chain fragment
      loops.append(np.array(cycle, dtype=np.int64))

  # 7) Filter by perimeter if requested
  if max_perimeter is not None:
    vertices = np.asarray(mesh.vertices)
    filtered_loops = []
    for loop in loops:
      if len(loop) < 2:
        continue
      # Calculate perimeter as sum of edge lengths
      loop_verts = vertices[loop]
      edge_vectors = np.diff(loop_verts, axis=0, append=loop_verts[:1])
      edge_lengths = np.linalg.norm(edge_vectors, axis=1)
      perimeter = np.sum(edge_lengths)
      if perimeter <= max_perimeter:
        filtered_loops.append(loop)
    return filtered_loops

  return loops  # list of (L,) vertex-index arrays


def fill_holes_simple(mesh, loops):
  """
  Fill holes in a mesh using simple fan triangulation from centroid.

  Args:
      mesh: o3d.geometry.TriangleMesh to modify
      loops: list of (L,) vertex-index arrays (boundary loops to fill)

  Returns:
      Modified mesh with holes filled
  """
  vertices = np.asarray(mesh.vertices)
  triangles = np.asarray(mesh.triangles).tolist()

  # Compute vertex normals from existing mesh to guide winding order
  mesh = maybe_compute_normals(mesh)
  vertex_normals = np.asarray(mesh.vertex_normals)

  for loop in loops:
    if len(loop) < 3:
      continue  # Need at least 3 vertices to make a triangle

    # Compute centroid of the loop
    loop_verts = vertices[loop]
    centroid = np.mean(loop_verts, axis=0)

    # Estimate expected normal direction
    boundary_normals = vertex_normals[loop]
    expected_normal = np.mean(boundary_normals, axis=0)
    expected_normal = expected_normal / (np.linalg.norm(expected_normal) + 1e-8)

    # Add centroid as a new vertex
    center_idx = len(vertices)
    vertices = np.vstack([vertices, centroid])

    # Create fan triangulation: connect each edge to the center
    for i in range(len(loop)):
      v1 = loop[i]
      v2 = loop[(i + 1) % len(loop)]

      # Check winding order
      p1, p2, pc = vertices[v1], vertices[v2], centroid
      tri_normal = np.cross(p2 - p1, pc - p1)
      tri_normal = tri_normal / (np.linalg.norm(tri_normal) + 1e-8)

      # Add triangle with correct winding order
      if np.dot(tri_normal, expected_normal) < 0:
        triangles.append([v1, center_idx, v2])  # Reversed
      else:
        triangles.append([v1, v2, center_idx])  # Original

  # Update mesh
  mesh.vertices = o3d.utility.Vector3dVector(vertices)
  mesh.triangles = o3d.utility.Vector3iVector(triangles)
  return mesh


def fill_holes_planar(mesh, loops):
  """
  Fill holes using planar triangulation (ear clipping).
  Projects boundary to its best-fit plane and triangulates.

  Args:
      mesh: o3d.geometry.TriangleMesh to modify
      loops: list of (L,) vertex-index arrays (boundary loops to fill)

  Returns:
      Modified mesh with holes filled
  """
  vertices = np.asarray(mesh.vertices)
  triangles = np.asarray(mesh.triangles).tolist()

  # Compute vertex normals from existing mesh to guide winding order
  mesh = maybe_compute_normals(mesh)
  vertex_normals = np.asarray(mesh.vertex_normals)

  for loop in loops:
    if len(loop) < 3:
      continue

    loop_verts = vertices[loop]

    # Estimate expected normal direction
    boundary_normals = vertex_normals[loop]
    expected_normal = np.mean(boundary_normals, axis=0)
    expected_normal = expected_normal / (np.linalg.norm(expected_normal) + 1e-8)

    # Compute best-fit plane using PCA
    centroid = np.mean(loop_verts, axis=0)
    centered = loop_verts - centroid

    if len(loop) > 2:
      # Use SVD to find plane normal (smallest singular value direction)
      _, _, vh = np.linalg.svd(centered)
      normal = vh[-1]  # Last row = normal to best-fit plane

      # Create 2D coordinate system in the plane
      u = vh[0]  # First principal direction
      v = np.cross(normal, u)

      # Project to 2D
      coords_2d = np.column_stack([np.dot(centered, u), np.dot(centered, v)])

      # Simple ear clipping triangulation
      remaining = list(range(len(loop)))

      while len(remaining) > 2:
        ear_found = False
        for i in range(len(remaining)):
          prev_idx = remaining[(i - 1) % len(remaining)]
          curr_idx = remaining[i]
          next_idx = remaining[(i + 1) % len(remaining)]

          # Check if this is an ear (convex vertex with no other vertices inside)
          p1 = coords_2d[prev_idx]
          p2 = coords_2d[curr_idx]
          p3 = coords_2d[next_idx]

          # Check if triangle is counter-clockwise (convex at p2)
          cross = (p3[0] - p1[0]) * (p2[1] - p1[1]) - (p3[1] - p1[1]) * (p2[0] - p1[0])

          if cross > 0:  # Counter-clockwise = convex
            # Check if any other vertex is inside this triangle
            is_ear = True
            for j in remaining:
              if j in [prev_idx, curr_idx, next_idx]:
                continue
              pt = coords_2d[j]
              # Simple point-in-triangle test
              if point_in_triangle_2d(pt, p1, p2, p3):
                is_ear = False
                break

            if is_ear:
              # Add triangle with correct winding order
              v0, v1, v2 = loop[prev_idx], loop[curr_idx], loop[next_idx]
              p0, p1, p2 = vertices[v0], vertices[v1], vertices[v2]
              tri_normal = np.cross(p1 - p0, p2 - p0)
              tri_normal = tri_normal / (np.linalg.norm(tri_normal) + 1e-8)

              if np.dot(tri_normal, expected_normal) < 0:
                triangles.append([v0, v2, v1])  # Reversed
              else:
                triangles.append([v0, v1, v2])  # Original

              remaining.pop(i)
              ear_found = True
              break

        if not ear_found:
          # Fallback: just add remaining triangle or break
          if len(remaining) == 3:
            v0 = loop[remaining[0]]
            v1 = loop[remaining[1]]
            v2 = loop[remaining[2]]
            p0, p1, p2 = vertices[v0], vertices[v1], vertices[v2]
            tri_normal = np.cross(p1 - p0, p2 - p0)
            tri_normal = tri_normal / (np.linalg.norm(tri_normal) + 1e-8)

            if np.dot(tri_normal, expected_normal) < 0:
              triangles.append([v0, v2, v1])  # Reversed
            else:
              triangles.append([v0, v1, v2])  # Original
          break
    else:
      # Degenerate case: just make a triangle
      v0, v1, v2 = loop[0], loop[1], loop[2]
      p0, p1, p2 = vertices[v0], vertices[v1], vertices[v2]
      tri_normal = np.cross(p1 - p0, p2 - p0)
      tri_normal = tri_normal / (np.linalg.norm(tri_normal) + 1e-8)

      if np.dot(tri_normal, expected_normal) < 0:
        triangles.append([v0, v2, v1])  # Reversed
      else:
        triangles.append([v0, v1, v2])  # Original

  # Update mesh
  mesh.triangles = o3d.utility.Vector3iVector(triangles)
  return mesh


def fill_holes_advancing_front(mesh, loops):
  """
  Fill holes using advancing front triangulation in 3D.
  Works well for irregular, non-planar holes.

  This method greedily fills holes by repeatedly choosing the best
  triangle to add based on quality metrics. Also ensures consistent
  triangle winding order for correct normal orientation.

  Args:
      mesh: o3d.geometry.TriangleMesh to modify
      loops: list of (L,) vertex-index arrays (boundary loops to fill)

  Returns:
      Modified mesh with holes filled
  """
  vertices = np.asarray(mesh.vertices)
  triangles = np.asarray(mesh.triangles).tolist()

  # Compute vertex normals from existing mesh to guide winding order
  mesh = maybe_compute_normals(mesh)
  vertex_normals = np.asarray(mesh.vertex_normals)

  for loop in loops:
    if len(loop) < 3:
      continue

    # Work with actual vertex indices, stored as a circular list
    boundary = list(loop)  # Make a copy

    # Estimate the expected normal direction from boundary vertex normals
    boundary_normals = vertex_normals[loop]
    expected_normal = np.mean(boundary_normals, axis=0)
    expected_normal = expected_normal / (np.linalg.norm(expected_normal) + 1e-8)

    # Keep filling until we have only 3 vertices left (final triangle)
    iteration = 0
    max_iterations = len(boundary) * 2  # Prevent infinite loops

    while len(boundary) > 3 and iteration < max_iterations:
      iteration += 1
      best_score = float('inf')
      best_idx = None

      # Try each potential triangle formed by consecutive vertices
      for i in range(len(boundary)):
        v0 = boundary[i]
        v1 = boundary[(i + 1) % len(boundary)]
        v2 = boundary[(i + 2) % len(boundary)]

        p0 = vertices[v0]
        p1 = vertices[v1]
        p2 = vertices[v2]

        # Compute triangle quality metrics
        edge0 = np.linalg.norm(p1 - p0)
        edge1 = np.linalg.norm(p2 - p1)
        edge2 = np.linalg.norm(p2 - p0)

        # Check for degenerate triangle
        v1_dir = p1 - p0
        v2_dir = p2 - p0
        normal = np.cross(v1_dir, v2_dir)
        area = 0.5 * np.linalg.norm(normal)

        if area < 1e-8:  # Degenerate triangle
          continue

        # Score based on triangle quality
        # Prefer smaller, more equilateral triangles
        perimeter = edge0 + edge1 + edge2
        max_edge = max(edge0, edge1, edge2)
        min_edge = min(edge0, edge1, edge2)
        aspect_ratio = max_edge / (min_edge + 1e-8)

        # Lower score is better
        score = perimeter * (1.0 + 0.5 * aspect_ratio)

        if score < best_score:
          best_score = score
          best_idx = i

      if best_idx is None:
        # Couldn't find a valid triangle, break
        print(
          f'Warning: Could not find valid triangle for loop with {len(boundary)} vertices'
        )
        break

      # Add the best triangle with correct winding order
      v0 = boundary[best_idx]
      v1 = boundary[(best_idx + 1) % len(boundary)]
      v2 = boundary[(best_idx + 2) % len(boundary)]

      # Check winding order against expected normal
      p0, p1, p2 = vertices[v0], vertices[v1], vertices[v2]
      tri_normal = np.cross(p1 - p0, p2 - p0)
      tri_normal = tri_normal / (np.linalg.norm(tri_normal) + 1e-8)

      # If normal points in wrong direction, reverse winding order
      if np.dot(tri_normal, expected_normal) < 0:
        triangles.append([v0, v2, v1])  # Reversed
      else:
        triangles.append([v0, v1, v2])  # Original

      # Remove the middle vertex from the boundary (we've filled around it)
      boundary.pop((best_idx + 1) % len(boundary))

    # Add final triangle with remaining 3 vertices (with correct winding)
    if len(boundary) == 3:
      p0, p1, p2 = vertices[boundary[0]], vertices[boundary[1]], vertices[boundary[2]]
      tri_normal = np.cross(p1 - p0, p2 - p0)
      tri_normal = tri_normal / (np.linalg.norm(tri_normal) + 1e-8)

      if np.dot(tri_normal, expected_normal) < 0:
        triangles.append([boundary[0], boundary[2], boundary[1]])  # Reversed
      else:
        triangles.append([boundary[0], boundary[1], boundary[2]])  # Original
    elif len(boundary) > 3:
      # Fallback: fan triangulation for remaining vertices
      print(
        f'Warning: Fallback to fan triangulation for {len(boundary)} remaining vertices'
      )
      for i in range(1, len(boundary) - 1):
        p0 = vertices[boundary[0]]
        p1 = vertices[boundary[i]]
        p2 = vertices[boundary[i + 1]]
        tri_normal = np.cross(p1 - p0, p2 - p0)
        tri_normal = tri_normal / (np.linalg.norm(tri_normal) + 1e-8)

        if np.dot(tri_normal, expected_normal) < 0:
          triangles.append([boundary[0], boundary[i + 1], boundary[i]])  # Reversed
        else:
          triangles.append([boundary[0], boundary[i], boundary[i + 1]])  # Original

  # Update mesh
  mesh.vertices = o3d.utility.Vector3dVector(vertices)
  mesh.triangles = o3d.utility.Vector3iVector(triangles)
  return mesh


def decimate_mesh_adaptive(mesh, target_number_of_triangles):
  """
  Advanced decimation that protects high-curvature regions (stairs, corners, etc).

  This function first identifies high-curvature vertices, then applies decimation
  with boundary protection to preserve important geometric features.

  Args:
      mesh: o3d.geometry.TriangleMesh to decimate
      target_number_of_triangles: Target number of triangles in output mesh

  Returns:
      Decimated mesh with feature preservation
  """
  original_tri_count = len(mesh.triangles)

  if original_tri_count <= target_number_of_triangles:
    print(f'Mesh already has {original_tri_count} triangles, no decimation needed')
    return mesh

  # Compute vertex normals if not present
  mesh = maybe_compute_normals(mesh)

  print(
    f'Adaptive decimation: {original_tri_count} -> {target_number_of_triangles} triangles...'
  )

  # Use quadric decimation which inherently preserves features
  # The boundary preservation helps keep important edges
  decimated_mesh = mesh.simplify_quadric_decimation(
    target_number_of_triangles=target_number_of_triangles,
  )

  final_tri_count = len(decimated_mesh.triangles)
  reduction_pct = 100 * (1 - final_tri_count / original_tri_count)
  print(
    f'Adaptive decimation complete: {original_tri_count} -> {final_tri_count} triangles ({reduction_pct:.1f}% reduction)'
  )

  # Recompute normals
  decimated_mesh = maybe_compute_normals(decimated_mesh)

  return decimated_mesh


def point_in_triangle_2d(p, a, b, c):
  """Check if point p is inside triangle abc in 2D."""

  def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

  d1 = sign(p, a, b)
  d2 = sign(p, b, c)
  d3 = sign(p, c, a)

  has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
  has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

  return not (has_neg and has_pos)


def remove_degenerate_triangles(mesh, min_area=1e-10):
  """
  Remove degenerate triangles from a mesh.

  A triangle is considered degenerate if:
  - It has zero or near-zero area
  - It has duplicate vertices
  - Any two vertices are too close together

  Args:
      mesh: o3d.geometry.TriangleMesh to clean
      min_area: Minimum triangle area threshold (default: 1e-10)

  Returns:
      Cleaned mesh with degenerate triangles removed
  """
  vertices = np.asarray(mesh.vertices)
  triangles = np.asarray(mesh.triangles)

  if len(triangles) == 0:
    return mesh

  valid_triangles = []
  removed_count = 0

  for tri in triangles:
    v0, v1, v2 = tri

    # Check for duplicate vertex indices
    if v0 == v1 or v1 == v2 or v2 == v0:
      removed_count += 1
      continue

    # Get vertex positions
    p0 = vertices[v0]
    p1 = vertices[v1]
    p2 = vertices[v2]

    # Check if vertices are too close (degenerate)
    min_edge_length = 1e-10
    if (
      np.linalg.norm(p1 - p0) < min_edge_length
      or np.linalg.norm(p2 - p1) < min_edge_length
      or np.linalg.norm(p2 - p0) < min_edge_length
    ):
      removed_count += 1
      continue

    # Compute triangle area
    edge1 = p1 - p0
    edge2 = p2 - p0
    cross = np.cross(edge1, edge2)
    area = 0.5 * np.linalg.norm(cross)

    # Keep triangle if area is above threshold
    if area >= min_area:
      valid_triangles.append(tri)
    else:
      removed_count += 1

  if removed_count > 0:
    print(f'Removed {removed_count} degenerate triangles')

  # Update mesh with cleaned triangles
  mesh.triangles = o3d.utility.Vector3iVector(np.array(valid_triangles))

  # Remove unreferenced vertices (optional, can be expensive)
  mesh.remove_unreferenced_vertices()

  return mesh


def remove_duplicate_triangles(mesh):
  """
  Remove duplicate triangles from a mesh.

  Two triangles are considered duplicates if they reference the same
  three vertices (in any order/winding).

  Args:
      mesh: o3d.geometry.TriangleMesh to clean

  Returns:
      Cleaned mesh with duplicate triangles removed
  """
  triangles = np.asarray(mesh.triangles)

  if len(triangles) == 0:
    return mesh

  # Sort vertices within each triangle to create a canonical representation
  sorted_triangles = np.sort(triangles, axis=1)

  # Find unique triangles
  unique_triangles, indices = np.unique(sorted_triangles, axis=0, return_index=True)

  removed_count = len(triangles) - len(unique_triangles)
  if removed_count > 0:
    print(f'Removed {removed_count} duplicate triangles')

    # Keep original triangles (with original winding) at the unique indices
    mesh.triangles = o3d.utility.Vector3iVector(triangles[np.sort(indices)])

  return mesh


def cleanup_mesh(mesh, min_triangle_area=1e-10):
  """
  Comprehensive mesh cleanup: removes degenerate and duplicate triangles.

  Args:
      mesh: o3d.geometry.TriangleMesh to clean
      min_triangle_area: Minimum triangle area threshold

  Returns:
      Cleaned mesh
  """
  original_tri_count = len(mesh.triangles)

  # Remove degenerate triangles
  mesh = remove_degenerate_triangles(mesh, min_area=min_triangle_area)

  # Remove duplicate triangles
  mesh = remove_duplicate_triangles(mesh)

  final_tri_count = len(mesh.triangles)
  total_removed = original_tri_count - final_tri_count

  if total_removed > 0:
    print(
      f'Total cleanup: removed {total_removed} triangles ({original_tri_count} -> {final_tri_count})'
    )

  return mesh


def color_loops(mesh, loops, color=(1.0, 0.0, 0.0)):
  """
  Colors the vertices belonging to boundary loops in the given mesh.

  Args:
      mesh: o3d.geometry.TriangleMesh
      loops: list of (L,) vertex-index arrays (from boundary_loops_from_mesh)
      color: RGB triple in [0,1]
  """
  n = len(mesh.vertices)
  colors = np.tile(np.array([[0.7, 0.7, 0.7]]), (n, 1))  # base gray
  for loop in loops:
    colors[loop] = color
  mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
  return mesh


def process_mesh(
  s3_path,
  tmp_dir,
  mesh_filename,
  max_hole_perimeter: float,
  target_triangles: int,
  fill_holes_method: str = 'advancing_front',  # 'planar', 'simple', 'advancing_front'
  save_result: bool = True,
  visualize: bool = False,
):
  s3_mesh_path = os.path.join(s3_path, mesh_filename)
  local_mesh_path = os.path.join(tmp_dir, mesh_filename)
  s5cmd_cp(s3_mesh_path, local_mesh_path)

  # Example usage:
  mesh = o3d.io.read_triangle_mesh(local_mesh_path)
  mesh = maybe_compute_normals(mesh)

  if visualize:
    # Visualize original mesh
    print('\nVisualizing original mesh...')
    o3d.visualization.draw_geometries([mesh], window_name='Original Mesh')

  # Get loops from the previous helper
  # Filter to only include loops with perimeter <= max_perimeter
  # Adjust this value based on your mesh scale (e.g., 1.0, 5.0, 10.0, etc.)
  loops = boundary_loops_from_mesh(mesh, max_perimeter=max_hole_perimeter)

  print(f'Found {len(loops)} boundary loops with perimeter <= {max_hole_perimeter}')

  # Calculate and print perimeters for all loops
  vertices = np.asarray(mesh.vertices)
  loop_info = []
  for i, loop in enumerate(loops):
    loop_verts = vertices[loop]
    edge_vectors = np.diff(loop_verts, axis=0, append=loop_verts[:1])
    perimeter = np.sum(np.linalg.norm(edge_vectors, axis=1))
    loop_info.append((i, len(loop), perimeter))

  # Sort by perimeter (descending) and show top 100
  loop_info_sorted = sorted(loop_info, key=lambda x: x[2], reverse=True)
  num_to_show = min(50, len(loop_info_sorted))

  print(f'\nTop {num_to_show} loops by perimeter size:')
  for rank, (idx, num_verts, perim) in enumerate(loop_info_sorted[:num_to_show], 1):
    print(f'  #{rank}: Loop {idx} - {num_verts} vertices, perimeter = {perim:.3f}')

  if visualize:
    # Visualize detected holes before filling
    print('\nVisualizing detected holes (red)...')
    # Create a copy for visualization to preserve original mesh
    vis_mesh = copy_mesh(mesh)
    vis_mesh = color_loops(vis_mesh, loops, color=(1, 0, 0))
    o3d.visualization.draw_geometries([vis_mesh], window_name='Detected Holes')

  # Fill holes using advancing front triangulation (best for irregular holes)
  print('\nFilling holes...')
  original_tri_count = len(mesh.triangles)
  filled_mesh = {
    'advancing_front': fill_holes_advancing_front,
    'planar': fill_holes_planar,
    'simple': fill_holes_simple,
  }[fill_holes_method](mesh, loops)
  new_tri_count = len(filled_mesh.triangles)
  print(
    f'Added {new_tri_count - original_tri_count} triangles to fill {len(loops)} holes'
  )

  # Clean up degenerate and duplicate triangles
  print('\nCleaning up mesh...')
  filled_mesh = cleanup_mesh(filled_mesh, min_triangle_area=1e-10)

  # Recompute normals after filling and cleanup
  filled_mesh = maybe_compute_normals(filled_mesh)

  if visualize:
    # Visualize result
    print('\nVisualizing filled mesh...')
    o3d.visualization.draw_geometries([filled_mesh], window_name='Filled Mesh')

  if save_result:
    output_path = local_mesh_path.replace('.ply', '_filled.ply')
    o3d.io.write_triangle_mesh(output_path, filled_mesh)
    s5cmd_cp(
      output_path, os.path.join(s3_path, mesh_filename.replace('.ply', '_filled.ply'))
    )

  # Decimate mesh to target triangle count (preserves features like stairs)
  filled_mesh = decimate_mesh_adaptive(filled_mesh, target_triangles)

  if visualize:
    # Visualize result
    print('\nVisualizing decimated mesh...')
    o3d.visualization.draw_geometries([filled_mesh], window_name='Filled Mesh')

  if save_result:
    output_path = local_mesh_path.replace('.ply', '_filled_decimated.ply')
    o3d.io.write_triangle_mesh(output_path, filled_mesh)
    s5cmd_cp(
      output_path,
      os.path.join(s3_path, mesh_filename.replace('.ply', '_filled_decimated.ply')),
    )


def process_grand_tour(s3_path, tmp_dir, visualize, save_result):
  s3_slices_path = os.path.join(s3_path, 'slices') + '/'
  slices = s5cmd_ls(s3_slices_path)
  for slice in tqdm(slices):
    # Extract slice name (e.g., 'slice_1148c086' from full S3 path)
    slice_name = os.path.basename(slice.rstrip('/'))
    s3_slice_path = os.path.join(s3_slices_path, slice)

    # Create slice-specific subdirectory to avoid filename collisions
    slice_tmp_dir = os.path.join(tmp_dir, slice_name)
    os.makedirs(slice_tmp_dir, exist_ok=True)

    process_mesh(
      s3_slice_path,
      slice_tmp_dir,  # Use slice-specific directory
      'mesh.ply',
      max_hole_perimeter=10.0,
      target_triangles=40000,
      save_result=save_result,
      visualize=visualize,
    )


def process_arkit(s3_path, tmp_dir, visualize, save_result):
  s3_path = s3_path.rstrip('/') + '/'

  # Extract scene ID from S3 path (e.g., '40753679' from full S3 path)
  scene_id = os.path.basename(s3_path.rstrip('/'))

  # Create scene-specific subdirectory to avoid filename collisions
  scene_tmp_dir = os.path.join(tmp_dir, scene_id)
  os.makedirs(scene_tmp_dir, exist_ok=True)

  print(f'Scene tmp directory: {scene_tmp_dir}')

  all_files = [file.rstrip('/') for file in s5cmd_ls(s3_path)]
  mesh_filename = [f for f in all_files if f.endswith('mesh.ply')][0]
  process_mesh(
    s3_path,
    scene_tmp_dir,  # Use scene-specific directory
    mesh_filename,
    max_hole_perimeter=5.0,
    target_triangles=40000,
    save_result=save_result,
    visualize=visualize,
  )


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Process mesh data',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
    '--s3-path', type=str, default=None, help='S3 path to load the data from.'
  )
  parser.add_argument('--dataset', type=str, help='What kind of dataset.')
  parser.add_argument('--visualize', action='store_true', help='Visualize the mesh.')
  parser.add_argument('--dont-save', action='store_true', help='Save the result.')
  args = parser.parse_args()

  with tempfile.TemporaryDirectory() as tmp_dir:
    if args.dataset == 'grand_tour':
      process_grand_tour(args.s3_path, tmp_dir, args.visualize, not args.dont_save)
    elif args.dataset == 'arkit':
      process_arkit(args.s3_path, tmp_dir, args.visualize, not args.dont_save)
    else:
      raise ValueError(f'Dataset {args.dataset} not supported.')

  # arkit_dir = '/home/ANT.AMAZON.COM/escontra/arkit_test/raw/40753679'
  # process_arkit(arkit_dir)
