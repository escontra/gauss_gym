from typing import Dict, List, Union

# import collections
import sys
import viser

import trimesh
import open3d as o3d

import numpy as np
import time
import json

import viser.transforms as vtf
import os


if sys.version_info < (3, 9):
  # Fixes importlib.resources for python < 3.9.
  # From: https://github.com/messense/typst-py/issues/12#issuecomment-1812956252
  import importlib.resources as importlib_res
  import importlib_resources

  setattr(importlib_res, 'files', importlib_resources.files)
  setattr(importlib_res, 'as_file', importlib_resources.as_file)


COLORS = [
  '#0022ff',
  '#33aa00',
  '#ff0011',
  '#ddaa00',
  '#cc44dd',
  '#0088aa',
  '#001177',
  '#117700',
  '#990022',
  '#885500',
  '#553366',
  '#006666',
  '#7777cc',
  '#999999',
  '#990099',
  '#888800',
  '#ff00aa',
  '#444444',
]

TO_IG_EULER_XYZ = np.array([np.pi, 0.0, 0.0])
ROTATION = (
  vtf.SO3.from_x_radians(TO_IG_EULER_XYZ[0]).inverse()
  @ vtf.SO3.from_y_radians(TO_IG_EULER_XYZ[1]).inverse()
  @ vtf.SO3.from_z_radians(TO_IG_EULER_XYZ[2]).inverse()
)

FRAME_SCALE = 0.25


def keep_largest_component(
  mesh: o3d.geometry.TriangleMesh,
) -> o3d.geometry.TriangleMesh:
  # Cluster triangles by connectivity
  triangle_clusters, cluster_n_tris, cluster_areas = mesh.cluster_connected_triangles()
  triangle_clusters = np.asarray(triangle_clusters)
  cluster_areas = np.asarray(cluster_areas)

  # Pick the cluster with the largest area (more robust than just #triangles)
  largest_id = int(np.argmax(cluster_areas))

  # Remove all triangles not in the largest cluster
  mask = triangle_clusters != largest_id
  mesh.remove_triangles_by_mask(mask.tolist())
  mesh.remove_unreferenced_vertices()

  return mesh


def clean_mesh_keep_largest(
  mesh, out_path=None, fill_small_holes=False, hole_size=5, smooth_iters=0
):
  mesh.remove_duplicated_vertices()
  mesh.remove_duplicated_triangles()
  mesh.remove_degenerate_triangles()
  mesh.remove_unreferenced_vertices()

  mesh = keep_largest_component(mesh)

  if fill_small_holes:
    mesh = mesh.fill_holes(hole_size)

  if smooth_iters > 0:
    mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iters)

  mesh.compute_vertex_normals()
  if out_path:
    o3d.io.write_triangle_mesh(out_path, mesh)
  return mesh


class DataVisualizer:
  """A robot visualizer using Viser, with the URDF attached under a /world root node."""

  global_servers: Dict[int, viser.ViserServer] = {}

  def __init__(
    self,
    data_path: str,
    port: int = 8080,
  ):
    """
    Initialize visualizer with a URDF model, loaded under a single /world node.

    Args:
        env: Environment instance
        urdf_path: Path to the URDF file
        port: Port number for the viser server
        dt: Desired update frequency in Hz
        force_dt: If True, force the update frequency to be dt Hz
    """
    # If there is an existing server on this port, shut it down
    if port in DataVisualizer.global_servers:
      print(f'Found existing server on port {port}, shutting it down.')
      DataVisualizer.global_servers.pop(port).stop()

    self.data_path = data_path

    self.server = viser.ViserServer(port=port)
    DataVisualizer.global_servers[port] = self.server

    # Also store mesh handles in case you want direct references
    self._gs_handle = None
    self._axes_handle = None
    self._mesh_handle = None
    self.setup_scene_selection()

  def setup_scene_selection(self):
    scene_names = os.listdir(self.data_path)
    init_scene = scene_names[0]

    slice_path = os.path.join(self.data_path, init_scene)
    init_meshes = os.listdir(slice_path)
    init_mesh = init_meshes[0]

    with self.server.gui.add_folder('Scene Selection'):
      self.scene_selection = self.server.gui.add_dropdown(
        'Select Scene',
        options=scene_names,
        initial_value=init_scene,
        hint='Select which scene to visualize',
      )

      self.slice_selection = self.server.gui.add_dropdown(
        'Select Slice',
        options=init_meshes,
        initial_value=init_mesh,
        hint='Select which environment to visualize',
      )

      @self.scene_selection.on_update
      def _(event) -> None:
        slice_path = os.path.join(self.data_path, self.scene_selection.value)
        init_meshes = os.listdir(slice_path)
        init_mesh = init_meshes[0]
        self.slice_selection.options = init_meshes
        self.slice_selection.initial_value = init_mesh

      @self.slice_selection.on_update
      def _(event) -> None:
        mean_vertex = self.add_mesh(
          self.scene_selection.value, self.slice_selection.value
        )
        self.add_camera_axes(
          self.scene_selection.value, self.slice_selection.value, mean_vertex
        )

    with self.server.gui.add_folder('Mesh processing'):
      self.fill_holes_button = self.server.gui.add_button(
        'Fill holes', hint='Fill holes in the mesh'
      )
      self.fix_mesh_button = self.server.gui.add_button('Fix mesh', hint='Fix the mesh')
      self.keep_largest_component_button = self.server.gui.add_button(
        'Keep largest component', hint='Keep the largest component of the mesh'
      )
      self.filter_smooth_taubin_button = self.server.gui.add_button(
        'Filter smooth taubin', hint='Filter the mesh with smooth taubin'
      )

      @self.fill_holes_button.on_click
      def _(event) -> None:
        self.fill_holes()

      @self.fix_mesh_button.on_click
      def _(event) -> None:
        self.fix_mesh()

      @self.keep_largest_component_button.on_click
      def _(event) -> None:
        self.keep_largest_component()

      @self.filter_smooth_taubin_button.on_click
      def _(event) -> None:
        self.filter_smooth_taubin()

  def add_mesh(self, scene_name: str, slice_name: str):
    slice_path = os.path.join(self.data_path, scene_name, slice_name)
    mesh_path = os.path.join(slice_path, 'mesh.ply')
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
      mesh = trimesh.util.concatenate(
        [m for m in mesh.geometry.values() if isinstance(m, trimesh.Trimesh)]
      )
    mean_vertex = np.mean(np.array(mesh.vertices), axis=0, keepdims=True)
    mesh.vertices -= mean_vertex
    mesh.vertices = np.dot(mesh.vertices, ROTATION.as_matrix().T)
    if self._mesh_handle is None:
      self._mesh_handle = self.server.scene.add_mesh_simple(
        name='/mesh',
        vertices=mesh.vertices,
        faces=mesh.faces,
        opacity=1.0,
        color=(0.282, 0.247, 0.361),
        side='double',
        visible=True,
      )
    else:
      self._mesh_handle.vertices = mesh.vertices
      self._mesh_handle.faces = mesh.faces
    return mean_vertex

  def fill_holes(self):
    vertices = self._mesh_handle.vertices
    faces = self._mesh_handle.faces

    # Convert to Open3D tensor-based mesh
    o3d_mesh = o3d.t.geometry.TriangleMesh()
    o3d_mesh.vertex.positions = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)
    o3d_mesh.triangle.indices = o3d.core.Tensor(faces, dtype=o3d.core.Dtype.Int32)

    # Fill holes using Open3D tensor API
    o3d_mesh_filled = o3d_mesh.fill_holes()

    # Convert back to numpy arrays
    filled_vertices = o3d_mesh_filled.vertex.positions.numpy()
    filled_faces = o3d_mesh_filled.triangle.indices.numpy()

    # Update the mesh handle
    self._mesh_handle.vertices = filled_vertices
    self._mesh_handle.faces = filled_faces

    # Convert to legacy mesh to check if watertight
    legacy_mesh = o3d_mesh_filled.to_legacy()
    is_watertight = legacy_mesh.is_watertight()
    print(f'Done filling holes. Is watertight: {is_watertight}')

  def fix_mesh(self):
    vertices = self._mesh_handle.vertices
    faces = self._mesh_handle.faces

    # Convert to Open3D legacy mesh (has the cleaning methods)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Apply cleaning operations
    o3d_mesh.remove_duplicated_vertices()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_unreferenced_vertices()

    # Convert back to numpy arrays
    filled_vertices = np.asarray(o3d_mesh.vertices)
    filled_faces = np.asarray(o3d_mesh.triangles)

    # Update the mesh handle
    self._mesh_handle.vertices = filled_vertices
    self._mesh_handle.faces = filled_faces

    # Check if watertight
    is_watertight = o3d_mesh.is_watertight()
    print(f'Done fixing mesh. Is watertight: {is_watertight}')

  def keep_largest_component(self):
    vertices = self._mesh_handle.vertices
    faces = self._mesh_handle.faces

    # Convert to Open3D legacy mesh (has the cleaning methods)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    triangle_clusters, cluster_n_tris, cluster_areas = (
      o3d_mesh.cluster_connected_triangles()
    )
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_areas = np.asarray(cluster_areas)

    # Pick the cluster with the largest area (more robust than just #triangles)
    largest_id = int(np.argmax(cluster_areas))

    # Remove all triangles not in the largest cluster
    mask = triangle_clusters != largest_id
    o3d_mesh.remove_triangles_by_mask(mask.tolist())
    o3d_mesh.remove_unreferenced_vertices()

    # Convert back to numpy arrays
    filled_vertices = np.asarray(o3d_mesh.vertices)
    filled_faces = np.asarray(o3d_mesh.triangles)

    # Update the mesh handle
    self._mesh_handle.vertices = filled_vertices
    self._mesh_handle.faces = filled_faces

    # Check if watertight
    is_watertight = o3d_mesh.is_watertight()
    print(f'Done keeping largest component. Is watertight: {is_watertight}')

  def filter_smooth_taubin(self):
    vertices = self._mesh_handle.vertices
    faces = self._mesh_handle.faces

    # Convert to Open3D legacy mesh (has the cleaning methods)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    o3d_mesh.filter_smooth_taubin(number_of_iterations=10)

    # Convert back to numpy arrays
    filled_vertices = np.asarray(o3d_mesh.vertices)
    filled_faces = np.asarray(o3d_mesh.triangles)

    # Update the mesh handle
    self._mesh_handle.vertices = filled_vertices
    self._mesh_handle.faces = filled_faces

    # Check if watertight
    is_watertight = o3d_mesh.is_watertight()
    print(f'Done filtering smooth taubin. Is watertight: {is_watertight}')

  def add_camera_axes(self, scene_name: str, slice_name: str, mean_vertex: np.ndarray):
    slice_path = os.path.join(self.data_path, scene_name, slice_name)
    transforms_path = os.path.join(slice_path, 'transforms.json')
    with open(transforms_path, 'r') as f:
      transforms = json.load(f)
    curr_cam_trans = []
    for frame in transforms['frames']:
      curr_cam_trans.append(np.array(frame['transform_matrix'])[:3, 3])
    curr_cam_trans = np.array(curr_cam_trans)
    curr_cam_trans -= mean_vertex
    curr_cam_trans = np.dot(curr_cam_trans, ROTATION.as_matrix().T)
    cam_quat = np.array([1.0, 0.0, 0.0, 0.0])[None].repeat(
      curr_cam_trans.shape[0], axis=0
    )
    if self._axes_handle is None:
      self._axes_handle = self.server.scene.add_batched_axes(
        '/cam_axes',
        batched_wxyzs=cam_quat,
        batched_positions=curr_cam_trans,
        batched_scales=np.ones((curr_cam_trans.shape[0], 3)) * FRAME_SCALE,
        visible=True,
      )
    else:
      self._axes_handle.batched_wxyzs = cam_quat
      self._axes_handle.batched_positions = curr_cam_trans

  def run(self):
    while True:
      time.sleep(1)


if __name__ == '__main__':
  visualizer = DataVisualizer(
    data_path='/home/ANT.AMAZON.COM/escontra/grand_tour_dataset'
  )
  visualizer.run()
