import yaml
import pathlib
import sys
import gc
import json
import numpy as np
import torch
from tqdm import tqdm
import viser.transforms as vtf
import open3d as o3d
import nksr
from nksr.configs import get_hparams
from pycg import vis
import argparse
from huggingface_hub import hf_hub_download

directory = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(directory.parent))
__package__ = directory.name

from .utils import s5cmd_cp  # noqa: E402


def generate_mesh(pcd, config, reconstructor, device, mesh_path):
  xyz_np = np.asarray(pcd.points)
  normals_np = np.asarray(pcd.normals)
  colors_np = np.asarray(pcd.colors)
  input_xyz = torch.from_numpy(xyz_np).float().to(device)
  input_normals = torch.from_numpy(normals_np).float().to(device)

  field = reconstructor.reconstruct(
    input_xyz,
    input_normals,
    detail_level=config['mesh_detail_level'],
    approx_kernel_grad=False,
    solver_tol=1.0e-5,
    fused_mode=True,
  )
  input_color = torch.from_numpy(colors_np).float().to(device)
  field.set_texture_field(nksr.fields.PCNNField(input_xyz, input_color))
  mesh = field.extract_dual_mesh(mise_iter=2)

  # Construct Open3D mesh.
  faces_np = mesh.f.cpu().numpy()
  vertices_np = mesh.v.cpu().numpy()
  colors_np = mesh.c.cpu().numpy()
  o3d_mesh = o3d.geometry.TriangleMesh()
  o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
  o3d_mesh.triangles = o3d.utility.Vector3iVector(faces_np)
  o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors_np)

  # Fix degenerate triangles.
  o3d_mesh.remove_duplicated_vertices()
  o3d_mesh.remove_duplicated_triangles()
  o3d_mesh.remove_degenerate_triangles()
  o3d_mesh.remove_unreferenced_vertices()

  # Keep largest component.
  triangle_clusters, _, cluster_areas = o3d_mesh.cluster_connected_triangles()
  triangle_clusters = np.asarray(triangle_clusters)
  cluster_areas = np.asarray(cluster_areas)
  largest_id = int(np.argmax(cluster_areas))
  mask = triangle_clusters != largest_id
  o3d_mesh.remove_triangles_by_mask(mask.tolist())
  o3d_mesh.remove_unreferenced_vertices()

  o3d.io.write_triangle_mesh(str(mesh_path), o3d_mesh)

  del input_xyz, input_normals, field, mesh
  torch.cuda.empty_cache()
  gc.collect()


class MeshGenerator:
  def __init__(
    self, config, mission_folder, output_folder, device=torch.device('cuda:0')
  ):
    self.config = config
    self.mission_folder = mission_folder

    self.frames_json_file = output_folder / 'transforms.json'
    assert output_folder.exists(), f'Root folder {output_folder} does not exist'
    self.output_folder = output_folder / 'slices'
    assert self.output_folder.exists(), (
      f'Output folder {self.output_folder} does not exist'
    )
    slice_dirs = list(self.output_folder.iterdir())
    self.slice_dirs = [d.name for d in slice_dirs if d.is_dir()]

    self.device = device
    nksr_config = get_hparams('ks')
    file_path = hf_hub_download(
      repo_id='escontra/ks',
      filename='ks.pth',
      # cache_dir="./cache"  # Optional: specify cache directory
    )
    nksr_config['url'] = file_path
    self.reconstructor = nksr.Reconstructor(device, nksr_config)
    self.reconstructor.chunk_tmp_device = torch.device('cpu')

  def run(self):
    for slice_idx in tqdm(self.slice_dirs):
      slice_folder = self.output_folder / slice_idx
      slice_folder.mkdir(parents=True, exist_ok=True)

      # 1. Load transforms for slice.
      # transforms_file = slice_folder / 'transforms.json'
      # assert transforms_file.exists(), (
      #   f'Transforms file {transforms_file} does not exist'
      # )
      # with transforms_file.open('r') as f:
      #   transforms_data = json.load(f)

      # 2. Load pointcloud for slice.
      pcd_file = slice_folder / 'pcd.ply'
      assert pcd_file.exists(), f'Pointcloud file {pcd_file} does not exist'
      pcd = o3d.io.read_point_cloud(str(pcd_file))
      # tqdm.write(f'Before cropping: {len(pcd.points)} points')

      # 3. Crop pointcloud along transforms.
      # cropped_pcd = crop_pointcloud(pcd, transforms_data, self.config)

      # 4. Remove outliers and small clusters.
      # cropped_pcd = remove_outliers_comprehensive(cropped_pcd,
      #                                             statistical_nb_neighbors=self.config["statistical_nb_neighbors"],
      #                                             statistical_std_ratio=self.config["statistical_std_ratio"])
      # cropped_pcd, _, _ = remove_small_clusters(cropped_pcd,
      #                                           eps=self.config["cluster_eps"],
      #                                           min_points=self.config["cluster_min_points"],
      #                                           min_cluster_size=self.config["cluster_min_size"])

      # vis = o3d.visualization.Visualizer()
      # vis.create_window()
      # vis.add_geometry(cropped_pcd)
      # vis.run()
      # tqdm.write(f'After cropping: {len(cropped_pcd.points)} points')
      tqdm.write(f'After cropping: {len(pcd.points)} points')
      if len(pcd.points) == 0:
        tqdm.write(f'No points after cropping for slice {slice_idx}')
        continue

      # 4. Generate mesh with NKSR.
      generate_mesh(
        pcd,
        self.config,
        self.reconstructor,
        self.device,
        slice_folder / 'mesh.ply',
      )
      # self.get_pointcloud_for_slice(slice_idx)
      # self.get_transforms_for_slice(slice_idx)
      del pcd  # , cropped_pcd


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

  slicer = MeshGenerator(
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
