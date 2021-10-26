import numpy as np
from ml_collections import config_dict


def get_config():
  config = config_dict.ConfigDict()

  config.format = 'ns'
  config.load_dir = '$HOME/ULI_DATA/apartment_to_grace/processed_data'
  config.output_dir = 'meshes'
  config.save_pc_only = False
  config.load_mesh = False

  # Visualization parameters.
  config.visualize = False

  # Size of slices to generate.
  config.bbox_slice_size = 1.5
  config.buffer_distance = 1.0
  config.slice_distance = 4.0
  config.slice_overlap = 2.0
  config.min_poses_per_segment = 10
  config.slice_direction = None

  # Position filtering.
  config.filter_keys = ['right']
  config.num_poses_after_smoothing = 40
  config.smoothing_factor = 12

  # Mesh generation parameters.
  config.voxel_size = 0.02
  config.block_resolution = 18
  config.block_count = 100_000
  config.sharpen_mesh = False
  config.sharpen_iterations = 1
  config.sharpen_strength = 0.05
  config.decimation_factor = 4
  config.density_threshold = 0.07
  config.poisson_depth = 8
  config.integrate_color = True
  config.depth_max = 2.5
  config.depth_scale = 1.0
  config.up_axis = 'y'

  # Transformations to IsaacGym coordinate frame.
  config.to_ig_euler_xyz = (-np.pi / 2, 0.0, np.pi)

  return config
