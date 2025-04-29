import numpy as np

from configs.base import get_config as base_get_config


def get_config():
  config = base_get_config()
  # config.arkit_path = '$HOME/ARKitScenes/raw_ARKitScenes/3dod/Training/43895956'
  # config.arkit_path = '$HOME/ARKitScenes/raw_ARKitScenes/3dod/Training/43649417'
  config.arkit_path = '$HOME/ARKitScenes/raw_ARKitScenes/3dod/Training/47334672'
  config.format = "arkit"
  config.visualize=True
  config.up_axis = 'z'
  config.decimation_factor = 1
  config.depth_max = 3.0
  # config.density_threshold = 0.03
  config.slice_distance = 20.0
  config.buffer_distance = 1.0
  config.slice_overlap = 0.0
  config.to_ig_euler_xyz = (0., 0., 0.)
  config.integrate_color = True
  config.poisson_depth = 6
  config.sharpen_mesh = False
  config.to_ig_euler_xyz = (np.pi, 0., 0.)
  config.slice_direction = '+'

  return config
