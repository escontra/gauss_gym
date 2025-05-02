import numpy as np

from configs.base import get_config as base_get_config


def get_config():
  config = base_get_config()
  config.load_dir = '$HOME/ARKitScenes/raw_ARKitScenes/3dod/Training/43895956'
  # config.load_dir = '$HOME/ARKitScenes/raw_ARKitScenes/3dod/Training/43649417'
  # config.load_dir = '$HOME/ARKitScenes/raw_ARKitScenes/3dod/Training/47334672'
  config.format = "arkit"
  config.up_axis = 'z'
  config.decimation_factor = 4
  config.depth_max = 5.0
  config.slice_distance = 3.0
  config.buffer_distance = 0.1
  config.slice_overlap = 0.0
  config.to_ig_euler_xyz = (np.pi, 0., 0.)
  config.slice_direction = '+'

  return config
