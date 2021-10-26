import numpy as np
from configs.base import get_config as base_get_config


def get_config():
  config = base_get_config()
  config.format = 'grandslam'
  config.up_axis = 'z'
  config.slice_direction = '+'
  config.decimation_factor = 2
  config.depth_max = 3.0
  config.slice_distance = 4.0
  config.slice_overlap = 3.0
  config.filter_keys = []
  config.buffer_distance = 0.75
  config.to_ig_euler_xyz = (np.pi, 0.0, 0.0)
  config.load_mesh = False
  return config
