import numpy as np
from configs.base import get_config as base_get_config

def get_config():
  config = base_get_config()
  config.format = 'ns'
  config.up_axis = 'y'
  config.slice_direction = '+'
  config.decimation_factor = 4
  config.depth_max = 2.5
  config.slice_distance = 4.0
  config.slice_overlap = 3.0
  config.buffer_distance = 1.0
  config.to_ig_euler_xyz = (-np.pi / 2, 0., np.pi)

  config.filter_keys = ['right']
  return config
