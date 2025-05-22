from configs.base import get_config as base_get_config

def get_config():
  config = base_get_config()
  config.up_axis = 'z'
  config.decimation_factor = 4
  config.depth_max = 5.0
  config.slice_distance = 5.0
  config.slice_overlap = 1.0
  config.buffer_distance = 1.0
  config.to_ig_euler_xyz = (0., 0., 0.)
  config.slice_direction = '-'
  config.load_mesh = True
  return config
