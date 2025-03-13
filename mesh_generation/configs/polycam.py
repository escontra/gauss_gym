from configs.base import get_config as base_get_config

def get_config():
  config = base_get_config()
  config.visualize=True
  config.up_axis = 'z'
  config.decimation_factor = 10
  config.depth_max = 3.0
  # config.density_threshold = 0.03
  config.slice_distance = 2.0
  config.to_ig_euler_xyz = (0., 0., 0.)

  return config
