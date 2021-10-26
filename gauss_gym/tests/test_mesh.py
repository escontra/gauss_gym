import numpy as np

import gauss_gym
from gauss_gym.utils import mesh_utils

# A1:
# URDF_PATH = gauss_gym.GAUSS_GYM_ROOT_DIR + "/resources/robots/a1/urdf/a1.urdf"
# link_names = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
# T1:
URDF_PATH = (
  gauss_gym.GAUSS_GYM_ROOT_DIR + '/resources/robots/t1/urdf/T1_locomotion.urdf'
)
link_names = ['right_foot_link', 'left_foot_link']

meshes = mesh_utils.get_mesh_for_links(URDF_PATH, link_names)
curr_sampled_points = None

for mesh in meshes:
  intersection_points, _ = mesh_utils.compute_mesh_ray_points(
    mesh, resolution_x=10, resolution_y=10, scan_direction='+z', visualize=True
  )
  sampled_points = mesh_utils.mesh_sampler_grid(mesh, 10, 10, scan_direction='z')
  sampled_points = np.pad(
    sampled_points, ((0, 0), (0, 1)), mode='constant', constant_values=0.0
  )
  if curr_sampled_points is None:
    curr_sampled_points = sampled_points
  else:
    assert curr_sampled_points.shape == sampled_points.shape
    assert np.allclose(curr_sampled_points, sampled_points)
    print('passed!')

  # import matplotlib.pyplot as plt
  # plt.scatter(sampled_points[:, 0], sampled_points[:, 1])
  # plt.axis('equal')
  # plt.show()
