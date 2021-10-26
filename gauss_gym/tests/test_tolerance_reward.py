import torch
import matplotlib.pyplot as plt
import numpy as np

from gauss_gym.utils.math import linear_or_gaussian_tolerance

# velocity tracking.
target_velocity = 0.5
test_range = [-1.0, 1.0]
lower = target_velocity - 0.2
upper = target_velocity
margin = np.abs(target_velocity)
sigmoid = 'linear'
value_at_margin = 0.0

num_values = 1000
values = torch.linspace(test_range[0], test_range[1], num_values)
value_at_margin = torch.full_like(values, value_at_margin)
target = torch.full_like(values, target_velocity)
is_gaussian = torch.full_like(values, True).bool()

tolerance_values = linear_or_gaussian_tolerance(
  values,
  target,
  0.3,
  1.0,
  is_gaussian,
)
plt.plot(values, tolerance_values)
plt.show()
