import torch
import matplotlib.pyplot as plt
import numpy as np

from legged_gym.utils.math import tolerance

# velocity tracking.
target_velocity = -0.75
test_range = [-2.0, 2.0]
lower = target_velocity - 0.2
upper = target_velocity 
margin = np.abs(target_velocity)
sigmoid = 'linear'
value_at_margin = 0.

# stand still.
# test_range = [-1.0, 1.0]
# lower = 0.0
# upper = 0.0
# margin = 1.0
# sigmoid = 'gaussian'
# value_at_margin = 0.1

num_values = 1000

values = torch.linspace(test_range[0], test_range[1], num_values)
lower = torch.full_like(values, lower)
upper = torch.full_like(values, upper)
margin = torch.full_like(values, margin)
value_at_margin = torch.full_like(values, value_at_margin)

tolerance_values = tolerance(values, lower=lower, upper=upper, margin=margin, sigmoid=sigmoid, value_at_margin=value_at_margin)

plt.plot(values, tolerance_values)
plt.show()

