import torch
from gauss_gym.utils import math_utils
import matplotlib.pyplot as plt

phase = torch.linspace(-2.0 * torch.pi, 2 * torch.pi, 100)

rz = math_utils.get_rz(phase, 0.08)

plt.plot(phase.cpu().numpy(), rz.cpu().numpy())
plt.show()
