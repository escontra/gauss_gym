import torch
from gauss_gym.rl import utils

trajectory = torch.randn(6, 2, 1)
dones = torch.zeros(6, 2)
# dones[:, 0] = 1
dones[3, :] = 1
# dones[4, 0] = 1
# dones[5, 1] = 1
# dones[:, 0] = 1
# dones[3, 1] = 1
dones = dones.bool()
# dones = torch.randint(0, 2, (10, 2))

print('Original')
print(f'trajectory.shape: {trajectory.shape}')
print(trajectory.squeeze(-1))
print(f'dones.shape: {dones.shape}')
print(dones)

padded_trajectories, trajectory_masks = utils.split_and_pad_trajectories(
  trajectory, dones
)

print('Padded')
print(f'padded_trajectories.shape: {padded_trajectories.shape}')
print(padded_trajectories.squeeze(-1))
print(f'trajectory_masks.shape: {trajectory_masks.shape}')
print(trajectory_masks)

unpadded_trajectories = utils.unpad_trajectories(padded_trajectories, trajectory_masks)

print('Unpadded')
print(f'unpadded_trajectories.shape: {unpadded_trajectories.shape}')
print(unpadded_trajectories.squeeze(-1))
