import numpy as np
import torch

from legged_gym.utils import voxel

np.set_printoptions(
    linewidth=np.inf,  # Print everything in one line
    precision=2,       # Show only 2 decimal places
)

def test_heightmap_to_voxels():
    # Test NumPy version
    heightmap_np = np.array([
        [-0.1, 0.2, 0.5],
        [0.0, 1.0, 2.0],
        [1.0, 2.0, 3.0],
        [2.0, 3.5, 4.0],
        [3.7, 4.0, 4.5],
    ], dtype=np.float32)
    
    num_voxels = 10 
    min_height = 0.0
    max_height = 4.0
    
    heightmap_torch = torch.from_numpy(heightmap_np)
    occupancy_grid_torch, centroid_grid_torch = voxel.heightmap_to_voxels(
        heightmap_torch, num_voxels, min_height, max_height
    )
    heightmap_torch = voxel.voxels_to_heightmap(occupancy_grid_torch, centroid_grid_torch, min_height, max_height)

    print('PyTorch - Occupancy grid shape:', occupancy_grid_torch.shape)
    print('PyTorch - Centroid grid shape:', centroid_grid_torch.shape)
    print('PyTorch - Heightmap shape:', heightmap_torch.shape)
    print(f'Orig heightmap:\n{heightmap_np}')
    print(f'Reconstructed heightmap:\n{heightmap_torch.cpu().numpy()}')

    unsaturated_mask = voxel.unsaturated_voxels_mask(occupancy_grid_torch)
    
    # Test batched input with PyTorch
    batched_heightmap = torch.stack([heightmap_torch, heightmap_torch + 1.0])
    batched_occupancy, batched_centroid = voxel.heightmap_to_voxels(
        batched_heightmap, num_voxels, min_height, max_height
    )
    
    print('PyTorch Batched - Occupancy grid shape:', batched_occupancy.shape)
    print('PyTorch Batched - Centroid grid shape:', batched_centroid.shape)

    # Check output shapes
    assert occupancy_grid_torch.shape == (*heightmap_torch.shape, num_voxels)
    assert centroid_grid_torch.shape == (*heightmap_torch.shape, num_voxels)
    assert batched_occupancy.shape == (*batched_heightmap.shape, num_voxels)
    assert batched_centroid.shape == (*batched_heightmap.shape, num_voxels)
    
    # Check that occupancy values are boolean
    assert occupancy_grid_torch.dtype == torch.bool
    
    # Check that centroid values are float32
    assert centroid_grid_torch.dtype == torch.float32
    
    # Check that each heightmap cell is occupied in exactly one voxel
    for i in range(3):
        for j in range(3):
            assert torch.sum(occupancy_grid_torch[i, j]) == 1

if __name__ == "__main__":
    test_heightmap_to_voxels()
    print("All tests passed!")
