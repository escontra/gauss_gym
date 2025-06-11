from typing import Tuple
import torch
import numpy as np


def heightmap_to_voxels_np(heightmap: np.ndarray, num_voxels: int, min_height: float, max_height: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a heightmap to voxel occupancy and centroid grids.
    
    Args:
        heightmap: [N, M] float32 array containing height values
        num_voxels: Number of voxels in the height dimension
        min_height: Minimum height value to consider
        max_height: Maximum height value to consider
        
    Returns:
        tuple of:
        - occupancy_grid: [N, M, H] bool array indicating voxel occupancy
        - centroid_grid: [N, M, H] float32 array containing normalized height within voxel
    """
    N, M = heightmap.shape
    voxel_height = (max_height - min_height) / num_voxels
    
    # Create voxel height levels
    voxel_levels = np.linspace(min_height, max_height, num_voxels + 1)
    
    # Initialize output grids
    occupancy_grid = np.zeros((N, M, num_voxels), dtype=bool)
    centroid_grid = np.zeros((N, M, num_voxels), dtype=np.float32)
    
    # Find which voxel each height belongs to
    voxel_indices = np.searchsorted(voxel_levels, heightmap) - 1
    voxel_indices = np.clip(voxel_indices, 0, num_voxels - 1)
    
    # Create index arrays for setting values
    n_indices = np.arange(N)[:, None, None]  # [N, 1, 1]
    m_indices = np.arange(M)[None, :, None]  # [1, M, 1]
    v_indices = voxel_indices[:, :, None]    # [N, M, 1]
    
    # Set occupancy
    occupancy_grid[n_indices, m_indices, v_indices] = True
    
    # Calculate normalized heights within voxels
    z_norm = (heightmap - voxel_levels[voxel_indices]) / voxel_height
    centroid_grid[n_indices, m_indices, v_indices] = z_norm[:, :, None]
    
    return occupancy_grid, centroid_grid


def heightmap_to_voxels_torch(heightmap: torch.Tensor, num_voxels: int, min_height: float=-1.0, max_height: float=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a heightmap to voxel occupancy and centroid grids using PyTorch.

    First and last height indices of the height voxels indicate saturated
    boundaries where height is <= min_height or >= max_height. Centroid grid
    values are set to 0 for these voxels. Centroid grid values are normalized
    to the range [0, 1] for the interior voxels.
    
    Args:
        heightmap: [..., N, M] tensor containing height values
        num_voxels: Number of voxels in the height dimension
        min_height: Minimum height value to consider
        max_height: Maximum height value to consider
        
    Returns:
        tuple of:
        - occupancy_grid: [..., N, M, H] bool tensor indicating voxel occupancy
        - centroid_grid: [..., N, M, H] float32 tensor containing normalized height within voxel
    """
    # Get the last two dimensions (N, M)
    *batch_dims, N, M = heightmap.shape
    # -2 because first and last voxels indicate saturated boundaries.
    voxel_height = (max_height - min_height) / (num_voxels - 2)

    # Create voxel height levels
    voxel_levels = torch.linspace(min_height, max_height, num_voxels - 1, device=heightmap.device)

    # Find which voxel each height belongs to using broadcasting
    # Expand heightmap and voxel_levels for comparison
    heightmap_expanded = heightmap.unsqueeze(-1)  # [..., N, M, 1]
    
    # Expand voxel_levels to match batch dimensions
    voxel_levels_expanded = voxel_levels
    for _ in batch_dims:
        voxel_levels_expanded = voxel_levels_expanded.unsqueeze(0)
    voxel_levels_expanded = voxel_levels_expanded.unsqueeze(-2).unsqueeze(-2)  # [1, ..., 1, 1, num_voxels+1]
    
    # Compare heights with levels to find voxel indices
    voxel_indices = torch.sum(heightmap_expanded > voxel_levels_expanded, dim=-1) # [..., N, M]
    voxel_indices[heightmap_expanded[..., 0] <= min_height] = 0
    voxel_indices[heightmap_expanded[..., 0] >= max_height] = num_voxels - 1
    voxel_indices = torch.clamp(voxel_indices, 0, num_voxels - 1)
    
    # Create a one-hot encoding of the voxel indices
    voxel_one_hot = torch.zeros((*batch_dims, N, M, num_voxels), dtype=torch.bool, device=heightmap.device)
    voxel_one_hot.scatter_(-1, voxel_indices.unsqueeze(-1), True)
    
    # Calculate normalized heights within voxels
    voxel_level_indices = torch.clamp(voxel_indices - 1, 0, num_voxels - 2)
    z_norm = (heightmap - voxel_levels[voxel_level_indices]) / voxel_height
    z_norm[voxel_indices == 0] = 0
    z_norm[voxel_indices == num_voxels - 1] = 0

    # Set the normalized heights in the occupied voxels
    centroid_grid = torch.where(voxel_one_hot, z_norm.unsqueeze(-1), torch.zeros_like(voxel_one_hot))
    
    return voxel_one_hot, centroid_grid

def unsaturated_voxels_mask(occupancy_grid: torch.Tensor):
    """
    Create a mask of the unsaturated voxels in the occupancy grid.

    Args:
        occupancy_grid: [..., N, M, H] bool tensor indicating voxel occupancy

    Returns:
        mask: [..., N, M, H] bool tensor indicating saturated voxels
    """
    saturated_mask = torch.zeros_like(occupancy_grid)
    saturated_mask[..., 0] = occupancy_grid[..., 0]
    saturated_mask[..., -1] = occupancy_grid[..., -1]
    return ~saturated_mask
