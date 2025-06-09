from typing import Tuple
import torch

def heightmap_to_voxels_torch(heightmap: torch.Tensor, num_voxels: int, min_height: float, max_height: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a heightmap to voxel occupancy and centroid grids using PyTorch.
    
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
    voxel_height = (max_height - min_height) / num_voxels

    # Create voxel height levels
    voxel_levels = torch.linspace(min_height, max_height, num_voxels + 1, device=heightmap.device)
    
    # Find which voxel each height belongs to using broadcasting
    # Expand heightmap and voxel_levels for comparison
    heightmap_expanded = heightmap.unsqueeze(-1)  # [..., N, M, 1]
    
    # Expand voxel_levels to match batch dimensions
    voxel_levels_expanded = voxel_levels
    for _ in batch_dims:
        voxel_levels_expanded = voxel_levels_expanded.unsqueeze(0)
    voxel_levels_expanded = voxel_levels_expanded.unsqueeze(-2).unsqueeze(-2)  # [1, ..., 1, 1, num_voxels+1]
    
    # Compare heights with levels to find voxel indices
    voxel_indices = torch.sum(heightmap_expanded > voxel_levels_expanded, dim=-1) - 1  # [..., N, M]
    voxel_indices = torch.clamp(voxel_indices, 0, num_voxels - 1)
    
    # Create a one-hot encoding of the voxel indices
    voxel_one_hot = torch.zeros((*batch_dims, N, M, num_voxels), dtype=torch.bool, device=heightmap.device)
    voxel_one_hot.scatter_(-1, voxel_indices.unsqueeze(-1), True)
    
    # Calculate normalized heights within voxels
    z_norm = (heightmap - voxel_levels[voxel_indices]) / voxel_height
    
    # Set the normalized heights in the occupied voxels
    centroid_grid = torch.where(voxel_one_hot, z_norm.unsqueeze(-1), torch.zeros_like(voxel_one_hot))
    
    return voxel_one_hot, centroid_grid
