from typing import Tuple
import torch


def heightmap_to_voxels(
  heightmap: torch.Tensor,
  num_voxels: int,
  min_height: float = -1.0,
  max_height: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
  voxel_levels = torch.linspace(
    min_height, max_height, num_voxels - 1, device=heightmap.device
  )

  # Find which voxel each height belongs to using broadcasting
  # Expand heightmap and voxel_levels for comparison
  heightmap_expanded = heightmap.unsqueeze(-1)  # [..., N, M, 1]

  # Expand voxel_levels to match batch dimensions
  voxel_levels_expanded = voxel_levels
  for _ in batch_dims:
    voxel_levels_expanded = voxel_levels_expanded.unsqueeze(0)
  voxel_levels_expanded = voxel_levels_expanded.unsqueeze(-2).unsqueeze(
    -2
  )  # [1, ..., 1, 1, num_voxels+1]

  # Compare heights with levels to find voxel indices
  voxel_indices = torch.sum(
    heightmap_expanded > voxel_levels_expanded, dim=-1
  )  # [..., N, M]
  voxel_indices[heightmap_expanded[..., 0] <= min_height] = 0
  voxel_indices[heightmap_expanded[..., 0] >= max_height] = num_voxels - 1
  voxel_indices = torch.clamp(voxel_indices, 0, num_voxels - 1)

  # Create a one-hot encoding of the voxel indices
  voxel_one_hot = torch.zeros(
    (*batch_dims, N, M, num_voxels), dtype=torch.bool, device=heightmap.device
  )
  voxel_one_hot.scatter_(-1, voxel_indices.unsqueeze(-1), True)

  # Calculate normalized heights within voxels
  voxel_level_indices = torch.clamp(voxel_indices - 1, 0, num_voxels - 2)
  z_norm = (heightmap - voxel_levels[voxel_level_indices]) / voxel_height
  z_norm[voxel_indices == 0] = 0
  z_norm[voxel_indices == num_voxels - 1] = 0

  # Set the normalized heights in the occupied voxels
  centroid_grid = torch.where(
    voxel_one_hot, z_norm.unsqueeze(-1), torch.zeros_like(voxel_one_hot)
  )

  return voxel_one_hot, centroid_grid


def unsaturated_voxels_mask(occupancy_grid: torch.Tensor):
  """
  Create a mask of the unsaturated columns in the occupancy grid.

  Args:
      occupancy_grid: [..., N, M, H] bool tensor indicating voxel occupancy

  Returns:
      mask: [..., N, M, H] bool tensor indicating saturated voxels
  """
  lower_saturated = occupancy_grid[..., :1].expand_as(occupancy_grid)
  upper_saturated = occupancy_grid[..., -1:].expand_as(occupancy_grid)
  saturated_columns = lower_saturated | upper_saturated
  unsaturated_mask = ~saturated_columns
  return unsaturated_mask


def voxels_to_heightmap(
  voxel_one_hot: torch.Tensor,
  centroid_grid: torch.Tensor,
  min_height: float = -1.0,
  max_height: float = 1.0,
) -> torch.Tensor:
  """
  Convert voxel occupancy and centroid grid back to heightmap.
  Args:
      voxel_one_hot: [..., N, M, H] bool tensor indicating voxel occupancy
      centroid_grid: [..., N, M, H] float tensor containing normalized height within voxel
      min_height: Minimum height value used when creating voxels
      max_height: Maximum height value used when creating voxels
  Returns:
      heightmap: [..., N, M] float tensor of reconstructed heights
  """
  H = voxel_one_hot.shape[-1]
  voxel_height = (max_height - min_height) / (H - 2)
  z_norm = centroid_grid.sum(dim=-1)  # [..., N, M]
  voxel_indices = voxel_one_hot.to(torch.int64).argmax(dim=-1)  # [..., N, M]
  heightmap = min_height + (voxel_indices.to(z_norm.dtype) - 1 + z_norm) * voxel_height
  heightmap = heightmap.clamp(min_height, max_height)
  return heightmap
