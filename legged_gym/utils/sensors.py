import torch
import torch.nn.functional as F
import os
from collections import defaultdict

from isaacgym.torch_utils import quat_apply
from legged_gym.utils.math import quat_apply_yaw
from typing import Tuple, Callable
from dataclasses import dataclass
import numpy as np
import warp as wp
from isaacgym import gymapi, gymutil
import math
from rendering.batch_gs_renderer import MultiSceneRenderer
import viser.transforms as vtf
import matplotlib.pyplot as plt
from PIL import Image
from isaacgym.torch_utils import to_torch, quat_rotate


class BatchWireframeSphereGeometry(gymutil.LineGeometry):
    """Draw multiple spheres without a for loop"""

    def __init__(self, num_spheres, radius=1.0, num_lats=8, num_lons=8, pose=None, color=None, color2=None):
        if color is None:
            color = (1, 0, 0)

        if color2 is None:
            color2 = color

        self.num_lines = 2 * num_lats * num_lons
        self.num_spheres = num_spheres

        verts = np.empty((self.num_lines, 2), gymapi.Vec3.dtype)
        colors = np.empty(self.num_lines, gymapi.Vec3.dtype)
        idx = 0

        ustep = 2 * math.pi / num_lats
        vstep = math.pi / num_lons

        u = 0.0
        for _i in range(num_lats):
            v = 0.0
            for _j in range(num_lons):
                x1 = radius * math.sin(v) * math.sin(u)
                y1 = radius * math.cos(v)
                z1 = radius * math.sin(v) * math.cos(u)

                x2 = radius * math.sin(v + vstep) * math.sin(u)
                y2 = radius * math.cos(v + vstep)
                z2 = radius * math.sin(v + vstep) * math.cos(u)

                x3 = radius * math.sin(v + vstep) * math.sin(u + ustep)
                y3 = radius * math.cos(v + vstep)
                z3 = radius * math.sin(v + vstep) * math.cos(u + ustep)

                verts[idx][0] = (x1, y1, z1)
                verts[idx][1] = (x2, y2, z2)
                colors[idx] = color

                idx += 1

                verts[idx][0] = (x2, y2, z2)
                verts[idx][1] = (x3, y3, z3)
                colors[idx] = color2

                idx += 1

                v += vstep
            u += ustep

        if pose is None:
            self.verts = np.tile(verts, (num_spheres, 1))
        else:
            self.verts = pose.transform_points(verts)

        self.verts_tmp = np.copy(self.verts)
        self._colors = np.tile(colors, (num_spheres,))

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors

    def draw(self, positions, gym, viewer, env, only_render_selected=-1, custom_colors=None):
        flat_pos = positions.repeat_interleave(self.num_lines, dim=0).view(-1, 3).cpu().numpy()
        if custom_colors is not None:
            colors_tmp = np.empty(self.colors().shape[0], gymapi.Vec3.dtype)
            custom_colors = custom_colors.repeat_interleave(self.num_lines, dim=0).view(-1, 3).cpu().numpy()
            colors_tmp['x'][:] = custom_colors[:, 0]
            colors_tmp['y'][:] = custom_colors[:, 1]
            colors_tmp['z'][:] = custom_colors[:, 2]
        else:
            colors_tmp = self.colors()
            

        self.verts_tmp["x"][:, 0] = self.verts["x"][:, 0] + flat_pos[:, 0]
        self.verts_tmp["x"][:, 1] = self.verts["x"][:, 1] + flat_pos[:, 0]
        self.verts_tmp["y"][:, 0] = self.verts["y"][:, 0] + flat_pos[:, 1]
        self.verts_tmp["y"][:, 1] = self.verts["y"][:, 1] + flat_pos[:, 1]
        self.verts_tmp["z"][:, 0] = self.verts["z"][:, 0] + flat_pos[:, 2]
        self.verts_tmp["z"][:, 1] = self.verts["z"][:, 1] + flat_pos[:, 2]

        if isinstance(only_render_selected, int):
          if only_render_selected < 0:
              start_frustrum = 0
              end_frustrum = self.num_spheres
          else:
              start_frustrum = only_render_selected
              end_frustrum = only_render_selected + 1
        else:
            start_frustrum = only_render_selected[0]
            end_frustrum = only_render_selected[1]

        verts_tmp = self.verts_tmp[start_frustrum * self.num_lines:end_frustrum * self.num_lines]
        colors_tmp = colors_tmp[start_frustrum * self.num_lines:end_frustrum * self.num_lines]
        total_lines = (end_frustrum - start_frustrum) * self.num_lines
        gym.add_lines(viewer, env, total_lines, verts_tmp, colors_tmp)


class BatchWireframeAxisGeometry(gymutil.LineGeometry):
    """Draw multiple spheres without a for loop"""

    def __init__(self, num_frustrums, length, thickness, lines_per_axis=64, color_x=(1, 0, 0), color_y=(0, 1, 0), color_z=(0, 0, 1)):

        self.num_lines = 3 * lines_per_axis
        self.num_frustrums = num_frustrums
        self.lines_per_axis = lines_per_axis

        verts = np.empty((self.num_lines, 2), gymapi.Vec3.dtype)
        colors = np.empty(self.num_lines, gymapi.Vec3.dtype)

        for i, value in enumerate(np.linspace(0, 2 * np.pi, lines_per_axis)):
            c1 = thickness * np.cos(value)
            c2 = thickness * np.sin(value)

            verts[i][0] = (0, c1, c2)
            verts[i][1] = (length, c1, c2)
            colors[i] = color_x

            verts[i + lines_per_axis][0] = (c1, 0, c2)
            verts[i + lines_per_axis][1] = (c1, length, c2)
            colors[i + lines_per_axis] = color_y

            verts[i + 2 * lines_per_axis][0] = (c1, c2, 0)
            verts[i + 2 * lines_per_axis][1] = (c1, c2, length)
            colors[i + 2 * lines_per_axis] = color_z

        self.verts = np.tile(verts, (num_frustrums, 1))
        self.verts_tmp = np.copy(self.verts)
        self._colors = np.tile(colors, (num_frustrums, 1))

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors

    def draw(self, positions, rotations, gym, viewer, env, axis_scales=None, only_render_selected=-1):
        flat_pos = positions.repeat_interleave(self.num_lines, dim=0).cpu().numpy()
        flat_quat = rotations.repeat_interleave(self.num_lines, dim=0) # (N, 4)

        if axis_scales is not None:
            # Repeat axis scales for each line.
            axis_scales_x = F.pad(axis_scales[..., :1], (0, 2), mode='constant', value=1)
            axis_scales_y = F.pad(axis_scales[..., 1:2], (1, 1), mode='constant', value=1)
            axis_scales_z = F.pad(axis_scales[..., 2:3], (2, 0), mode='constant', value=1)
            axis_scales = torch.stack((axis_scales_x, axis_scales_y, axis_scales_z), dim=1).reshape(-1, 3)
            axis_scales = axis_scales.repeat_interleave(self.lines_per_axis, dim=0)
        verts_stacked = np.stack((
            self.verts["x"], self.verts["y"], self.verts["z"]
        ), axis=-1)
        if axis_scales is not None:
            verts_stacked[:, 1, :] *= axis_scales.cpu().numpy()
        rotated_verts = quat_apply(
          flat_quat[:, None].repeat(1, 2, 1),
          torch.tensor(verts_stacked, device=positions.device)
        ).cpu().numpy()

        self.verts_tmp["x"][:, 0] = rotated_verts[:, 0, 0] + flat_pos[:, 0]
        self.verts_tmp["x"][:, 1] = rotated_verts[:, 1, 0] + flat_pos[:, 0]
        self.verts_tmp["y"][:, 0] = rotated_verts[:, 0, 1] + flat_pos[:, 1]
        self.verts_tmp["y"][:, 1] = rotated_verts[:, 1, 1] + flat_pos[:, 1]
        self.verts_tmp["z"][:, 0] = rotated_verts[:, 0, 2] + flat_pos[:, 2]
        self.verts_tmp["z"][:, 1] = rotated_verts[:, 1, 2] + flat_pos[:, 2]

        if isinstance(only_render_selected, int):
          if only_render_selected < 0:
              start_frustrum = 0
              end_frustrum = self.num_frustrums
          else:
              start_frustrum = only_render_selected
              end_frustrum = only_render_selected + 1
        else:
            start_frustrum = only_render_selected[0]
            end_frustrum = only_render_selected[1]

        verts_tmp = self.verts_tmp[start_frustrum * self.num_lines:end_frustrum * self.num_lines]
        colors_tmp = self._colors[start_frustrum * self.num_lines:end_frustrum * self.num_lines]
        total_lines = (end_frustrum - start_frustrum) * self.num_lines
        gym.add_lines(viewer, env, total_lines, verts_tmp, colors_tmp)


class BatchWireframeFrustumGeometry(gymutil.LineGeometry):
    """Draw multiple wireframe frustums (e.g. camera frustums) without a for loop.

    The frustum is defined in camera space (origin at (0,0,0) with the camera looking along +x).
    The near plane is located at distance `near` and the far plane at `far`. The horizontal field-of-view
    is given by `fov` (in radians) and the aspect ratio (width/height) by `aspect`.

    The 8 frustum corners are computed as:
      - Near plane corners (at x = near):
          v0: (near,  half_height_near,  half_width_near)  # top-right
          v1: (near,  half_height_near, -half_width_near)  # top-left
          v2: (near, -half_height_near, -half_width_near)  # bottom-left
          v3: (near, -half_height_near,  half_width_near)  # bottom-right
      - Far plane corners (at x = far):
          v4: (far,  half_height_far,  half_width_far)      # top-right
          v5: (far,  half_height_far, -half_width_far)      # top-left
          v6: (far, -half_height_far, -half_width_far)      # bottom-left
          v7: (far, -half_height_far,  half_width_far)      # bottom-right

    Then the edges are defined as:
      - Near plane: v0-v1, v1-v2, v2-v3, v3-v0
      - Far plane:  v4-v5, v5-v6, v6-v7, v7-v4
      - Sides:      v0-v4, v1-v5, v2-v6, v3-v7
    """
    def __init__(self, num_frustums, near, far, image_width, image_height, focal_length, thickness, num_lines_per_edge=32):
        self.num_frustums = num_frustums
        
        # Compute aspect ratio from image dimensions
        aspect = image_width / image_height
        
        # Derive horizontal fov from focal length and image width
        fov = 2 * np.arctan(image_width / (2 * focal_length))
        
        # Compute half widths/heights at near and far planes
        half_width_near = near * np.tan(fov / 2)
        half_height_near = half_width_near / aspect
        half_width_far = far * np.tan(fov / 2)
        half_height_far = half_width_far / aspect
        
        # Define the 8 corners of the frustum in camera space (camera looks along +x)
        v0 = (half_width_near, half_height_near, near)   # top-right
        v1 = (-half_width_near, half_height_near, near)  # top-left
        v2 = (-half_width_near, -half_height_near, near) # bottom-left
        v3 = (half_width_near, -half_height_near, near)  # bottom-right

        v4 = (half_width_far, half_height_far, far)      # top-right
        v5 = (-half_width_far, half_height_far, far)     # top-left
        v6 = (-half_width_far, -half_height_far, far)    # bottom-left
        v7 = (half_width_far, -half_height_far, far)     # bottom-right

        # Define the edges of the frustum - including redundant edges for reliability
        edges = [
            # Near plane
            (v0, v1), (v1, v2), (v2, v3), (v3, v0),  # Primary edges clockwise
            (v2, v1), (v0, v3),  # Redundant bottom and top edges
            
            # Far plane
            (v4, v5), (v5, v6), (v6, v7), (v7, v4),  # Primary edges clockwise
            (v6, v5), (v4, v7),  # Redundant bottom and top edges
            
            # Connecting edges
            (v0, v4), (v1, v5), (v2, v6), (v3, v7)   # Connect near & far
        ]
        self.num_lines = len(edges) * num_lines_per_edge
        
        verts = np.empty((self.num_lines, 2), gymapi.Vec3.dtype)
        colors = np.empty(self.num_lines, gymapi.Vec3.dtype)
        
        # For each edge, create multiple lines arranged in a circular pattern
        for edge_idx, (start, end) in enumerate(edges):
            # Set color based on which part of the frustum this edge belongs to
            if edge_idx < 6:  # Near plane edges (including redundant)
                color = (1, 0.5, 0)  # Orange
            elif edge_idx < 12:  # Far plane edges (including redundant)
                color = (0, 1, 1)  # Cyan
            else:  # Connecting edges
                color = (1, 0, 1)  # Purple
            
            # Create a coordinate system for the circular pattern
            direction = np.array(end) - np.array(start)
            direction = direction / np.linalg.norm(direction)
            
            # Find perpendicular vectors to create the circular pattern
            if np.allclose(direction, [1, 0, 0]):
                perp1 = np.array([0, 1, 0])
            else:
                perp1 = np.cross(direction, [1, 0, 0])
                perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(direction, perp1)
            
            # Create lines around the edge in a circular pattern
            for i in range(num_lines_per_edge):
                angle = 2 * np.pi * i / num_lines_per_edge
                offset = thickness * (perp1 * np.cos(angle) + perp2 * np.sin(angle))
                
                line_idx = edge_idx * num_lines_per_edge + i
                verts[line_idx][0] = tuple(np.array(start) + offset)
                verts[line_idx][1] = tuple(np.array(end) + offset)
                colors[line_idx] = color

        self.verts = np.tile(verts, (num_frustums, 1))
        self.verts_tmp = np.copy(self.verts)
        self._colors = np.tile(colors, (num_frustums, 1))

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors

    def draw(self, positions, rotations, gym, viewer, env, only_render_selected=-1):
        flat_pos = positions.repeat_interleave(self.num_lines, dim=0).cpu().numpy()
        flat_quat = rotations.repeat_interleave(self.num_lines, dim=0).cpu().numpy() # (N, 4)

        # Stack the vertex components for rotation
        verts_stacked = np.stack((
            self.verts["x"], self.verts["y"], self.verts["z"]
        ), axis=-1)
        # Apply quaternion rotation to the vertices.
        # (Assumes a function `quat_apply` is available that applies the quaternion rotation.)
        rotated_verts = quat_apply(
            torch.tensor(flat_quat[:, None].repeat(2, 1), device=positions.device),
            torch.tensor(verts_stacked, device=positions.device)
        ).cpu().numpy()

        # Translate the rotated vertices by the instance positions
        self.verts_tmp["x"][:, 0] = rotated_verts[:, 0, 0] + flat_pos[:, 0]
        self.verts_tmp["x"][:, 1] = rotated_verts[:, 1, 0] + flat_pos[:, 0]
        self.verts_tmp["y"][:, 0] = rotated_verts[:, 0, 1] + flat_pos[:, 1]
        self.verts_tmp["y"][:, 1] = rotated_verts[:, 1, 1] + flat_pos[:, 1]
        self.verts_tmp["z"][:, 0] = rotated_verts[:, 0, 2] + flat_pos[:, 2]
        self.verts_tmp["z"][:, 1] = rotated_verts[:, 1, 2] + flat_pos[:, 2]

        if only_render_selected >= 0:
            verts_tmp = self.verts_tmp[only_render_selected * self.num_lines:(only_render_selected + 1) * self.num_lines]
            colors_tmp = self._colors[only_render_selected * self.num_lines:(only_render_selected + 1) * self.num_lines]
            total_lines = self.num_lines
        else:
            verts_tmp = self.verts_tmp
            colors_tmp = self._colors
            total_lines = self.num_frustums * self.num_lines

        gym.add_lines(viewer, env, total_lines, verts_tmp, colors_tmp)


wp.init()

@wp.kernel
def raycast_kernel(
    mesh: wp.uint64,
    ray_starts_world: wp.array(dtype=wp.vec3),
    ray_directions_world: wp.array(dtype=wp.vec3),
    ray_hits_world: wp.array(dtype=wp.vec3),
):

    tid = wp.tid()

    t = float(0.0)  # hit distance along ray
    u = float(0.0)  # hit face barycentric u
    v = float(0.0)  # hit face barycentric v
    sign = float(0.0)  # hit face sign
    n = wp.vec3()  # hit face normal
    f = int(0)  # hit face index
    max_dist = float(1e6)  # max raycast disance
    # ray cast against the mesh
    if wp.mesh_query_ray(mesh, ray_starts_world[tid], ray_directions_world[tid], max_dist, t, u, v, sign, n, f):
        ray_hits_world[tid] = ray_starts_world[tid] + t * ray_directions_world[tid]


def ray_cast(ray_starts_world, ray_directions_world, wp_mesh):
    """Performs ray casting on the terrain mesh.

    Args:
        ray_starts_world (Torch.tensor): The starting position of the ray.
        ray_directions_world (Torch.tensor): The ray direction.

    Returns:
        [Torch.tensor]: The ray hit position. Returns float('inf') for missed hits.
    """
    shape = ray_starts_world.shape
    ray_starts_world = ray_starts_world.view(-1, 3)
    ray_directions_world = ray_directions_world.view(-1, 3)
    num_rays = len(ray_starts_world)
    ray_starts_world_wp = wp.types.array(
        ptr=ray_starts_world.data_ptr(),
        dtype=wp.vec3,
        shape=(num_rays,),
        copy=False,
        owner=False,
        device=wp_mesh.device,
    )
    ray_directions_world_wp = wp.types.array(
        ptr=ray_directions_world.data_ptr(),
        dtype=wp.vec3,
        shape=(num_rays,),
        copy=False,
        owner=False,
        device=wp_mesh.device,
    )
    ray_hits_world = torch.zeros((num_rays, 3), device=ray_starts_world.device)
    ray_hits_world[:] = float("inf")
    ray_hits_world_wp = wp.types.array(
        ptr=ray_hits_world.data_ptr(), dtype=wp.vec3, shape=(num_rays,), copy=False, owner=False, device=wp_mesh.device
    )
    wp.launch(
        kernel=raycast_kernel,
        dim=num_rays,
        inputs=[wp_mesh.id, ray_starts_world_wp, ray_directions_world_wp, ray_hits_world_wp],
        device=wp_mesh.device,
    )
    wp.synchronize()
    return ray_hits_world.view(shape)


@wp.kernel
def nearest_point_kernel(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    mesh_points: wp.array(dtype=wp.vec3),
):

    tid = wp.tid()

    max_dist = float(1e6)  # max raycast disance
    query = wp.mesh_query_point(mesh, points[tid], max_dist=max_dist)
    if not query.result:
        return

    # Evaluate the position of the nearest location found.
    mesh_points[tid] = wp.mesh_eval_position(
        mesh,
        query.face,
        query.u,
        query.v,
    )


def nearest_point(points, wp_mesh):
    """Performs ray casting on the terrain mesh.

    Args:
        point (Torch.tensor): The near point.

    Returns:
        [Torch.tensor]: The ray hit position. Returns float('inf') for missed hits.
    """
    shape = points.shape
    points = points.view(-1, 3)
    num_points = len(points)
    points_wp = wp.types.array(
        ptr=points.data_ptr(),
        dtype=wp.vec3,
        shape=(num_points,),
        copy=False,
        owner=False,
        device=wp_mesh.device,
    )
    mesh_points = torch.zeros((num_points, 3), device=points.device)
    mesh_points[:] = float("inf")
    mesh_points_wp = wp.types.array(
        ptr=mesh_points.data_ptr(), dtype=wp.vec3, shape=(num_points,), copy=False, owner=False, device=wp_mesh.device
    )
    wp.launch(
        kernel=nearest_point_kernel,
        dim=num_points,
        inputs=[wp_mesh.id, points_wp, mesh_points_wp],
        device=wp_mesh.device,
    )
    wp.synchronize()
    return mesh_points.view(shape)


def convert_to_wp_mesh(vertices, triangles, device):
    return wp.Mesh(
        points=wp.array(vertices.astype(np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(triangles.astype(np.int32).flatten(), dtype=int, device=device),
    )


def grid_pattern(pattern_cfg: "GridPatternCfg", device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """A regular grid pattern for ray casting.

    Args:
        pattern_cfg (GridPatternCfg): The config for the pattern.
        device (str): The device
    Returns:
        ray_starts (torch.Tensor): The starting positions of the rays
        ray_directions (torch.Tensor): The ray directions

    """
    y = torch.arange(
        start=-pattern_cfg.width / 2, end=pattern_cfg.width / 2 + 1.0e-9, step=pattern_cfg.resolution, device=device
    )
    x = torch.arange(
        start=-pattern_cfg.length / 2, end=pattern_cfg.length / 2 + 1.0e-9, step=pattern_cfg.resolution, device=device
    )
    grid_x, grid_y = torch.meshgrid(x, y)

    num_rays = grid_x.numel()
    ray_starts = torch.zeros(num_rays, 3, device=device)
    ray_starts[:, 0] = grid_x.flatten()
    ray_starts[:, 1] = grid_y.flatten()

    ray_directions = torch.zeros_like(ray_starts)
    ray_directions[..., :] = torch.tensor(list(pattern_cfg.direction), device=device)
    return ray_starts, ray_directions

@dataclass
class GridPatternLocomotionCfg:
    resolution: float = 0.1
    width: float = 1.0 #  for rmp, we use 8.0; for rl policy, should be 3.2
    length: float = 1.6 #  for rmp, we use 8.0; for rl policy, should be 3.2
    max_xy_drift: float = 0.05
    direction: Tuple = (0.0, 0.0, -1.0)
    pattern_func: Callable = grid_pattern

def base_height(pattern_cfg: "BaseHeightLocomotionCfg", device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """A regular grid pattern for ray casting.

    Args:
        pattern_cfg (GridPatternCfg): The config for the pattern.
        device (str): The device
    Returns:
        ray_starts (torch.Tensor): The starting positions of the rays
        ray_directions (torch.Tensor): The ray directions

    """
    ray_starts = torch.zeros(1, 3, device=device)

    ray_directions = torch.zeros_like(ray_starts)
    ray_directions[..., :] = torch.tensor(list(pattern_cfg.direction), device=device)
    return ray_starts, ray_directions


@dataclass
class BaseHeightLocomotionCfg:
    direction: Tuple = (0.0, 0.0, -1.0)
    pattern_func: Callable = base_height


class RayCaster():
    def __init__(self, env):        
        self.attachement_pos =(0.0, 0.0, 20.0)
        self.attachement_quat = (0.0, 0.0, 0.0, 1.0)
        self.attach_yaw_only = True
        self.pattern_cfg = GridPatternLocomotionCfg()
        self.body_attachement_name = "base"
        self.default_hit_value = 10
        # self.terrain_mesh = convert_to_wp_mesh(env.scene_manager.all_vertices_mesh, env.scene_manager.all_triangles_mesh, env.device)
        self.terrain_mesh = env.scene_manager.terrain_mesh
        self.num_envs = env.num_envs
        self.device = env.device

        self.ray_starts, self.ray_directions = self.pattern_cfg.pattern_func(self.pattern_cfg, self.device)
        self.num_rays = len(self.ray_directions)

        offset_pos = torch.tensor(list(self.attachement_pos), device=self.device)
        offset_quat = torch.tensor(list(self.attachement_quat), device=env.device)
        self.ray_directions = quat_apply(offset_quat.repeat(len(self.ray_directions), 1), self.ray_directions)
        self.ray_starts += offset_pos

        self.ray_starts = self.ray_starts.repeat(self.num_envs, 1, 1)
        self.ray_directions = self.ray_directions.repeat(self.num_envs, 1, 1)

        self.ray_hits_world = torch.zeros(self.num_envs, self.num_rays, 3, device=self.device)
        self.env = env
        self.sphere_geom = None

    def update(self, dt, env_ids=...):
        """Perform raycasting on the terrain.

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the ray hits. Defaults to ....
        """
        states = self.env.root_states[env_ids, :].squeeze(1) 
        pos = states[..., :3]
        quats = states[..., 3:7]
        if self.attach_yaw_only:
            ray_starts_world = quat_apply_yaw(quats.repeat(1, self.num_rays), self.ray_starts[env_ids]) + pos.unsqueeze(
                1
            )
            ray_directions_world = self.ray_directions[env_ids]
        else:
            ray_starts_world = quat_apply(quats.repeat(1, self.num_rays), self.ray_starts[env_ids]) + pos.unsqueeze(1)
            ray_directions_world = quat_apply(quats.repeat(1, self.num_rays), self.ray_directions[env_ids])

        self.ray_hits_world[env_ids] = ray_cast(ray_starts_world, ray_directions_world, self.terrain_mesh)

    def get_data(self):
        return torch.nan_to_num(self.ray_hits_world, posinf=self.default_hit_value)
    
    def debug_vis(self, env):
        if self.sphere_geom is None:
            self.sphere_geom = BatchWireframeSphereGeometry(
                self.num_envs * self.num_rays, 0.02, 4, 4, None, color=(0, 1, 0)
            )
        if self.env.selected_environment >= 0:
          only_render_selected_range = [self.env.selected_environment * self.num_rays, (self.env.selected_environment + 1) * self.num_rays]
          self.sphere_geom.draw(self.ray_hits_world, env.gym, env.viewer, env.envs[0], only_render_selected=only_render_selected_range)
        else:
          self.sphere_geom.draw(self.ray_hits_world, env.gym, env.viewer, env.envs[0])


class FootContactSensor():
    def __init__(self, env):        
        self.feet_edge_pos = env.cfg["asset"]["feet_edge_pos"]
        self.attach_yaw_only = True
        self.pattern_cfg = BaseHeightLocomotionCfg()
        self.default_hit_value = 10
        # self.terrain_mesh = convert_to_wp_mesh(env.scene_manager.all_vertices_mesh, env.scene_manager.all_triangles_mesh, env.device)
        self.terrain_mesh = env.scene_manager.terrain_mesh
        self.num_envs = env.num_envs
        self.num_feet = len(env.feet_indices)
        self.device = env.device

        self.ray_starts, self.ray_directions = self.pattern_cfg.pattern_func(self.pattern_cfg, self.device)
        self.ray_starts = self.ray_starts.repeat(self.num_feet, 1)
        self.ray_directions = self.ray_directions.repeat(self.num_feet, 1)

        self.ray_starts = self.ray_starts.repeat(self.num_envs, 1, 1)
        self.ray_directions = self.ray_directions.repeat(self.num_envs, 1, 1)
        feet_edge_relative_pos = to_torch(env.cfg["asset"]["feet_edge_pos"], device=env.device, requires_grad=False)
        self.num_edge_points = feet_edge_relative_pos.shape[0]
        self.feet_edge_relative_pos = (
            feet_edge_relative_pos.unsqueeze(0).unsqueeze(0)
            .expand(self.num_envs, self.num_feet, self.num_edge_points, 3)
        )
        self.total_edges = self.num_feet * self.num_edge_points

        self.feet_edge_pos = torch.zeros(self.num_envs, self.num_feet, self.num_edge_points, 3, device=env.device)
        self.feet_ground_distance = torch.zeros(self.num_envs, self.num_feet, self.num_edge_points, device=env.device)
        self.feet_contact = torch.zeros(self.num_envs, self.num_feet, self.num_edge_points, dtype=torch.bool, device=env.device)
        self.env = env
        self.sphere_geom = None

    def update(self, dt, env_ids=...):
        """Perform raycasting on the terrain.

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the ray hits. Defaults to ....
        """
        feet_pos, feet_quat = self.env.get_feet_pos_quat()
        expanded_feet_pos = feet_pos.unsqueeze(2).expand(self.num_envs, self.num_feet, self.num_edge_points, 3)
        expanded_feet_quat = feet_quat.unsqueeze(2).expand(self.num_envs, self.num_feet, self.num_edge_points, 4)
        feet_edge_pos = expanded_feet_pos.reshape(-1, 3) + quat_rotate(expanded_feet_quat.reshape(-1, 4), self.feet_edge_relative_pos.reshape(-1, 3))
        self.feet_edge_pos[env_ids] = feet_edge_pos.reshape(self.num_envs, self.num_feet, self.num_edge_points, 3)

        nearest_points = nearest_point(feet_edge_pos, self.terrain_mesh)
        dist = torch.norm(nearest_points - feet_edge_pos, dim=-1)
        self.feet_ground_distance[env_ids] = dist.view(self.num_envs, self.num_feet, self.num_edge_points)
        self.feet_contact[env_ids] = self.feet_ground_distance[env_ids] < self.env.cfg["asset"]["feet_contact_radius"]

    def get_data(self):
        foot_contact = torch.any(self.feet_contact, dim=-1)
        return foot_contact
    
    def debug_vis(self, env):
        if self.sphere_geom is None:
            self.sphere_geom = BatchWireframeSphereGeometry(
                self.num_envs * self.num_feet * self.num_edge_points, self.env.cfg["asset"]["feet_contact_radius"], 16, 16, None, color=(1, 1, 0)
            )
        feet_edge_pos = self.feet_edge_pos.view(-1, 3)
        feet_contact = self.feet_contact.view(-1)
        color_red = to_torch(np.array([1., 0., 0.])[None].repeat(feet_contact.shape[0], 0), device=self.device, requires_grad=False)
        color_green = to_torch(np.array([0., 1., 0.])[None].repeat(feet_contact.shape[0], 0), device=self.device, requires_grad=False)
        colors = torch.where(feet_contact[..., None], color_green, color_red)
        if self.env.selected_environment >= 0:
          only_render_selected_range = [self.env.selected_environment * self.total_edges, (self.env.selected_environment + 1) * self.total_edges]
          self.sphere_geom.draw(feet_edge_pos, env.gym, env.viewer, env.envs[0], only_render_selected=only_render_selected_range, custom_colors=colors)
        else:
          self.sphere_geom.draw(feet_edge_pos, env.gym, env.viewer, env.envs[0], custom_colors=colors)


def update_image(new_image, fig, im):
    # To visualize environment RGB.
    if len(new_image.shape) == 4:
        rows = cols = int(np.floor(np.sqrt(new_image.shape[0])))
        new_image = new_image[:rows**2]
        def image_grid(imgs, rows, cols):
            assert len(imgs) == rows*cols
            img = Image.fromarray(imgs[0])

            w, h = img.size
            grid = Image.new('RGB', size=(cols*w, rows*h))
            grid_w, grid_h = grid.size
            
            for i, img in enumerate(imgs):
                img = Image.fromarray(img)
                grid.paste(img, box=(i%cols*w, i//cols*h))
            return grid
        to_plot = image_grid(new_image, rows, cols)
    else:
        to_plot = new_image

    im.set_data(np.array(to_plot))
    fig.canvas.flush_events()
    fig.canvas.draw()
    plt.pause(0.001)


class GaussianSplattingRenderer():
    def __init__(self, env, scene_manager):        
        self.num_envs = env.num_envs
        self.device = env.device
        self.scene_manager = scene_manager
        self.terrain = scene_manager._terrain

        print('Loading Gaussian Splatting Renderer...')
        self.scene_path_map = {s: os.path.join(os.path.dirname(os.path.dirname(self.terrain.get_mesh(s, f).filepath)), 'splat', 'splat.ply') for s, f in self.terrain.mesh_keys}
        self.path_scene_map = {v: k for k, v in self.scene_path_map.items()}
        print(torch.cuda.device_count())
        self._gs_renderer = MultiSceneRenderer(
            list(self.scene_path_map.values()),
            renderer_gpus=[self.device],
            # renderer_gpus=[1],
            output_gpu=self.device
        )

        self.fig, self.process_image_fn, self.im = None, None, None
        self.camera_positions = torch.zeros(self.num_envs, 3, device=self.device)
        self.camera_quats_xyzw = torch.zeros(self.num_envs, 4, device=self.device)
        self.env = env
        self.renders = torch.zeros(self.num_envs, self.env.cfg["env"]["camera_params"]["cam_height"], self.env.cfg["env"]["camera_params"]["cam_width"], 3, device=self.device, dtype=torch.uint8)
        self.frustrum_geom = None
        self.axis_geom = None

        self.local_offset = torch.tensor(
            np.array(self.env.cfg["env"]["camera_params"]["cam_xyz_offset"])[None].repeat(self.num_envs, 0),
            dtype=torch.float, device=self.device, requires_grad=False
        )

    def update(self, dt, env_ids=...):
        """Render images with Gaussian splatting.

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the ray hits. Defaults to ....
        """
        cam_trans, cam_quat = self.scene_manager.get_cam_pose_world_frame()
        self.camera_positions = cam_trans
        cam_trans = cam_trans.cpu().numpy()
        cam_lin_vel, cam_ang_vel = self.scene_manager.get_cam_velocity_world_frame()
        cam_lin_vel = cam_lin_vel.cpu().numpy()
        cam_ang_vel = cam_ang_vel.cpu().numpy()

        self.camera_quats_xyzw = cam_quat
        cam_rot = vtf.SO3.from_quaternion_xyzw(cam_quat.cpu().numpy())

        # From OpenCV to OpenGL camera convention for GS renderer.
        R_x = vtf.SO3.from_x_radians(-np.pi / 2)
        R_z = vtf.SO3.from_z_radians(np.pi)
        cam_trans -= self.env.env_origins.cpu().numpy()
        # TODO: FIX THIS.
        # cam_trans = np.dot(cam_trans, R_z.as_matrix())
        # cam_trans = np.dot(cam_trans, R_x.as_matrix())
        # cam_trans += self.scene_manager.cam_offset.cpu().numpy()
        # cam_rot = R_z.inverse() @ cam_rot
        # cam_rot  = R_x.inverse() @ cam_rot

        # Batch render multiple scenes.
        c2ws = vtf.SE3.from_rotation_and_translation(cam_rot, cam_trans).as_matrix()
        scene_poses = defaultdict(list)
        scene_linear_velocities = defaultdict(list)
        scene_angular_velocities = defaultdict(list)
        for k, pose, lin_vel, ang_vel in zip(self.scene_manager.scenes, c2ws, cam_lin_vel, cam_ang_vel):
            scene_poses[self.scene_path_map[k]].append(pose)
            scene_linear_velocities[self.scene_path_map[k]].append(lin_vel)
            scene_angular_velocities[self.scene_path_map[k]].append(ang_vel)
        scene_poses = {k: np.array(v) for k, v in scene_poses.items()}
        scene_linear_velocities = {k: np.array(v) for k, v in scene_linear_velocities.items()}
        scene_angular_velocities = {k: np.array(v) for k, v in scene_angular_velocities.items()}
        start, end = 0, 0
        for k, v in scene_poses.items():
            poses = vtf.SE3.from_matrix(v)
            t = poses.translation()
            end = start + t.shape[0]
            q = poses.rotation()
            if 'stairwell' not in k:
              t = np.dot(t, R_z.as_matrix())
              t = np.dot(t, R_x.as_matrix())
              t += self.scene_manager.cam_offset.cpu().numpy()[start:end]
              q = R_z.inverse() @ q
              q  = R_x.inverse() @ q
            else:
              t += self.scene_manager.cam_offset.cpu().numpy()[start:end]
            start = end
            scene_poses[k] = vtf.SE3.from_rotation_and_translation(q, t).as_matrix()
              

        renders = self._gs_renderer.batch_render(
            scene_poses,
            focal=self.env.cfg["env"]["camera_params"]["focal_length"],
            h=self.env.cfg["env"]["camera_params"]["cam_height"],
            w=self.env.cfg["env"]["camera_params"]["cam_width"],
            camera_linear_velocity=scene_linear_velocities,
            camera_angular_velocity=scene_angular_velocities,
            minibatch=1024,
        )
        renders = torch.cat(list(renders.values()), dim=0)

        self.renders[env_ids] = renders # (255 * renders[env_ids]).to(torch.uint8)

    def get_data(self):
        return self.renders
    
    
    def debug_vis(self, env):
        if self.frustrum_geom is None:
            self.frustrum_geom = BatchWireframeFrustumGeometry(
                self.num_envs,
                0.1,
                0.2,
                self.env.cfg["env"]["camera_params"]["cam_width"],
                self.env.cfg["env"]["camera_params"]["cam_height"],
                self.env.cfg["env"]["camera_params"]["focal_length"],
                0.005,
                32)
            self.axis_geom = BatchWireframeAxisGeometry(self.num_envs, 0.25, 0.005, 32)
        if self.fig is None:
            plt.ion()
            self.fig, ax = plt.subplots()
            n = int(np.floor(np.sqrt(self.num_envs)))
            def process_image_fn(tensor, selected_env):
              if selected_env >= 0:
                return tensor[selected_env].cpu().numpy()
              else:
                return tensor.cpu().numpy()
            self.process_image_fn = process_image_fn

            if self.env.selected_environment >= 0:
                self.im = ax.imshow(np.zeros((self.env.cfg["env"]["camera_params"]["cam_height"], self.env.cfg["env"]["camera_params"]["cam_width"], 3), dtype=np.uint8))
            else:
                self.im = ax.imshow(np.zeros((self.env.cfg["env"]["camera_params"]["cam_height"] * n, self.env.cfg["env"]["camera_params"]["cam_width"] * n, 3), dtype=np.uint8))
            plt.show(block=False)

        self.frustrum_geom.draw(self.camera_positions, self.camera_quats_xyzw, env.gym, env.viewer, env.envs[0], self.env.selected_environment)
        self.axis_geom.draw(self.camera_positions, self.camera_quats_xyzw, env.gym, env.viewer, env.envs[0], only_render_selected=self.env.selected_environment)
        update_image(self.process_image_fn(self.renders, self.env.selected_environment), self.fig, self.im)


class RayCasterBaseHeight(RayCaster):
    def __init__(self, env):        
        self.attachement_pos =(0.0, 0.0, 20.0)
        self.attachement_quat = (0.0, 0.0, 0.0, 1.0)
        self.attach_yaw_only = True
        self.pattern_cfg = BaseHeightLocomotionCfg()
        self.body_attachement_name = "base"
        self.default_hit_value = 10
        # self.terrain_mesh = convert_to_wp_mesh(env.scene_manager.all_vertices_mesh, env.scene_manager.all_triangles_mesh, env.device)
        self.terrain_mesh = env.scene_manager.terrain_mesh
        self.num_envs = env.num_envs
        self.device = env.device

        self.ray_starts, self.ray_directions = self.pattern_cfg.pattern_func(self.pattern_cfg, self.device)
        self.num_rays = len(self.ray_directions)

        offset_pos = torch.tensor(list(self.attachement_pos), device=self.device)
        offset_quat = torch.tensor(list(self.attachement_quat), device=env.device)
        self.ray_directions = quat_apply(offset_quat.repeat(len(self.ray_directions), 1), self.ray_directions)
        self.ray_starts += offset_pos

        self.ray_starts = self.ray_starts.repeat(self.num_envs, 1, 1)
        self.ray_directions = self.ray_directions.repeat(self.num_envs, 1, 1)

        self.ray_hits_world = torch.zeros(self.num_envs, self.num_rays, 3, device=self.device)
        self.env = env
        self.sphere_geom = None

    def debug_vis(self, env):
        if self.sphere_geom is None:
            self.sphere_geom = BatchWireframeSphereGeometry(
                self.num_envs * self.num_rays, 0.06, 8, 8, None, color=(1, 0, 0)
            )
        self.sphere_geom.draw(self.ray_hits_world, env.gym, env.viewer, env.envs[0], only_render_selected=self.env.selected_environment)
        
class LinkHeightSensor():
    def __init__(self, env, link_names, color=(1, 0, 0), attach_yaw_only=True):        
        self.attachement_pos =(0.0, 0.0, 20.0)
        self.attachement_quat = (0.0, 0.0, 0.0, 1.0)
        direction = (0.0, 0.0, -1.0)
        self.attach_yaw_only = attach_yaw_only
        self.link_names = link_names
        self.link_indices = [env.gym.find_asset_rigid_body_index(env.robot_asset, name) for name in link_names]
        self.color = color

        self.default_hit_value = 10
        self.terrain_mesh = env.scene_manager.terrain_mesh
        self.num_envs = env.num_envs
        self.device = env.device

        self.ray_starts = torch.zeros(len(self.link_indices), 3, device=self.device)
        self.ray_directions = torch.zeros_like(self.ray_starts)
        self.ray_directions[..., :] = torch.tensor(list(direction), device=self.device)
        self.num_rays = len(self.ray_directions)

        offset_pos = torch.tensor(list(self.attachement_pos), device=self.device)
        offset_quat = torch.tensor(list(self.attachement_quat), device=env.device)
        self.ray_directions = quat_apply(offset_quat.repeat(len(self.ray_directions), 1), self.ray_directions)
        self.ray_starts += offset_pos

        self.ray_starts = self.ray_starts.repeat(self.num_envs, 1, 1)
        self.ray_directions = self.ray_directions.repeat(self.num_envs, 1, 1)

        self.ray_hits_world = torch.zeros(self.num_envs, self.num_rays, 3, device=self.device)
        self.link_heights = torch.zeros(self.num_envs, len(self.link_indices), device=self.device)
        self.env = env
        self.sphere_geom = None

    def update(self, dt, env_ids=...):
        """Perform raycasting on the terrain.

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the ray hits. Defaults to ....
        """
        pos = self.env.rigid_body_state.view(self.num_envs, self.env.num_bodies, 13)[env_ids, self.link_indices, 0:3]
        quats = self.env.rigid_body_state.view(self.num_envs, self.env.num_bodies, 13)[env_ids, self.link_indices, 3:7]

        if self.attach_yaw_only:
            ray_starts_world = quat_apply_yaw(quats, self.ray_starts[env_ids]) + pos
            ray_directions_world = self.ray_directions[env_ids]
        else:
            ray_starts_world = quat_apply(quats, self.ray_starts[env_ids]) + pos
            ray_directions_world = quat_apply(quats, self.ray_directions[env_ids])

        self.ray_hits_world[env_ids] = ray_cast(ray_starts_world, ray_directions_world, self.terrain_mesh)
        self.link_heights[env_ids] = pos[..., 2] - self.ray_hits_world[..., 2]

    def get_data(self):
        return torch.nan_to_num(self.link_heights, posinf=self.default_hit_value, neginf=self.default_hit_value)
    
    def debug_vis(self, env):
        if self.sphere_geom is None:
            self.sphere_geom = BatchWireframeSphereGeometry(
                self.num_envs * self.num_rays, 0.02, 20, 20, None, color=self.color
            )
        if self.env.selected_environment >= 0:
          only_render_selected_range = [self.env.selected_environment * self.num_rays, (self.env.selected_environment + 1) * self.num_rays]
          self.sphere_geom.draw(self.ray_hits_world, env.gym, env.viewer, env.envs[0], only_render_selected=only_render_selected_range)
        else:
          self.sphere_geom.draw(self.ray_hits_world, env.gym, env.viewer, env.envs[0])
