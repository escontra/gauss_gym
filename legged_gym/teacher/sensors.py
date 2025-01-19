import torch

from isaacgym.torch_utils import quat_apply
from legged_gym.utils.math import quat_apply_yaw
from typing import Tuple, Callable
from dataclasses import dataclass
import numpy as np
import torch
import warp as wp
from isaacgym import gymapi, gymutil
import math


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
            self.verts = np.repeat(verts, num_spheres, axis=0)
        else:
            self.verts = pose.transform_points(verts)

        self.verts_tmp = np.copy(self.verts)
        self._colors = np.repeat(colors, num_spheres, axis=0)

    def vertices(self):
        return self.verts

    def colors(self):
        return self._colors

    def draw(self, positions, gym, viewer, env):
        if len(positions.shape) == 2:
            positions = positions.unsqueeze(0)
        flat_pos = positions.unsqueeze(0).repeat(self.num_lines, 1, 1, 1).view(-1, 3).cpu().numpy()
        self.verts_tmp["x"][:, 0] = self.verts["x"][:, 0] + flat_pos[:, 0]
        self.verts_tmp["x"][:, 1] = self.verts["x"][:, 1] + flat_pos[:, 0]
        self.verts_tmp["y"][:, 0] = self.verts["y"][:, 0] + flat_pos[:, 1]
        self.verts_tmp["y"][:, 1] = self.verts["y"][:, 1] + flat_pos[:, 1]
        self.verts_tmp["z"][:, 0] = self.verts["z"][:, 0] + flat_pos[:, 2]
        self.verts_tmp["z"][:, 1] = self.verts["z"][:, 1] + flat_pos[:, 2]
        gym.add_lines(viewer, env, self.num_spheres * self.num_lines, self.verts_tmp, self.colors())


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
    
class RayCaster():
    def __init__(self, env):        
        self.attachement_pos =(0.0, 0.0, 20.0)
        self.attachement_quat = (0.0, 0.0, 0.0, 1.0)
        self.attach_yaw_only = True
        self.pattern_cfg = GridPatternLocomotionCfg()
        self.body_attachement_name = "base"
        self.default_hit_value = 10
        self.terrain_mesh = convert_to_wp_mesh(env.all_vertices_mesh, env.all_triangles_mesh, env.device)
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
        self.sphere_geom.draw(self.ray_hits_world, env.gym, env.viewer, env.envs[0])
