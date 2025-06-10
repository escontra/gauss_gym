import warp as wp
import torch
import numpy as np

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
