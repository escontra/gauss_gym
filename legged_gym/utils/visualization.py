import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from isaacgym import gymapi, gymutil

from legged_gym.utils.math import quat_apply


def plot_occupancy_grid(env, env_id, occupancy_grids, titles):
    heights = env.sensors["raycast_grid"].ray_starts.clone()

    fig = plt.figure(figsize=(10, 8))

    for i, (occupancy_grid, title) in enumerate(zip(occupancy_grids, titles)):
      heights = env.sensors["raycast_grid"].ray_starts.clone()
      first_nonzero = torch.argmax(occupancy_grid.to(torch.int32), dim=-1)  # Find first non-zero in each row
      heights[..., -1] = first_nonzero
      heights = heights[env_id].reshape(-1, 3)
    
      heights_np = heights.cpu().numpy()
      x = heights_np[:, 0]
      y = heights_np[:, 1]
      z = heights_np[:, 2]

      ax = fig.add_subplot(len(occupancy_grids), 1, i+1, projection='3d')
      ax.scatter(x, y, z, c=z, cmap='viridis', s=50, vmin=0, vmax=occupancy_grid.shape[-1])
      ax.set_xlim([x.min(), x.max()])
      ax.set_ylim([y.min(), y.max()])
      ax.set_zlim([0, occupancy_grid.shape[-1]])
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')
      ax.set_title(title)
      ax.set_box_aspect([1, 1, 1])
    
    plt.show()


def update_image(new_image, fig, im):

    # Convert from channels first to channels last format
    if len(new_image.shape) == 4:
        new_image = new_image.transpose(0, 2, 3, 1)
    elif len(new_image.shape) == 3:
        new_image = new_image.transpose(1, 2, 0)
    else:
        raise ValueError(f'Invalid image shape: {new_image.shape}')

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
    def __init__(self, num_frustums, near, far, image_width, image_height, fl_x, fl_y, thickness, num_lines_per_edge=32):
        self.num_frustums = num_frustums
        
        # Derive horizontal fov from focal length and image width
        fov_x = 2 * np.arctan(image_width / (2 * fl_x))
        fov_y = 2 * np.arctan(image_height / (2 * fl_y))
        
        # Compute half widths/heights at near and far planes
        half_width_near = near * np.tan(fov_x / 2)
        half_height_near = near * np.tan(fov_y / 2)
        half_width_far = far * np.tan(fov_x / 2)
        half_height_far = far * np.tan(fov_y / 2)
        
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
