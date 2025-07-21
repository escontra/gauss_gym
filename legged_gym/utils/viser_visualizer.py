from typing import Dict, List, Union
import collections
import sys
import viser
from viser.extras import ViserUrdf
import yourdfpy
import trimesh
import torch
import numpy as np
import time
import cv2
import viser.transforms as vtf
import plotly.graph_objects as go

from legged_gym import utils


if sys.version_info < (3, 9):
    # Fixes importlib.resources for python < 3.9.
    # From: https://github.com/messense/typst-py/issues/12#issuecomment-1812956252
    import importlib.resources as importlib_res
    import importlib_resources
    setattr(importlib_res, "files", importlib_resources.files)
    setattr(importlib_res, "as_file", importlib_resources.as_file)


VEL_SCALE = 0.25
HEADING_SCALE = 0.2
FRAME_SCALE = 0.25
PLOT_TIME_WINDOW = 5.0

COLORS = [
    '#0022ff', '#33aa00', '#ff0011', '#ddaa00', '#cc44dd', '#0088aa',
    '#001177', '#117700', '#990022', '#885500', '#553366', '#006666',
    '#7777cc', '#999999', '#990099', '#888800', '#ff00aa', '#444444',
]


class LeggedRobotViser:
    """A robot visualizer using Viser, with the URDF attached under a /world root node."""

    global_servers: Dict[int, viser.ViserServer] = {}

    def __init__(self, env, urdf_path: str, port: int = 8080, dt: float = 1.0 / 60.0, force_dt: bool = True):
        """
        Initialize visualizer with a URDF model, loaded under a single /world node.
        
        Args:
            env: Environment instance
            urdf_path: Path to the URDF file
            port: Port number for the viser server
            dt: Desired update frequency in Hz
            force_dt: If True, force the update frequency to be dt Hz
        """
        self.env = env
        # If there is an existing server on this port, shut it down
        if port in LeggedRobotViser.global_servers:
            utils.print(f"Found existing server on port {port}, shutting it down.", color="red")
            LeggedRobotViser.global_servers.pop(port).stop()

        self.server = viser.ViserServer(port=port)
        LeggedRobotViser.global_servers[port] = self.server

        self.dt = dt
        self.force_dt = force_dt

        self._isaac_world_node = self.server.scene.add_frame("/isaac_world", show_axes=False)

        # Load URDF for both simulators
        self.urdf = yourdfpy.URDF.load(urdf_path, load_collision_meshes=True)

        # Also store mesh handles in case you want direct references
        self._gs_handle = None
        self._axes_handle = None
        self._frustrum_handle = None
        self._vel_handle = None
        self._robot_camera_handle = None
        self._contact_handles = None
        self._mesh_handle = None

        self.scene_manager = self.env.scene_manager
        self.current_rendered_env_id = 0
        self.last_rendered_env_id = -1 # set to -1 to force camera update on first render

        self.add_gui_elements()

        # Attach URDF under both world nodes
        self.isaac_urdf = ViserUrdf(
            target=self.server,
            urdf_or_path=self.urdf,
            root_node_name="/isaac_world"
        )

    def add_gui_elements(self):
        # Add simulation control buttons
        with self.server.gui.add_folder("Simulation Control"):
            self.play_pause = self.server.gui.add_checkbox(
                "Play",
                initial_value=True,
                hint="Toggle simulation play/pause"
            )
            self.step_button = self.server.gui.add_button(
                "Step",
                hint="Step the simulation forward by one frame"
            )
            self.reset_button = self.server.gui.add_button(
                "Reset",
                hint="Reset both simulators to initial state"
            )
            self.step_requested = False

            @self.reset_button.on_click
            def _(_):
                """Reset both simulators when reset button is clicked"""
                # Reset IsaacGym environment
                self.env.reset_idx(torch.tensor([self.current_rendered_env_id], device=self.env.device), time_out_buf=torch.zeros(self.env.num_envs, device=self.env.device))
                utils.print(f'[PLAY] Resetting Isaac', color="cyan")
            
            @self.step_button.on_click
            def _(_):
                if not self.play_pause.value:  # Only allow stepping when paused
                    self.step_requested = True

        with self.server.gui.add_folder("Plot Control", expand_by_default=False, order=100):
            self.joint_selection = self.server.gui.add_dropdown(
                "Select joint",
                options=['all'] + self.env.dof_names,
                initial_value='all',
                hint="Which joint to plot."
            )
            self.show_action_plot = self.server.gui.add_checkbox(
                "Show Action Plot",
                initial_value=False,
                hint="Toggle action plot visibility"
            )
            self.show_dof_pos_plot = self.server.gui.add_checkbox(
                "Show DOF Pos Plot",
                initial_value=False,
                hint="Toggle DOF pos plot visibility"
            )
            self.reward_selection = self.server.gui.add_dropdown(
                "Select Reward",
                options=['all'] + self.env.reward_names,
                initial_value='all',
                hint="Which reward to plot."
            )
            self.show_rew_plot = self.server.gui.add_checkbox(
                "Show Reward Plot",
                initial_value=False,
                hint="Toggle reward plot visibility"
            )

            self.action_plot = None
            self.dof_pos_plot = None
            self.rew_plot = None

            @self.show_action_plot.on_update
            def _(event) -> None:
                if self.show_action_plot.value:
                    # Create the action plot if it doesn't exist
                    if self.action_plot is None:
                        self.action_plot = self.server.gui.add_plotly(
                            figure=go.Figure(),
                            aspect=1.0,
                            visible=True
                        )
                else:
                    # Remove the plot if it exists
                    if self.action_plot is not None:
                        self.action_plot.remove()
                        self.action_plot = None

            @self.show_dof_pos_plot.on_update
            def _(event) -> None:
                if self.show_dof_pos_plot.value:
                    # Create the action plot if it doesn't exist
                    if self.dof_pos_plot is None:
                        self.dof_pos_plot = self.server.gui.add_plotly(
                            figure=go.Figure(),
                            aspect=1.0,
                            visible=True
                        )
                else:
                    # Remove the plot if it exists
                    if self.dof_pos_plot is not None:
                        self.dof_pos_plot.remove()
                        self.dof_pos_plot = None

            @self.show_rew_plot.on_update
            def _(event) -> None:
                if self.show_rew_plot.value:
                    # Create the action plot if it doesn't exist
                    if self.rew_plot is None:
                        self.rew_plot = self.server.gui.add_plotly(
                            figure=go.Figure(),
                            aspect=1.0,
                            visible=True
                        )
                else:
                    # Remove the plot if it exists
                    if self.rew_plot is not None:
                        self.rew_plot.remove()
                        self.rew_plot = None


        # Initialize history for plots.
        self.history_length = int(PLOT_TIME_WINDOW / self.dt)
        self.current_time = 0.0
        self.time_history = []
        self.action_history = []
        self.dof_pos_history = []
        self.rew_history = collections.defaultdict(list)

        self.setup_scene_selection()

        self.camera_viz_folder = self.server.gui.add_folder("Visualization")
        with self.camera_viz_folder:
            self.show_robot_frustum = self.server.gui.add_checkbox(
                "Show Robot Frustum",
                initial_value=False,
                hint="Toggle robot frustum visibility"
            )
            self.show_robot_velocities = self.server.gui.add_checkbox(
                "Show Robot Velocities",
                initial_value=False,
                hint="Toggle robot velocities visibility"
            )
            self.show_camera_axes = self.server.gui.add_checkbox(
                "Show Camera Axes",
                initial_value=False,
                hint="Toggle camera axes visibility"
            )
            self.show_contacts = self.server.gui.add_checkbox(
                "Show Contacts",
                initial_value=False,
                hint="Toggle contacts visibility"
            )

        self.gaussian_splatting_viz_folder = self.server.gui.add_folder("Gaussian Splatting")
        with self.gaussian_splatting_viz_folder:
            self.show_gaussian_splatting = self.server.gui.add_checkbox(
                "Show Gaussian Splat",
                initial_value=True,
                hint="Toggle gaussian splats"
            )

        self.mesh_viz_folder = self.server.gui.add_folder("Mesh Visualization")
        with self.mesh_viz_folder:
            self.show_mesh = self.server.gui.add_checkbox(
                "Show Mesh",
                initial_value=True,
                hint="Toggle robot mesh visibility"
            )

    def add_gaussians(self, env_idx: int):
        if "gs_renderer" not in self.env.sensors:
            return
        ply_renderers = self.env.sensors["gs_renderer"].get_gs_renderers()
        mesh_name = self.scene_manager.mesh_names[self.scene_manager.mesh_id_for_env_id(env_idx)]
        splat_name = '/'.join(mesh_name.split('/')[:-1])
        renderer = ply_renderers[splat_name]
        centers = renderer.means.cpu().numpy() / renderer.dataparser_scale
        colors = renderer.colors_viser.cpu().numpy()
        opacities = renderer.opacities.cpu().numpy()[..., None]
        scales = renderer.scales.cpu().numpy() / renderer.dataparser_scale
        Rs = vtf.SO3(renderer.quats.cpu().numpy()).as_matrix()
        covariances = np.einsum(
            "nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
        )

        cam_offset = self.scene_manager.cam_offset[env_idx].cpu().numpy()
        env_offset = self.env.env_origins[env_idx].cpu().numpy()

        ig_to_orig_rot = vtf.SO3.from_matrix(self.scene_manager.ig_to_orig_rot[env_idx].cpu().numpy())
        dataparser_transform = vtf.SE3.from_matrix(renderer.dataparser_transform.cpu().numpy())

        # Move from GS frame to IG frame.
        cumulative_transform = dataparser_transform.inverse()
        cumulative_transform = vtf.SE3.from_rotation_and_translation(
            vtf.SO3.identity(),
            cam_offset
        ).inverse().multiply(cumulative_transform)
        cumulative_transform = vtf.SE3.from_rotation_and_translation(
            ig_to_orig_rot,
            np.zeros(3)
        ).inverse().multiply(cumulative_transform)
        cumulative_transform = vtf.SE3.from_rotation_and_translation(
              vtf.SO3.identity(),
              env_offset
        ).multiply(cumulative_transform)

        if self._gs_handle is not None:
            self._gs_handle.remove()
        self._gs_handle = self.server.scene.add_gaussian_splats(
            "/gs",
            centers=centers,
            covariances=covariances,
            rgbs=colors,
            opacities=opacities,
            wxyz=cumulative_transform.rotation().wxyz,
            position=cumulative_transform.translation(),
        )

    def add_camera_axes(self, env_idx: int):
        mesh_idx = self.scene_manager.mesh_id_for_env_id(env_idx)
        cam_trans = self.scene_manager.cam_trans_viz[mesh_idx].cpu().numpy()
        cam_quat = self.scene_manager.cam_quat_xyzw_viz[mesh_idx].cpu().numpy()
        if self._axes_handle is None:
            self._axes_handle = self.server.scene.add_batched_axes(
                "/cam_axes",
                batched_wxyzs=np.roll(cam_quat, 1, axis=1),
                batched_positions=cam_trans,
                batched_scales=np.ones(cam_trans.shape[0]) * FRAME_SCALE,
                visible=self.show_camera_axes.value,
            )
        else:
            self._axes_handle.visible = self.show_camera_axes.value
            self._axes_handle.batched_wxyz = np.roll(cam_quat, 1, axis=1)
            self._axes_handle.batched_positions = cam_trans

    def add_mesh(self, env_idx: int):
        # Get vertices and faces from the scene manager.
        mesh_idx = self.scene_manager.mesh_id_for_env_id(env_idx)
        vertices = self.scene_manager.all_vertices[mesh_idx]
        faces = self.scene_manager.all_triangles[mesh_idx]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        if self._mesh_handle is None:
            self._mesh_handle = self.server.scene.add_mesh_simple(
                name="/terrain",
                vertices=mesh.vertices,
                faces=mesh.faces,
                color=(0.282, 0.247, 0.361),
                side='double',
                visible=self.show_mesh.value,
            )
        else:
            self._mesh_handle.visible = self.show_mesh.value
            self._mesh_handle.vertices = mesh.vertices
            self._mesh_handle.faces = mesh.faces

    def add_everything(self, env_idx: int):
        self.add_mesh(env_idx)
        self.add_gaussians(env_idx)
        self.add_camera_axes(env_idx)

    def setup_scene_selection(self):

        mesh_names = self.scene_manager.mesh_names
        sorted_indices = sorted(range(len(mesh_names)), key=lambda i: mesh_names[i])
        mesh_names = [mesh_names[i] for i in sorted_indices]

        with self.server.gui.add_folder("Scene Selection"):
            self.scene_selection = self.server.gui.add_dropdown(
                "Select Scene",
                options=mesh_names,
                initial_value=mesh_names[0],
                hint="Select which scene to visualize"
            )
        
            # Add callback for clip selection
            @self.scene_selection.on_update
            def _(event) -> None:

                for i, mesh_name in zip(sorted_indices, mesh_names):
                    if mesh_name == self.scene_selection.value:
                        possible_env_ids = self.scene_manager.env_ids_for_mesh_id(i)
                        self.current_rendered_env_id = possible_env_ids[0].item()
                        self.add_everything(self.current_rendered_env_id)
                        break

    def set_viewer_camera(self, position: Union[np.ndarray, List[float]], lookat: Union[np.ndarray, List[float]]):
        """
        Set the camera position and look-at point.
        
        Args:
            position: Camera position in world coordinates
            lookat: Point to look at in world coordinates
        """
        clients = self.server.get_clients()
        for id, client in clients.items():
            client.camera.position = position
            client.camera.look_at = lookat

    def get_camera_position_for_robot(self, env_idx: int):
        """
        Calculate camera position to look at the robot from 3m away.
        
        Args:
            env_offset: The environment offset
            root_pos: Current root position of the robot
        
        Returns:
            camera_pos: Position for the camera
            lookat_pos: Position to look at (robot position)
        """
        # Get actual robot position in world frame (root_pos is in env frame, need to transform to world frame)
        # world_pos = root_pos + env_offset
        world_pos = self.env.root_states[env_idx, :3].cpu().numpy()
        
        # Position to look at (robot position plus small height offset)
        lookat_pos = world_pos + np.array([0.0, 0.0, 0.5])
        
        # Calculate camera position 3m away at 45 degree angle
        distance = 3.0
        # camera_offset = np.array([distance/np.sqrt(2), -distance/np.sqrt(2), 1.5])  # 45 degrees in xy-plane
        camera_offset = np.array([0, -distance/np.sqrt(2), 1.5])  # 45 degrees in xy-plane
        camera_pos = world_pos + camera_offset
        
        return camera_pos, lookat_pos
    
    def set_viewer_camera(self, position: Union[np.ndarray, List[float]], lookat: Union[np.ndarray, List[float]]):
        """
        Set the camera position and look-at point.
        
        Args:
            position: Camera position in world coordinates
            lookat: Point to look at in world coordinates
        """
        clients = self.server.get_clients()
        for id, client in clients.items():
            client.camera.position = position
            client.camera.look_at = lookat

    def update_contacts(self, env_idx: int):
        foot_contact_sensor = self.env.sensors["foot_contact_sensor"]
        feet_edge_pos = foot_contact_sensor.feet_edge_pos[env_idx].cpu().numpy().reshape(-1, 3)
        feet_contact = foot_contact_sensor.feet_contact_viz[env_idx].cpu().numpy().reshape(-1)
        if self._contact_handles is None:
            contact_handles = []
            for i in range(feet_edge_pos.shape[0]):
                color = (0, 255, 0) if feet_contact[i] else (255, 0, 0)
                contact_handles.append(
                    self.server.scene.add_icosphere(
                        name=f"/contact/{i}",
                        radius=self.env.cfg["asset"]["feet_contact_radius"] + 0.005,
                        color=color,
                        position=feet_edge_pos[i],
                        side='double',
                        cast_shadow=False,
                        receive_shadow=False,
                        opacity=0.5,
                        visible=self.show_contacts.value,
                    )
                )
            self._contact_handles = contact_handles
        else:
            for i, handle in enumerate(self._contact_handles):
                handle.visible = self.show_contacts.value
                handle.position = feet_edge_pos[i]
                handle.color = (0, 255, 0) if feet_contact[i] else (255, 0, 0)

    def update_velocities(self, env_idx: int):
        robot_rot = vtf.SO3.from_quaternion_xyzw(self.env.root_states[env_idx, 3:7].cpu().numpy())
        vel_origin = self.env.root_states[env_idx, :3].cpu().numpy() + robot_rot.apply(np.array([0, 0, 0.25]))
        vel_x_scale = VEL_SCALE * self.scene_manager.velocity_command[env_idx, 0].item() / self.env.command_ranges["lin_vel"][1]
        vel_y_scale = VEL_SCALE * self.scene_manager.velocity_command[env_idx, 1].item() / self.env.command_ranges["lin_vel"][1]
        vel_x_world = robot_rot.apply(np.array([vel_x_scale, 0, 0]))
        vel_y_world = robot_rot.apply(np.array([0, vel_y_scale, 0]))
        vel_x_segment = np.stack([vel_origin, vel_origin + vel_x_world], axis=0)
        vel_y_segment = np.stack([vel_origin, vel_origin + vel_y_world], axis=0)
        heading_rot = vtf.SO3.from_rpy_radians(0., 0., self.scene_manager.heading_command[env_idx].item())
        heading_segment = np.stack([vel_origin, vel_origin + heading_rot.apply(np.array([HEADING_SCALE, 0.0, 0.0]))], axis=0)
        vel_segments = np.stack([vel_x_segment, vel_y_segment, heading_segment], axis=0)
        colors = np.array([[255, 255, 255], [255, 255, 255], [255, 255, 0]])[:, None].repeat(2, axis=1)
        if self._vel_handle is None:
            self._vel_handle = self.server.scene.add_line_segments(
                "/vel_command",
                points=vel_segments,
                colors=colors,
                visible=self.show_robot_velocities.value,
                line_width=4.0,
            )
        else:
            self._vel_handle.visible = self.show_robot_velocities.value
            self._vel_handle.points = vel_segments

    def update_camera_frustum(self, env_idx: int):

        cam_pos = self.scene_manager.renderer.camera_positions[env_idx].cpu().numpy()
        cam_quat = self.scene_manager.renderer.camera_quats_xyzw[env_idx].cpu().numpy()
        cam_quat_wxyz = np.array([cam_quat[3], cam_quat[0], cam_quat[1], cam_quat[2]])
        cam_image = self.scene_manager.renderer.renders[env_idx].cpu().numpy().transpose(1, 2, 0)

        fov = self.scene_manager.renderer.fov
        aspect = self.scene_manager.renderer.aspect
        scale = 0.1
        final_quat = cam_quat_wxyz
        body_pos = cam_pos
        rgb_image = cam_image

        if self._frustrum_handle is None:
            self._frustrum_handle = self.server.scene.add_camera_frustum(
                    "/cam",
                    fov=fov,
                    aspect=aspect,
                    scale=scale,
                    line_width=3.0,  # Thicker lines for better visibility
                    color=(0, 0, 0),  # Bright magenta color for visibility
                    wxyz=final_quat,
                    position=body_pos,
                    image=rgb_image,
                    # format='rgb',  # Use RGB format which is supported by Viser
                    format='jpeg',
                    jpeg_quality=90,
                    visible=self.show_robot_frustum.value,
                    cast_shadow=False,
                    receive_shadow=False,
                )
        else:
            self._frustrum_handle.image = rgb_image
            self._frustrum_handle.wxyz = final_quat
            self._frustrum_handle.position = body_pos
            self._frustrum_handle.visible = self.show_robot_frustum.value

        # upscale the image by a factor of 3 to be more visible
        rgb_image_upscaled = cv2.resize(rgb_image, (rgb_image.shape[1] * 3, rgb_image.shape[0] * 3))
        if self._robot_camera_handle is None:
            with self.camera_viz_folder:
                self._robot_camera_handle = self.server.gui.add_image(
                    rgb_image_upscaled,
                    label="Robot Camera",
                    format='jpeg',
                    jpeg_quality=90,
                )
        else:
            self._robot_camera_handle.image = rgb_image_upscaled
  
    def update_action_plot(self):
        """Update the action plot with the current history."""
        if self.action_plot is None:
            return

        # Convert histories to numpy arrays for easier slicing
        times = np.array(self.time_history)
        actions = np.array(self.action_history)
        
        # Make times relative to current time
        current_time = self.current_time
        relative_times = times - current_time
        
        # Create a new figure
        fig = go.Figure()
        
        if self.joint_selection.value == 'all':
          for i, joint_name in enumerate(self.env.dof_names):
              fig.add_trace(go.Scatter(
                  x=relative_times,
                  y=actions[:, i],
                  name=joint_name,
                  line=dict(color=COLORS[i % len(COLORS)]),
                  showlegend=False
              ))
        else:
          fig.add_trace(go.Scatter(
              x=relative_times,
              y=actions[:, self.env.dof_names.index(self.joint_selection.value)],
              name=self.joint_selection.value,
              line=dict(color=COLORS[0]),
              showlegend=False
          ))
        
        # Update layout
        fig.update_layout(
            title="Joint Actions",
            xaxis_title="Time (seconds ago)",
            yaxis_title="Action Value",
            xaxis=dict(
                range=[-PLOT_TIME_WINDOW, 0],  # Fixed window of last 5 seconds
                autorange=False  # Disable autoranging
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
        )
        
        # Update the plot
        self.action_plot.figure = fig

    def update_dof_pos_plot(self):
        """Update the DOF pos plot with the current history."""
        if self.dof_pos_plot is None:
            return

        # Convert histories to numpy arrays for easier slicing
        times = np.array(self.time_history)
        dof_pos = np.array(self.dof_pos_history)
        
        # Make times relative to current time
        current_time = self.current_time
        relative_times = times - current_time
        
        # Create a new figure
        fig = go.Figure()
        
        if self.joint_selection.value == 'all':
          for i, joint_name in enumerate(self.env.dof_names):
              fig.add_trace(go.Scatter(
                  x=relative_times,
                  y=dof_pos[:, i],
                  name=joint_name,
                  line=dict(color=COLORS[i % len(COLORS)]),
                  showlegend=False
              ))
        else:
          fig.add_trace(go.Scatter(
              x=relative_times,
              y=dof_pos[:, self.env.dof_names.index(self.joint_selection.value)],
              name=self.joint_selection.value,
              line=dict(color=COLORS[0]),
              showlegend=False
          ))
          dof_pos_lower = self.env.dof_pos_limits[self.env.dof_names.index(self.joint_selection.value), 0].item()
          dof_pos_upper = self.env.dof_pos_limits[self.env.dof_names.index(self.joint_selection.value), 1].item()
          fig.add_hline(y=dof_pos_upper, line=dict(color='red', width=2, dash='dash'), name='Upper Limit')
          fig.add_hline(y=dof_pos_lower, line=dict(color='red', width=2, dash='dash'), name='Lower Limit')

        # Update layout
        fig.update_layout(
            title="DOF Positions",
            xaxis_title="Time (seconds ago)",
            yaxis_title="Action Value",
            xaxis=dict(
                range=[-PLOT_TIME_WINDOW, 0],  # Fixed window of last 5 seconds
                autorange=False  # Disable autoranging
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
        )
        
        # Update the plot
        self.dof_pos_plot.figure = fig

    def update_rew_plot(self):
        """Update the reward plot with the current history."""
        if self.rew_plot is None:
            return

        # Convert histories to numpy arrays for easier slicing
        times = np.array(self.time_history)

        # Make times relative to current time
        current_time = self.current_time
        relative_times = times - current_time
        
        # Create a new figure
        fig = go.Figure()
        
        if self.reward_selection.value == 'all':
          for i, name in enumerate(self.rew_history.keys()):
              fig.add_trace(go.Scatter(
                  x=relative_times,
                  y=self.rew_history[name],
                  name=name,
                  line=dict(color=COLORS[i % len(COLORS)]),
                  showlegend=False
              ))
        else:
          fig.add_trace(go.Scatter(
              x=relative_times,
              y=self.rew_history[self.reward_selection.value],
              name=self.reward_selection.value,
              line=dict(color=COLORS[0]),
              showlegend=False
          ))

        # Update layout
        fig.update_layout(
            title="Rewards",
            xaxis_title="Time (seconds ago)",
            yaxis_title="Action Value",
            xaxis=dict(
                range=[-PLOT_TIME_WINDOW, 0],  # Fixed window of last 5 seconds
                autorange=False  # Disable autoranging
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
        )
        
        # Update the plot
        self.rew_plot.figure = fig


    def update(self, root_states: torch.Tensor, dof_pos: torch.Tensor):
        """
        Update IsaacGym viz.
        
        Args:
            root_states: (num_envs, 13) -> pos(3), quat(4), linvel(3), angvel(3)
            dof_pos: (num_envs, num_dofs)
            env_idx: Which environment to read from
        """

        env_idx = self.current_rendered_env_id

        if env_idx != self.last_rendered_env_id:
            camera_pos, lookat_pos = self.get_camera_position_for_robot(env_idx)
            self.set_viewer_camera(position=camera_pos, lookat=lookat_pos)
            self.time_history = []
            self.action_history = []
            self.dof_pos_history = []
            self.rew_history = collections.defaultdict(list)
            self.current_time = 0.0

        if self.last_rendered_env_id == -1:
          self.add_everything(env_idx)

        if self._gs_handle is not None:
            self._gs_handle.visible = self.show_gaussian_splatting.value

        if self._axes_handle is not None:
            self._axes_handle.visible = self.show_camera_axes.value

        if self._mesh_handle is not None:
            self._mesh_handle.visible = self.show_mesh.value

        self.current_time += self.dt
        self.time_history.append(self.current_time)
        self.action_history.append(self.env.actions[env_idx].cpu().numpy())
        self.dof_pos_history.append(dof_pos[env_idx].cpu().numpy())
        for name in self.env.rew_dict:
            self.rew_history[name].append(self.env.rew_dict[name][env_idx].item())

        if len(self.time_history) > self.history_length:
            self.time_history = self.time_history[-self.history_length:]
            self.action_history = self.action_history[-self.history_length:]
            self.dof_pos_history = self.dof_pos_history[-self.history_length:]
            for name in self.rew_history:
                self.rew_history[name] = self.rew_history[name][-self.history_length:]

        self.update_action_plot()
        self.update_dof_pos_plot()
        self.update_rew_plot()

        # Block until either play is true or step is requested
        while not (self.play_pause.value or self.step_requested):
            time.sleep(0.01)  # Small sleep to prevent CPU spinning
            
        root_pos = root_states[env_idx, :3].cpu().numpy()
        root_quat = root_states[env_idx, 3:7].cpu().numpy()
        dof_dict = dict(zip(self.env.dof_names, dof_pos[env_idx].cpu().numpy().tolist()))
        dof_pos_np = np.array([dof_dict[name] for name in self.isaac_urdf.get_actuated_joint_names()])
        
        # Convert quaternion from (x,y,z,w) to (w,x,y,z) for Viser
        viser_quat = np.array([root_quat[3], root_quat[0], root_quat[1], root_quat[2]])

        self.update_camera_frustum(env_idx)
        self.update_velocities(env_idx)
        self.update_contacts(env_idx)
        # Update IsaacGym visualization
        self._isaac_world_node.position = root_pos
        self._isaac_world_node.wxyz = viser_quat

        if self.force_dt:
            if not hasattr(self, 'last_time'):
                self.last_time = time.monotonic()
            else:
                dt = time.monotonic() - self.last_time
                self.last_time = time.monotonic()
                if dt < self.dt:
                    time.sleep(self.dt - dt)

        # Now let yourdfpy update the relative link transforms based on dof_pos
        with self.server.atomic():
            self.isaac_urdf.update_cfg(dof_pos_np)
        
        self.step_requested = False
        self.last_rendered_env_id = env_idx