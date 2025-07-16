import viser
from viser.extras import ViserUrdf
import yourdfpy
from typing import Dict, List, Union, Optional, Tuple
import trimesh
import torch
import numpy as np
import time
import cv2
import viser.transforms as vtf


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
            print(f"Found existing server on port {port}, shutting it down.")
            LeggedRobotViser.global_servers.pop(port).stop()

        self.server = viser.ViserServer(port=port)
        LeggedRobotViser.global_servers[port] = self.server

        self.dt = dt
        self.force_dt = force_dt

        self._isaac_world_node = self.server.scene.add_frame("/isaac_world", show_axes=False)

        # Load URDF for both simulators
        self.urdf = yourdfpy.URDF.load(urdf_path, load_collision_meshes=True)


        # Also store mesh handles in case you want direct references
        self._mesh_handles = {}
        self._gs_handle = None


        self.scene_manager = self.env.scene_manager
        self.current_rendered_env_id = 0
        self.last_rendered_env_id = -1 # set to -1 to force camera update on first render

        self.add_terrain_meshes()
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
                print(f'[PLAY] Resetting Isaac')
            
            @self.step_button.on_click
            def _(_):
                if not self.play_pause.value:  # Only allow stepping when paused
                    self.step_requested = True

        self.setup_scene_selection()
            

        self.camera_viz_folder = self.server.gui.add_folder("Camera Visualization")
        with self.camera_viz_folder:
            self.show_robot_frustum = self.server.gui.add_checkbox(
                "Show Robot Frustum",
                initial_value=False,
                hint="Toggle robot frustum visibility"
            )

        self.gaussian_splatting_viz_folder = self.server.gui.add_folder("Gaussian Splatting")
        with self.gaussian_splatting_viz_folder:
            self.show_gaussian_splatting = self.server.gui.add_checkbox(
                "Show Gaussian Splat",
                initial_value=True,
                hint="Toggle robot frustum visibility"
            )

        self.mesh_viz_folder = self.server.gui.add_folder("Mesh Visualization")
        with self.mesh_viz_folder:
            self.show_mesh = self.server.gui.add_checkbox(
                "Show Mesh",
                initial_value=True,
                hint="Toggle robot frustum visibility"
            )

    def add_terrain_meshes(self):
        self.add_mesh(
            "terrain",
            self.scene_manager.all_vertices_mesh,
            self.scene_manager.all_triangles_mesh,
            color=(0.282, 0.247, 0.361),
        )

    def add_gaussians(self, mesh_name: str, env_idx: int):
        if "gs_renderer" not in self.env.sensors:
            return
        ply_renderers = self.env.sensors["gs_renderer"].get_gs_renderers()
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

        self._gs_handle = self.server.scene.add_gaussian_splats(
            "gs",
            centers=centers,
            covariances=covariances,
            rgbs=colors,
            opacities=opacities,
            wxyz=cumulative_transform.rotation().wxyz,
            position=cumulative_transform.translation(),
        )

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
                        self.current_rendered_env_id = possible_env_ids[0]
                        self.add_gaussians(self.scene_selection.value, self.current_rendered_env_id)
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

    def get_camera_position_for_robot(self, root_pos):
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
        world_pos = root_pos
        
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

    
    def add_mesh(self, 
                name: str,
                vertices: np.ndarray,
                faces: np.ndarray,
                color: Union[Tuple[float, float, float], List[float]] = (1.0, 0.5, 0.5),
                transform: Optional[np.ndarray] = None):
        """
        Add a mesh to the scene under root "/".  (You can also attach to "/world" if you wish.)
        """
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        if transform is not None:
            mesh.apply_transform(transform)
        
        handle = self.server.scene.add_mesh_simple(
            name,
            mesh.vertices,
            mesh.faces,
            color=color,
            side='double',
            # light blue colour
        )
        self._mesh_handles[name.lstrip('/')] = handle
        return handle
    
    def update_camera_frustum(self):

        env_idx = self.current_rendered_env_id

        cam_pos = self.scene_manager.renderer.camera_positions[env_idx].cpu().numpy()
        cam_quat = self.scene_manager.renderer.camera_quats_xyzw[env_idx].cpu().numpy()
        cam_quat_wxyz = np.array([cam_quat[3], cam_quat[0], cam_quat[1], cam_quat[2]])
        cam_image = self.scene_manager.renderer.renders[env_idx].cpu().numpy().transpose(1, 2, 0)

        fov = self.scene_manager.renderer.fov
        aspect = self.scene_manager.renderer.aspect
        scale = 0.15
        final_quat = cam_quat_wxyz
        body_pos = cam_pos
        rgb_image = cam_image

        self.server.scene.add_camera_frustum(
                f"/cam",
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
            )

        # upscale the image by a factor of 3 to be more visible
        rgb_image_upscaled = cv2.resize(rgb_image, (rgb_image.shape[1] * 3, rgb_image.shape[0] * 3))
        if not hasattr(self, 'robot_camera_handle'):
            with self.camera_viz_folder:
                self.robot_camera_handle = self.server.gui.add_image(
                    rgb_image_upscaled,
                    label="Robot Camera",
                    format='jpeg',
                    jpeg_quality=90,
                )
        else:
            self.robot_camera_handle.image = rgb_image_upscaled

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

            camera_pos, lookat_pos = self.get_camera_position_for_robot(
                # self.scene_manager.env_origins[env_idx],
                root_states[env_idx, :3].cpu().numpy()
            )
            self.set_viewer_camera(position=camera_pos, lookat=lookat_pos)

        if self._gs_handle is not None:
            self._gs_handle.visible = self.show_gaussian_splatting.value

        for _, v in self._mesh_handles.items():
            v.visible = self.show_mesh.value

        # Block until either play is true or step is requested
        while not (self.play_pause.value or self.step_requested):
            time.sleep(0.01)  # Small sleep to prevent CPU spinning
            
        root_pos = root_states[env_idx, :3].cpu().numpy()
        root_quat = root_states[env_idx, 3:7].cpu().numpy()
        dof_dict = dict(zip(self.env.dof_names, dof_pos[env_idx].cpu().numpy().tolist()))
        dof_pos_np = np.array([dof_dict[name] for name in self.isaac_urdf.get_actuated_joint_names()])
        
        # Convert quaternion from (x,y,z,w) to (w,x,y,z) for Viser
        viser_quat = np.array([root_quat[3], root_quat[0], root_quat[1], root_quat[2]])

        self.update_camera_frustum()
        # Update IsaacGym visualization
        self._isaac_world_node.position = root_pos
        self._isaac_world_node.wxyz = viser_quat
        # self._mujoco_world_node.position = root_pos
        # self._mujoco_world_node.wxyz = viser_quat

        if self.force_dt:
            if not hasattr(self, 'last_time'):
                self.last_time = time.monotonic()
            else:
                dt = time.monotonic() - self.last_time
                self.last_time = time.monotonic()
                if dt < self.dt:
                    # print(f'sleeping for {desired_dt - dt} seconds')
                    time.sleep(self.dt - dt)

        # Now let yourdfpy update the relative link transforms based on dof_pos
        with self.server.atomic():
            self.isaac_urdf.update_cfg(dof_pos_np)
        
        self.step_requested = False
        self.last_rendered_env_id = env_idx