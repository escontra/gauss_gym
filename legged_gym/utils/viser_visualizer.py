import viser
from viser.extras import ViserUrdf
import yourdfpy
from typing import Dict, List, Union, Optional, Tuple
import trimesh
import torch
import numpy as np
import time


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

        self._isaac_world_node = self.server.scene.add_frame("/isaac_world", show_axes=False)

        # Load URDF for both simulators
        self.urdf = yourdfpy.URDF.load(urdf_path, load_collision_meshes=True)

        # Attach URDF under both world nodes
        self.isaac_urdf = ViserUrdf(
            target=self.server,
            urdf_or_path=self.urdf,
            root_node_name="/isaac_world"
        )


        # Also store mesh handles in case you want direct references
        self._mesh_handles = {}

        @self.reset_button.on_click
        def _(_):
            """Reset both simulators when reset button is clicked"""
            # Reset IsaacGym environment
            self.env.reset_idx(torch.tensor([0], device=self.env.device), time_out_buf=torch.zeros(self.env.num_envs, device=self.env.device))
            print(f'[PLAY] Resetting Isaac')
        
        @self.step_button.on_click
        def _(_):
            if not self.play_pause.value:  # Only allow stepping when paused
                self.step_requested = True
    

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

    def get_camera_position_for_robot(self, env_offset, root_pos):
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
        world_pos = root_pos + env_offset
        
        # Position to look at (robot position plus small height offset)
        lookat_pos = world_pos + np.array([0.0, 0.0, 0.5])
        
        # Calculate camera position 3m away at 45 degree angle
        distance = 3.0
        # camera_offset = np.array([distance/np.sqrt(2), -distance/np.sqrt(2), 1.5])  # 45 degrees in xy-plane
        camera_offset = np.array([0, -distance/np.sqrt(2), 1.5])  # 45 degrees in xy-plane
        camera_pos = world_pos + camera_offset
        
        return camera_pos, lookat_pos
    
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

    def update(self, root_states: torch.Tensor, dof_pos: torch.Tensor, env_idx: int = 0):
        """
        Update IsaacGym viz.
        
        Args:
            root_states: (num_envs, 13) -> pos(3), quat(4), linvel(3), angvel(3)
            dof_pos: (num_envs, num_dofs)
            env_idx: Which environment to read from
        """
        # Block until either play is true or step is requested
        while not (self.play_pause.value or self.step_requested):
            time.sleep(0.01)  # Small sleep to prevent CPU spinning
            
        root_pos = root_states[env_idx, :3].cpu().numpy()
        root_quat = root_states[env_idx, 3:7].cpu().numpy()
        dof_pos_np = dof_pos[env_idx].cpu().numpy()
        
        # Convert quaternion from (x,y,z,w) to (w,x,y,z) for Viser
        viser_quat = np.array([root_quat[3], root_quat[0], root_quat[1], root_quat[2]])

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