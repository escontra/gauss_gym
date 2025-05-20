import torch
from typing import Tuple, List, Union, Dict
from pathlib import Path
import numpy as np
from plyfile import PlyData
from gsplat.rendering import rasterization
from torch import Tensor
from jaxtyping import Float
import tqdm
import argparse
import json


class BatchPLYRenderer:
    def __init__(self, ply_path: Path, device: str = "cuda"):
        """
        Initialize renderer with Gaussians loaded from a PLY file
        
        Args:
            ply_path: Path to the .ply file containing Gaussian parameters
            device: Device to run computations on ("cuda" or "cpu")
        """
        self.device = torch.device(device)
        
        # Load Gaussian parameters from PLY
        self.means, self.scales, self.quats, self.colors, self.opacities, self.sh_degree = self._load_ply_gaussians(ply_path)
        
        # Set dummy transform parameters (replace these with actual values if needed)
        # Read these values from the corresponding .json file of the same name as the splat file
        json_path = ply_path.with_suffix(".json")
        if json_path.exists():
            with open(json_path, "r") as f:
                json_data = json.load(f)
                self.dataparser_scale = json_data["scale"]
                self.dataparser_transform = torch.tensor(json_data["transform"]).float().to(self.device)
                # append a row of 0001 to it
                self.dataparser_transform = torch.cat([self.dataparser_transform, torch.tensor([[0, 0, 0, 1]], device=self.device, dtype=torch.float32)], dim=0)
        else:
            raise RuntimeError(f"Could not find JSON file for {ply_path}")

    def _load_ply_gaussians(self, ply_path: Path):
        """Load Gaussian parameters from a PLY file."""
        plydata = PlyData.read(ply_path)
        v = plydata["vertex"]
        
        # Load positions
        means = torch.from_numpy(np.stack([v["x"], v["y"], v["z"]], axis=-1)).float().to(self.device)
        
        # Load scales
        scales = torch.from_numpy(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1)).float().to(self.device)
        scales = torch.exp(scales)  # Convert from log-space
        
        # Load rotations
        quats = torch.from_numpy(np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1)).float().to(self.device)
        
        # Load spherical harmonics
        sh_components = [
            np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=-1)
        ]
        
        num_rest_coeffs = sum(1 for name in v.dtype().names if name.startswith("f_rest_"))
        num_bases = num_rest_coeffs // 3
        for i in range(num_bases):
            sh_3chan = np.zeros((len(v), 3))
            for j in range(3):
                sh_3chan[:, j] = v[f"f_rest_{i + num_bases*j}"]
            sh_components.append(sh_3chan)
        
        colors = torch.from_numpy(np.stack(sh_components, axis=1)).float().to(self.device)
        # Load opacity
        opacities = torch.from_numpy(v["opacity"]).float().to(self.device)
        opacities = torch.sigmoid(opacities)  # Convert from logit-space
        
        # Calculate SH degree
        sh_degree = int(np.sqrt(colors.shape[1]) - 1)
        
        return means, scales, quats, colors, opacities, sh_degree

    @torch.no_grad()
    def batch_render(
        self,
        c2ws: Float[Tensor, "num_cameras 4 4"],
        fl_x: float,
        fl_y: float,
        pp_x: float,
        pp_y: float,
        h: int,
        w: int,
        camera_linear_velocity: Union[Float[Tensor, "num_cameras 3"], None] = None,
        camera_angular_velocity: Union[Float[Tensor, "num_cameras 3"], None] = None,
        motion_blur_frac: float = 0.0,
        minibatch: int = 15,
        out_device: str = 'cpu',
    ) -> Tuple[Tensor, Tensor]:
        """
        Batch render a set of cameras

        Args:
            c2ws: Camera-to-world matrices (4x4 homogeneous matrices, OpenCV format)
            focal: Focal length for all cameras
            h: Image height
            w: Image width
            minibatch: Number of images to render in each batch
            device: Device to output tensors to (defaults to self.device)

        Returns:
            Tuple containing:
            - RGB images tensor of shape (num_cameras, h, w, 3)
            - Depth images tensor of shape (num_cameras, h, w, 1)
        """
        out_device = torch.device(out_device) if out_device else self.device
        
        # Apply dataset transform and scale
        # c2ws is a Bx4x4 matrix of camera-to-world matrices
        c2ws = torch.bmm(self.dataparser_transform.repeat(c2ws.shape[0], 1, 1), c2ws)
        c2ws[:, :3, 3] *= self.dataparser_scale
        # Invert camera-to-world matrices
        # convert it back to opencv format
        w2cs = torch.inverse(c2ws)
        
        # Create camera intrinsics matrix
        K = torch.tensor([
            [fl_x, 0, pp_x],
            [0, fl_y, pp_y],
            [0, 0, 1]
        ], device=self.device, dtype=torch.float32)
        
        # Initialize output tensors
        num_images = c2ws.shape[0]
        imgs_out = torch.empty((num_images, h, w, 3), device=out_device, dtype=torch.uint8)
        # depth_out = torch.empty((num_images, h, w, 1), device=out_device, dtype=torch.float32)
        
        # Render in batches
        for i in range(0, num_images, minibatch):
            batch_w2cs = w2cs[i:i + minibatch]
            # Render using gsplat
            colors, _, _ = rasterization(
                self.means,
                self.quats,
                self.scales,
                self.opacities,
                self.colors,
                batch_w2cs,
                K[None].repeat(len(batch_w2cs), 1, 1),
                w,
                h,
                radius_clip=3.0,
                sh_degree=self.sh_degree,
                render_mode="RGB",
                rasterize_mode='antialiased'
            )
            colors.clamp_(0, 1)  # Clamp colors to [0, 1]
            
            imgs_out[i:i + minibatch] = (255 * colors[..., :3].to(out_device, non_blocking=True)).to(torch.uint8)
            # depth_out[i:i + minibatch] = (depths / self.dataparser_scale).to(out_device, non_blocking=True)
        
        return imgs_out# , depth_out

def create_spiral_poses(radius: float, num_frames: int) -> torch.Tensor:
    """Create camera poses for a spiral path around the origin."""
    poses = []
    for i in range(num_frames):
        # Calculate angle and height
        theta = (i / num_frames) * 2 * np.pi
        
        # Calculate camera position
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = 0.2
        
        # Create look-at matrix
        pos = np.array([x, y, z])
        look_at = np.array([0, 0, 0])  # Look at origin
        up = np.array([0, 0, 1])  # Up vector
        
        # Calculate camera orientation
        forward = look_at - pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Create camera-to-world matrix
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = -up
        c2w[:3, 2] = forward
        c2w[:3, 3] = pos
        
        poses.append(c2w)
    
    return torch.from_numpy(np.stack(poses, axis=0)).float()

class MultiSceneRenderer:
    def __init__(self, ply_paths: List[Union[str, Path]], renderer_gpus: List[Union[int, str]], output_gpu: Union[int, str]):
        """
        Initialize multi-scene renderer that distributes rendering across multiple GPUs
        
        Args:
            ply_paths: List of paths to PLY files to be rendered
            renderer_gpus: List of GPU device IDs to distribute renderers across
            output_gpu: GPU device ID where output tensors will be placed
        """
        self.renderer_gpus = [f"cuda:{gpu}" if isinstance(gpu, int) else gpu for gpu in renderer_gpus]
        self.output_gpu = f"cuda:{output_gpu}" if isinstance(output_gpu, int) else output_gpu
        self.renderers = {}
        print(f'Rendering on GPUs: {self.renderer_gpus}, Outputting to GPU: {self.output_gpu}')
        
        # Instantiate renderers for each PLY file, distributing across GPUs
        for i, ply_path in enumerate(ply_paths):
            gpu_device = self.renderer_gpus[i % len(self.renderer_gpus)]
            print(f"Initializing renderer for {ply_path} on {gpu_device}")
            self.renderers[ply_path] = BatchPLYRenderer(Path(ply_path), device=gpu_device)
    
    @torch.no_grad()
    def batch_render(
        self,
        scene_poses: Dict[str, Float[Tensor, "num_cameras 4 4"]],
        fl_x: float,
        fl_y: float,
        pp_x: float,
        pp_y: float,
        h: int,
        w: int,
        camera_linear_velocity: Union[Dict[str, Float[Tensor, "num_cameras 3"]], None] = None,
        camera_angular_velocity: Union[Dict[str, Float[Tensor, "num_cameras 3"]], None] = None,
        motion_blur_frac: float = 0.0,
        minibatch: int = 15
    ) -> Dict[str, Tuple[Float[Tensor, "num_cameras h w 3"], Float[Tensor, "num_cameras h w 1"]]]:
        """
        Render multiple scenes across different GPUs
        
        Args:
            scene_poses: Dictionary mapping PLY paths to camera poses (c2ws)
            camera_linear_velocity: Dictionary mapping PLY paths to camera linear velocities.
            camera_angular_velocity: Dictionary mapping PLY paths to camera angular velocities.
              Angular velocity is in RPY radians per second.
            motion_blur_frac: Fraction of samples in the batch to blur over [0, 1].
            focal: Focal length for all cameras
            h: Image height
            w: Image width
            minibatch: Number of images to render in each batch
        
        Returns:
            Dictionary mapping PLY paths to (images, depth) tuples on output_gpu
        """
        results = {}
        with torch.no_grad():
          
          # Render each scene on its assigned GPU
          for ply_path, c2ws in scene_poses.items():
              if camera_linear_velocity is not None:
                cam_lin_vel = camera_linear_velocity[ply_path]
                cam_ang_vel = camera_angular_velocity[ply_path]
              if ply_path not in self.renderers:
                  raise KeyError(f"No renderer initialized for {ply_path}. Available renderers: {list(self.renderers.keys())}")
              
              # Render on the assigned GPU and move results to output GPU
              renderer = self.renderers[ply_path]
              c2ws = torch.tensor(c2ws, dtype=torch.float32, device=renderer.device)
              # imgs, depths = renderer.batch_render(c2ws, focal, h, w, minibatch, out_device=self.output_gpu)
              imgs = renderer.batch_render(
                  c2ws, fl_x, fl_y, pp_x, pp_y, h, w,
                  camera_linear_velocity=cam_lin_vel,
                  camera_angular_velocity=cam_ang_vel,
                  minibatch=minibatch, out_device=self.output_gpu)
              
              # Store results
              results[ply_path] = imgs
              
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_path", type=str, required=True, help="Path to the .ply file")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--num_frames", type=int, default=240, help="Number of frames in the spiral")
    parser.add_argument("--radius", type=float, default=10.0, help="Radius of the spiral")
    parser.add_argument("--height", type=int, default=800, help="Output image height")
    parser.add_argument("--width", type=int, default=800, help="Output image width")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--json_path", type=str, default=None, help="Path to the JSON file with camera poses")
    args = parser.parse_args()

    # Initialize renderer
    renderer = BatchPLYRenderer(Path(args.ply_path))
    
    if args.json_path: 
        print("Trying to render a dataset render...")
        json_path = "../uli/bww_stepswithfeet/processed_data_complete/transforms.json"
        
        with open(json_path, "r") as f:
            json_data = json.load(f)
        c2ws = []
        for f in json_data['frames']:
            mat = np.array(f['transform_matrix'])
            mat[:3,1] = -mat[:3,1]
            mat[:3,2] = -mat[:3,2]
            c2ws.append(mat)
        c2ws = torch.tensor(np.array(c2ws)).float().to(renderer.device)
    else:
        print("Trying to render a spiral...")
        c2ws = create_spiral_poses(args.radius, args.num_frames)
        c2ws = c2ws.to(renderer.device)

    focal = 500
    
    # Render frames
    print("Rendering frames...")
    rgb_frames, _ = renderer.batch_render(c2ws, focal, args.height, args.width)
    
    
    from moviepy.editor import ImageSequenceClip

    # Convert to video
    print("Creating video...")

    # Convert frames to uint8 if they're float32
    frames_np = [(frame.cpu().numpy() * 255).astype(np.uint8) for frame in rgb_frames]

    # Create clip from sequence of frames
    clip = ImageSequenceClip(frames_np, fps=args.fps)

    # Write video file
    clip.write_videofile(args.output, 
                        fps=args.fps,
                        codec='libx264',  # More widely supported than mp4v
                        verbose=False,     # Reduces output noise
                        logger=None)       # Prevents printing progress bar

    print(f"Video saved to {args.output}")