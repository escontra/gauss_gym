import torch
from typing import Tuple, List, Union, Dict
from pathlib import Path
import numpy as np
from plyfile import PlyData
from gsplat.rendering import rasterization
from torch import Tensor
from jaxtyping import Float
import json
from einops import rearrange, repeat

from legged_gym import utils


def exp_map_SE3(tangent_vector: Float[Tensor, "b 6"]) -> Float[Tensor, "b 3 4"]:
    """Compute the exponential map `se(3) -> SE(3)`.

    This can be used for learning pose deltas on `SE(3)`.

    Args:
        tangent_vector: A tangent vector from `se(3)`.

    Returns:
        [R|t] transformation matrices.
    """

    tangent_vector_lin = tangent_vector[:, :3].view(-1, 3, 1)
    tangent_vector_ang = tangent_vector[:, 3:].view(-1, 3, 1)

    theta = torch.linalg.norm(tangent_vector_ang, dim=1).unsqueeze(1)
    theta2 = theta**2
    theta3 = theta**3

    near_zero = theta < 1e-2
    non_zero = torch.ones(1, dtype=tangent_vector.dtype, device=tangent_vector.device)
    theta_nz = torch.where(near_zero, non_zero, theta)
    theta2_nz = torch.where(near_zero, non_zero, theta2)
    theta3_nz = torch.where(near_zero, non_zero, theta3)

    # Compute the rotation
    sine = theta.sin()
    cosine = torch.where(near_zero, 8 / (4 + theta2) - 1, theta.cos())
    sine_by_theta = torch.where(near_zero, 0.5 * cosine + 0.5, sine / theta_nz)
    one_minus_cosine_by_theta2 = torch.where(near_zero, 0.5 * sine_by_theta, (1 - cosine) / theta2_nz)
    ret = torch.zeros(tangent_vector.shape[0], 3, 4).to(dtype=tangent_vector.dtype, device=tangent_vector.device)
    ret[:, :3, :3] = one_minus_cosine_by_theta2 * tangent_vector_ang @ tangent_vector_ang.transpose(1, 2)

    ret[:, 0, 0] += cosine.view(-1)
    ret[:, 1, 1] += cosine.view(-1)
    ret[:, 2, 2] += cosine.view(-1)
    temp = sine_by_theta.view(-1, 1) * tangent_vector_ang.view(-1, 3)
    ret[:, 0, 1] -= temp[:, 2]
    ret[:, 1, 0] += temp[:, 2]
    ret[:, 0, 2] += temp[:, 1]
    ret[:, 2, 0] -= temp[:, 1]
    ret[:, 1, 2] -= temp[:, 0]
    ret[:, 2, 1] += temp[:, 0]

    # Compute the translation
    sine_by_theta = torch.where(near_zero, 1 - theta2 / 6, sine_by_theta)
    one_minus_cosine_by_theta2 = torch.where(near_zero, 0.5 - theta2 / 24, one_minus_cosine_by_theta2)
    theta_minus_sine_by_theta3_t = torch.where(near_zero, 1.0 / 6 - theta2 / 120, (theta - sine) / theta3_nz)

    ret[:, :, 3:] = sine_by_theta * tangent_vector_lin
    ret[:, :, 3:] += one_minus_cosine_by_theta2 * torch.cross(tangent_vector_ang, tangent_vector_lin, dim=1)
    ret[:, :, 3:] += theta_minus_sine_by_theta3_t * (
        tangent_vector_ang @ (tangent_vector_ang.transpose(1, 2) @ tangent_vector_lin)
    )
    ret = torch.cat([ret, torch.zeros(ret.shape[0], 1, 4, device=ret.device)], dim=1)
    ret[:, 3, 3] = 1.0 # homogeneous
    return ret


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
        self.means, self.scales, self.quats, self.colors, self.colors_viser, self.opacities, self.sh_degree = self._load_ply_gaussians(ply_path)
        
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

        SH_C0 = 0.28209479177387814
        colors_viser = torch.from_numpy(0.5 + SH_C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)).float().to(self.device)
        
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
        
        return means, scales, quats, colors, colors_viser, opacities, sh_degree

    @torch.no_grad()
    def batch_render(
        self,
        c2ws: Float[Tensor, "num_cameras 4 4"],
        fl_x: Union[float, Tensor],
        fl_y: Union[float, Tensor],
        pp_x: Union[float, Tensor],
        pp_y: Union[float, Tensor],
        h: int,
        w: int,
        camera_linear_velocity: Union[Float[Tensor, "num_cameras 3"], None] = None,
        camera_angular_velocity: Union[Float[Tensor, "num_cameras 3"], None] = None,
        motion_blur_frac: float = 0.0,
        blur_samples: int = 5,
        blur_dt: float = 0.03,
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
        num_images = c2ws.shape[0]
        if isinstance(fl_x, float):
          fl_x = torch.full((num_images,), fl_x, device=self.device)
        if isinstance(fl_y, float):
          fl_y = torch.full((num_images,), fl_y, device=self.device)
        if isinstance(pp_x, float):
          pp_x = torch.full((num_images,), pp_x, device=self.device)
        if isinstance(pp_y, float):
          pp_y = torch.full((num_images,), pp_y, device=self.device)

        # Create camera intrinsics matrix
        #     [fl_x, 0, pp_x],
        #     [0, fl_y, pp_y],
        #     [0, 0, 1]
        K = torch.eye(3, device=self.device, dtype=torch.float32)[None].repeat(num_images, 1, 1)
        K[:, 0, 0] = fl_x
        K[:, 1, 1] = fl_y
        K[:, 0, 2] = pp_x
        K[:, 1, 2] = pp_y

        # Initialize output tensors
        imgs_out = torch.empty((num_images, h, w, 3), device=out_device, dtype=torch.uint8)
        # depth_out = torch.empty((num_images, h, w, 1), device=out_device, dtype=torch.float32)
        
        # Render in batches
        for i in range(0, num_images, minibatch):
            batch_w2cs = w2cs[i:i + minibatch]
            batch_K = K[i:i + minibatch]

            original_B_size = len(batch_w2cs)
            # for simplicity either blur the whole minibatch or none. in expectation this is the same as 
            # a ratio of motion blur fraction
            blur_batch = torch.rand(1).item() < motion_blur_frac
            if blur_batch:
                # compute `blur_samples` more cameras along the velocity vectors to render from
                # negative dt because we want to move backwards in time to render blur from the past
                dts = -torch.linspace(0, blur_dt, blur_samples, device=self.device)
                # interleave the original cameras to render more
                batch_w2cs = repeat(batch_w2cs, 'batch mat1 mat2 -> (batch r) mat1 mat2', r = blur_samples)
                # grab the 6-dof velocity vectors
                batch_vels = torch.cat([camera_linear_velocity[i:i + minibatch], camera_angular_velocity[i:i + minibatch]], dim=-1) # B, 6
                # also repeat the velocity vectors for each of the blur samples
                batch_vels = repeat(batch_vels, 'batch vel -> (batch r) vel', r = blur_samples)
                # repeat the dt for each of the blur samples
                dts = repeat(dts, 'samples -> (batch samples) 1', batch = original_B_size)
                # compute the delta matrices
                batch_delta_mats = exp_map_SE3(batch_vels * dts)# (B r) 4 4
                # apply the delta matrices to the original cameras in world frame
                # this is equivalent to DT @ C2w, BUT gsplat takes in a w2c matrix so we need to do (DT @ C2w)^-1 = (C2w)^-1 @ DT^-1
                batch_w2cs = torch.bmm(batch_w2cs,torch.inverse(batch_delta_mats))
            # If this OOMs, decrease minibatch size (blur uses 5x more memory)

            # Render using gsplat
            with torch.no_grad():
              colors, _, _ = rasterization(
                  self.means,
                  self.quats,
                  self.scales,
                  self.opacities,
                  self.colors,
                  batch_w2cs,
                  batch_K,
                  w,
                  h,
                  radius_clip=3.0,
                  sh_degree=self.sh_degree,
                  render_mode="RGB",
              )
            colors.clamp_(0, 1)  # Clamp colors to [0, 1]
            if blur_batch:
                # if we blurred then re-average the colors along the blur dimension
                colors = rearrange(colors, '(b r) h w c -> b r h w c', r = blur_samples)
                colors = colors.mean(dim=1)
            assert colors.shape == (original_B_size, h, w, 3), f"Colors shape is {colors.shape} but should be {original_B_size, h, w, 3}"
            
            imgs_out[i:i + minibatch] = (255 * colors[..., :3].to(out_device, non_blocking=True)).to(torch.uint8)
            # depth_out[i:i + minibatch] = (depths / self.dataparser_scale).to(out_device, non_blocking=True)
        
        return imgs_out# , depth_out


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
        utils.print(f'Rendering on GPUs: {self.renderer_gpus}, Outputting to GPU: {self.output_gpu}', color='cyan')
        
        # Instantiate renderers for each PLY file, distributing across GPUs
        for i, ply_path in enumerate(ply_paths):
            gpu_device = self.renderer_gpus[i % len(self.renderer_gpus)]
            utils.print(f"Initializing renderer for {ply_path} on {gpu_device}", color='cyan')
            self.renderers[ply_path] = BatchPLYRenderer(Path(ply_path), device=gpu_device)
    
    @torch.no_grad()
    def batch_render(
        self,
        scene_poses: Dict[str, Float[Tensor, "num_cameras 4 4"]],
        fl_x: Union[float, Dict[str, float]],
        fl_y: Union[float, Dict[str, float]],
        pp_x: Union[float, Dict[str, float]],
        pp_y: Union[float, Dict[str, float]],
        h: int,
        w: int,
        camera_linear_velocity: Union[Dict[str, Float[Tensor, "num_cameras 3"]], None] = None,
        camera_angular_velocity: Union[Dict[str, Float[Tensor, "num_cameras 3"]], None] = None,
        motion_blur_frac: float = 0.0,
        blur_samples: int = 5,
        blur_dt: float = 0.03,
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
              renderer = self.renderers[ply_path]
              if camera_linear_velocity is not None:
                cam_lin_vel = camera_linear_velocity[ply_path].to(renderer.device)
                cam_ang_vel = camera_angular_velocity[ply_path].to(renderer.device)
              if ply_path not in self.renderers:
                  raise KeyError(f"No renderer initialized for {ply_path}. Available renderers: {list(self.renderers.keys())}")

              if isinstance(fl_x, dict):
                curr_fl_x = fl_x[ply_path]
              else:
                curr_fl_x = fl_x
              if isinstance(fl_y, dict):
                curr_fl_y = fl_y[ply_path]
              else:
                curr_fl_y = fl_y
              if isinstance(pp_x, dict):
                curr_pp_x = pp_x[ply_path]
              else:
                curr_pp_x = pp_x
              if isinstance(pp_y, dict):
                curr_pp_y = pp_y[ply_path]
              else:
                curr_pp_y = pp_y

              # Render on the assigned GPU and move results to output GPU
              c2ws = c2ws.to(renderer.device)
              imgs = renderer.batch_render(
                  c2ws, curr_fl_x, curr_fl_y, curr_pp_x, curr_pp_y, h, w,
                  camera_linear_velocity=cam_lin_vel,
                  camera_angular_velocity=cam_ang_vel,
                  motion_blur_frac=motion_blur_frac,
                  blur_samples=blur_samples, blur_dt=blur_dt,
                  minibatch=minibatch, out_device=self.output_gpu)
              
              # Store results
              results[ply_path] = imgs
              
        return results
