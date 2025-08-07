from pathlib import Path
import zarr
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
import torch
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
from PIL import Image
import cv2
from scipy.ndimage import grey_dilation, grey_erosion
import matplotlib.pyplot as plt
import tqdm
import open3d as o3d
import viser.transforms as vtf
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp
from pytictac import Timer

TOLERANCE = 100 # 20


def pq_to_se3(p, q):
    se3 = np.eye(4, dtype=np.float32)
    try:
        se3[:3, :3] = R.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
        se3[:3, 3] = [p["x"], p["y"], p["z"]]
    except:
        se3[:3, :3] = R.from_quat(q).as_matrix()
        se3[:3, 3] = p
    return se3


class FastGetClosestTf:
    def __init__(self, odom: zarr.Group, return_se3: bool = False):
        self.odom = odom
        self.timestamps = odom["timestamp"][:]
        self.pose_pos = odom["pose_pos"][:]
        self.pose_orien = odom["pose_orien"][:]
        self.return_se3 = return_se3

    def __call__(self, timestamp: float, interpolate: bool = False) -> np.ndarray:
        assert interpolate is False, "Interpolation not implemented yet"
        idx = np.argmin(np.abs(self.timestamps - timestamp))

        # Handle boundary cases
        if idx == 0 or idx == len(self.timestamps) - 1 or self.timestamps[idx] == timestamp:
            print(f"Requested timestamp {timestamp} is at border of available times or exact match.")
            p = self.pose_pos[idx]
            q = self.pose_orien[idx]
            if self.return_se3:
                return pq_to_se3(p, q)
            return p, q

        # Normal case: determine which two points to interpolate between
        if timestamp <= self.timestamps[idx]:
            # Interpolate between previous and current
            idx1, idx2 = idx - 1, idx
        else:
            # Interpolate between current and next
            idx1, idx2 = idx, idx + 1

        # Get the two poses for interpolation
        t1, t2 = self.timestamps[idx1], self.timestamps[idx2]
        pos1, pos2 = self.pose_pos[idx1], self.pose_pos[idx2]
        quat1, quat2 = self.pose_orien[idx1], self.pose_orien[idx2]

        # Create timestamps array for interpolation
        timestamps = np.array([t1, t2])
        target_time = np.array([timestamp])

        # Linear interpolation for position
        positions = np.array([pos1, pos2])
        translation_interpolator = interp1d(
            timestamps, positions, kind="linear", axis=0, bounds_error=False, fill_value=(pos1, pos2)
        )
        interpolated_position = translation_interpolator(target_time)[0]

        # SLERP interpolation for rotation
        rotations = R.from_quat([quat1, quat2])
        slerp_interpolator = Slerp(timestamps, rotations)
        interpolated_rotation = slerp_interpolator(target_time)
        interpolated_quat = interpolated_rotation.as_quat()[0]

        if self.return_se3:
            return pq_to_se3(interpolated_position, interpolated_quat)
        return interpolated_position, interpolated_quat

        return p, q


# class FastGetClosestTf:
#     def __init__(self, odom: zarr.Group, return_se3: bool = False):
#         self.odom = odom
#         self.timestamps = odom["timestamp"][:]
#         self.pose_pos = odom["pose_pos"][:]
#         self.pose_orien = odom["pose_orien"][:]
#         self.return_se3 = return_se3

#     def __call__(self, timestamp: float, interpolate: bool = False) -> np.ndarray:
#         assert interpolate is False, "Interpolation not implemented yet"
#         idx = np.argmin(np.abs(self.timestamps - timestamp))
#         p = self.pose_pos[idx]
#         q = self.pose_orien[idx]
#         if self.return_se3:
#             return pq_to_se3(p, q)
#         return p, q


def attrs_to_se3(attrs):
    se3 = np.eye(4, dtype=np.float32)
    q, t = attrs["transform"]["rotation"], attrs["transform"]["translation"]
    se3[:3, :3] = R.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    se3[:3, 3] = [t["x"], t["y"], t["z"]]
    return se3


def project_lidar_to_camera(
    lidar_points, K, lidar_to_camera_transform, image_width, image_height, D=None, distortion_model="pinhole"
):
    """Project LiDAR points onto camera image plane with distortion correction (OpenCV)"""
    lidar_points_homo = np.hstack([lidar_points, np.ones((lidar_points.shape[0], 1))])
    camera_points_homo = (lidar_to_camera_transform @ lidar_points_homo.T).T
    camera_points = camera_points_homo[:, :3]

    # Project to image plane
    if distortion_model == "pinhole":
        image_points = (K @ camera_points.T).T
        image_points[:, 0] /= image_points[:, 2]
        image_points[:, 1] /= image_points[:, 2]

    elif distortion_model == "radtan":
        # OpenCV expects points in shape (N, 1, 3)
        objectPoints = camera_points.reshape(-1, 1, 3).astype(np.float32)
        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.zeros((3, 1), dtype=np.float32)
        image_points, _ = cv2.projectPoints(objectPoints, rvec, tvec, K, D)
        image_points = image_points.reshape(-1, 2)

    elif distortion_model == "equidistant":
        # Undistorted (pinhole) model
        objectPoints = camera_points.reshape(-1, 1, 3).astype(np.float32)
        rvec = np.zeros((3, 1), dtype=np.float32)
        tvec = np.zeros((3, 1), dtype=np.float32)

        image_points, _ = cv2.fisheye.projectPoints(
            objectPoints,
            rvec,
            tvec,
            np.asarray(K, dtype=np.float64).reshape(3, 3),
            np.asarray(D, dtype=np.float64).reshape(-1, 1) if D is not None else None,
        )
        image_points = image_points.reshape(-1, 2)

    else:
        raise ValueError(f"Unsupported distortion model: {distortion_model}")

    # Filter points within image bounds
    valid_pixels = (
        (image_points[:, 0] >= 0)
        & (image_points[:, 0] < image_width)
        & (image_points[:, 1] >= 0)
        & (image_points[:, 1] < image_height)
        & (camera_points[:, 2] > 0)
    )

    mapping_idx = np.arange(len(image_points))[valid_pixels]

    valid_image_points = image_points[valid_pixels]
    valid_depths = camera_points[valid_pixels, 2]

    # Create depth image
    depth_image = np.full((image_height, image_width), -1, dtype=np.float32)

    mapping_image = np.full((image_height, image_width), -1, dtype=np.int32)

    if len(valid_image_points) > 0:
        pixel_coords = valid_image_points[:, :2].astype(int)
        for i, (x, y) in enumerate(pixel_coords):
            if 0 <= x < image_width and 0 <= y < image_height:
                # Use closest depth if multiple points project to same pixel
                if depth_image[y, x] == -1 or valid_depths[i] < depth_image[y, x]:
                    depth_image[y, x] = valid_depths[i]
                    mapping_image[y, x] = mapping_idx[i]

    return depth_image, mapping_image


def filter_lidar(mission_root, lidar_tags, image_tags, mission_folder):
    visu_3d = False
    visu = False
    add_to_zarr = False
    visu_pcd = True
    ODOM_TAG = "dlio_map_odometry"

    get_closest_tf = FastGetClosestTf(mission_root[ODOM_TAG], return_se3=True)

    pcd = o3d.geometry.PointCloud()

    base_to_box_base = pq_to_se3(
        mission_root["tf"].attrs["tf"]["box_base"]["translation"],
        mission_root["tf"].attrs["tf"]["box_base"]["rotation"],
    )

    # prefetch image timestamps
    image_timestamps = {}
    for image_tag in image_tags:
        image_timestamps[image_tag] = mission_root[image_tag]["timestamp"][:]

    for lidar_tag in lidar_tags:
        bar = tqdm.tqdm(range(0, mission_root[lidar_tag]["sequence_id"].shape[0]))

        # prefetch lidar data
        lidar_timestamps = mission_root[lidar_tag]["timestamp"][:]
        lidar_points_pre = mission_root[lidar_tag]["points"][:, :]
        valid_points = mission_root[lidar_tag]["valid"][:, 0]

        for lidar_id in bar:
            lidar_timestamp = lidar_timestamps[lidar_id]
            lidar_points = lidar_points_pre[lidar_id, : valid_points[lidar_id]]
            lidar_colors = np.zeros_like(lidar_points)
            lidar_points_mask = np.zeros(lidar_points.shape[0], dtype=bool)

            tf_t_lidar = get_closest_tf(lidar_timestamp)

            image_idx_lookup = {}
            for image_tag in image_tags:
                # find closest image based on timestamp
                idx = np.argmin(np.abs(image_timestamps[image_tag] - lidar_timestamp))
                image_idx_lookup[image_tag] = idx

            for image_tag, idx in image_idx_lookup.items():
                # TODO check if it works under motion
                print(f"Motion compensation and tf lookup interpolation not verified!!")

                if ODOM_TAG == "dlio_map_odometry":
                  dlio_world_to_hesai = tf_t_lidar  # FYI
                  odom_to_box_base = dlio_world_to_hesai @ attrs_to_se3(
                                  mission_root["hesai_points_undistorted"].attrs
                              )  # hesai to box_base
                  tf_t_lidar = odom_to_box_base @ np.linalg.inv(base_to_box_base)  # box_base to box_base

                image_timestamp = image_timestamps[image_tag][idx]
                tf_t_camera = get_closest_tf(image_timestamp)
                # compute relate pointcloud motion
                t1_t2_motion = tf_t_camera @ np.linalg.inv(tf_t_lidar)
                box_base_image = attrs_to_se3(mission_root[image_tag].attrs)
                box_base_lidar = attrs_to_se3(mission_root[lidar_tag].attrs)

                # if ODOM_TAG == "anymal_state_odometry":
                lidar_to_box_base = attrs_to_se3(mission_root[lidar_tag].attrs)
                box_base_to_lidar = np.linalg.inv(lidar_to_box_base)
                odom_to_lidar = tf_t_lidar @ base_to_box_base @ box_base_to_lidar
                # elif ODOM_TAG == "dlio_map_odometry":
                #   odom_to_lidar = np.linalg.inv(tf_t_lidar)

                # get translation from lidar_timestamp to image timestamp
                box_lidar_image = np.linalg.inv(box_base_lidar @ np.linalg.inv(box_base_image))
                lidar_to_camera = t1_t2_motion @ box_lidar_image

                K = mission_root[image_tag].attrs["camera_info"]["K"]
                D = mission_root[image_tag].attrs["camera_info"]["D"]
                W = mission_root[image_tag].attrs["camera_info"]["width"]
                H = mission_root[image_tag].attrs["camera_info"]["height"]
                distortion_model = mission_root[image_tag].attrs["camera_info"]["distortion_model"]

                # with Timer('project_lidar_to_camera'):
                depth_image, mapping_image = project_lidar_to_camera(
                    lidar_points.copy(), K, lidar_to_camera.copy(), W, H, D, distortion_model=distortion_model
                )

                # with Timer('load_images'):
                mask_image = Image.open(mission_folder / "images" / (image_tag + "_mask") / f"{idx:06d}.png")
                mask_image = np.array(mask_image).astype(bool)
                # 0 == dynamic, 1 == not dynamic
                mask_image = grey_erosion(mask_image, size=(TOLERANCE, TOLERANCE))


                # with Timer('valid ray stuff'):
                valid_rays = np.unique(mapping_image[mask_image])
                valid_rays = valid_rays[valid_rays >= 0]  # Remove -1 values
                lidar_points_mask[valid_rays] = True

                # with Timer('load rgb stuff'):
                rgb_image = Image.open(mission_folder / "images" / image_tag / f"{idx:06d}.jpeg")
                rgb_image = np.array(rgb_image)
                
                # with Timer('color stuff 1'):
                ray_indices = mapping_image[mask_image]

                # with Timer('color stuff 2'):
                rgb_image = torch.from_numpy(rgb_image).to('cuda:0')
                mask_image = torch.from_numpy(mask_image).to('cuda:0')
                valid_colors = rgb_image[mask_image] / 255.0
                
                # with Timer('color stuff 3'):
                lidar_colors = torch.from_numpy(lidar_colors).to('cuda:0')
                ray_indices = torch.from_numpy(ray_indices).to('cuda:0')
                lidar_colors[ray_indices] = valid_colors
                lidar_colors = lidar_colors.cpu().numpy()

                # with Timer('transform stuff'):
                lidar_odom_vtf = vtf.SE3.from_matrix(odom_to_lidar)
                lidar_points_transformed = lidar_odom_vtf.apply(lidar_points[lidar_points_mask])

                # with Timer('pcf stuff'):
                pcd_keep = o3d.geometry.PointCloud()
                pcd_keep.points = o3d.utility.Vector3dVector(lidar_points_transformed)
                pcd_keep.colors = o3d.utility.Vector3dVector(lidar_colors[lidar_points_mask])
                pcd += pcd_keep

                if bar.n % 50 == 0:
                  pcd = pcd.voxel_down_sample(voxel_size=0.02)
                  pcd = pcd.remove_duplicated_points()

                if visu_3d:

                    print(f"Visualizing lidar points for lidar_id {lidar_id} and image_tag {image_tag}")
                    # Point cloud with mask (e.g. non-human points) - red, larger, opaque
                    pcd_keep.paint_uniform_color([0, 1, 0])  # green

                    # Original point cloud - green, smaller, semi-transparent
                    pcd_removed = o3d.geometry.PointCloud()
                    pcd_removed.points = o3d.utility.Vector3dVector(lidar_points[~lidar_points_mask])
                    pcd_removed.paint_uniform_color([1, 0, 0])  # red

                    vis = o3d.visualization.Visualizer()
                    vis.create_window(window_name="Lidar Points", width=920, height=580)

                    # Add original point cloud first (smaller, semi-transparent green)
                    vis.add_geometry(pcd_removed)
                    vis.add_geometry(pcd_keep)
                    vis.add_geometry(pcd)
                    vis.run()
                    vis.destroy_window()

                if bar.n % 50 == 0 and visu_pcd:
                    vis = o3d.visualization.Visualizer()
                    vis.create_window(window_name="Lidar Points", width=920, height=580)
                    # Add original point cloud first (smaller, semi-transparent green)
                    vis.add_geometry(pcd)
                    vis.run()
                    vis.destroy_window()

                if visu:
                    rgb_image = Image.open(mission_folder / "images" / image_tag / f"{idx:06d}.jpeg")

                    rgb_image = np.array(rgb_image)

                    # Normalize depth image for visualization - max range 10m
                    depth_normalized = (depth_image.clip(0, 10.0) / 10.0 * 255).astype(np.uint8)

                    # Dilate the depth image to increase pixel width to 3
                    depth_normalized = grey_dilation(depth_normalized, size=(3, 3))

                    # rgb_image H,W,3
                    alpha = 0

                    cmap = plt.get_cmap("turbo").reversed()
                    color_depth = cmap(depth_normalized)  # H,W,4

                    # Set alpha to 0 where depth is 0
                    color_depth[..., 3] = np.where(depth_normalized == 0, 0, color_depth[..., 3])

                    # Convert color_depth from float [0,1] to uint8 [0,255] and remove alpha channel
                    color_depth_rgb = (color_depth[..., :3] * 255).astype(np.uint8)

                    # Use alpha channel for blending: where alpha==0, keep rgb_image pixel
                    alpha_mask = color_depth[..., 3][..., None]
                    overlay = (alpha * rgb_image + (1 - alpha) * color_depth_rgb).astype(np.uint8)
                    overlay = np.where(alpha_mask == 0, rgb_image, overlay)

                    # Convert overlay to BGR for cv2 if needed
                    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"depth_overlay_{lidar_id}_{image_tag}.png", overlay_bgr)

            if add_to_zarr:
                # Add the filtered lidar points to the zarr store
                lidar_points_filtered = mission_root[lidar_tag]["points"][lidar_id][:].copy()
                nr_points = lidar_points_mask.sum()

                lidar_points_filtered[:nr_points] = lidar_points[lidar_points_mask]
                lidar_points_filtered[nr_points:] = 0.0  # Fill the rest with

                if f"{lidar_tag}_filtered" not in mission_root:
                    # overwrite existing sub group
                    zarr_group = mission_root.create_group(f"{lidar_tag}_filtered", overwrite=True)
                    zarr_group.create_dataset(
                        "points",
                        shape=(lidar_timestamps.shape[0],) + lidar_points_filtered.shape,
                        dtype=lidar_points_filtered.dtype,
                        overwrite=True,
                        chunks=(1,) + lidar_points_filtered.shape,
                    )
                    zarr_group.create_dataset(
                        "valid",
                        shape=(lidar_timestamps.shape[0], 1),
                        dtype=np.uint32,
                        overwrite=True,
                    )
                    zarr_group.create_dataset(
                        "timestamp",
                        shape=lidar_timestamps.shape,
                        dtype=lidar_timestamps.dtype,
                        overwrite=True,
                    )
                    mission_root[f"{lidar_tag}_filtered"]["timestamp"] = lidar_timestamps
                    for key, value in mission_root[lidar_tag].attrs.items():
                        mission_root[f"{lidar_tag}_filtered"].attrs[key] = value

                mission_root[f"{lidar_tag}_filtered"]["points"][lidar_id] = lidar_points_filtered
                mission_root[f"{lidar_tag}_filtered"]["valid"][lidar_id] = nr_points



if __name__ == "__main__":
    mission = "2024-11-04-10-57-34"
    image_tags = ["hdr_front", "hdr_left", "hdr_right"]  #
    lidar_tags = ["livox_points_undistorted", "hesai_points_undistorted"]

    grand_tour_folder = Path("~/grand_tour_dataset").expanduser()
    mission_folder = grand_tour_folder / mission

    # Remove lidar points intersect with human masks
    print("Start filtering lidar...")
    mission_root = zarr.open_group(store=mission_folder / "data", mode="a")
    filter_lidar(mission_root, lidar_tags, image_tags, mission_folder)
