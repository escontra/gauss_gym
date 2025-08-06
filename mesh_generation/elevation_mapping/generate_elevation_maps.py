from elevation_mapping_cupy import ElevationMap
from elevation_mapping_cupy import Parameter
import elevation_mapping_cupy

# General
from pathlib import Path
import os
import numpy as np
import zarr
import matplotlib.pyplot as plt
import tqdm
import zarr
import open3d as o3d
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import interp1d
from PIL import Image
import trimesh
import numpy as np
from scipy import ndimage
import plotly.graph_objects as go
import imageio


def pq_to_se3(p, q):
    se3 = np.eye(4, dtype=np.float32)
    try:
        se3[:3, :3] = R.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
        se3[:3, 3] = [p["x"], p["y"], p["z"]]
    except:
        se3[:3, :3] = R.from_quat(q).as_matrix()
        se3[:3, 3] = p
    return se3


def attrs_to_se3(attrs):
    return pq_to_se3(attrs["transform"]["translation"], attrs["transform"]["rotation"])


# Auxilary function for tf lookup replacement in zarr file
def get_closest_tf(
    timestamp: float, odom: zarr.Group, return_se3: bool = False, interpolate: bool = False
) -> np.ndarray:
    assert interpolate is False, "Interpolation not implemented yet"
    idx = np.argmin(np.abs(odom["timestamp"] - timestamp))
    p = odom["pose_pos"][idx]
    q = odom["pose_orien"][idx]
    if return_se3:
        return pq_to_se3(p, q)
    return p, q


# Function to convert gridmap to mesh
def get_mesh(elevation_map, resolution, height_error, max_triangles=50000):
    min_val = elevation_map.min()
    span = elevation_map.max() - elevation_map.min()
    scaled_16bit = ((elevation_map - min_val) / (span) * 65535).astype(np.uint16)
    Image.fromarray(scaled_16bit, mode="I;16").save("input.png")
    error = height_error / span
    # Installation via: https://github.com/fogleman/hmm/tree/master
    os.system(f"/usr/local/bin/hmm input.png /tmp/output.stl -z 1 -e {error} -t {max_triangles}")
    mesh = trimesh.load("/tmp/output.stl")
    vertices = mesh.vertices

    x, y, z = np.split(np.array(vertices), 3, axis=1)
    x -= int(elevation_map.shape[0] / 2.0)
    y -= int(elevation_map.shape[1] / 2.0)

    x *= resolution
    y *= resolution
    z = z * span + min_val
    mesh.vertices = np.concatenate((x, y, z), axis=1)
    return mesh


# Function to convert gridmap to mesh
def remove_points_vectorized(elevation, number_of_cells, radius_in_cells, delta_height):
    """
    Vectorized version for better performance on large arrays.
    """
    elevation_removed = elevation.copy()
    rows, cols = elevation.shape

    # Create structure element for morphological operations
    struct_elem = np.zeros((2 * radius_in_cells + 1, 2 * radius_in_cells + 1))
    y, x = np.ogrid[-radius_in_cells : radius_in_cells + 1, -radius_in_cells : radius_in_cells + 1]
    mask = x * x + y * y <= radius_in_cells * radius_in_cells
    struct_elem[mask] = 1

    # For each point, count neighbors that are delta_height lower
    for i in range(rows):
        for j in range(cols):
            if np.isnan(elevation[i, j]):
                continue

            current_val = elevation[i, j]

            # Extract neighborhood
            i_min = max(0, i - radius_in_cells)
            i_max = min(rows, i + radius_in_cells + 1)
            j_min = max(0, j - radius_in_cells)
            j_max = min(cols, j + radius_in_cells + 1)

            neighborhood = elevation[i_min:i_max, j_min:j_max]

            # Create mask for this neighborhood
            ni, nj = neighborhood.shape
            center_i, center_j = i - i_min, j - j_min

            # Count valid lower neighbors
            lower_count = 0
            for ni_idx in range(ni):
                for nj_idx in range(nj):
                    if ni_idx == center_i and nj_idx == center_j:
                        continue  # Skip center point

                    # Check if within circular radius
                    dist_sq = (ni_idx - center_i) ** 2 + (nj_idx - center_j) ** 2
                    if dist_sq > radius_in_cells**2:
                        continue

                    neighbor_val = neighborhood[ni_idx, nj_idx]
                    if not np.isnan(neighbor_val) and neighbor_val <= (current_val - delta_height):
                        lower_count += 1

            # Mark as outlier if too many lower neighbors
            if lower_count >= number_of_cells:
                elevation_removed[i, j] = np.nan

    return elevation_removed


# Function to convert gridmap to mesh
def nearest_neighbor_inpaint(image, mask):
    _, indices = ndimage.distance_transform_edt(mask, return_indices=True)
    filled = image[tuple(indices)]
    return filled


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


class ElevationMapWrapper:
    def __init__(self, map_length, resolution):
        self.root = Path(elevation_mapping_cupy.__file__).parent.parent
        weight_file = self.root / "config/core/weights.dat"
        plugin_config_file = self.root / "config/core/plugin_config.yaml"
        self.param = Parameter(use_chainer=False, weight_file=weight_file, plugin_config_file=plugin_config_file)
        self.param.enable_drift_compensation = False

        self.param.subscriber = {"front_upper_depth": {"topic_name": "/integrated_depth", "data_type": "pointcloud"}}
        self.param.map_length = map_length
        self.param.resolution = resolution
        self.param.update()

        self._pointcloud_process_counter = 0
        self._image_process_counter = 0
        self._map = ElevationMap(self.param)

    def integrate_pointcloud(self, pts, trans, rot, position_noise: float = 0.0, orientation_noise: float = 0.0):
        channels = ["x", "y", "z"]
        self._map.input_pointcloud(pts, channels, rot, trans, position_noise, orientation_noise)

    def move_map(self, trans, rot):
        self._map.move_to(trans, rot)

    def update_variance(self):
        # should be called every
        self._map.update_variance()

    def update_time(self):
        # should be called every
        self._map.update_time()


class SynchronizedTopic:
    def __init__(self, mission_root, lidar_tags, odom_tag="dlio_map_odometry", load_runtime=False):
        self.load_runtime = load_runtime
        self.lidar_tags = lidar_tags
        self.mission_root = mission_root
        self.lidar_timestamps = []
        self.valid_lidars = []
        self.all_lidar_points = []
        self.fast_get_closest_tf = []

        self.element_idx = []
        self.tag_idx = []

        for i, lidar_tag in enumerate(lidar_tags):
            if "depth_camera" in lidar_tag:
                # Load for depth image

                mission_root[lidar_tag].attrs["camera_info"]
                NR = mission_root[lidar_tag]["timestamp"].shape[0]
                self.lidar_timestamps.append(mission_root[lidar_tag]["timestamp"][:])
                self.element_idx.append(np.arange(NR, dtype=np.int32))
                self.tag_idx.append(np.full((NR,), i, dtype=np.int32))

                image_folder = (
                    Path(str(mission_root.store_path).replace("file://", "").replace("/data", "/images")) / lidar_tag
                )
                K = np.array(mission_root[lidar_tag].attrs["camera_info"]["K"]).reshape(3, 3)
                intrinsic = o3d.camera.PinholeCameraIntrinsic()
                intrinsic.set_intrinsics(
                    width=mission_root[lidar_tag].attrs["camera_info"]["width"],
                    height=mission_root[lidar_tag].attrs["camera_info"]["height"],
                    fx=K[0, 0],
                    fy=K[1, 1],
                    cx=K[0, 2],
                    cy=K[1, 2],
                )
                valid_lidars = np.full((NR,), -1, dtype=np.int32)
                lidar_points = []
                if not self.load_runtime:
                    valid_lidars = np.empty((NR,), dtype=np.uint32)
                    for i in range(NR):
                        image_path = image_folder / f"{i:06d}.png"
                        depth_image = np.float32(imageio.imread(image_path)) / 1000
                        depth_o3d = o3d.geometry.Image(depth_image)
                        pcd = o3d.geometry.PointCloud.create_from_depth_image(
                            depth_o3d,
                            intrinsic,
                            depth_scale=1.0,  # Already in meters
                            depth_trunc=30.0,  # Max depth (clipping)
                            stride=1,  # Use every pixel
                        )
                        pcd = pcd.remove_non_finite_points()
                        points = np.asarray(pcd.points)
                        valid_lidars[i] = points.shape[0]
                        lidar_points.append(points)

                self.valid_lidars.append(valid_lidars)
                self.all_lidar_points.append(lidar_points)

            else:
                self.tag_idx.append(
                    np.full(
                        (mission_root[lidar_tag.replace("_filtered", "")]["timestamp"].shape[0],), i, dtype=np.int32
                    )
                )
                self.element_idx.append(
                    np.arange(mission_root[lidar_tag.replace("_filtered", "")]["timestamp"].shape[0], dtype=np.int32)
                )
                self.lidar_timestamps.append(mission_root[lidar_tag.replace("_filtered", "")]["timestamp"][:])
                self.valid_lidars.append(mission_root[lidar_tag]["valid"][:, 0])
                self.all_lidar_points.append(mission_root[lidar_tag]["points"][:])

            self.fast_get_closest_tf.append(FastGetClosestTf(mission_root[odom_tag], return_se3=True))

        self.element_idx = np.concatenate(self.element_idx, axis=0)
        self.tag_idx = np.concatenate(self.tag_idx, axis=0)

        self.index = np.argsort(np.concatenate(self.lidar_timestamps, axis=0))
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.index):
            raise StopIteration

        tag_idx = self.tag_idx[self.index[self.current_index]]
        element_idx = self.element_idx[self.index[self.current_index]]
        lidar_tag = self.lidar_tags[tag_idx]
        valid_points = self.valid_lidars[tag_idx][element_idx]

        if valid_points < 0 and self.load_runtime:
            # Load for depth image
            image_folder = (
                Path(str(self.mission_root.store_path).replace("file://", "").replace("/data", "/images")) / lidar_tag
            )
            K = np.array(self.mission_root[lidar_tag].attrs["camera_info"]["K"]).reshape(3, 3)
            intrinsic = o3d.camera.PinholeCameraIntrinsic()
            intrinsic.set_intrinsics(
                width=self.mission_root[lidar_tag].attrs["camera_info"]["width"],
                height=self.mission_root[lidar_tag].attrs["camera_info"]["height"],
                fx=K[0, 0],
                fy=K[1, 1],
                cx=K[0, 2],
                cy=K[1, 2],
            )
            image_path = image_folder / f"{element_idx:06d}.png"
            depth_image = np.float32(imageio.imread(image_path)) / 1000
            depth_o3d = o3d.geometry.Image(depth_image)
            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                depth_o3d,
                intrinsic,
                depth_scale=1.0,  # Already in meters
                depth_trunc=30.0,  # Max depth (clipping)
                stride=1,  # Use every pixel
            )
            pcd = pcd.remove_non_finite_points()
            lidar_points = np.asarray(pcd.points)

        else:
            lidar_points = self.all_lidar_points[tag_idx][element_idx][:valid_points]

        timestamp = self.lidar_timestamps[tag_idx][element_idx]
        odom_to_base = self.fast_get_closest_tf[tag_idx](timestamp)

        self.current_index += 1
        return (lidar_tag, lidar_points, odom_to_base, timestamp)

    def __len__(self):
        return self.index.shape[0]


if __name__ == "__main__":
    mission = "2024-11-04-10-57-34"
    lidar_tags = [
        "depth_camera_front_upper",
        # "depth_camera_left",
        # "depth_camera_right",
        # "depth_camera_rear_upper",
        "livox_points_undistorted_filtered",
        "hesai_points_undistorted_filtered",
    ]
    integration_settings = {
        "depth_camera_front_upper": (0.005, 10),
        # "depth_camera_rear_upper": (0.005, 5),
        # "depth_camera_left": (0.005, 5),
        # "depth_camera_right": (0.005, 5),
        "hesai_points_undistorted_filtered": (0.005, 5),
        "livox_points_undistorted_filtered": (0.015, 5),
    }
    VISU_ELEVATION = False
    CONVERT_MESH = True
    PLOT_MESH = False
    DEBUG_DEPTH_CAM = False
    ODOM_TAG = "dlio_map_odometry"

    # Open GrandTour Dataset
    grand_tour_folder = Path("~/grand_tour_dataset").expanduser()
    mission_folder = grand_tour_folder / mission
    mission_root = zarr.open_group(store=mission_folder / "data", mode="r")
    base_to_box_base = pq_to_se3(
        mission_root["tf"].attrs["tf"]["box_base"]["translation"],
        mission_root["tf"].attrs["tf"]["box_base"]["rotation"],
    )

    emw = ElevationMapWrapper(map_length=16, resolution=0.04)
    pcd_synced = SynchronizedTopic(mission_root, lidar_tags, odom_tag=ODOM_TAG, load_runtime=True)

    DEPTH_CAM = None
    LIDAR = None

    elevation_maps = []
    for j, (lidar_tag, lidar_points, tf, timestamp) in enumerate(
        tqdm.tqdm(pcd_synced, total=len(pcd_synced), desc="Processing Lidar Data")
    ):
        if j < 8000:
            continue
        valid_points = np.linalg.norm(lidar_points[:, :2], axis=1) < integration_settings[lidar_tag][1]
        lidar_points = lidar_points[valid_points]

        if ODOM_TAG == "anymal_state_odometry":
            odom_to_base = tf
        elif ODOM_TAG == "dlio_map_odometry":
            dlio_world_to_hesai = tf  # FYI
            odom_to_box_base = dlio_world_to_hesai @ attrs_to_se3(
                mission_root["hesai_points_undistorted"].attrs
            )  # hesai to box_base
            odom_to_base = odom_to_box_base @ np.linalg.inv(base_to_box_base)  # box_base to box_base
        else:
            raise ValueError(f"Unknown odometry tag: {ODOM_TAG}")

        if "depth_camera" in lidar_tag:
            lidar_to_base = attrs_to_se3(mission_root[lidar_tag].attrs)
            base_to_lidar = np.linalg.inv(lidar_to_base)
            odom_to_lidar = odom_to_base @ base_to_lidar
        else:
            lidar_to_box_base = attrs_to_se3(mission_root[lidar_tag].attrs)
            box_base_to_lidar = np.linalg.inv(lidar_to_box_base)
            odom_to_lidar = odom_to_base @ base_to_box_base @ box_base_to_lidar

        t = odom_to_lidar[:3, 3].copy()
        t[2] = 0.0

        emw.move_map(t, np.eye(3))
        # Only integrate upto a certain distance
        emw.integrate_pointcloud(
            lidar_points, odom_to_lidar[:3, 3], odom_to_lidar[:3, :3], integration_settings[lidar_tag][0]
        )
        emw.update_variance()
        emw.update_time()

        if DEBUG_DEPTH_CAM:
            if lidar_tag == "depth_camera_front_upper":
                DEPTH_CAM = (lidar_points, odom_to_lidar)
            if lidar_tag == "livox_points_undistorted_filtered":
                LIDAR = (lidar_points, odom_to_lidar)
            if DEPTH_CAM is not None and LIDAR is not None:
                depth_points, depth_tf = DEPTH_CAM
                lidar_points, lidar_tf = LIDAR

                # Transform depth camera points
                depth_points_transformed = (depth_tf[:3, :3] @ depth_points.T).T + depth_tf[:3, 3]
                # Transform lidar points
                lidar_points_transformed = (lidar_tf[:3, :3] @ lidar_points.T).T + lidar_tf[:3, 3]

                # Create Open3D point clouds
                pcd_depth = o3d.geometry.PointCloud()
                pcd_depth.points = o3d.utility.Vector3dVector(depth_points_transformed)
                pcd_depth.paint_uniform_color([0, 0, 1])  # Blue

                pcd_lidar = o3d.geometry.PointCloud()
                pcd_lidar.points = o3d.utility.Vector3dVector(lidar_points_transformed)
                pcd_lidar.paint_uniform_color([1, 0, 0])  # Red

                print(f"Visualizing depth camera and lidar point clouds for step {j}")
                o3d.visualization.draw_geometries(
                    [pcd_depth, pcd_lidar], window_name="Depth vs Lidar", width=920, height=580
                )

        if j % 200 == 0 and CONVERT_MESH:
            elevation = emw._map.get_layer("elevation").get()
            is_valid = emw._map.get_layer("is_valid").get()
            elevation[is_valid == 0] = np.nan
            elevation = elevation[1:-1, 1:-1]

            np.save(f"debug/{mission}/{j:04d}_elevation_map.npy", emw._map.elevation_map.get())
            metadata = {
                "timestamp": timestamp,
                "pose": odom_to_lidar,
                "translation": t,
                "lidar_tag": lidar_tag,
                "resolution": emw._map.resolution,
                "map_length": emw._map.map_length,
            }
            np.save(f"debug/{mission}/{j:04d}_elevation_map_metadata.npy", metadata, allow_pickle=True)

            elevation_map = emw._map.elevation_map.get()[0]
            is_valid = emw._map.elevation_map.get()[2]
            elevation_map[is_valid == 0] = np.nan

            # elevation_map = remove_points_vectorized(
            #     elevation_map, number_of_cells=2, radius_in_cells=int(0.1 / metadata["resolution"]), delta_height=0.4
            # )
            # elevation_map = nearest_neighbor_inpaint(image=elevation_map, mask=np.isnan(elevation_map))

            val = np.nanmax(elevation_map) + 0.1
            elevation_map[np.isnan(elevation_map)] = val
            mesh = get_mesh(elevation_map, metadata["resolution"], 0.0001)

            rotation_matrix = R.from_euler("z", 90, degrees=True).as_matrix()
            mesh.vertices = (rotation_matrix @ mesh.vertices.T).T
            mesh.vertices[:, :2] += metadata["translation"][:2]

            # filter for invalid points
            valid_faces = []
            for face in mesh.faces:
                if np.all(mesh.vertices[face, 2] < val - 0.0001):
                    valid_faces.append(face)
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=np.array(valid_faces), process=False)
            mesh.export(f"debug/{mission}/{j:04d}_elevation_map.ply")

            if PLOT_MESH:
                fig = go.Figure()
                T = odom_to_lidar  # odom_to_lidar  # np.linalg.inv(odom_to_lidar)
                transformed_points = (T[:3, :3] @ lidar_points.T).T + T[:3, 3]

                fig.add_trace(
                    go.Scatter3d(
                        x=transformed_points[:, 0],
                        y=transformed_points[:, 1],
                        z=transformed_points[:, 2],
                        mode="markers",
                        marker=dict(size=2),
                        name="Transformed Point Cloud",
                    )
                )
                fig.add_trace(
                    go.Mesh3d(
                        x=mesh.vertices[:, 0],
                        y=mesh.vertices[:, 1],
                        z=mesh.vertices[:, 2],
                        i=mesh.faces[:, 0],
                        j=mesh.faces[:, 1],
                        k=mesh.faces[:, 2],
                        intensity=mesh.vertices[:, 2],
                        colorscale="viridis",
                        showscale=True,
                        colorbar=dict(title="Elevation"),
                        flatshading=False,
                        opacity=0.95,
                    )
                )

                fig.update_layout(
                    title="3D Mesh from Elevation Map",
                    scene=dict(
                        xaxis_title="X Axis",
                        yaxis_title="Y Axis",
                        zaxis_title="Z Axis",
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                        aspectmode="data",
                    ),
                    width=800,
                    height=600,
                )
                fig.show()
                print(f"Saved mesh for step {j} with lidar tag {lidar_tag}")

        if VISU_ELEVATION:
            if j % 10 == 0:
                elevation = emw._map.get_layer("elevation").get()
                is_valid = emw._map.get_layer("is_valid").get()
                elevation[is_valid == 0] = np.nan
                elevation = elevation[1:-1, 1:-1]
                elevation_maps.append(elevation)

            if j % 100 == 0 and j != 0:
                fig, axes = plt.subplots(5, 2, figsize=(8, 20))
                axes = axes.flatten()
                for i in range(min(len(elevation_maps), 10)):
                    ax = axes[i]
                    im = ax.imshow(elevation_maps[i], cmap="terrain")
                    ax.axis("off")

                    ax.set_title(f"Step {j - 100 + i*10 } Elevation Map")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.savefig(f"debug/{mission}/{j:04d}_elevation_map.png")
                plt.close(fig)
                elevation_maps = []

                if False:
                    idx_y, idx_x = np.where(~np.isnan(elevation))
                    # Convert grid indices to coordinates (centered at map origin)
                    cx = elevation.shape[1] // 2
                    cy = elevation.shape[0] // 2
                    x_coords = (idx_x - cx) * emw._map.resolution
                    y_coords = (idx_y - cy) * emw._map.resolution
                    z_coords = elevation[idx_y, idx_x]
                    points = np.stack([x_coords, y_coords, z_coords], axis=1)

                    # Color by elevation using terrain colormap
                    norm_elev = (z_coords - np.nanmin(z_coords)) / (np.nanmax(z_coords) - np.nanmin(z_coords) + 1e-8)
                    cmap = plt.get_cmap("terrain")
                    colors = cmap(norm_elev)[:, :3]  # RGB

                    # Create Open3D point cloud
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    pcd.colors = o3d.utility.Vector3dVector(colors)

                    print(f"Visualizing elevation map pointcloud for step {j}")
                    o3d.visualization.draw_geometries([pcd], window_name="Elevation Map Points", width=920, height=580)
