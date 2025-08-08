import yaml
from pathlib import Path
import json
import cv2
import numpy as np
import torch
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import zarr
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import interp1d


def ros_to_gl_transform(transform_ros):
    cv_to_gl = np.eye(4)
    cv_to_gl[1:3, 1:3] = np.array([[-1, 0], [0, -1]])
    transform_gl = cv_to_gl @ transform_ros @ np.linalg.inv(cv_to_gl)
    return transform_gl


def gl_to_ros_transform(transform_gl):
    cv_to_gl = np.eye(4)
    cv_to_gl[1:3, 1:3] = np.array([[-1, 0], [0, -1]])
    transform_ros = np.linalg.inv(cv_to_gl) @ transform_gl @ cv_to_gl
    return transform_ros


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


class FastGetClosestOdomToBaseTf:
    def __init__(self, odom_key, mission_root):
        self.odom_key = odom_key
        self.mission_root = mission_root
        odom = mission_root[odom_key]
        self.odom = mission_root[odom_key]
        self.timestamps = odom["timestamp"][:]
        self.pose_pos = odom["pose_pos"][:]
        self.pose_orien = odom["pose_orien"][:]

    def __call__(self, timestamp: float, interpolate_leg_odometry: bool = False) -> np.ndarray:
        assert interpolate_leg_odometry is False, "Interpolation not implemented yet"
        idx = np.argmin(np.abs(self.timestamps - timestamp))

        # Handle boundary cases
        if idx == 0 or idx == len(self.timestamps) - 1 or self.timestamps[idx] == timestamp:
            print(f"Requested timestamp {timestamp} is at border of available times or exact match.")
            p = self.pose_pos[idx]
            q = self.pose_orien[idx]
            odom_to_base = pq_to_se3(p, q)
        else:
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
            odom_to_base = pq_to_se3(interpolated_position, interpolated_quat)

        if self.odom_key == "dlio_map_odometry":
            # This odometry topic is from: dlio_world_to_hesai frame
            dlio_world_to_hesai = odom_to_base
            odom_to_box_base = dlio_world_to_hesai @ attrs_to_se3(self.mission_root["hesai_points_undistorted"].attrs)
            base_to_box_base = pq_to_se3(
                self.mission_root["tf"].attrs["tf"]["box_base"]["translation"],
                self.mission_root["tf"].attrs["tf"]["box_base"]["rotation"],
            )
            odom_to_base = odom_to_box_base @ np.linalg.inv(base_to_box_base)

        return odom_to_base


class Masking:
    def __init__(self, nc):
        self.processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-panoptic")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-large-coco-panoptic"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        (nc.output_folder / "mask").mkdir(parents=True, exist_ok=True)
        self.nc = nc

    def __call__(self, *args, **kwds):
        cv_image, image_path, invalid_mask, frame_data = args
        mask_file_path = str(image_path).replace("rgb", "mask").replace(".jpeg", ".png")
        pil_image = Image.fromarray(cv_image)

        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        predicted_segmentation_maps = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[pil_image.size[::-1]]
        )
        segmentation_map = predicted_segmentation_maps[0]
        human_mask = (segmentation_map == 0).cpu().numpy()  # Person class is 0
        binary_mask = (~human_mask * 255).astype(np.uint8)

        # Apply the mask from the rectification
        binary_mask[invalid_mask] = 0

        # Save the binary mask as PNG
        Image.fromarray(binary_mask, mode="L").save(mask_file_path)

        # Logging percentage human pixels
        # human_pixel_count = np.sum(human_mask)
        # total_pixels = human_mask.size
        # coverage_percent = (human_pixel_count / total_pixels) * 100
        # print(f"Human coverage in frame: {coverage_percent:.2f}%")

        # Add metadata
        frame_data["mask_file_path"] = frame_data["file_path"].replace("rgb", "mask")
        return frame_data


class Depth:
    def __init__(self, nc):
        pass

    def __call__(self, *args, **kwds):
        pass


class NerfstudioConverter:
    def __init__(self, config, mission_root: zarr.Group, mission_folder, output_folder, mission_name):
        self.config = config
        self.mission_root = mission_root
        self.mission_folder = mission_folder

        self.tf_lookup = FastGetClosestOdomToBaseTf("dlio_map_odometry", mission_root)

        self.output_folder = output_folder / f"{mission_name}_nerfstudio"
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.frames_json_file = self.output_folder / "transforms.json"
        self.images_folder = self.output_folder / "rgb"
        self.images_folder.mkdir(parents=True, exist_ok=True)

        self.image_counters = {key["tag"]: 0 for key in self.config["cameras"]}
        self.image_last_stored = {key["tag"]: 0 for key in self.config["cameras"]}

        self.plugins = []
        if self.config["create_mask_based_on_semantics"]:
            self.plugins.append(Masking(self))
        if self.config["create_depth_based_on_lidar"]:
            self.plugins.append(Depth(self))

        self.undist_helpers = {key["tag"]: {} for key in self.config["cameras"]}

    def undistort_image(self, image, config):
        K = np.array(self.mission_root[config["tag"]].attrs["camera_info"]["K"]).reshape((3, 3))
        D = np.array(self.mission_root[config["tag"]].attrs["camera_info"]["D"])
        h, w = image.shape[:2]

        helper = self.undist_helpers[config["tag"]]

        # Fill in auxiliary data for undistortion
        if not hasattr(helper, "new_camera_info"):
            if self.mission_root[config["tag"]].attrs["camera_info"]["distortion_model"] == "equidistant":
                helper["new_camera_matrix"] = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    K, D, (w, h), np.eye(3), balance=1.0, fov_scale=1.0
                )
                helper["D_new"] = [0, 0, 0, 0]
                helper["map1"], helper["map2"] = cv2.fisheye.initUndistortRectifyMap(
                    K, D, np.eye(3), helper["new_camera_matrix"], (w, h), cv2.CV_16SC2
                )
            else:
                helper["new_camera_matrix"], _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
                helper["D_new"] = [0, 0, 0, 0, 0]
                helper["map1"], helper["map2"] = cv2.initUndistortRectifyMap(
                    K, D, None, helper["new_camera_matrix"], (w, h), cv2.CV_16SC2
                )
            helper["invalid_mask"] = (
                cv2.remap(
                    np.ones(image.shape[:2], dtype=np.uint8),
                    helper["map1"],
                    helper["map2"],
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                == 0
            )

        if not self.mission_root[config["tag"]].attrs["camera_info"]["distortion_model"] == "equidistant":
            # HACKY way to deal with wrong camera info of zed2i
            image = cv2.flip(image, 0)
            undistorted_image = cv2.remap(
                image, helper["map1"], helper["map2"], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
            )
            undistorted_image = cv2.flip(undistorted_image, 0)
        else:
            undistorted_image = cv2.remap(
                image, helper["map1"], helper["map2"], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
            )
        return undistorted_image

    def run(self):
        base_to_box_base = pq_to_se3(
            self.mission_root["tf"].attrs["tf"]["box_base"]["translation"],
            self.mission_root["tf"].attrs["tf"]["box_base"]["rotation"],
        )
        frames_data = {"frames": []}

        for camera in self.config["cameras"]:
            camera_tag = camera["tag"]
            data = self.mission_root[camera_tag]
            timestamps = data["timestamp"][:]
            seqs = data["sequence_id"][:]
            last_t = None
            last_pos = None
            for i in tqdm(range(timestamps.shape[0]), desc=f"Processing {camera_tag}"):
                timestamp = timestamps[i]
                if last_t is not None and timestamp - last_t < 1 / camera["hz"] + 0.001:
                    continue
                last_t = timestamp

                odom_to_base__t_camera = self.tf_lookup(timestamp)
                if (
                    last_pos is not None
                    and np.linalg.norm(last_pos - odom_to_base__t_camera[:2, 3]) < camera["distance_threshold"]
                ):
                    continue
                last_pos = odom_to_base__t_camera[:2, 3]

                odom_to_cam__t_camera = odom_to_base__t_camera @ base_to_box_base @ attrs_to_se3(data.attrs)

                cv_image = cv2.imread(self.mission_folder / "images" / camera_tag / f"{i:06d}.jpeg")

                blur = cv2.Laplacian(cv_image, cv2.CV_64F).var()
                if blur < camera["blur_threshold"]:
                    print(f"Warning: Image too blurry (blur value: {blur}). Skipping.")
                    continue

                image_filename = f"{camera_tag}_{seqs[i]:05d}.png"
                image_path = self.images_folder / image_filename

                cv_image = self.undistort_image(cv_image, camera)

                cv2.imwrite(str(image_path), cv_image)

                # Convert to OpenGL convention
                odom_to_cam__t_camera_gl = ros_to_gl_transform(odom_to_cam__t_camera)

                timestamp = timestamps[i]
                secs = int(timestamp)
                nsecs = int((timestamp - secs) * 1e9)

                frame_data = {
                    "file_path": f"./rgb/{image_filename}",
                    "transform_matrix": odom_to_cam__t_camera_gl,
                    "camera_frame_id": seqs[i],
                    "fl_x": str(data.attrs["camera_info"]["K"][0]),
                    "fl_y": str(data.attrs["camera_info"]["K"][4]),
                    "cx": str(data.attrs["camera_info"]["K"][2]),
                    "cy": str(data.attrs["camera_info"]["K"][5]),
                    "w": str(data.attrs["camera_info"]["width"]),
                    "h": str(data.attrs["camera_info"]["height"]),
                    "k1": str(data.attrs["camera_info"]["D"][0]),
                    "k2": str(data.attrs["camera_info"]["D"][1]),
                    "p1": str(data.attrs["camera_info"]["D"][2]),
                    "p2": str(data.attrs["camera_info"]["D"][3]),
                    "timestamp": str(secs) + "_" + str(nsecs),
                }

                invalid_mask = self.undist_helpers[camera["tag"]]["invalid_mask"]
                for plugin in self.plugins:
                    frame_data = plugin(cv_image, image_path, invalid_mask, frame_data)

                frames_data["frames"].append(frame_data)

        with open(self.frames_json_file, "w") as f:
            json.dump(frames_data, f, indent=2)


if __name__ == "__main__":
    CONFIG_FILE = Path("~/git/gauss-gym/gaussian_splatting/grand_tour_release.yaml").expanduser()
    MISSION_FOLDER = Path("~/grand_tour_dataset/2024-11-04-10-57-34").expanduser()
    OUTPUT_FOLDER = Path("/tmp/nerfstudio_output")

    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)

    mission_root = zarr.open_group(store=MISSION_FOLDER / "data", mode="r")

    converter = NerfstudioConverter(
        config=config,
        mission_root=mission_root,
        mission_folder=MISSION_FOLDER,
        output_folder=OUTPUT_FOLDER,
        mission_name=MISSION_FOLDER.stem,
    )
    converter.run()
