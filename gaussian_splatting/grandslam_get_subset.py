import pathlib
import copy
import json
import argparse
from sortedcontainers import SortedDict
import numpy as np

import viser.transforms as vtf

CAMERA_MODEL_MAP = {
   'hdr': 'OPENCV',
   'zed': 'OPENCV'
}

def find_nearest_key(d, key):
    if key in d:
        return key
    keys = d.keys()
    pos = d.bisect_left(key)
    if pos == 0:
        return keys[0]
    if pos == len(keys):
        return keys[-1]
    before = keys[pos - 1]
    after = keys[pos]
    if after - key < key - before:
        return after
    else:
        return before


def load_transforms_ns(json_path: pathlib.Path) -> dict:
  print(f"Opening: {json_path}")
  with json_path.open("r") as f:
    transforms = json.load(f)
  return transforms

def get_camera_and_timestamp_from_frame(frame: dict) -> tuple[str, int]:
  file_path = pathlib.Path(frame["file_path"])
  camera_name = "_".join(file_path.stem.split("_")[:-1])
  idx = int(file_path.stem.split("_")[-1])
  timestamp = int(frame["timestamp"].replace("_", ""))
  return camera_name, idx, timestamp

def main(argv=None):
  parser = argparse.ArgumentParser(description='Process some integers')
  parser.add_argument('--json_path', type=str, default='', help='Path to the JSON file')
  parser.add_argument('--exclude_cameras', nargs='+', default=[], help='List of cameras to exclude')
  parser.add_argument('--scene_start', type=float, default=0.0, help='Where to start indexing frames')
  parser.add_argument('--scene_end', type=float, default=1.0, help='Where to end indexing frames')
  parser.add_argument('--dist_thresh', type=float, default=0.3, help='Distance threshold for skipping frames [m]')
  parser.add_argument('--rot_thresh', type=float, default=0.1, help='Rotation threshold for skipping frames [rad]')

  parsed, other = parser.parse_known_args(argv)
  json_path = pathlib.Path(parsed.json_path)
  transforms = load_transforms_ns(json_path)

  camera_map = {}
  for frame in transforms["frames"]:
    camera_name, _, timestamp = get_camera_and_timestamp_from_frame(frame)
    if camera_name not in camera_map:
      camera_map[camera_name] = SortedDict()
    camera_map[camera_name][timestamp] = frame

  valid_cameras = []
  print(f'Excluding cameras: {parsed.exclude_cameras}')
  for camera_name in camera_map.keys():
    excluded = False
    for exclude_camera in parsed.exclude_cameras:
      if exclude_camera in camera_name:
        excluded = True
    if excluded:
      continue
    # if not any(exclude_camera in camera_name for exclude_camera in parsed.exclude_cameras):
    valid_cameras.append(camera_name)

  print(f'Valid cameras: {valid_cameras}')

  camera_model_map = {}
  for camera_name in valid_cameras:
    for k in CAMERA_MODEL_MAP.keys():
      if k in camera_name:
        camera_model_map[camera_name] = CAMERA_MODEL_MAP[k]

  camera_models = list(set(camera_model_map.values()))
  if len(camera_models) > 1:
    raise ValueError(f'Multiple camera models found: {camera_models}')
  camera_model = camera_models[0]
  print(f'Camera model: {camera_model}')

  timestamp_per_camera = {k: [] for k in valid_cameras}

  new_transforms = copy.deepcopy(transforms)
  new_transforms["camera_model"] = camera_model
  new_transforms["frames"] = []

  anchor_camera = valid_cameras[0]
  num_skipped, num_added = 0, 0
  prev_anchor_frame = None

  cam_idx = int(parsed.scene_start * len(camera_map[anchor_camera]))
  max_idx = min(int(parsed.scene_end * len(camera_map[anchor_camera])), len(camera_map[anchor_camera]) - 1)

  while cam_idx < max_idx:

    anchor_timestamp = camera_map[anchor_camera].keys()[cam_idx]
    anchor_frame = camera_map[anchor_camera][anchor_timestamp]
    if prev_anchor_frame is not None:
      curr_transform = vtf.SE3.from_matrix(np.array(anchor_frame["transform_matrix"]))
      prev_transform = vtf.SE3.from_matrix(np.array(prev_anchor_frame["transform_matrix"]))
      dist = np.linalg.norm(curr_transform.translation() - prev_transform.translation())
      rot_diff = np.abs(
         np.array(curr_transform.rotation().as_rpy_radians())
         - np.array(prev_transform.rotation().as_rpy_radians()))

      if dist < parsed.dist_thresh and np.all(rot_diff < parsed.rot_thresh):
        cam_idx += 1
        continue

    prev_anchor_frame = anchor_frame
    cam_idx += 1

    for camera_name in valid_cameras:
      nearest_timestamp = find_nearest_key(camera_map[camera_name], anchor_timestamp)
      if nearest_timestamp in timestamp_per_camera[camera_name]:
        num_skipped += 1
        continue
      timestamp_per_camera[camera_name].append(nearest_timestamp)
      frame = camera_map[camera_name][nearest_timestamp]
      num_added += 1

      new_transforms["frames"].append(frame)

  print(f'Added: {num_added}, Skipped: {num_skipped}')
  filename = f"{json_path.stem}_subset_{parsed.scene_start}_{parsed.scene_end}".replace(".", "p")
  new_json_path = json_path.parent / f"{filename}.json"
  with open(new_json_path, "w") as f:
    json.dump(new_transforms, f)
  print(f"Saved to {new_json_path}")

if __name__ == "__main__":
    main()