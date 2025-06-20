import os
import pathlib
import types
import numpy as np
import functools

import rospy
import ros_numpy
import cv2
import pyrealsense2 as rs
from sensor_msgs.msg import Image
from unitree_legged_msgs.msg import Float32MultiArrayStamped

import legged_gym
from legged_gym.utils import flags, config
from legged_gym.rl import deployment_runner_onnx


def get_input_filter(cfg):
  """This is the filter different from the simulator, but try to close the gap."""

  def input_filter(
    image: np.ndarray,
    downscale_factor: float,
  ):
    """Processes [H, W, 3] image."""
    image = cv2.resize(
      image, (0, 0), fx=1 / downscale_factor, fy=1 / downscale_factor
    )
    return image

  return functools.partial(
    input_filter,
    downscale_factor=cfg["env"]["camera_params"]["downscale_factor"],
  )


def get_started_pipeline(
  cfg,
):
  # By default, rgb is not used.
  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(
    rs.stream.color,
    cfg["env"]["camera_params"]["cam_width"],
    cfg["env"]["camera_params"]["cam_height"],
    rs.format.rgb8,
    1.0 / cfg["env"]["camera_params"]["refresh_duration"],
  )
  profile = pipeline.start(config)
  return pipeline, profile


def main(argv=None):
  log_root = pathlib.Path(os.path.join(legged_gym.GAUSS_GYM_ROOT_DIR, "logs"))
  load_run_path = None
  parsed, other = flags.Flags(
    {
      "runner": {"load_run": ""},
      "mode": "jetson",
      "debug": False,
      "namespace": "/a112138",
    }
  ).parse_known(argv)

  if parsed.runner.load_run != "":
    load_run_path = log_root / parsed.runner.load_run
  else:
    load_run_path = sorted(
      [item for item in log_root.iterdir() if item.is_dir()],
      key=lambda path: path.stat().st_mtime,
    )[-1]

  print(f"Loading run from: {load_run_path}...")
  cfg = config.Config.load(load_run_path / "config.yaml")
  cfg = cfg.update({"runner.load_run": load_run_path.name})
  cfg = cfg.update({"runner.resume": True})

  cfg = flags.Flags(cfg).parse(other)
  print(cfg)
  cfg = types.MappingProxyType(dict(cfg))

  deploy_cfg = config.Config.load(load_run_path / "deploy_config.yaml")
  print(deploy_cfg)
  deploy_cfg = types.MappingProxyType(dict(deploy_cfg))

  if cfg["logdir"] == "default":
    log_root = pathlib.Path(legged_gym.GAUSS_GYM_ROOT_DIR) / "logs"
  elif cfg["logdir"] != "":
    log_root = pathlib.Path(cfg["logdir"])
  else:
    raise ValueError("Must specify logdir as 'default' or a path.")

  runner = deployment_runner_onnx.DeploymentRunner(deploy_cfg, cfg)

  if cfg["runner"]["resume"]:
    assert cfg["runner"]["load_run"] != "", (
      "Must specify load_run when resuming."
    )
    runner.load(log_root)

  log_level = rospy.DEBUG if parsed.debug else rospy.INFO
  rospy.init_node("a1_visual_" + parsed.mode, log_level=log_level)

  rs_filter = get_input_filter(cfg)
  rs_pipeline, rs_profile = get_started_pipeline(cfg)
  print(f'RS Profile:\n{rs_profile}')
  # embedding_publisher = rospy.Publisher(
  #   parsed.namespace + "/visual_embedding",
  #   Float32MultiArrayStamped,
  #   queue_size=1,
  # )
  if parsed.debug:
    image_publisher = rospy.Publisher(
      parsed.namespace + "/camera/depth/image_rect_raw",
      Image,
      queue_size=1,
    )

  ros_rate = rospy.Rate(1.0 / cfg["env"]["camera_params"]["refresh_duration"])
  rospy.loginfo(
    "Using refresh duration {}s".format(
      cfg["env"]["camera_params"]["refresh_duration"]
    )
  )

  rospy.loginfo("RealSense initialized.")
  try:
    # embedding_msg = Float32MultiArrayStamped()
    # embedding_msg.header.frame_id = parsed.namespace + "/camera_color_optical_frame"
    frame_got = False

    while not rospy.is_shutdown():
      frames = rs_pipeline.wait_for_frames(
        int(cfg["env"]["camera_params"]["refresh_duration"] * 1000)
      )
      # embedding_msg.header.stamp = rospy.Time.now()
      color_frame = frames.get_color_frame()
      if not color_frame:
        continue
      if not frame_got:
        frame_got = True
        rospy.loginfo("Realsense frame recieved. Sending embeddings...")

      color_frame = np.asanyarray(color_frame.get_data())
      color_frame = rs_filter(color_frame)

      if parsed.debug:
        rgb_image_msg = ros_numpy.msgify(Image, color_frame, encoding="rgb8")
        rgb_image_msg.header.stamp = rospy.Time.now()
        rgb_image_msg.header.frame_id = (
          parsed.namespace + "/camera_color_optical_frame"
        )
        image_publisher.publish(rgb_image_msg)

      ros_rate.sleep()
  finally:
    rs_pipeline.stop()


if __name__ == "__main__":
  main()
