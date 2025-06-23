import os
import pathlib
import types
import numpy as np
import functools

import cv2
import rospy
import ros_numpy
# import torch
# import torch.nn.functional as F
import pyrealsense2 as rs
from sensor_msgs.msg import Image
from unitree_legged_msgs.msg import Float32MultiArrayStamped

import legged_gym
from legged_gym.utils import flags, config, visualization, when
from deployment.a1.a1_real import UnitreeA1Real
from legged_gym.rl import deployment_runner_onnx


def get_input_filter(cfg):
  """This is the filter different from the simulator, but try to close the gap."""

  def input_filter(
    image: np.ndarray,
    width: int,
    height: int
  ):
    """Processes [H, W, 3] image."""
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized_image
 
  downscale_factor = cfg["env"]["camera_params"]["downscale_factor"]
  orig_height = cfg["env"]["camera_params"]["cam_height"]
  orig_width = cfg["env"]["camera_params"]["cam_width"]
  new_height = int(orig_height / downscale_factor)
  new_width = int(orig_width / downscale_factor)

  return functools.partial(
    input_filter,
    width=new_width,
    height=new_height
  )


def get_started_pipeline(
  cfg,
  serial_number: str = '',
):
  # By default, rgb is not used.
  pipeline = rs.pipeline()
  config = rs.config()
  if serial_number != '':
    print(f'Enabling device: {serial_number}')
    config.enable_device(serial_number)
  config.enable_stream(
    rs.stream.color,
    cfg["env"]["camera_params"]["cam_width"],
    cfg["env"]["camera_params"]["cam_height"],
    rs.format.rgb8,
    60
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
      "serial_number": '',
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
  # print(cfg)
  cfg = types.MappingProxyType(dict(cfg))

  deploy_cfg = config.Config.load(load_run_path / "deploy_config.yaml")
  # print(deploy_cfg)
  deploy_cfg = types.MappingProxyType(dict(deploy_cfg))

  if cfg["logdir"] == "default":
    log_root = pathlib.Path(legged_gym.GAUSS_GYM_ROOT_DIR) / "logs"
  elif cfg["logdir"] != "":
    log_root = pathlib.Path(cfg["logdir"])
  else:
    raise ValueError("Must specify logdir as 'default' or a path.")

  runner = deployment_runner_onnx.DeploymentRunner(
          cfg,
          model_name='image_encoder',
          execution_provider='CUDAExecutionProvider')

  if cfg["runner"]["resume"]:
    assert cfg["runner"]["load_run"] != "", (
      "Must specify load_run when resuming."
    )
    runner.load(log_root)

  log_level = rospy.DEBUG if parsed.debug else rospy.INFO
  rospy.init_node("a1_visual_" + parsed.mode, log_level=log_level)

  unitree_real_env = UnitreeA1Real(
    robot_namespace=parsed.namespace,
    cfg=cfg,
    forward_depth_topic=None,
    forward_depth_embedding_dims=None,
    move_by_wireless_remote=False,
    move_by_gamepad=False)
  unitree_real_env.start_ros()
  unitree_real_env.wait_untill_ros_working()

  rs_filter = get_input_filter(cfg)
  rs_pipeline, rs_profile = get_started_pipeline(cfg, parsed.serial_number)
  print(f'RS streams:\n{rs_profile.get_streams()}')
  # embedding_publisher = rospy.Publisher(
  #   parsed.namespace + "/visual_embedding",
  #   Float32MultiArrayStamped,
  #   queue_size=1,
  # )
  if parsed.debug:
    image_publisher = rospy.Publisher(
      parsed.namespace + "/camera/image_rect_raw",
      Image,
      queue_size=1,
    )

  duration = cfg["sim"]["dt"] * cfg["control"]["decimation"] # in sec
  ros_rate = rospy.Rate(1.0 / duration)
  get_frame = when.Clock(every=cfg["env"]["camera_params"]["refresh_duration"], first=True)
  rospy.loginfo(
    "Using refresh duration {}s".format(
      cfg["env"]["camera_params"]["refresh_duration"]
    )
  )

  rospy.loginfo("RealSense initialized.")
  frames = rs_pipeline.wait_for_frames(2000)
  print('Initial frames received!')
  occupancy_fig_state = (None, None)
  os.makedirs('visualizations', exist_ok=True)
  step = 0
  try:
    # embedding_msg = Float32MultiArrayStamped()
    # embedding_msg.header.frame_id = parsed.namespace + "/camera_color_optical_frame"

    while not rospy.is_shutdown():
      inference_start_time = rospy.get_time()
      get_new_frame = get_frame()
      if get_new_frame:
        frames = rs_pipeline.wait_for_frames(
          int(cfg["env"]["camera_params"]["refresh_duration"] * 1000)
        )
        # embedding_msg.header.stamp = rospy.Time.now()
        color_frame = frames.get_color_frame()
        color_frame = np.asanyarray(color_frame.get_data())
        color_frame = rs_filter(color_frame)

      obs = unitree_real_env.get_obs()
      projected_gravity = obs['policy']['projected_gravity']

      encoder_input = {
              'projected_gravity': projected_gravity,
              'camera_image': np.transpose(color_frame, (2, 0, 1))[None]
      }
      model_preds, _ = runner.predict(encoder_input)
      inference_duration = rospy.get_time() - inference_start_time
      rospy.loginfo_throttle(10, "inference duration: {:.3f}".format(inference_duration))
      if parsed.debug and get_new_frame:
          occupancy_grid_state = visualization.update_occupancy_grid(
                  None,
                  *occupancy_fig_state,
                  0,
                  [model_preds['out_critic/ray_cast/occupancy_grid']],
                  ['pred'],
                  show=False
                  )
          occupancy_grid_state[0].savefig(f"visualizations/occupancy_{step:06}.png")
          from PIL import Image as PILImage
          img = PILImage.fromarray(color_frame)
          img.save(f"visualizations/image_{step:06}.png")

          rgb_image_msg = ros_numpy.msgify(Image, color_frame, encoding="rgb8")
          rgb_image_msg.header.stamp = rospy.Time.now()
          rgb_image_msg.header.frame_id = (
            parsed.namespace + "/camera_color_optical_frame"
          )
          image_publisher.publish(rgb_image_msg)
      step += 1
      ros_rate.sleep()
  finally:
    rs_pipeline.stop()


if __name__ == "__main__":
  main()
