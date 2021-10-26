import os
import pathlib
import types
import numpy as np  # noqa: F401
import functools  # noqa: F401

import cv2  # noqa: F401
import rospy
import ros_numpy  # noqa: F401
from sensor_msgs.msg import Image
from unitree_legged_msgs.msg import Float32MultiArrayStamped  # noqa: F401

import gauss_gym
from gauss_gym.utils import flags, config, visualization  # noqa: F401
from deployment.a1.a1_real import UnitreeA1Real
from gauss_gym.rl import deployment_runner_onnx


def main(argv=None):
  log_root = pathlib.Path(os.path.join(gauss_gym.GAUSS_GYM_ROOT_DIR, 'logs'))
  load_run_path = None
  parsed, other = flags.Flags(
    {
      'runner': {'load_run': ''},
      'debug': False,
      'save_step': False,
      'namespace': '/a112138',
      'serial_number': '',
    }
  ).parse_known(argv)

  if parsed.runner.load_run != '':
    load_run_path = log_root / parsed.runner.load_run
  else:
    load_run_path = sorted(
      [item for item in log_root.iterdir() if item.is_dir()],
      key=lambda path: path.stat().st_mtime,
    )[-1]

  print(f'Loading run from: {load_run_path}...')
  cfg = config.Config.load(load_run_path / 'train_config.yaml')
  deploy_cfg = config.Config.load(load_run_path / 'deploy_config.yaml')
  cfg = cfg.update({'runner.load_run': load_run_path.name})
  cfg = cfg.update({'runner.resume': True})

  cfg = flags.Flags(cfg).parse(other)
  cfg = types.MappingProxyType(dict(cfg))
  deploy_cfg = types.MappingProxyType(dict(deploy_cfg))

  if cfg['logdir'] == 'default':
    log_root = pathlib.Path(gauss_gym.GAUSS_GYM_ROOT_DIR) / 'logs'
  elif cfg['logdir'] != '':
    log_root = pathlib.Path(cfg['logdir'])
  else:
    raise ValueError("Must specify logdir as 'default' or a path.")

  runner = deployment_runner_onnx.DeploymentRunner(
    cfg, model_name='image_encoder', execution_provider='CUDAExecutionProvider'
  )

  if cfg['runner']['resume']:
    assert cfg['runner']['load_run'] != '', 'Must specify load_run when resuming.'
    runner.load(log_root)

  log_level = rospy.DEBUG if parsed.debug else rospy.INFO
  rospy.init_node('a1_visual', log_level=log_level)

  unitree_real_env = UnitreeA1Real(
    robot_namespace=parsed.namespace,
    cfg=cfg,
    deploy_cfg=deploy_cfg,
    move_by_gamepad=False,
    vision_only=True,
    vision_model=runner,
  )
  unitree_real_env.start_ros()
  unitree_real_env.wait_untill_ros_working()

  if parsed.debug:
    image_publisher = rospy.Publisher(  # noqa: F841
      parsed.namespace + '/camera/image_rect_raw',
      Image,
      queue_size=1,
    )

  duration = cfg['sim']['dt'] * cfg['control']['decimation']  # in sec
  ros_rate = rospy.Rate(1.0 / duration)

  occupancy_fig_state = (None, None)  # noqa: F841
  os.makedirs('visualizations', exist_ok=True)
  step = 0
  if parsed.save_step:
    save_path = '/home/unitree/a1_data_vision'
    import pickle  # noqa: F401

    os.makedirs(save_path, exist_ok=False)
  # embedding_msg = Float32MultiArrayStamped()
  # embedding_msg.header.frame_id = parsed.namespace + "/camera_color_optical_frame"

  while not rospy.is_shutdown():
    total_step_time = rospy.get_time()

    env_start_time = rospy.get_time()
    obs = unitree_real_env.get_obs()
    projected_gravity = obs['policy']['projected_gravity']
    print(f'Projected gravity: {projected_gravity}')
    env_duration = rospy.get_time() - env_start_time
    rospy.loginfo_throttle(10, 'env duration: {:.3f}'.format(env_duration))

    # encoder_start_time = rospy.get_time()
    # encoder_input = {
    #         'projected_gravity': projected_gravity,
    #         'camera_image': np.transpose(color_frame, (2, 0, 1))[None]
    # }
    # model_preds, visual_embedding = runner.predict(encoder_input, rnn_only=False)
    # if parsed.save_step:
    #     with open(os.path.join(save_path, f'{rospy.Time.now()}.pkl'), 'wb') as f:
    #         pickle.dump({'inputs': encoder_input, 'model_preds': model_preds, 'visual_embedding': visual_embedding}, f, protocol=5)
    # encoder_duration = rospy.get_time() - encoder_start_time
    # rospy.loginfo_throttle(10, "encoder duration: {:.3f}".format(encoder_duration))

    inference_duration = rospy.get_time() - total_step_time
    rospy.loginfo_throttle(10, 'inference duration: {:.3f}'.format(inference_duration))
    # if parsed.debug and frames.is_frame():
    # occupancy_grid_state = visualization.update_occupancy_grid(
    #         None,
    #         *occupancy_fig_state,
    #         0,
    #         [model_preds['critic/ray_cast/occupancy_grid']],
    #         ['pred'],
    #         show=False
    #         )
    # occupancy_grid_state[0].savefig(f"visualizations/occupancy_{step:06}.png")
    # from PIL import Image as PILImage
    # img = PILImage.fromarray(color_frame)
    # img.save(f"visualizations/image_{step:06}.png")

    # rgb_image_msg = ros_numpy.msgify(Image, color_frame, encoding="rgb8")
    # rgb_image_msg.header.stamp = rospy.Time.now()
    # rgb_image_msg.header.frame_id = (
    #   parsed.namespace + "/camera_color_optical_frame"
    # )
    # image_publisher.publish(rgb_image_msg)
    step += 1
    ros_rate.sleep()


if __name__ == '__main__':
  main()
