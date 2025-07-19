import os
import pathlib
import types
import numpy as np
import rospy

from deployment.a1.a1_real import UnitreeA1Real
import legged_gym
from legged_gym.utils import flags, config


def standup_procedure(
        env,
        ros_rate,
        cfg,
        angle_tolerance= 0.1,
        kp= None,
        kd= None,
        warmup_timesteps= 25,
        policy=None,
    ):
    """
    Args:
        warmup_timesteps: the number of timesteps to linearly increase the target position
    """
    rospy.loginfo("Robot standing up, please wait ...")

    target_pos = np.zeros((1, 12), dtype= np.float32)
    standup_timestep_i = 0
    while not rospy.is_shutdown():
        dof_pos = [env.low_state_buffer.motorState[env.dof_map[i]].q for i in range(12)]
        diff = [env.default_dof_pos[i].item() - dof_pos[i] for i in range(12)]
        direction = [1 if i > 0 else -1 for i in diff]
        if standup_timestep_i < warmup_timesteps:
            direction = [standup_timestep_i / warmup_timesteps * i for i in direction]
        if all([abs(i) < angle_tolerance for i in diff]):
            break
        print("max joint error (rad):", max([abs(i) for i in diff]), end= "\r")
        for i in range(12):
            target_pos[0, i] = dof_pos[i] + direction[i] * angle_tolerance if abs(diff[i]) > angle_tolerance else env.default_dof_pos[i]
        env.publish_legs_cmd(target_pos,
            kp= kp,
            kd= kd,
        )
        if policy is not None:
            _ = policy(env.get_obs()[cfg["policy"]["obs_key"]])
        ros_rate.sleep()
        standup_timestep_i += 1

    rospy.loginfo("Robot stood up! press A on the gamepad to continue")
    while not rospy.is_shutdown():
        if env.start_pressed:
            break
        if env.quit_pressed:
            env.publish_legs_cmd(env.default_dof_pos[None], kp= 0, kd= 0.5)
            rospy.signal_shutdown("Controller send stop signal, exiting")
            exit(0)
        env.publish_legs_cmd(env.default_dof_pos[None], kp= kp, kd= kd)
        if policy is not None:
            _ = policy(env.get_obs()[cfg["policy"]["obs_key"]])
        ros_rate.sleep()
    rospy.loginfo("Robot standing up procedure finished!")

def main(argv = None):
    log_root = pathlib.Path(os.path.join(legged_gym.GAUSS_GYM_ROOT_DIR, 'logs'))
    load_run_path = None
    parsed, other = flags.Flags({
        'runner': {'load_run': ''},
        'debug': False,
        'namespace': '/a112138',
    }).parse_known(argv)

    if parsed.runner.load_run != '':
      load_run_path = log_root / parsed.runner.load_run
    else:
      load_run_path = sorted(
        [item for item in log_root.iterdir() if item.is_dir()],
        key=lambda path: path.stat().st_mtime,
      )[-1]

    print(f'Loading run from: {load_run_path}...')
    cfg = config.Config.load(load_run_path / 'train_config.yaml')
    cfg = cfg.update({'runner.load_run': load_run_path.name})
    cfg = cfg.update({'runner.resume': True})

    cfg = flags.Flags(cfg).parse(other)
    print(cfg)
    cfg = types.MappingProxyType(dict(cfg))

    if cfg["logdir"] == "default":
        log_root = pathlib.Path(legged_gym.GAUSS_GYM_ROOT_DIR) / 'logs'
    elif cfg["logdir"] != "":
        log_root = pathlib.Path(cfg["logdir"])
    else:
        raise ValueError("Must specify logdir as 'default' or a path.")

    log_level = rospy.DEBUG if parsed.debug else rospy.INFO
    rospy.init_node("a1_policy", log_level= log_level)


    unitree_real_env = UnitreeA1Real(
        robot_namespace= parsed.namespace,
        cfg= cfg,
        forward_depth_topic=None,
        forward_depth_embedding_dims=None,
        move_by_wireless_remote= False,
        move_by_gamepad=True,
        use_vision=False,
    )
    unitree_real_env.start_ros()
    unitree_real_env.wait_untill_ros_working()
    duration = cfg["sim"]["dt"] * cfg["control"]["decimation"] # in sec
    rate = rospy.Rate(1 / duration)

    # standup_procedure(
    #     unitree_real_env,
    #     rate,
    #     cfg,
    #     angle_tolerance= 0.2,
    #     kp= 80,
    #     kd= 1.5,
    #     warmup_timesteps= 100,
    #     policy=None,
    # )
    excite_duration = 5.0
    while not rospy.is_shutdown():
        excite_procedure(unitree_real_env, rate, cfg, excite_duration)
    # while not rospy.is_shutdown():
    #     excite_start = rospy.get_time()
    #     inference_start_time = rospy.get_time()
    #     obs = unitree_real_env.get_obs()
    #     unitree_real_env.send_action(np.zeros_like(unitree_real_env.default_dof_pos[None]), kp=20, kd=0.5)
    #     # unitree_real_env.send_action(actions)
    #     motor_temperatures = [motor_state.temperature for motor_state in unitree_real_env.low_state_buffer.motorState]
    #     rospy.loginfo_throttle(10, " ".join(["motor_temperatures:"] + ["{:d},".format(t) for t in motor_temperatures[:12]]))
    #     inference_duration = rospy.get_time() - inference_start_time
    #     rospy.loginfo_throttle(10, "inference duration: {:.3f}".format(inference_duration))
    #     rate.sleep()
    #     if unitree_real_env.quit_pressed:
    #         unitree_real_env.publish_legs_cmd(unitree_real_env.default_dof_pos[None], kp=20, kd=0.5)
    #         rospy.signal_shutdown("Controller send stop signal, exiting")
    # return

def excite_procedure(env, rate, cfg, excite_duration):
  # Get the robot to its default position.
  print('Choose the joint to excite:')
  for i, name in range(env.extra_cfg["dof_names"]):
      print(f'\t {name}: [{i}]')
  joint_idx = int(input('Enter the joint index: '))
  real_joint_idx = env.extra_cfg["dof_map"][joint_idx]
  gain = float(input('Enter the desired gain:'))
  damping = float(input('Enter the desired damping:'))
  action = float(input('Enter the desired action:'))

  times = []
  actions = []
  dof_pos = []

  # Get the robot to its default position.
  standup_procedure(
      env,
      rate,
      cfg,
      angle_tolerance= 0.2,
      kp= 80,
      kd= 1.5,
      warmup_timesteps= 100,
      policy=None,
  )

  excite_start = rospy.get_time()
  while not rospy.is_shutdown():
    if rospy.get_time() - excite_start > excite_duration:
        break
    times.append(rospy.get_time() - excite_start)
    actions.append(action)
    actions = np.zeros_like(env.default_dof_pos[None])
    actions[0, real_joint_idx] = action
    dof_pos.append(env.dof_pos[0, real_joint_idx])
    env.send_action(actions, kp=gain, kd=damping)
    rate.sleep()
    if env.quit_pressed:
        env.publish_legs_cmd(env.default_dof_pos[None], kp=20, kd=0.5)
        rospy.signal_shutdown("Controller send stop signal, exiting")


if __name__ == "__main__":
    main()
