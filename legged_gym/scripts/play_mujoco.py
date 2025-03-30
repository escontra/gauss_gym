import os
import sys
import select
import numpy as np

import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.helpers import get_args

import torch
import mujoco, mujoco.viewer
import pathlib
import pickle
import warnings

def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c


if __name__ == "__main__":
    args = get_args()
    log_root = pathlib.Path(os.path.join(LEGGED_GYM_ROOT_DIR, 'logs'))
    load_run_path = None
    if args.load_run is not None:
      load_run_path = log_root / args.load_run
    else:
      load_run_path = sorted(
        [item for item in log_root.iterdir() if item.is_dir()],
        key=lambda path: path.stat().st_mtime,
      )[-1]

    print(f'Loading run from: {load_run_path}...')

    with open(load_run_path / 'train_config.pkl', 'rb') as f:
      train_cfg = pickle.load(f)

    with open(load_run_path / 'env_config.pkl', 'rb') as f:
      env_cfg = pickle.load(f)

    if args.task is not None:
      if args.task != env_cfg.task_name:
         warnings.warn(f'Args and cfg task name mismatch: {args.task} != {env_cfg.task_name}. Overriding...')
      env_cfg.task_name = args.task

    print(f'\tTask name: {env_cfg.task_name}')

    for attr_name in dir(env_cfg.domain_rand):
       if isinstance(getattr(env_cfg.domain_rand, attr_name), bool):
          setattr(env_cfg.domain_rand, attr_name, False)

    env, _ = task_registry.make_env(name=env_cfg.task_name, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    train_cfg.runner_class_name = "MuJoCoRunner"
    mujoco_runner, train_cfg = task_registry.make_alg_runner(env, name=env_cfg.task_name, args=args, train_cfg=train_cfg)

    mj_model = mujoco.MjModel.from_xml_path(env_cfg.asset.mujoco_file)
    mj_model.opt.timestep = env_cfg.sim.dt
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    default_dof_pos = np.zeros(mj_model.nu, dtype=np.float32)
    dof_stiffness = np.zeros(mj_model.nu, dtype=np.float32)
    dof_damping = np.zeros(mj_model.nu, dtype=np.float32)
    for i in range(mj_model.nu):
        found = False
        for name in env_cfg.init_state.default_joint_angles.keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                default_dof_pos[i] = env_cfg.init_state.default_joint_angles[name]
                found = True
        if not found:
            default_dof_pos[i] = env_cfg.init_state.default_joint_angles["default"]

        found = False
        for name in env_cfg.control.stiffness.keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                dof_stiffness[i] = env_cfg.control.stiffness[name]
                dof_damping[i] = env_cfg.control.damping[name]
                found = True
        if not found:
            raise ValueError(f"PD gain of joint {mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)} were not defined")
    mj_data.qpos = np.concatenate(
        [
            np.array(env_cfg.init_state.pos, dtype=np.float32),
            np.array(env_cfg.init_state.rot[3:4] + env_cfg.init_state.rot[0:3], dtype=np.float32),
            default_dof_pos,
        ]
    )
    mujoco.mj_forward(mj_model, mj_data)

    actions = np.zeros((env_cfg.env.num_actions), dtype=np.float32)
    dof_targets = np.zeros(default_dof_pos.shape, dtype=np.float32)
    gait_frequency = gait_process = 0.0
    lin_vel_x = lin_vel_y = ang_vel_yaw = 0.0
    it = 0

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.elevation = -20
        print(f"Set command (x, y, yaw): ")
        while viewer.is_running():
            if select.select([sys.stdin], [], [], 0)[0]:
                try:
                    parts = sys.stdin.readline().strip().split()
                    if len(parts) == 3:
                        lin_vel_x, lin_vel_y, ang_vel_yaw = map(float, parts)
                        if lin_vel_x == 0 and lin_vel_y == 0 and ang_vel_yaw == 0:
                            gait_frequency = 0
                        else:
                            gait_frequency = np.average(env_cfg.commands.gait_frequency)
                        print(
                            f"Updated command to: x={lin_vel_x}, y={lin_vel_y}, yaw={ang_vel_yaw}\nSet command (x, y, yaw): ",
                            end="",
                        )
                    else:
                        raise ValueError
                except ValueError:
                    print("Invalid input. Enter three numeric values.\nSet command (x, y, yaw): ", end="")
            dof_pos = mj_data.qpos.astype(np.float32)[7:]
            dof_vel = mj_data.qvel.astype(np.float32)[6:]
            quat = mj_data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
            base_ang_vel = mj_data.sensor("angular-velocity").data.astype(np.float32)
            projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
            if it % env_cfg.control.decimation == 0:
                obs = np.zeros(env_cfg.num_observations, dtype=np.float32)
                obs[0:3] = projected_gravity * env_cfg.normalization.gravity
                obs[3:6] = base_ang_vel * env_cfg.normalization.ang_vel
                obs[6] = lin_vel_x * env_cfg.normalization.lin_vel
                obs[7] = lin_vel_y * env_cfg.normalization.lin_vel
                obs[8] = ang_vel_yaw * env_cfg.normalization.ang_vel
                obs[9] = np.cos(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
                obs[10] = np.sin(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
                obs[11:23] = (dof_pos - default_dof_pos) * env_cfg.normalization.dof_pos
                obs[23:35] = dof_vel * env_cfg.normalization.dof_vel
                obs[35:47] = actions
                dist = mujoco_runner.act(torch.tensor(obs).unsqueeze(0))
                actions[:] = dist.loc.detach().numpy()
                actions[:] = np.clip(actions, -env_cfg.normalization.clip_actions, env_cfg.normalization.clip_actions)
                dof_targets[:] = default_dof_pos + env_cfg.control.action_scale * actions
            mj_data.ctrl = np.clip(
                dof_stiffness * (dof_targets - dof_pos) - dof_damping * dof_vel,
                mj_model.actuator_ctrlrange[:, 0],
                mj_model.actuator_ctrlrange[:, 1],
            )
            mujoco.mj_step(mj_model, mj_data)
            viewer.cam.lookat[:] = mj_data.qpos.astype(np.float32)[0:3]
            viewer.sync()
            it += 1
            gait_process = np.fmod(gait_process + env_cfg.sim.dt * gait_frequency, 1.0)
