import os
import sys
import select
import numpy as np

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils.task_registry import task_registry
from legged_gym.teacher import observation_groups as observation_groups_teacher
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi
from legged_gym.utils import helpers
from legged_gym.utils import flags, config

from isaacgym import torch_utils as tu

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

""" Get obs components and cat to a single obs input """
def compute_observation(env_cfg, observation_groups, mj_data, command, gait_frequency, gait_process, default_dof_pos, actions):
    
    obs_dict = {}
    for group in observation_groups:
        if "teacher" in group.name:
            continue
        obs_dict[group.name] = {}
        for observation in group.observations:
            if observation.name == "projected_gravity":
                quat = mj_data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
                obs = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
                obs = torch.tensor(obs, dtype=torch.float32)
            elif observation.name == "base_ang_vel":
                obs = torch.tensor(mj_data.sensor("angular-velocity").data.astype(np.float32))
            elif observation.name == "lin_vel":
                obs = torch.tensor(mj_data.sensor("velocity").data.astype(np.float32))
            elif observation.name == "dof_pos":
                obs = torch.tensor(mj_data.qpos.astype(np.float32)[7:]) - torch.tensor(default_dof_pos)
            elif observation.name == "dof_vel":
                obs = torch.tensor(mj_data.qvel.astype(np.float32)[6:])
            elif observation.name == "velocity_commands":
                obs = torch.tensor(command)
            elif observation.name == "gait_progress":
                obs = torch.cat((
                    (torch.cos(2 * torch.pi * torch.tensor(gait_process)) * (torch.tensor(gait_frequency) > 1.0e-8).float()).unsqueeze(-1),
                    (torch.sin(2 * torch.pi * torch.tensor(gait_process)) * (torch.tensor(gait_frequency) > 1.0e-8).float()).unsqueeze(-1),
                ), dim = -1)
            elif observation.name == "actions":
                obs = torch.tensor(actions)
            obs = obs.unsqueeze(0)
            if observation.clip:
                obs = obs.clip(min=observation.clip[0], max=observation.clip[1])
            if observation.scale is not None:
                scale = observation.scale
                if isinstance(scale, list):
                    scale = torch.tensor(scale, device=obs.device)[None]
                obs = scale * obs
            obs_dict[group.name][observation.name] = obs

    return obs_dict


def main(argv=None):
    log_root = pathlib.Path(os.path.join(LEGGED_GYM_ROOT_DIR, 'logs'))
    load_run_path = None
    parsed, other = flags.Flags(load_run='', checkpoint=-1).parse_known(argv)
    if parsed.load_run != '':
      load_run_path = log_root / parsed.load_run
    else:
      load_run_path = sorted(
        [item for item in log_root.iterdir() if item.is_dir()],
        key=lambda path: path.stat().st_mtime,
      )[-1]

    print(f'Loading run from: {load_run_path}...')
    cfg = config.Config.load(load_run_path / 'config.yaml')
    cfg = cfg.update({'runner.load_run': load_run_path.name})
    cfg = cfg.update({'runner.checkpoint': parsed.checkpoint})
    cfg = cfg.update({'runner.resume': True})

    
    print(f'\tTask name: {cfg["task"]}')

    for attr_name in dir(cfg.domain_rand):
       if isinstance(getattr(cfg.domain_rand, attr_name), bool):
          setattr(cfg.domain_rand, attr_name, False)

    # TODO: can we get rid of the env?
    cfg = flags.Flags(cfg).parse(other)
    print(cfg)
    cfg = dict(cfg)
    task_class = task_registry.get_task_class(cfg["task"])
    helpers.set_seed(cfg["seed"])
    env = task_class(cfg=cfg)
    cfg["runner"]["resume"] = True
    cfg["runner"]["class_name"] = "MuJoCoRunner"
    mujoco_runner = task_registry.make_alg_runner(env, cfg=cfg)

    observation_groups = [getattr(observation_groups_teacher, name) for name in cfg["observations"]["observation_groups"]]

    print("Starting MuJoCo viewer...")

    mj_model = mujoco.MjModel.from_xml_path(cfg["asset"]["mujoco_file"].format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
    mj_model.opt.timestep = cfg["sim"]["dt"]
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    default_dof_pos = np.zeros(mj_model.nu, dtype=np.float32)
    dof_stiffness = np.zeros(mj_model.nu, dtype=np.float32)
    dof_damping = np.zeros(mj_model.nu, dtype=np.float32)
    for i in range(mj_model.nu):
        found = False
        for name in cfg["init_state"]["default_joint_angles"].keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                default_dof_pos[i] = cfg["init_state"]["default_joint_angles"][name]
                found = True
        if not found:
            default_dof_pos[i] = cfg["init_state"]["default_joint_angles"]["default"]

        found = False
        for name in cfg["control"]["stiffness"].keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                dof_stiffness[i] = cfg["control"]["stiffness"][name]
                dof_damping[i] = cfg["control"]["damping"][name]
                found = True
        if not found:
            raise ValueError(f"PD gain of joint {mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)} were not defined")
    mj_data.qpos = np.concatenate(
        [
            np.array(cfg["init_state"]["pos"], dtype=np.float32),
            np.array(cfg["init_state"]["rot"][3:4] + cfg["init_state"]["rot"][0:3], dtype=np.float32),
            default_dof_pos,
        ]
    )
    mujoco.mj_forward(mj_model, mj_data)

    actions = np.zeros(12, dtype=np.float32) # TODO: get from env
    dof_targets = np.zeros(default_dof_pos.shape, dtype=np.float32)
    gait_frequency = gait_process = 0.0
    lin_vel_x = lin_vel_y = ang_vel_yaw = 0.0
    it = 0

    print("Launching MuJoCo viewer...")
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
                            gait_frequency = np.average(cfg["commands"]["gait_frequency"])
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
            # forward = tu.quat_apply(torch.tensor(quat).unsqueeze(0), torch.tensor([1.0, 0.0, 0.0]).unsqueeze(0))
            # heading = torch.atan2(forward[:, 1], forward[:, 0])
            # ang_vel_yaw = torch.clip(0.5*wrap_to_pi(ang_vel_yaw - heading), -1., 1.)
            if it % cfg["control"]["decimation"] == 0:
                obs = compute_observation(cfg, observation_groups, mj_data, [lin_vel_x, lin_vel_y, ang_vel_yaw], gait_frequency, gait_process, default_dof_pos, actions)
                dist = mujoco_runner.act(obs['student_observations'])
                actions[:] = dist.detach().cpu().numpy()
                actions[:] = np.clip(actions, -cfg["normalization"]["clip_actions"], cfg["normalization"]["clip_actions"])
                dof_targets[:] = default_dof_pos + cfg["control"]["action_scale"] * actions
            mj_data.ctrl = np.clip(
                dof_stiffness * (dof_targets - dof_pos) - dof_damping * dof_vel,
                mj_model.actuator_ctrlrange[:, 0],
                mj_model.actuator_ctrlrange[:, 1],
            )
            mujoco.mj_step(mj_model, mj_data)
            viewer.cam.lookat[:] = mj_data.qpos.astype(np.float32)[0:3]
            viewer.sync()
            it += 1
            gait_process = float(np.fmod(gait_process + cfg["sim"]["dt"] * gait_frequency, 1.0))



if __name__ == '__main__':
    main()