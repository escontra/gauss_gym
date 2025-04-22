import os
import sys
import select
import types
import numpy as np
import torch
import pathlib
import mujoco
import mujoco.viewer

import legged_gym
from legged_gym.utils import flags, config, helpers, observation_groups, math
from legged_gym.rl.deployment_runner import DeploymentRunner


def apply_map(data, map):
    return torch.tensor(data[list(map.keys())], dtype=torch.float32)[list(map.values())]


""" Get obs components and cat to a single obs input """
def compute_observation(env_cfg, obs_groups, mj_data, command, gait_frequency, gait_process, default_dof_pos, actions, mj_ig_map):
    
    obs_dict = {}
    for group in obs_groups:
        if group.name != env_cfg['policy']['obs_key']:
          continue
        obs_dict[group.name] = {}
        for observation in group.observations:
            if observation.name == "projected_gravity":
                quat = mj_data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
                obs = math.quat_rotate_inverse(
                    torch.from_numpy(quat[None]),
                    torch.from_numpy(np.array([0.0, 0.0, -1.0]).astype(np.float32)[None]))[0].numpy()
                obs = torch.tensor(obs, dtype=torch.float32)
            elif observation.name == "base_ang_vel":
                obs = torch.tensor(mj_data.sensor("angular-velocity").data.astype(np.float32))
            elif observation.name == "lin_vel":
                obs = torch.tensor(mj_data.sensor("velocity").data.astype(np.float32))
            elif observation.name == "dof_pos":
                obs = torch.tensor(mj_data.qpos.astype(np.float32)[7:]) - torch.tensor(default_dof_pos)
                obs = apply_map(obs, mj_ig_map)
            elif observation.name == "dof_vel":
                obs = torch.tensor(mj_data.qvel.astype(np.float32)[6:])
                obs = apply_map(obs, mj_ig_map)
            elif observation.name == "velocity_commands":
                obs = torch.tensor(command)
            elif observation.name == "gait_progress":
                obs = torch.cat((
                    (torch.cos(2 * torch.pi * torch.tensor(gait_process)) * (torch.tensor(gait_frequency) > 1.0e-8).float()).unsqueeze(-1),
                    (torch.sin(2 * torch.pi * torch.tensor(gait_process)) * (torch.tensor(gait_frequency) > 1.0e-8).float()).unsqueeze(-1),
                ), dim = -1)
            elif observation.name == "actions":
                obs = torch.tensor(actions)
                obs = apply_map(obs, mj_ig_map)
            else:
                raise ValueError(f"Observation {observation.name} not found")
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
    log_root = pathlib.Path(os.path.join(legged_gym.GAUSS_GYM_ROOT_DIR, 'logs'))
    load_run_path = None
    parsed, other = flags.Flags({'runner': {'load_run': ''}}).parse_known(argv)
    if parsed.runner.load_run != '':
      load_run_path = log_root / parsed.runner.load_run
    else:
      load_run_path = sorted(
        [item for item in log_root.iterdir() if item.is_dir()],
        key=lambda path: path.stat().st_mtime,
      )[-1]

    print(f'Loading run from: {load_run_path}...')
    cfg = config.Config.load(load_run_path / 'config.yaml')
    cfg = cfg.update({'runner.load_run': load_run_path.name})
    cfg = cfg.update({'runner.resume': True})
    cfg = cfg.update({'headless': False})
    cfg = cfg.update({'runner.class_name': "DeploymentRunner"})

    deploy_cfg = config.Config.load(load_run_path / 'deploy_config.yaml')

    print(f'\tTask name: {cfg["task"]}')

    cfg = flags.Flags(cfg).parse(other)
    print('Config:')
    print(cfg)
    print('Deploy config:')
    print(deploy_cfg)
    cfg = types.MappingProxyType(dict(cfg))
    deploy_cfg = types.MappingProxyType(dict(deploy_cfg))
    helpers.set_seed(cfg["seed"])

    if "mujoco_file" not in cfg["asset"]:
        raise ValueError(f"mujoco_file not specified in config.yaml: {cfg['asset']}")
    
    mj_model = mujoco.MjModel.from_xml_path(cfg["asset"]["mujoco_file"].format(GAUSS_GYM_ROOT_DIR=legged_gym.GAUSS_GYM_ROOT_DIR))
    mj_model.opt.timestep = cfg["sim"]["dt"]
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)

    # Map from MuJoCo dof index to IsaacGym dof index. Used for re-ordering observations.
    for i in range(mj_model.nu):
        mj_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)

    mj_ig_map = {}
    default_dof_pos_mj = np.zeros(mj_model.nu, dtype=np.float32)
    dof_stiffness_mj = np.zeros(mj_model.nu, dtype=np.float32)
    dof_damping_mj = np.zeros(mj_model.nu, dtype=np.float32)
    print(f'{"MJ Index":<10}{"IG Index":<10}{"Name":<20}{"Default Pos":<15}{"Stiffness":<15}{"Damping":<15}')
    print(f'{"-"*10:<10}{"-"*10:<10}{"-"*20:<20}{"-"*15:<15}{"-"*15:<15}{"-"*15:<15}')
    for i in range(mj_model.nu):
        mj_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        ig_idx = deploy_cfg["deploy"]["dof_names"].index(mj_name)
        mj_ig_map[i] = ig_idx
        default_dof_pos_mj[i] = cfg["init_state"]["default_joint_angles"][mj_name]
        found_stiffness = False
        for cfg_name in cfg["control"]["stiffness"].keys():
            if cfg_name in mj_name:
                dof_stiffness_mj[i] = cfg["control"]["stiffness"][cfg_name]
                dof_damping_mj[i] = cfg["control"]["damping"][cfg_name]
                found_stiffness = True
        if not found_stiffness:
            raise ValueError(f"PD gain of joint {mj_name} were not defined")
        print(f'{i:<10}{ig_idx:<10}{mj_name:<20}{default_dof_pos_mj[i]:<15.1f}{dof_stiffness_mj[i]:<15.1f}{dof_damping_mj[i]:<15.1f}')
    ig_mj_map = {v: k for k, v in mj_ig_map.items()}

    mj_data.qpos = np.concatenate(
        [
            np.array(cfg["init_state"]["pos"], dtype=np.float32),
            np.array(cfg["init_state"]["rot"][3:4] + cfg["init_state"]["rot"][0:3], dtype=np.float32),
            default_dof_pos_mj,
        ]
    )
    mujoco.mj_forward(mj_model, mj_data)

    runner = DeploymentRunner(deploy_cfg, cfg, device=cfg["rl_device"])
    runner.load(log_root)
    obs_groups = observation_groups.observation_groups_from_dict(cfg["observations"])
    use_gait_frequency = "GAIT_PROGRESS" in cfg["observations"][cfg["policy"]["obs_key"]]["observations"]

    print("Starting MuJoCo viewer...")
    actions = np.zeros(deploy_cfg["deploy"]["num_actions"], dtype=np.float32)
    dof_targets = np.zeros(default_dof_pos_mj.shape, dtype=np.float32)
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

                        if use_gait_frequency:
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
            if it % cfg["control"]["decimation"] == 0:
                obs = compute_observation(cfg, obs_groups, mj_data, [lin_vel_x, lin_vel_y, ang_vel_yaw], gait_frequency, gait_process, default_dof_pos_mj, actions, mj_ig_map)
                dist = runner.act(obs[cfg["policy"]["obs_key"]])
                actions[:] = dist.detach().cpu().numpy()
                actions[:] = np.clip(actions, -cfg["normalization"]["clip_actions"], cfg["normalization"]["clip_actions"])
                actions[:] = actions * cfg["control"]["action_scale"]
                actions[:] = apply_map(actions, ig_mj_map)
                dof_targets[:] = default_dof_pos_mj + actions
            mj_data.ctrl = np.clip(
                dof_stiffness_mj * (dof_targets - dof_pos) - dof_damping_mj * dof_vel,
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