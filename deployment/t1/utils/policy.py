import numpy as np
import pathlib

import legged_gym
from legged_gym.utils import helpers
from legged_gym.utils import observation_groups
from legged_gym.rl.deployment_runner_onnx import DeploymentRunner


""" Get obs components and cat to a single obs input """
def compute_observation(env_cfg, obs_groups, dof_pos, dof_vel, base_ang_vel, projected_gravity, vx, vy, vyaw, default_dof_pos, gait_frequency, gait_process, actions):
    
    obs_dict = {}
    for group in obs_groups:
        if group.name != env_cfg['policy']['obs_key']:
          continue
        obs_dict[group.name] = {}
        for observation in group.observations:
            if observation.name == "projected_gravity":
                obs = np.array(projected_gravity, dtype=np.float32)
            elif observation.name == "base_ang_vel":
                obs = np.array(base_ang_vel, dtype=np.float32)
            elif observation.name == "dof_pos":
                obs = np.array((dof_pos - default_dof_pos)[11:], dtype=np.float32)
            elif observation.name == "dof_vel":
                obs = np.array(dof_vel[11:], dtype=np.float32)
            elif observation.name == "velocity_commands":
                obs = np.array([vx, vy, vyaw], dtype=np.float32)
            elif observation.name == "gait_progress":
                obs = np.concatenate((
                    (np.cos(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8))[..., None],
                    (np.sin(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8))[..., None],
                ), axis=-1).astype(np.float32)
            elif observation.name == "actions":
                obs = np.array(actions, dtype=np.float32)
            else:
                raise ValueError(f"Observation {observation.name} not found")
            obs = obs[None]
            if observation.clip:
                obs = np.clip(obs, observation.clip[0], observation.clip[1])
            if observation.scale is not None:
                scale = observation.scale
                if isinstance(scale, list):
                    scale = np.array(scale)[None]
                obs = scale * obs
            obs_dict[group.name][observation.name] = obs

    return obs_dict

class Policy:
    def __init__(self, cfg, deploy_cfg, onboard_cfg):
        self.cfg = cfg
        self.onboard_cfg = onboard_cfg

        try:
            helpers.set_seed(cfg["seed"])
            if cfg["logdir"] == "default":
                log_root = pathlib.Path(legged_gym.GAUSS_GYM_ROOT_DIR) / 'logs'
            elif cfg["logdir"] != "":
                log_root = pathlib.Path(cfg["logdir"])
            else:
                raise ValueError("Must specify logdir as 'default' or a path.")
            
            self.runner = DeploymentRunner(deploy_cfg, cfg)

            if cfg["runner"]["resume"]:
                assert cfg["runner"]["load_run"] != "", "Must specify load_run when resuming."
                self.runner.load(log_root)
            self.observation_groups = observation_groups.observation_groups_from_config(cfg["observations"])
        except Exception as e:
            print(f"Failed to start runner: {e}")
            raise
        self._init_inference_variables()

    def get_policy_interval(self):
        return self.policy_interval

    def _init_inference_variables(self):
        self.default_dof_pos = np.array(self.onboard_cfg["common"]["default_qpos"], dtype=np.float32)
        self.stiffness = np.array(self.onboard_cfg["common"]["stiffness"], dtype=np.float32)
        self.damping = np.array(self.onboard_cfg["common"]["damping"], dtype=np.float32)

        self.commands = np.zeros(3, dtype=np.float32)
        self.smoothed_commands = np.zeros(3, dtype=np.float32)

        self.gait_frequency = np.average(self.onboard_cfg["commands"]["gait_frequency"])
        self.gait_process = 0.0
        self.dof_targets = np.copy(self.default_dof_pos)
        self.obs = None
        self.actions = np.zeros(12, dtype=np.float32) # TODO: get from cfg
        self.policy_interval = self.onboard_cfg["common"]["dt"] * self.cfg["control"]["decimation"]

    def inference(self, time_now, dof_pos, dof_vel, base_ang_vel, projected_gravity, vx, vy, vyaw):
        self.gait_process = np.fmod(time_now * self.gait_frequency, 1.0)
        self.commands[0] = vx
        self.commands[1] = vy
        self.commands[2] = vyaw
        clip_range = (-self.policy_interval, self.policy_interval)
        self.smoothed_commands += np.clip(self.commands - self.smoothed_commands, *clip_range)

        if np.linalg.norm(self.smoothed_commands) < 1e-5:
            self.gait_frequency = 0.0
        else:
            self.gait_frequency = np.average(self.onboard_cfg["commands"]["gait_frequency"])

        # self.obs[0:3] = projected_gravity * self.cfg["policy"]["normalization"]["gravity"]
        # self.obs[3:6] = base_ang_vel * self.cfg["policy"]["normalization"]["ang_vel"]
        # self.obs[6] = (
        #     self.smoothed_commands[0] * self.cfg["policy"]["normalization"]["lin_vel"] * (self.gait_frequency > 1.0e-8)
        # )
        # self.obs[7] = (
        #     self.smoothed_commands[1] * self.cfg["policy"]["normalization"]["lin_vel"] * (self.gait_frequency > 1.0e-8)
        # )
        # self.obs[8] = (
        #     self.smoothed_commands[2] * self.cfg["policy"]["normalization"]["ang_vel"] * (self.gait_frequency > 1.0e-8)
        # )
        # self.obs[9] = np.cos(2 * np.pi * self.gait_process) * (self.gait_frequency > 1.0e-8)
        # self.obs[10] = np.sin(2 * np.pi * self.gait_process) * (self.gait_frequency > 1.0e-8)
        # self.obs[11:23] = (dof_pos - self.default_dof_pos)[11:] * self.cfg["policy"]["normalization"]["dof_pos"]
        # self.obs[23:35] = dof_vel[11:] * self.cfg["policy"]["normalization"]["dof_vel"]
        # self.obs[35:47] = self.actions

        self.obs = compute_observation(self.cfg, self.observation_groups, dof_pos, dof_vel, base_ang_vel, projected_gravity, vx, vy, vyaw, self.default_dof_pos, self.gait_frequency, self.gait_process, self.actions)
        outs = self.runner.act(self.obs[self.cfg['policy']['obs_key']])
        self.actions[:] = outs['actions']
        self.actions[:] = np.clip(self.actions, -self.cfg["normalization"]["clip_actions"], self.cfg["normalization"]["clip_actions"])
        
        self.dof_targets[:] = self.default_dof_pos
        self.dof_targets[11:] += self.cfg["control"]["action_scale"] * self.actions

        return self.dof_targets
