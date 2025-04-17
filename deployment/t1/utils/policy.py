import numpy as np
import torch

from legged_gym.utils.task_registry import task_registry
from legged_gym.utils import helpers
from legged_gym.utils import observation_groups as observation_groups_teacher

from legged_gym import GAUSS_GYM_ROOT_DIR
import pathlib
from legged_gym.rl.mujoco_runner import MuJoCoRunner

def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c

""" Get obs components and cat to a single obs input """
def compute_observation(env_cfg, observation_groups, dof_pos, dof_vel, base_ang_vel, projected_gravity, vx, vy, vyaw, default_dof_pos, gait_frequency, gait_process, actions):
    
    obs_dict = {}
    for group in observation_groups:
        if "teacher" in group.name:
            continue
        obs_dict[group.name] = {}
        for observation in group.observations:
            if observation.name == "projected_gravity":
                obs = projected_gravity
                obs = torch.tensor(obs, dtype=torch.float32)
            elif observation.name == "base_ang_vel":
                obs = torch.tensor(base_ang_vel, dtype=torch.float32)
            elif observation.name == "dof_pos":
                obs = torch.tensor((dof_pos - default_dof_pos)[11:], dtype=torch.float32)
            elif observation.name == "dof_vel":
                obs = torch.tensor(dof_vel[11:], dtype=torch.float32)
            elif observation.name == "velocity_commands":
                obs = torch.tensor([vx, vy, vyaw], dtype=torch.float32)
            elif observation.name == "gait_progress":
                obs = torch.cat((
                    (torch.cos(2 * torch.pi * torch.tensor(gait_process)) * (torch.tensor(gait_frequency) > 1.0e-8).float()).unsqueeze(-1),
                    (torch.sin(2 * torch.pi * torch.tensor(gait_process)) * (torch.tensor(gait_frequency) > 1.0e-8).float()).unsqueeze(-1),
                ), dim = -1).to(torch.float32)
            elif observation.name == "actions":
                obs = torch.tensor(actions, dtype=torch.float32)
            else:
                raise ValueError(f"Observation {observation.name} not found")
            obs = obs.unsqueeze(0)
            if observation.clip:
                obs = obs.clip(min=observation.clip[0], max=observation.clip[1])
            if observation.scale is not None:
                scale = observation.scale
                if isinstance(scale, list):
                    scale = torch.tensor(scale, device=obs.device, dtype=torch.float32)[None]
                obs = scale * obs
            obs_dict[group.name][observation.name] = obs

    return obs_dict

class Policy:
    def __init__(self, cfg, onboard_cfg):
        self.cfg = cfg
        self.onboard_cfg = onboard_cfg

        try:
            helpers.set_seed(cfg["seed"])
            cfg["runner"]["resume"] = True
            cfg["runner"]["class_name"] = "MuJoCoRunner"
            cfg["rl_device"] = "cpu"
            # self.runner = task_registry.make_alg_runner(None, cfg=cfg)
            if cfg["logdir"] == "default":
                log_root = pathlib.Path(GAUSS_GYM_ROOT_DIR) / 'logs'
            elif cfg["logdir"] != "":
                log_root = pathlib.Path(cfg["logdir"])
            else:
                raise ValueError("Must specify logdir as 'default' or a path.")
            
            @dataclasses.dataclass
            class DummyEnv:
                num_actions: int = 12
            env = DummyEnv()

            runner = eval(cfg["runner"]["class_name"])(env, cfg, device=cfg["rl_device"])

            if cfg["runner"]["resume"]:
                assert cfg["runner"]["load_run"] != "", "Must specify load_run when resuming."
                runner.load(log_root)
            self.observation_groups = [getattr(observation_groups_teacher, name) for name in cfg["observations"]["observation_groups"]]
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
        # for key in self.obs['student_observations']:
            # print(key, self.obs['student_observations'][key].dtype)

        dist = self.runner.act(self.obs['student_observations'])
        self.actions[:] = dist.detach().cpu().numpy()
        self.actions[:] = np.clip(self.actions, -self.cfg["normalization"]["clip_actions"], self.cfg["normalization"]["clip_actions"])
        
        self.dof_targets[:] = self.default_dof_pos
        self.dof_targets[11:] += self.cfg["control"]["action_scale"] * self.actions

        return self.dof_targets
