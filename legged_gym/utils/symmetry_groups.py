from typing import List, Dict
import torch
from typing import Callable
import dataclasses
from legged_gym.utils import observation_groups
from legged_gym.rl import utils


@dataclasses.dataclass(frozen=True)
class SymmetryModifier:
  observation: observation_groups.Observation
  symmetry_fn: Callable

  def __call__(self, env, obs):
    return self.symmetry_fn(env, obs)


@dataclasses.dataclass(frozen=True)
class SymmetryGroup:
  name: str
  symmetries: List[SymmetryModifier]


def symmetry_groups_from_config(config: Dict) -> List[SymmetryGroup]:
  symmetry_groups = []
  for name, cfg in config.items():
    symmetries = [eval(obs_name) for obs_name in cfg["symmetries"]]
    symmetry_groups.append(
      SymmetryGroup(
        name=name,
        symmetries=symmetries
      )
    )
  return symmetry_groups


def symmetry_dict_from_config(config: Dict) -> Dict[str, Dict[str, SymmetryModifier]]:
  symmetry_dict = {}
  for name, cfg in config.items():
    symmetry_dict[name] = {}
    for symmetry_cls in cfg["symmetries"]:
      symmetry = eval(symmetry_cls)
      symmetry_dict[name][symmetry.observation.name] = symmetry
  return symmetry_dict


def base_lin_vel_symmetry(env, obs):
  x_vel = obs[..., 0]
  y_vel = obs[..., 1]
  z_vel = obs[..., 2]
  return torch.stack([x_vel, -y_vel, z_vel], dim=-1)


def base_ang_vel_symmetry(env, obs):
  roll_rate = obs[..., 0]
  pitch_rate = obs[..., 1]
  yaw_rate = obs[..., 2]
  return torch.stack([-roll_rate, pitch_rate, -yaw_rate], dim=-1)


def projected_gravity_symmetry(env, obs):
  return torch.stack([obs[..., 0], -obs[..., 1], obs[..., 2]], dim=-1)


def pushing_forces_symmetry(env, obs):
  return torch.stack([obs[..., 0], -obs[..., 1], obs[..., 2]], dim=-1)


def pushing_torques_symmetry(env, obs):
  return torch.stack([-obs[..., 0], obs[..., 1], -obs[..., 2]], dim=-1)


def base_mass_scaled_symmetry(env, obs):
  return torch.stack([obs[..., 0], -obs[..., 1], obs[..., 2], obs[..., 3]], dim=-1)


def velocity_commands_symmetry(env, obs):
  x_command = obs[..., 0]
  y_command = obs[..., 1]
  ang_vel_command = obs[..., 2]
  return torch.stack([x_command, -y_command, -ang_vel_command], dim=-1)


def ray_cast_symmetry(env, obs):
  return torch.flip(obs, dims=[-1])


def camera_image_symmetry(env, obs):
  return torch.flip(obs, dims=[-2])


def identity_symmetry(env, obs):
  return obs.clone()


def t1_joint_symmetry(env, joint_val, use_multipliers=True):
  joint_map = {'Left': 'Right', 'Right': 'Left'}
  multipliers = {
    'AAHead_yaw': -1.0,
    'Head_pitch': 1.0,
    'Waist': -1.0,
    'Elbow_Yaw': -1.0,
    'Elbow_Pitch': 1.0,
    'Shoulder_Pitch': 1.0,
    'Shoulder_Roll': -1.0,
    'Ankle_Pitch': 1.0,
    'Ankle_Roll': -1.0,
    'Hip_Pitch': 1.0,
    'Hip_Roll': -1.0,
    'Hip_Yaw': -1.0,
    'Knee_Pitch': 1.0}
  joint_val_sym = torch.zeros_like(joint_val)
  for dof_name in env.dof_names:
    if dof_name.startswith('Left') or dof_name.startswith('Right'):
      name_parts = dof_name.split('_')
      new_name = dof_name.replace(name_parts[0], joint_map[name_parts[0]])
      multiplier = multipliers['_'.join(name_parts[1:])] if use_multipliers else 1.0
    else:
      new_name = dof_name
      multiplier = multipliers[dof_name] if use_multipliers else 1.0
    joint_val_sym[..., env.dof_names.index(dof_name)] = multiplier * joint_val[..., env.dof_names.index(new_name)]
  return joint_val_sym


def t1_dof_pos_symmetry(env, obs):
  return t1_joint_symmetry(env, obs)


def t1_dof_vel_symmetry(env, obs):
  return t1_joint_symmetry(env, obs)


def t1_actions_symmetry(env, obs):
  return t1_joint_symmetry(env, obs)


def t1_stiffness_symmetry(env, obs):
  return t1_joint_symmetry(env, obs, use_multipliers=False)

def t1_damping_symmetry(env, obs):
  return t1_joint_symmetry(env, obs, use_multipliers=False)


def a1_joint_symmetry(env, joint_val, use_multipliers=True):
  joint_map = {'FL': 'FR', 'RL': 'RR', 'FR': 'FL', 'RR': 'RL'}
  multipliers = {'calf': 1.0, 'hip': -1.0, 'thigh': 1.0}
  joint_val_sym = torch.zeros_like(joint_val)
  for dof_name in env.dof_names:
    new_name = dof_name.replace(dof_name[:2], joint_map[dof_name[:2]])
    multiplier = multipliers[dof_name.split('_')[1]] if use_multipliers else 1.0
    # print(f'Mapping {new_name} (idx {env.dof_names.index(new_name)}) '
    #       f'to {dof_name} (idx {env.dof_names.index(dof_name)}) '
    #       f'with multiplier {multiplier}.')
    joint_val_sym[..., env.dof_names.index(dof_name)] = multiplier * joint_val[..., env.dof_names.index(new_name)]
  return joint_val_sym


def a1_hip_height_symmetry(env, hip_heights):
  hip_map = {'FL': 'FR', 'RL': 'RR', 'FR': 'FL', 'RR': 'RL'}
  hip_val_sym = torch.zeros_like(hip_heights)
  for hip_name in env.hip_names:
    new_name = hip_name.replace(hip_name[:2], hip_map[hip_name[:2]])
    # print(f'Mapping {hip_name} (idx {env.hip_names.index(hip_name)}) '
    #       f'to {new_name} (idx {env.hip_names.index(new_name)}).')
    hip_val_sym[..., env.hip_names.index(hip_name)] = hip_heights[..., env.hip_names.index(new_name)]
  return hip_val_sym


def a1_feet_symmetry(env, feet_value):
  foot_map = {'FL': 'FR', 'RL': 'RR', 'FR': 'FL', 'RR': 'RL'}
  feet_value_sym = torch.zeros_like(feet_value)
  for foot_name in env.feet_names:
    new_name = foot_name.replace(foot_name[:2], foot_map[foot_name[:2]])
    # print(f'Mapping {foot_name} (idx {env.feet_names.index(foot_name)}) '
    #       f'to {new_name} (idx {env.feet_names.index(new_name)}).')
    feet_value_sym[..., env.feet_names.index(foot_name)] = feet_value[..., env.feet_names.index(new_name)]
  return feet_value_sym


def a1_dof_pos_symmetry(env, obs):
  return a1_joint_symmetry(env, obs)


def a1_dof_vel_symmetry(env, obs):
  return a1_joint_symmetry(env, obs)


def a1_actions_symmetry(env, obs):
  return a1_joint_symmetry(env, obs)


def a1_stiffness_symmetry(env, obs):
  return a1_joint_symmetry(env, obs, use_multipliers=False)


def a1_damping_symmetry(env, obs):
  return a1_joint_symmetry(env, obs, use_multipliers=False)


def a1_motor_strength_symmetry(env, obs):
  return a1_joint_symmetry(env, obs, use_multipliers=False)


def a1_motor_error_symmetry(env, obs):
  return a1_joint_symmetry(env, obs, use_multipliers=True)


def a1_dof_friction_curriculum_values_symmetry(env, obs):
  return a1_joint_symmetry(env, obs, use_multipliers=False)


RAY_CAST = SymmetryModifier(
  observation=observation_groups.RAY_CAST,
  symmetry_fn=ray_cast_symmetry,
)

CAMERA_IMAGE = SymmetryModifier(
  observation=observation_groups.CAMERA_IMAGE,
  symmetry_fn=camera_image_symmetry,
)

BASE_LIN_VEL = SymmetryModifier(
  observation=observation_groups.BASE_LIN_VEL,
  symmetry_fn=base_lin_vel_symmetry,
)

BASE_ANG_VEL = SymmetryModifier(
  observation=observation_groups.BASE_ANG_VEL,
  symmetry_fn=base_ang_vel_symmetry,
)

PROJECTED_GRAVITY = SymmetryModifier(
  observation=observation_groups.PROJECTED_GRAVITY,
  symmetry_fn=projected_gravity_symmetry,
)

PUSHING_FORCES = SymmetryModifier(
  observation=observation_groups.PUSHING_FORCES,
  symmetry_fn=pushing_forces_symmetry,
)

PUSHING_TORQUES = SymmetryModifier(
  observation=observation_groups.PUSHING_TORQUES,
  symmetry_fn=pushing_torques_symmetry,
)

BASE_HEIGHT = SymmetryModifier(
  observation=observation_groups.BASE_HEIGHT,
  symmetry_fn=identity_symmetry,
)

HIP_HEIGHTS = SymmetryModifier(
  observation=observation_groups.HIP_HEIGHTS,
  symmetry_fn=a1_hip_height_symmetry,
)

VELOCITY_COMMANDS = SymmetryModifier(
  observation=observation_groups.VELOCITY_COMMANDS,
  symmetry_fn=velocity_commands_symmetry,
)

A1_DOF_POS = SymmetryModifier(
  observation=observation_groups.DOF_POS,
  symmetry_fn=a1_dof_pos_symmetry,
)

A1_DOF_VEL = SymmetryModifier(
  observation=observation_groups.DOF_VEL,
  symmetry_fn=a1_dof_vel_symmetry,
)

A1_ACTIONS = SymmetryModifier(
  observation=observation_groups.ACTIONS,
  symmetry_fn=a1_actions_symmetry,
)

A1_STIFFNESS = SymmetryModifier(
  observation=observation_groups.STIFFNESS,
  symmetry_fn=a1_stiffness_symmetry,
)

A1_DAMPING = SymmetryModifier(
  observation=observation_groups.DAMPING,
  symmetry_fn=a1_damping_symmetry,
)

A1_MOTOR_STRENGTH = SymmetryModifier(
  observation=observation_groups.MOTOR_STRENGTH,
  symmetry_fn=a1_motor_strength_symmetry,
)

A1_MOTOR_ERROR = SymmetryModifier(
  observation=observation_groups.MOTOR_ERROR,
  symmetry_fn=a1_motor_error_symmetry,
)

A1_DOF_FRICTION_CURRICULUM_VALUES = SymmetryModifier(
  observation=observation_groups.DOF_FRICTION_CURRICULUM_VALUES,
  symmetry_fn=a1_dof_friction_curriculum_values_symmetry,
)

A1_FEET_AIR_TIME = SymmetryModifier(
  observation=observation_groups.FEET_AIR_TIME,
  symmetry_fn=a1_feet_symmetry,
)

A1_FEET_CONTACT_TIME = SymmetryModifier(
  observation=observation_groups.FEET_CONTACT_TIME,
  symmetry_fn=a1_feet_symmetry,
)

A1_FEET_CONTACT = SymmetryModifier(
  observation=observation_groups.FEET_CONTACT,
  symmetry_fn=a1_feet_symmetry,
)

T1_DOF_POS = SymmetryModifier(
  observation=observation_groups.DOF_POS,
  symmetry_fn=t1_dof_pos_symmetry,
)

T1_DOF_VEL = SymmetryModifier(
  observation=observation_groups.DOF_VEL,
  symmetry_fn=t1_dof_vel_symmetry,
)

T1_ACTIONS = SymmetryModifier(
  observation=observation_groups.ACTIONS,
  symmetry_fn=t1_actions_symmetry,
)

T1_STIFFNESS = SymmetryModifier(
  observation=observation_groups.STIFFNESS,
  symmetry_fn=t1_stiffness_symmetry,
)

T1_DAMPING = SymmetryModifier(
  observation=observation_groups.DAMPING,
  symmetry_fn=t1_damping_symmetry,
)

GAIT_PROGRESS = SymmetryModifier(
  observation=observation_groups.GAIT_PROGRESS,
  symmetry_fn=identity_symmetry,
)

IMAGE_ENCODER_LATENT = SymmetryModifier(
  observation=observation_groups.IMAGE_ENCODER_LATENT,
  symmetry_fn=lambda env, obs: utils.mirror_latent(obs),
)

BASE_MASS_SCALED = SymmetryModifier(
  observation=observation_groups.BASE_MASS_SCALED,
  symmetry_fn=base_mass_scaled_symmetry,
)
