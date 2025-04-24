import torch

# solves circular imports of LeggedEnv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legged_gym.envs.locomotion import LeggedEnv, NavEnv

from legged_gym.utils.math import quat_rotate_inverse

def projected_gravity(env: "LeggedEnv", params, is_real=False):
    if is_real:
        return quat_rotate_inverse(
            torch.tensor(env.low_state_buffer.imu.quaternion).unsqueeze(0),
            env.gravity_vec,
        ).to(env.model_device)
    else:
      return env.projected_gravity

def pushing_forces(env: "LeggedEnv", params, is_real=False):
    return env.pushing_forces[:, env.base_link_index, :]

def pushing_torques(env: "LeggedEnv", params, is_real=False):
    return env.pushing_torques[:, env.base_link_index, :]

def base_mass_scaled(env: "LeggedEnv", params, is_real=False):
    return env.base_mass_scaled

def dof_friction_curriculum_values(env: "LeggedEnv", params, is_real=False):
    return env.dof_friction_curriculum_values

def gait_progress(env: "LeggedEnv", params):
    return torch.cat((
      (torch.cos(2 * torch.pi * env.gait_process) * (env.gait_frequency > 1.0e-8).float()).unsqueeze(-1),
      (torch.sin(2 * torch.pi * env.gait_process) * (env.gait_frequency > 1.0e-8).float()).unsqueeze(-1),
    ), dim = -1)

def base_lin_vel(env: "LeggedEnv", params, is_real=False):
    return env.base_lin_vel

def base_ang_vel(env: "LeggedEnv", params, is_real=False):
    if is_real:
        return torch.tensor(env.low_state_buffer.imu.gyroscope, device= env.model_device).unsqueeze(0)
    else:
        return env.base_ang_vel

def velocity_commands(env: "LeggedEnv", params, is_real=False):
    if is_real:
      return env.command_buf
    else:
      return env.commands[:, :3]

def dof_pos(env: "ANY_ENV", params, is_real=False):
    if is_real:
        return torch.tensor([
            env.low_state_buffer.motorState[env.dof_map[i]].q for i in range(12)
        ], dtype= torch.float32, device= env.model_device).unsqueeze(0) - env.default_dof_pos
    else:
        return env.dof_pos - env.default_dof_pos

def dof_vel(env: "ANY_ENV", params, is_real=False):
    if is_real:
      return torch.tensor([
          env.low_state_buffer.motorState[env.dof_map[i]].dq for i in range(12)
      ], dtype= torch.float32, device= env.model_device).unsqueeze(0)
    else:
      return env.dof_vel

def actions(env: "ANY_ENV", params, is_real=False):
    return env.actions

def stiffness(env: "ANY_ENV", params, is_real=False):
    return env.stiffness * env.dof_stiffness_multiplier

def damping(env: "ANY_ENV", params, is_real=False):
    return env.damping * env.dof_damping_multiplier

def motor_strength(env: "ANY_ENV", params, is_real=False):
    return env.motor_strength_multiplier

def ray_cast(env: "ANY_ENV", params, is_real=False):
    sensor = env.sensors[params.sensor]
    heights = env.root_states[:, 2].unsqueeze(1) - 0.5 - sensor.get_data()[..., 2]
    return heights

def gs_render(env: "ANY_ENV", params, is_real=False):
    sensor = env.sensors[params.sensor]
    return sensor.get_data()

def base_height(env: "ANY_ENV", params, is_real=False):
    sensor = env.sensors[params.sensor]
    return sensor.get_data()

def hip_heights(env: "ANY_ENV", params, is_real=False):
    sensor = env.sensors[params.sensor]
    return sensor.get_data()

def feet_air_time(env: "ANY_ENV", params, is_real=False):
    return env.feet_air_time

def feet_contact_time(env: "ANY_ENV", params, is_real=False):
    return env.feet_contact_time

def feet_contact(env: "ANY_ENV", params, is_real=False):
    return env.feet_contact.to(torch.float32)
