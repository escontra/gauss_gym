import torch

# solves circular imports of LeggedEnv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legged_gym.envs.locomotion import LeggedEnv, NavEnv

def projected_gravity(env: "LeggedEnv", params):
    return env.projected_gravity


def base_lin_vel(env: "LeggedEnv", params):
    return env.base_lin_vel

def base_ang_vel(env: "LeggedEnv", params):
    return env.base_ang_vel

def velocity_commands(env: "LeggedEnv", params):
    return env.commands[:, :3] # env.command_generator.get_command()

def dof_pos(env: "ANY_ENV", params):
    return env.dof_pos - env.default_dof_pos

def dof_vel(env: "ANY_ENV", params):
    return env.dof_vel

def actions(env: "ANY_ENV", params):
    return env.actions

def ray_cast(env: "ANY_ENV", params):
    sensor = env.sensors[params["sensor"]]
    heights = env.root_states[:, 2].unsqueeze(1) - 0.5 - sensor.get_data()[..., 2]
    return heights

def gs_render(env: "ANY_ENV", params):
    sensor = env.sensors[params["sensor"]]
    return sensor.get_data()