import dataclasses
from legged_gym.utils import observations as O
from typing import Callable, Union, Tuple, List, Dict


@dataclasses.dataclass(frozen=True)
class Observation:
  name: str
  func: Callable
  sensor: Union[str, None] = None
  noise: Union[float, None] = None
  clip: Union[Tuple[float, float], None] = None
  scale: Union[float, None] = None
  latency_range: Tuple[float, float] = (0., 0.)
  refresh_duration: float = 0.


@dataclasses.dataclass(frozen=True)
class ObservationGroup:
  name: str
  observations: List[Observation]
  add_noise: bool = False
  add_latency: bool = False
  latency_resampling_interval_s: Union[float, None] = None


def observation_groups_from_dict(config: Dict) -> List[ObservationGroup]:
  observation_groups = []
  for name, cfg in config.items():
    observations = [eval(obs_name) for obs_name in cfg['observations']]
    observation_groups.append(
      ObservationGroup(
        name=name,
        **{**cfg, 'observations': observations},
      )
    )
  return observation_groups


# Common observations.
BASE_LIN_VEL = Observation(
  name="base_lin_vel",
  func=O.base_lin_vel,
  scale=2.0,
)

BASE_ANG_VEL = Observation(
  name="base_ang_vel",
  func=O.base_ang_vel,
  noise=0.2,
  scale=0.25,
  latency_range=(0.04-0.0025, 0.04+0.0075),
)

PROJECTED_GRAVITY = Observation(
  name="projected_gravity",
  func=O.projected_gravity,
  noise=0.05,
  latency_range=(0.04-0.0025, 0.04+0.0075),
)

VELOCITY_COMMANDS = Observation(
  name="velocity_commands", func=O.velocity_commands,
  scale=[2.0, 2.0, 0.25],
  latency_range=(0.04-0.0025, 0.04+0.0075),
)

DOF_POS = Observation(
  name="dof_pos",
  func=O.dof_pos,
  noise=0.01,
  scale=1.0,
  latency_range=(0.04-0.0025, 0.04+0.0075),
)

DOF_VEL = Observation(
  name="dof_vel",
  func=O.dof_vel,
  noise=1.5,
  scale=0.1,
  latency_range=(0.04-0.0025, 0.04+0.0075),
)

ACTIONS = Observation(
  name="actions",
  func=O.actions,
)

CAMERA_IMAGES = Observation(
  name="camera_images",
  func=O.gs_render,
  sensor="gs_renderer",
  latency_range=(0.25, 0.30),
  refresh_duration=1/10,
)

GAIT_PROGRESS = Observation(
  name="gait_progress",
  func=O.gait_progress,
)

RAY_CAST = Observation(
  name="ray_cast",
  func=O.ray_cast,
  sensor="raycast_grid",
  clip=(-1.0, 1.0),
)

BASE_MASS_SCALED = Observation(
  name="base_mass_scaled",
  func=O.base_mass_scaled,
)

DOF_FRICTION_CURRICULUM_VALUES = Observation(
  name="dof_friction_curriculum_values",
  func=O.dof_friction_curriculum_values,
)

PUSHING_FORCES = Observation(
  name="pushing_forces",
  func=O.pushing_forces,
  scale=0.1,
)

PUSHING_TORQUES = Observation(
  name="pushing_torques",
  func=O.pushing_torques,
  scale=0.5,
)

BASE_HEIGHT = Observation(
  name="base_height",
  func=O.base_height,
  scale=1.0,
  sensor="base_height_raycaster",
)

HIP_HEIGHTS = Observation(
  name="hip_heights",
  func=O.hip_heights,
  scale=1.0,
  sensor="hip_height_raycaster",
)

FEET_AIR_TIME = Observation(
  name="feet_air_time",
  func=O.feet_air_time,
)

FEET_CONTACT_TIME = Observation(
  name="feet_contact_time",
  func=O.feet_contact_time,
)

FEET_CONTACT = Observation(
  name="feet_contact",
  func=O.feet_contact,
)