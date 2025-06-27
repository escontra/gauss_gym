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
  ignore_in_observation_manager: bool = False


@dataclasses.dataclass(frozen=True)
class ObservationGroup:
  name: str
  observations: List[Observation]
  add_noise: bool = False

  # Latency.
  add_latency: bool = False
  latency_resampling_interval_s: Union[float, None] = None

  # Sync latencies for these observations. Requires that each Observation has the same
  # latency range.
  sync_latency: Union[List[Observation], None] = None


def observation_groups_from_dict(config: Dict) -> List[ObservationGroup]:
  observation_groups = []
  for name, cfg in config.items():
    observations = [eval(obs_name) for obs_name in cfg['observations']]
    if 'sync_latency' in cfg:
      sync_latency = [eval(obs_name) for obs_name in cfg['sync_latency']]
      assert all(obs.latency_range == sync_latency[0].latency_range for obs in sync_latency), \
        "All observations in sync_latency must have the same latency range."
    else:
      sync_latency = None
    observation_groups.append(
      ObservationGroup(
        name=name,
        **{**cfg, 'observations': observations, 'sync_latency': sync_latency},
      )
    )
  return observation_groups


# Common observations.
BASE_LIN_VEL = Observation(
  name="base_lin_vel",
  func=O.base_lin_vel,
)

BASE_ANG_VEL = Observation(
  name="base_ang_vel",
  func=O.base_ang_vel,
  noise=0.2,
  latency_range=(0.04-0.0125, 0.04+0.0075),
)

PROJECTED_GRAVITY = Observation(
  name="projected_gravity",
  func=O.projected_gravity,
  noise=0.05,
  latency_range=(0.04-0.0125, 0.04+0.0075),
)

VELOCITY_COMMANDS = Observation(
  name="velocity_commands", func=O.velocity_commands,
  latency_range=(0.04-0.0125, 0.04+0.0075),
)

DOF_POS = Observation(
  name="dof_pos",
  func=O.dof_pos,
  noise=0.01,
  latency_range=(0.04-0.0125, 0.04+0.0075),
)

DOF_VEL = Observation(
  name="dof_vel",
  func=O.dof_vel,
  noise=1.5,
  latency_range=(0.04-0.0125, 0.04+0.0075),
)

ACTIONS = Observation(
  name="actions",
  func=O.actions,
)

STIFFNESS = Observation(
  name="stiffness",
  func=O.stiffness,
)

DAMPING = Observation(
  name="damping",
  func=O.damping,
)

MOTOR_STRENGTH = Observation(
  name="motor_strength",
  func=O.motor_strength,
)

CAMERA_IMAGE = Observation(
  name="camera_image",
  func=O.gs_render,
  sensor="gs_renderer",
  latency_range=(0.12, 0.3),
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
)

PUSHING_TORQUES = Observation(
  name="pushing_torques",
  func=O.pushing_torques,
)

BASE_HEIGHT = Observation(
  name="base_height",
  func=O.base_height,
  sensor="base_height_raycaster",
)

HIP_HEIGHTS = Observation(
  name="hip_heights",
  func=O.hip_heights,
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

IMAGE_ENCODER_LATENT = Observation(
  name="image_encoder",
  func=None,
  ignore_in_observation_manager=True,
)
