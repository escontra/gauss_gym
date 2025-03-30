import dataclasses
from legged_gym.utils import observations as O
from typing import Callable, Union, Tuple, List
import copy


@dataclasses.dataclass
class Observation:
  name: str
  func: Callable
  sensor: Union[str, None] = None
  noise: Union[float, None] = None
  clip: Union[Tuple[float, float], None] = None
  scale: Union[float, None] = None
  latency_range: Tuple[float, float] = (0., 0.)
  refresh_duration: float = 0.


@dataclasses.dataclass
class ObservationGroup:
  name: str
  observations: List[Observation]
  add_noise: bool
  add_latency: bool
  is_recurrent: bool


TEACHER_OBSERVATION_GROUP = ObservationGroup(
  name="teacher_observations",
  observations=[
    Observation(
      name="base_lin_vel",
      func=O.base_lin_vel,
      scale=2.0,
    ),
    Observation(
      name="base_ang_vel",
      func=O.base_ang_vel,
      scale=0.25,
    ),
    Observation(
      name="projected_gravity",
      func=O.projected_gravity,
    ),
    Observation(
      name="velocity_commands",
      func=O.velocity_commands,
      scale=[2.0, 2.0, 0.25],
    ),
    Observation(
      name="dof_pos",
      func=O.dof_pos,
    ),
    Observation(
      name="dof_vel",
      func=O.dof_vel,
      scale=0.05,
    ),
    Observation(
      name="actions",
      func=O.actions,
    ),
    Observation(
      name="ray_cast",
      func=O.ray_cast,
      sensor="raycast_grid",
      clip=(-1.0, 1.0),
    ),
    Observation(
      name="base_mass_scaled",
      func=O.base_mass_scaled,
    ),
    Observation(
      name="dof_friction_curriculum_values",
      func=O.dof_friction_curriculum_values,
    ),
    Observation(
      name="pushing_forces",
      func=O.pushing_forces,
      scale=0.1,
    ),
    Observation(
      name="pushing_torques",
      func=O.pushing_torques,
      scale=0.5,
    ),
  ],
  add_noise=False,
  add_latency=False,
  is_recurrent=True,
)

STUDENT_OBSERVATION_GROUP = ObservationGroup(
  name="student_observations",
  observations=[
    Observation(
      name="base_ang_vel",
      func=O.base_ang_vel,
      noise=0.2,
      scale=0.25,
      latency_range=(0.04-0.0025, 0.04+0.0075),
    ),
    Observation(
      name="projected_gravity",
      func=O.projected_gravity,
      noise=0.05,
      latency_range=(0.04-0.0025, 0.04+0.0075),
    ),
    Observation(
      name="velocity_commands", func=O.velocity_commands, scale=[2.0, 2.0, 0.25],
      latency_range=(0.04-0.0025, 0.04+0.0075),
    ),
    Observation(
      name="dof_pos",
      func=O.dof_pos,
      noise=0.01,
      scale=1.0,
      latency_range=(0.04-0.0025, 0.04+0.0075),
    ),
    Observation(
      name="dof_vel",
      func=O.dof_vel,
      noise=1.5,
      scale=0.05,
      latency_range=(0.04-0.0025, 0.04+0.0075),
    ),
    Observation(
      name="actions",
      func=O.actions,
    ),
  ],
  add_noise=False,
  add_latency=False,
  is_recurrent=True,
)


# A1.
TEACHER_OBSERVATION_GROUP_A1_IMAGE = copy.deepcopy(TEACHER_OBSERVATION_GROUP)
STUDENT_OBSERVATION_GROUP_A1_IMAGE = copy.deepcopy(STUDENT_OBSERVATION_GROUP)
STUDENT_OBSERVATION_GROUP_A1_IMAGE.observations.append(
  Observation(
    name="camera_images",
    func=O.gs_render,
    sensor="gs_renderer",
    latency_range=(0.25, 0.30),
    refresh_duration=1/10,
  )
)

# T1.
TEACHER_OBSERVATION_GROUP_T1 = copy.deepcopy(TEACHER_OBSERVATION_GROUP)
TEACHER_OBSERVATION_GROUP_T1.observations.extend([
  Observation(
    name="gait_progress",
    func=O.gait_progress,
  )
])

STUDENT_OBSERVATION_GROUP_T1_PROPRIO = copy.deepcopy(STUDENT_OBSERVATION_GROUP)
STUDENT_OBSERVATION_GROUP_T1_PROPRIO.observations.append(
  Observation(
    name="gait_progress",
    func=O.gait_progress,
  )
)
