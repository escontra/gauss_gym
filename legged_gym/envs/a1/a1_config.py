# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR
import math

class A1RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_actions = 12
        env_spacing = 8.0

        # Camera parameters.
        focal_length = 40
        cam_height = 64
        cam_width = 64
        cam_xyz_offset = [0.26, 0.0, 0.03] # Local frame: [x, y, z] meters.
        cam_rpy_offset = [math.pi / 2, math.pi / 2, math.pi] # Local frame[roll, pitch, yaw] radians.

        # Distance / angle from camera trajectory based termination conditions.
        # max_traj_pos_distance = 0.5
        # max_traj_yaw_distance_rad = 0.75

        max_traj_pos_distance = 1.0
        max_traj_yaw_distance_rad = 1.0

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'gaussian'

        # Terrain parameters.
        scene_root = f"{LEGGED_GYM_ROOT_DIR}/scenes"
        curriculum = False
        measure_heights = False

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.5] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # stiffness = {'joint': 20.}  # [N*m/rad]
        # damping = {'joint': 0.5}     # [N*m*s/rad]
        stiffness = {'joint': 50.}  # [N*m/rad]
        damping = {'joint': 1.}     # [N*m*s/rad]
        # stiffness = {'joint': 35.}  # [N*m/rad]
        # damping = {'joint': 0.75}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        torque_limits = 25 # override the urdf

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        name = "a1"
        camera_link_name = "base"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base", "imu"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        feet_edge_pos = [[0., 0., 0.]] # x,y,z [m]
        feet_contact_radius = 0.02
  
    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_motor = True
        leg_motor_strength_range = [0.9, 1.1]
        randomize_com = True
        class com_range:
            x = [-0.05, 0.15]
            y = [-0.1, 0.1]
            z = [-0.05, 0.05]
        randomize_base_mass = True
        added_mass_range = [0., 3.]

    # class rewards( LeggedRobotCfg.rewards ):
    #     only_positive_rewards = True
    #     soft_dof_pos_limit = 0.9
    #     terminate_height = 0.08
    #     base_height_target = 0.40
    #     class scales( LeggedRobotCfg.rewards.scales ):
    #         tracking_lin_vel = 0.
    #         tracking_lin_vel_x = 1.
    #         tracking_lin_vel_y = 1.
    #         tracking_ang_vel = 0.5
    #         lin_vel_z = -0.5

    #         # Added:
    #         dof_acc = -1.25e-7
    #         torques = -0.0001
    #         # dof_pos_limits = -10.0
    #         base_height = -2.0
  
  
    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = False
        terminate_height = 0.08
        soft_dof_pos_limit = 0.8
        max_contact_force = 100.0
        base_height_target = 0.35
        class scales:
            tracking_lin_vel = 0.
            tracking_lin_vel_x = 1.
            tracking_lin_vel_y = 1.
            tracking_ang_vel = 0.5
  
            # tracking_ang_vel = 0.05 # TODO: what?
            # TODO: Need to add these.
            # world_vel_l2norm = -1.
            energy_substeps = -1e-6
            energy = -0.
            alive = 2.
            # penalty for hardware safety
            exceed_dof_pos_limits = -1e-1
            exceed_torque_limits_i = -2e-1

    class commands( LeggedRobotCfg.commands ):
        heading_command = True # if true: compute ang vel command from heading error
        class ranges ( LeggedRobotCfg.commands.ranges ):
            lin_vel = [0.0, 1.0] # min max [m/s]


class A1RoughCfgPPO( LeggedRobotCfgPPO ):

    class policy( LeggedRobotCfgPPO.policy ):
        init_noise_std = 0.5
        # mu_activation = 'tanh'
        mu_activation = None

    runner_class_name = 'Runner'
    # runner_class_name = 'StudentTeacherRunner'
    # runner_class_name = 'OnPolicyRunner'
    class runner( LeggedRobotCfgPPO.runner ):
        # policy_class_name = 'ActorCriticRecurrentWithImages'
        policy_class_name = 'ActorCriticRecurrent'
        algorithm_class_name = 'PPO'
        # algorithm_class_name = 'BehaviorCloning'
        # algorithm_class_name = 'PPO'
        teacher_iterations = 250
        student_teacher_mix_iterations = 750
        run_name = ''
        experiment_name = ''
        load_run = -1
        max_iterations = 10000
    # class algorithm:
    #     # training params
    #     num_learning_epochs = 5
    #     num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
    #     learning_rate = 1.e-3 #5.e-4
    #     max_grad_norm = 1.

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        learning_rate = 1.e-5
        # gamma = 0.995
        gamma = 0.99
        lam = 0.95
        entropy_coef = -0.01
        symmetric_coef = 10.
        num_learning_epochs = 20
        clip_min_std = 0.2
        # value_loss_coef = 1.0
        # use_clipped_value_loss = True
        # clip_param = 0.2
        # num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        # schedule = 'adaptive' # could be adaptive, fixed
        # desired_kl = 0.01
        # # bound_coef = 1.0
        # max_grad_norm = 1.

# class A1RoughCfgPPO( LeggedRobotCfgPPO ):
#     class algorithm( LeggedRobotCfgPPO.algorithm ):
#         entropy_coef = 0.01
#     class runner( LeggedRobotCfgPPO.runner ):
#         run_name = ''
#         experiment_name = 'rough_a1'

  