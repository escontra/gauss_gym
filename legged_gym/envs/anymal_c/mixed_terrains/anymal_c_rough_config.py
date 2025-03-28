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

anymal_c_action_scale = 0.5
anymal_c_default_joint_angles = { # = target angles [rad] when action = 0.0
    "LF_HAA": 0.0,
    "LH_HAA": 0.0,
    "RF_HAA": -0.0,
    "RH_HAA": -0.0,

    "LF_HFE": 0.4,
    "LH_HFE": -0.4,
    "RF_HFE": 0.4,
    "RH_HFE": -0.4,

    "LF_KFE": -0.8,
    "LH_KFE": 0.8,
    "RF_KFE": -0.8,
    "RH_KFE": 0.8,
}


class AnymalCRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_actions = 12
        env_spacing = 8.0

        # Camera parameters.
        focal_length = 40
        cam_height = 64
        cam_width = 64
        cam_xyz_offset = [0.44665475921048414, 0.009719827812061002, 0.17764090819492523] # Local frame: [x, y, z] meters.
        cam_rpy_offset = [math.pi / 2, math.pi / 2, math.pi] # Local frame[roll, pitch, yaw] radians.

        # Distance / angle from camera trajectory based termination conditions.
        max_traj_pos_distance = 1.0
        max_traj_yaw_distance_rad = 1.0

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'gaussian'

        # Terrain parameters.
        scene_root = f"{LEGGED_GYM_ROOT_DIR}/scenes"
        curriculum = False
        measure_heights = False

        cams_yaw_only = False

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.7] # x,y,z [m]
        default_joint_angles = anymal_c_default_joint_angles

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {'HAA': 80., 'HFE': 80., 'KFE': 80.}  # [N*m/rad]
        damping = {'HAA': 2., 'HFE': 2., 'KFE': 2.}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = anymal_c_action_scale
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = True
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"
        computer_clip_torque = False
        motor_clip_torque = False

    class asset( LeggedRobotCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/anymal_c/urdf/anymal_c.urdf"
        name = "anymal_c"
        camera_link_name = "base"
        base_link_name = "base"
        foot_name = "FOOT"
        penalize_contacts_on = ["SHANK", "THIGH"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        # feet_edge_pos = [[0., 0., -0.0225]] # x,y,z [m]
        feet_edge_pos = [[0., 0., 0.0225]] # x,y,z [m]
        feet_contact_radius = 0.03 + 1e-3

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-5., 5.]
        randomize_motor = True
        leg_motor_strength_range = [0.9, 1.1]
        randomize_com = True
        class com_range:
            x = [-0.05, 0.15]
            y = [-0.1, 0.1]
            z = [-0.05, 0.05]
        push_robots = True
        push_interval_s = 10.
        max_push_vel_xy = 1.
        randomize_friction = True
        friction_range = [0.5, 1.25]

    class normalization( LeggedRobotCfg.normalization ):
        # clip_actions = 1.
        clip_actions_method = None # "hard"
  
    class rewards( LeggedRobotCfg.rewards ):
        base_height_target = 0.5
        max_contact_force = 500.
        only_positive_rewards = True
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 0.
            tracking_lin_vel_x = 1.
            tracking_lin_vel_y = 1.
            tracking_ang_vel = 0.5
            lin_vel_z = -0.5
            pass

    class commands( LeggedRobotCfg.commands ):
        heading_command = True # if true: compute ang vel command from heading error
        class ranges ( LeggedRobotCfg.commands.ranges ):
            lin_vel = [0.0, 1.0] # min max [m/s]

class AnymalCRoughCfgPPO( LeggedRobotCfgPPO ):

    class policy( LeggedRobotCfgPPO.policy ):
        init_noise_std = 0.5
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

        # value_loss_coef = 1.0
        # use_clipped_value_loss = True
        # clip_param = 0.2
        # num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        # schedule = 'adaptive' # could be adaptive, fixed
        # desired_kl = 0.01
        # # bound_coef = 1.0
        # max_grad_norm = 1.