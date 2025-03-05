from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR
import math

class T1RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 64
        num_actions = 12
        num_observations = 48
        env_spacing = 12.0

        # Camera parameters.
        focal_length = 100
        cam_height = 156
        cam_width = 156
        cam_xyz_offset = [0.1, 0.0, 0.1]  # Local frame: [x, y, z] meters.
        cam_rpy_offset = [0.0, math.pi / 4, 0.0]  # Local frame[roll, pitch, yaw] radians.
        debug_viz_single_image = True

        # Distance / angle from camera trajectory based termination conditions.
        max_traj_pos_distance = 1.0
        max_traj_yaw_distance_rad = 1.0

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'custom'

        # Terrain parameters.
        scene_root = f"{LEGGED_GYM_ROOT_DIR}/scenes/bridge"
        height_offset = -1.7
        curriculum = False
        measure_heights = False

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'Left_Hip_Pitch': -0.2,
            'Left_Hip_Roll': 0.,
            'Left_Hip_Yaw': 0.,
            'Left_Knee_Pitch': 0.4,
            'Left_Ankle_Pitch': 0.,
            'Left_Ankle_Roll': -0.2,

            'Right_Hip_Pitch': -0.2,
            'Right_Hip_Roll': 0.,
            'Right_Hip_Yaw': 0.,
            'Right_Knee_Pitch': 0.4,
            'Right_Ankle_Pitch': 0.,
            'Right_Ankle_Roll': -0.2
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {   'Hip': 200.0, 
                        'Knee': 200.0,
                        'Ankle': 50.0}  # [N*m/rad]
        damping = { 'Hip': 5.0, 'Knee': 5.0,
                    'Ankle': 1.0}  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10
        
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/t1/urdf/T1_locomotion.urdf'
        name = "t1"
        foot_name = 'foot'
        penalize_contacts_on = ["Trunk", "H1", "H2", "AL", "AR", "Waist", "Hip", "Shank", "Ankle"]
        terminate_after_contacts_on = ['Waist']
        flip_visual_attachments = False
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
    
    # class domain_rand( LeggedRobotCfg.domain_rand):
    #     randomize_base_mass = True
    #     added_mass_range = [-5., 5.]
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 1.
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        max_contact_force = 500.
        only_positive_rewards = False
        swing_period = 0.2
        feet_distance_ref = 0.2
        base_height_target: 0.68
        class scales( LeggedRobotCfg.rewards.scales ):
            survival = 0.25
            tracking_lin_vel_x = 1.0
            tracking_lin_vel_y = 1.0
            tracking_ang_vel = 0.5
            base_height = -20.
            orientation = -5.
            torques = -2.e-4
            torque_tiredness = -1.e-2
            power = -2.e-3
            lin_vel_z = -2.
            ang_vel_xy = -0.2
            dof_vel = -1.e-4
            dof_acc = -1.e-7
            root_acc = -1.e-4
            action_rate = -1.
            dof_pos_limits = -1.
            dof_vel_limits = -0.
            torque_limits = -0.
            collision = -1.
            feet_slip = -0.1
            feet_vel_z = -0.
            feet_yaw_diff = -1.
            feet_yaw_mean = -1.
            feet_roll = -0.1
            feet_distance = -1.
            feet_swing = 3.
    
    class commands( LeggedRobotCfg.commands ):
        heading_command = True # if true: compute ang vel command from heading error
        gait_frequency = 1.5
        class ranges ( LeggedRobotCfg.commands.ranges ):
            lin_vel = [0.0, 1.0] # min max [m/s]

class T1RoughCfgPPO( LeggedRobotCfgPPO ):
    
    runner_class_name = 'OnPolicyRunner'
    class runner( LeggedRobotCfgPPO.runner ):
        # policy_class_name = 'ActorCriticRecurrentWithImages'
        policy_class_name = 'ActorCritic'
        # algorithm_class_name = 'BehaviorCloning'
        algorithm_class_name = 'PPO'
        teacher_iterations = 250
        student_teacher_mix_iterations = 750
        run_name = ''
        experiment_name = 'rough_t1'
        load_run = -1

    class algorithm:
        # training params
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        max_grad_norm = 1.



  