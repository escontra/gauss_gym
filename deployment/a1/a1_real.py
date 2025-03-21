import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import json
import os
import os.path as osp
from collections import OrderedDict
from typing import Tuple

import rospy
from unitree_legged_msgs.msg import LowState
from unitree_legged_msgs.msg import LegsCmd
from unitree_legged_msgs.msg import Float32MultiArrayStamped
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

import ros_numpy
from legged_gym.utils.math import quat_rotate_inverse
from legged_gym.teacher import observation_groups

@torch.no_grad()
def resize2d(img, size):
    return (F.adaptive_avg_pool2d(Variable(img), size)).data

class UnitreeA1Real:
    """ This is the handler that works for ROS 1 on unitree. """
    def __init__(self,
            robot_namespace= "a112138",
            low_state_topic= "/low_state",
            legs_cmd_topic= "/legs_cmd",
            forward_depth_topic = "/camera/depth/image_rect_raw",
            forward_depth_embedding_dims = None,
            odom_topic= "/odom/filtered",
            lin_vel_deadband= 0.1,
            ang_vel_deadband= 0.1,
            move_by_wireless_remote= False, # if True, command will not listen to move_cmd_subscriber, but wireless remote.
            cfg= dict(),
            extra_cfg= dict(),
            model_device= torch.device("cpu"),
        ):
        """
        NOTE:
            * Must call start_ros() before using this class's get_obs() and send_action()
            * Joint order of simulation and of real A1 protocol are different, see dof_names
            * We store all joints values in the order of simulation in this class
        Args:
            forward_depth_embedding_dims: If a real number, the obs will not be built as a normal env.
                The segment of obs will be subsituted by the embedding of forward depth image from the
                ROS topic.
            cfg: same config from a1_config but a dict object.
            extra_cfg: some other configs that is hard to load from file.
        """
        self.model_device = model_device
        self.num_envs = 1
        self.robot_namespace = robot_namespace
        self.low_state_topic = low_state_topic
        self.legs_cmd_topic = legs_cmd_topic
        self.forward_depth_topic = forward_depth_topic
        self.forward_depth_embedding_dims = forward_depth_embedding_dims
        self.odom_topic = odom_topic
        self.lin_vel_deadband = lin_vel_deadband
        self.ang_vel_deadband = ang_vel_deadband
        self.move_by_wireless_remote = move_by_wireless_remote
        self.cfg = cfg
        self.extra_cfg = dict(
            torque_limits= torch.tensor([33.5] * 12, dtype= torch.float32, device= self.model_device, requires_grad= False), # Nm
            # torque_limits= torch.tensor([1, 5, 5] * 4, dtype= torch.float32, device= self.model_device, requires_grad= False), # Nm
            dof_map= [ # from isaacgym simulation joint order to URDF order
                3, 4, 5,
                0, 1, 2,
                9, 10,11,
                6, 7, 8,
            ], # real_joint_idx = dof_map[sim_joint_idx]
            dof_names= [ # NOTE: order matters. This list is the order in simulation.
                "FL_hip_joint",
                "FL_thigh_joint",
                "FL_calf_joint",

                "FR_hip_joint",
                "FR_thigh_joint",
                "FR_calf_joint",
                
                "RL_hip_joint",
                "RL_thigh_joint",
                "RL_calf_joint",
                
                "RR_hip_joint",
                "RR_thigh_joint",
                "RR_calf_joint",
            ],
            # motor strength is multiplied directly to the action.
            motor_strength= torch.ones(12, dtype= torch.float32, device= self.model_device, requires_grad= False),
        ); self.extra_cfg.update(extra_cfg)
        if "torque_limits" in self.cfg["control"]:
            if isinstance(self.cfg["control"]["torque_limits"], (tuple, list)):
                for i in range(len(self.cfg["control"]["torque_limits"])):
                    self.extra_cfg["torque_limits"][i] = self.cfg["control"]["torque_limits"][i]
            else:
                self.extra_cfg["torque_limits"][:] = self.cfg["control"]["torque_limits"]
        self.command_buf = torch.zeros((self.num_envs, 3,), device= self.model_device, dtype= torch.float32) # zeros for initialization
        self.actions = torch.zeros((1, 12), device= model_device, dtype= torch.float32)

        self.process_configs()
    
    def start_ros(self):
        # initialze several buffers so that the system works even without message update.
        # self.low_state_buffer = LowState() # not initialized, let input message update it.
        self.base_position_buffer = torch.zeros((self.num_envs, 3), device= self.model_device, requires_grad= False)
        self.legs_cmd_publisher = rospy.Publisher(
            self.robot_namespace + self.legs_cmd_topic,
            LegsCmd,
            queue_size= 1,
        )
        # self.debug_publisher = rospy.Publisher(
        #     "/DNNmodel_debug",
        #     Float32MultiArray,
        #     queue_size= 1,
        # )
        # NOTE: this launches the subscriber callback function
        self.low_state_subscriber = rospy.Subscriber(
            self.robot_namespace + self.low_state_topic,
            LowState,
            self.update_low_state,
            queue_size= 1,
        )
        self.odom_subscriber = rospy.Subscriber(
            self.robot_namespace + self.odom_topic,
            Odometry,
            self.update_base_pose,
            queue_size= 1,
        )
        if not self.move_by_wireless_remote:
            self.move_cmd_subscriber = rospy.Subscriber(
                "/cmd_vel",
                Twist,
                self.update_move_cmd,
                queue_size= 1,
            )
        self.pose_cmd_subscriber = rospy.Subscriber(
            "/body_pose",
            Pose,
            self.dummy_handler,
            queue_size= 1,
        )
    
    def wait_untill_ros_working(self):
        rate = rospy.Rate(100)
        while not hasattr(self, "low_state_buffer"):
            rate.sleep()
        rospy.loginfo("UnitreeA1Real.low_state_buffer acquired, stop waiting.")
        
    def process_configs(self):
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.gravity_vec = torch.zeros((self.num_envs, 3), dtype= torch.float32)
        self.gravity_vec[:, self.up_axis_idx] = -1

        self.observation_groups = [getattr(observation_groups, name) for name in self.cfg["observations"]["observation_groups"]]

        self.d_gains = torch.tensor(self.cfg["control"]["damping"]["joint"], device= self.model_device, dtype= torch.float32)
        self.p_gains = torch.tensor(self.cfg["control"]["stiffness"]["joint"], device= self.model_device, dtype= torch.float32)

        if not isinstance(self.cfg["control"]["damping"]["joint"], (list, tuple)):
            self.cfg["control"]["damping"]["joint"] = [self.cfg["control"]["damping"]["joint"]] * 12
        if not isinstance(self.cfg["control"]["stiffness"]["joint"], (list, tuple)):
            self.cfg["control"]["stiffness"]["joint"] = [self.cfg["control"]["stiffness"]["joint"]] * 12
        self.default_dof_pos = torch.zeros(12, device= self.model_device, dtype= torch.float32)
        for i in range(12):
            name = self.extra_cfg["dof_names"][i]
            default_joint_angle = self.cfg["init_state"]["default_joint_angles"][name]
            self.default_dof_pos[i] = default_joint_angle

        self.computer_clip_torque = self.cfg["control"].get("computer_clip_torque", True)
        rospy.loginfo("Computer Clip Torque (onboard) is " + str(self.computer_clip_torque))
        if self.computer_clip_torque:
            self.torque_limits = self.extra_cfg["torque_limits"]
            rospy.loginfo("[Env] torque limit: {:.1f} {:.1f} {:.1f}".format(*self.torque_limits[:3]))
        
        # store config values to attributes to improve speed
        self.clip_obs = self.cfg["normalization"]["clip_observations"]
        self.control_type = self.cfg["control"]["control_type"]
        self.action_scale = self.cfg["control"]["action_scale"]
        rospy.loginfo("[Env] action scale: {:.1f}".format(self.action_scale))
        self.clip_actions = self.cfg["normalization"]["clip_actions"]
        if self.cfg["normalization"].get("clip_actions_method", None) == "hard":
            rospy.loginfo("clip_actions_method with hard mode")
            rospy.loginfo("clip_actions: " + str(self.cfg["normalization"]["clip_actions"]))
            self.clip_actions_method = "hard"
            self.clip_actions_low = torch.tensor(-1 * self.cfg["normalization"]["clip_actions"], device= self.model_device, dtype= torch.float32)
            self.clip_actions_high = torch.tensor(self.cfg["normalization"]["clip_actions"], device= self.model_device, dtype= torch.float32)
        else:
            rospy.loginfo("clip_actions_method is " + str(self.cfg["normalization"].get("clip_actions_method", None)))
        self.dof_map = self.extra_cfg["dof_map"]

        # get ROS params for hardware configs
        self.joint_limits_high = torch.tensor([
            rospy.get_param(self.robot_namespace + "/joint_limits/{}_max".format(s)) \
            for s in ["hip", "thigh", "calf"] * 4
        ])
        self.joint_limits_low = torch.tensor([
            rospy.get_param(self.robot_namespace + "/joint_limits/{}_min".format(s)) \
            for s in ["hip", "thigh", "calf"] * 4
        ])
    
    def clip_action_before_scale(self, actions):
        actions = torch.clip(actions, -self.clip_actions, self.clip_actions)
        if getattr(self, "clip_actions_method", None) == "hard":
            actions = torch.clip(actions, self.clip_actions_low, self.clip_actions_high)
        return actions

    def clip_by_torque_limit(self, actions_scaled):
        """ Different from simulation, we reverse the process and clip the actions directly,
        so that the PD controller runs in robot but not our script.
        """
        control_type = self.cfg["control"]["control_type"]
        if control_type == "P":
            p_limits_low = (-self.torque_limits) + self.d_gains*self.dof_vel
            p_limits_high = (self.torque_limits) + self.d_gains*self.dof_vel
            actions_low = (p_limits_low/self.p_gains) - self.default_dof_pos + self.dof_pos
            actions_high = (p_limits_high/self.p_gains) - self.default_dof_pos + self.dof_pos
        else:
            raise NotImplementedError

        return torch.clip(actions_scaled, actions_low, actions_high)

    """ Get obs components and cat to a single obs input """
    def compute_observation(self):
        """Use the updated low_state_buffer to compute observation vector."""
        assert hasattr(self, "legs_cmd_publisher"), "start_ros() not called, ROS handlers are not initialized!"
        obs_dict = {}
        for group in self.observation_groups:
          if "teacher" in group.name:
              continue
          obs_dict[group.name] = {}
          for observation in group.observations:
              obs = observation.func(self, observation, is_real=True)
              if observation.name == "dof_pos":
                  self.dof_pos = obs + self.default_dof_pos
              if observation.name == "dof_vel":
                  self.dof_vel = obs
              if observation.clip:
                  obs = obs.clip(min=observation.clip[0], max=observation.clip[1])
              if observation.scale is not None:
                  scale = observation.scale
                  if isinstance(scale, list):
                      scale = torch.tensor(scale, device=obs.device)[None]
                  obs = scale * obs
              obs_dict[group.name][observation.name] = obs

        self.obs_dict = obs_dict


    """ The methods combined with outer model forms the step function
    NOTE: the outer user handles the loop frequency.
    """
    def send_action(self, actions):
        """ The function that send commands to the real robot.
        """
        self.actions = self.clip_action_before_scale(actions)
        if self.computer_clip_torque:
            robot_coordinates_action = self.clip_by_torque_limit(actions * self.action_scale) + self.default_dof_pos.unsqueeze(0)
        else:
            rospy.logwarn_throttle(60, "You are using control without any torque clip. The network might output torques larger than the system can provide.")
            robot_coordinates_action = self.actions * self.action_scale + self.default_dof_pos.unsqueeze(0)

        # debugging and logging
        # transfered_action = torch.zeros_like(self.actions[0])
        # for i in range(12):
        #     transfered_action[self.dof_map[i]] = self.actions[0, i] + self.default_dof_pos[i]
        # self.debug_publisher.publish(Float32MultiArray(data=
        #     transfered_action\
        #     .cpu().numpy().astype(np.float32).tolist()
        # ))

        # restrict the target action delta in order to avoid robot shutdown (maybe there is another solution)
        # robot_coordinates_action = torch.clip(
        #     robot_coordinates_action,
        #     self.dof_pos - 0.3,
        #     self.dof_pos + 0.3,
        # )

        # wrap the message and publish
        self.publish_legs_cmd(robot_coordinates_action)

    def publish_legs_cmd(self, robot_coordinates_action, kp= None, kd= None):
        """ publish the joint position directly to the robot. NOTE: The joint order from input should
        be in simulation order. The value should be absolute value rather than related to dof_pos.
        """
        robot_coordinates_action = torch.clip(
            robot_coordinates_action.cpu(),
            self.joint_limits_low,
            self.joint_limits_high,
        )
        legs_cmd = LegsCmd()
        for sim_joint_idx in range(12):
            real_joint_idx = self.dof_map[sim_joint_idx]
            legs_cmd.cmd[real_joint_idx].mode = 10
            legs_cmd.cmd[real_joint_idx].q = robot_coordinates_action[0, sim_joint_idx] if self.control_type == "P" else rospy.get_param(self.robot_namespace + "/PosStopF", (2.146e+9))
            legs_cmd.cmd[real_joint_idx].dq = 0.
            legs_cmd.cmd[real_joint_idx].tau = 0.
            legs_cmd.cmd[real_joint_idx].Kp = self.p_gains[sim_joint_idx] if kp is None else kp
            legs_cmd.cmd[real_joint_idx].Kd = self.d_gains[sim_joint_idx] if kd is None else kd
        self.legs_cmd_publisher.publish(legs_cmd)

    def get_obs(self):
        """ The function that refreshes the buffer and return the observation vector.
        """
        self.compute_observation()
        return self.obs_dict


    """ ROS callbacks and handlers that update the buffer """
    def update_low_state(self, ros_msg):
        self.low_state_buffer = ros_msg
        if self.move_by_wireless_remote:
            self.command_buf[0, 0] = self.low_state_buffer.wirelessRemote.ly
            self.command_buf[0, 1] = -self.low_state_buffer.wirelessRemote.lx # right-moving stick is positive
            self.command_buf[0, 2] = -self.low_state_buffer.wirelessRemote.rx # right-moving stick is positive
            # set the command to zero if it is too small
            if np.linalg.norm(self.command_buf[0, :2]) < self.lin_vel_deadband:
                self.command_buf[0, :2] = 0.
            if np.abs(self.command_buf[0, 2]) < self.ang_vel_deadband:
                self.command_buf[0, 2] = 0.
        self.low_state_get_time = rospy.Time.now()

    def update_base_pose(self, ros_msg):
        """ update robot odometry for position """
        self.base_position_buffer[0, 0] = ros_msg.pose.pose.position.x
        self.base_position_buffer[0, 1] = ros_msg.pose.pose.position.y
        self.base_position_buffer[0, 2] = ros_msg.pose.pose.position.z

    def update_move_cmd(self, ros_msg):
        self.command_buf[0, 0] = ros_msg.linear.x
        self.command_buf[0, 1] = ros_msg.linear.y
        self.command_buf[0, 2] = ros_msg.angular.z

    def dummy_handler(self, ros_msg):
        """ To meet the need of teleop-legged-robots requirements """
        pass
