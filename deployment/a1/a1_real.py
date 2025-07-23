import numpy as np
import os
import pygame

os.environ["SDL_VIDEODRIVER"] = "dummy"

import rospy
from unitree_legged_msgs.msg import LowState
from unitree_legged_msgs.msg import LegsCmd
from unitree_legged_msgs.msg import Float32MultiArrayStamped
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy

from legged_gym.utils import observation_groups


ROBOT_JOINT_ORDER = [ # a1_description joint order.
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",

    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",

    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",

    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]


class UnitreeA1Real:
    """ This is the handler that works for ROS 1 on unitree. """
    def __init__(self,
            robot_namespace: str = "a112138",
            low_state_topic: str = "/low_state",
            legs_cmd_topic: str = "/legs_cmd",
            image_topic: str = "/camera/color/image_raw",
            image_embedding_topic: str = "/image_encoder",
            odom_topic: str = "/odom/filtered",
            gamepad_topic: str = "/sdl_gamepad/joy",
            move_by_gamepad=True, # if True, command will not listen to move_cmd_subscriber, but gamepad.
            cfg=dict(),
            deploy_cfg=dict(),
            vision_only: bool = False,
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
        self.num_envs = 1
        self.robot_namespace = robot_namespace
        self.low_state_topic = low_state_topic
        self.legs_cmd_topic = legs_cmd_topic
        self.image_topic = image_topic
        self.image_embedding_topic = image_embedding_topic
        self.odom_topic = odom_topic
        self.gamepad_topic = gamepad_topic
        self.move_by_gamepad = move_by_gamepad
        self.vision_only = vision_only

        self.cfg = cfg
        self.deploy_cfg = deploy_cfg

        # Buffers for commands and actions.
        self.command_buf = np.zeros((self.num_envs, 3,), dtype=np.float32) # zeros for initialization
        self.actions = np.zeros((self.num_envs, 12), dtype=np.float32)

        self.process_configs()
        self.policy_uses_vision = observation_groups.IMAGE_ENCODER_LATENT in self.observation_groups[self.cfg["policy"]["obs_key"]].observations
        if self.vision_only:
            assert self.policy_uses_vision, "IMAGE_ENCODER_LATENT is not in the observation group."

    def process_configs(self):
        self.sim_dof_order = self.deploy_cfg["dof_names"]
        # Map from isaacgym joint order to robot joint order.
        self.dof_map = [ROBOT_JOINT_ORDER.index(name) for name in self.sim_dof_order]
        self.torque_limits = np.array([33.5] * 12, dtype=np.float32)
        if "torque_limits" in self.cfg["control"]:
            if isinstance(self.cfg["control"]["torque_limits"], (tuple, list)):
                for i in range(len(self.cfg["control"]["torque_limits"])):
                    self.torque_limits[i] = self.cfg["control"]["torque_limits"][i]
            else:
                self.torque_limits[:] = self.cfg["control"]["torque_limits"]

        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.gravity_vec = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.gravity_vec[:, self.up_axis_idx] = -1

        self.observation_groups = observation_groups.observation_groups_from_config(self.cfg["observations"])

        if not isinstance(self.cfg["control"]["damping"]["joint"], (list, tuple)):
            self.cfg["control"]["damping"]["joint"] = [self.cfg["control"]["damping"]["joint"]] * 12
        if not isinstance(self.cfg["control"]["stiffness"]["joint"], (list, tuple)):
            self.cfg["control"]["stiffness"]["joint"] = [self.cfg["control"]["stiffness"]["joint"]] * 12
        self.d_gains = np.array(self.cfg["control"]["damping"]["joint"], dtype=np.float32)
        self.p_gains = np.array(self.cfg["control"]["stiffness"]["joint"], dtype=np.float32)

        self.default_dof_pos = np.zeros(12, dtype=np.float32)
        for i in range(12):
            name = self.extra_cfg["dof_names"][i]
            default_joint_angle = self.cfg["init_state"]["default_joint_angles"][name]
            self.default_dof_pos[i] = default_joint_angle

        self.computer_clip_torque = self.cfg["control"]["computer_clip_torque"]
        
        # store config values to attributes to improve speed
        # self.clip_obs = self.cfg["normalization"]["clip_observations"]
        self.control_type = self.cfg["control"]["control_type"]
        self.action_scale = self.cfg["control"]["action_scale"]
        self.clip_actions = self.cfg["normalization"]["clip_actions"]
        if self.cfg["normalization"]["clip_actions_method"] == "hard":
            rospy.loginfo("clip_actions_method with hard mode")
            rospy.loginfo("clip_actions: " + str(self.cfg["normalization"]["clip_actions"]))
            self.clip_actions_method = "hard"
            self.clip_actions_low = np.array(self.cfg["normalization"]["clip_actions_low"], dtype=np.float32)
            self.clip_actions_high = np.array(self.cfg["normalization"]["clip_actions_high"], dtype=np.float32)
        else:
            rospy.loginfo("clip_actions_method is " + str(self.cfg["normalization"]["clip_actions_method"]))

        # get ROS params for hardware configs
        self.joint_limits_high = np.array([
            rospy.get_param(self.robot_namespace + "/joint_limits/{}_max".format(s)) \
            for s in ["hip", "thigh", "calf"] * 4
        ])
        self.joint_limits_low = np.array([
            rospy.get_param(self.robot_namespace + "/joint_limits/{}_min".format(s)) \
            for s in ["hip", "thigh", "calf"] * 4
        ])
        self.joint_limits_low_sim = np.array(self.deploy_cfg["dof_pos_limits_low"])
        self.joint_limits_high_sim = np.array(self.deploy_cfg["dof_pos_limits_high"])

        rospy.loginfo('UnitreeA1Real:')
        rospy.loginfo(f'SIM JOINT ORDER: {self.sim_dof_order}')
        rospy.loginfo(f'ROBOT JOINT ORDER: {ROBOT_JOINT_ORDER}')
        rospy.loginfo(f'Dof map: {self.dof_map}')
        rospy.loginfo(f'Torque limits: {self.torque_limits.tolist()}')
        rospy.loginfo(f"Joint limits high: {self.joint_limits_high.tolist()}")
        rospy.loginfo(f"Joint limits low: {self.joint_limits_low.tolist()}")
        rospy.loginfo(f"Joint limits low sim: {self.joint_limits_low_sim.tolist()}")
        rospy.loginfo(f"Joint limits high sim: {self.joint_limits_high_sim.tolist()}")
        rospy.loginfo(f"P Gains: {self.p_gains.tolist()}")
        rospy.loginfo(f"D Gains: {self.d_gains.tolist()}")
        rospy.loginfo(f"Default dof pos: {self.default_dof_pos.tolist()}")
        rospy.loginfo("Computer Clip Torque (onboard) is " + str(self.computer_clip_torque))
        if self.computer_clip_torque:
            rospy.loginfo("[Env] torque limit: {:.1f} {:.1f} {:.1f}".format(*self.torque_limits[:3]))
        rospy.loginfo("[Env] action scale: {:.1f}".format(self.action_scale))

    def start_ros(self):
        # initialze several buffers so that the system works even without message update.
        # self.low_state_buffer = LowState() # not initialized, let input message update it.
        self.base_position_buffer = np.zeros((self.num_envs, 3), dtype=np.float32)
        if not self.vision_only:
          self.legs_cmd_publisher = rospy.Publisher(
              self.robot_namespace + self.legs_cmd_topic,
              LegsCmd,
              queue_size=1,
          )
        self.low_state_subscriber = rospy.Subscriber(
            self.robot_namespace + self.low_state_topic,
            LowState,
            self.update_low_state,
            queue_size=1,
        )
        self.odom_subscriber = rospy.Subscriber(
            self.robot_namespace + self.odom_topic,
            Odometry,
            self.update_base_pose,
            queue_size=1,
        )
        self.gamepad_subscriber = rospy.Subscriber(
            self.robot_namespace + self.gamepad_topic,
            Joy,
            self.update_gamepad,
            queue_size=1,
        )
        self.pose_cmd_subscriber = rospy.Subscriber(
            "/body_pose",
            Pose,
            self.dummy_handler,
            queue_size=1,
        )
        if self.vision_only:
            self.visual_embedding_publisher = rospy.Publisher(
                self.robot_namespace + self.image_embedding_topic,
                Float32MultiArrayStamped,
                queue_size=1,
            )
        if not self.vision_only and self.policy_uses_vision:
            self.visual_embedding_subscriber = rospy.Subscriber(
                self.robot_namespace + self.image_embedding_topic,
                Float32MultiArrayStamped,
                self.update_visual_embedding,
                queue_size= 1,
            )
    
    def wait_untill_ros_working(self):
        rate = rospy.Rate(100)
        while not hasattr(self, "low_state_buffer"):
            rate.sleep()
        rate = rospy.Rate(100)
        rospy.loginfo("UnitreeA1Real.low_state_buffer acquired, stop waiting.")
        if not self.vision_only and self.policy_uses_vision:
            while not hasattr(self, "visual_embedding_buffer"):
                print("Waiting for visual embedding buffer...")
                rate.sleep()
        rospy.loginfo("UnitreeA1Real.visual_embedding_buffer acquired, stop waiting.")
        
    def process_action(self, actions):
        if getattr(self, "clip_actions_method", None) == "hard":
            actions = np.clip(actions, self.clip_actions_low, self.clip_actions_high)
        else:
            actions = np.clip(actions, -self.clip_actions, self.clip_actions)
        self.actions[:] = actions
        actions = actions * self.action_scale
        if self.cfg["control"]["computer_clip_torque"]:
            dof_vel = self.dof_vel
            dof_pos_ = self.dof_pos - self.default_dof_pos
            p_limits_low = (-self.torque_limits) + self.d_gains * dof_vel
            p_limits_high = (self.torque_limits) + self.d_gains * dof_vel
            actions_low = (p_limits_low / self.p_gains) + dof_pos_
            actions_high = (p_limits_high / self.p_gains) + dof_pos_
            actions = np.clip(actions, actions_low, actions_high)
        return actions
           
    """ Get obs components and cat to a single obs input """
    def compute_observation(self):
        """Use the updated low_state_buffer to compute observation vector."""
        assert hasattr(self, "legs_cmd_publisher") or self.read_only, "start_ros() not called, ROS handlers are not initialized!"
        if self.move_by_gamepad:
            self._poll_gamepad()
        obs_dict = {}
        for group in self.observation_groups:
          if group.name != self.cfg["policy"]["obs_key"]:
              continue
          obs_dict[group.name] = {}
          for observation in group.observations:
              if observation == observation_groups.IMAGE_ENCODER_LATENT:
                  continue
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
                      scale = np.array(scale)[None]
                  obs = scale * obs
              obs_dict[group.name][observation.name] = obs

        if hasattr(self, "visual_embedding_buffer") and self.use_vision:
            obs_dict["policy"]["image_encoder"] = self.visual_embedding_buffer
        self.obs_dict = obs_dict


    """ The methods combined with outer model forms the step function
    NOTE: the outer user handles the loop frequency.
    """
    def send_action(self, actions, kp=None, kd=None):
        """ The function that send commands to the real robot.
        """
        robot_coordinates_action = self.process_action(actions) + self.default_dof_pos[None]
        self.publish_legs_cmd(robot_coordinates_action, kp=kp, kd=kd)

    def publish_legs_cmd(self, robot_coordinates_action, kp= None, kd= None):
        """ publish the joint position directly to the robot. NOTE: The joint order from input should
        be in simulation order. The value should be absolute value rather than related to dof_pos.
        """
        assert not self.read_only, "Cannot publish legs cmd in read-only mode."
        robot_coordinates_action = np.clip(
            robot_coordinates_action,
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
            if np.linalg.norm(self.command_buf[0, :2]) < self.cfg["commands"]["small_lin_vel_threshold"]:
                self.command_buf[0, :2] = 0.
            if np.abs(self.command_buf[0, 2]) < self.cfg["commands"]["small_ang_vel_threshold"]:
                self.command_buf[0, 2] = 0.
        self.low_state_get_time = rospy.Time.now()
  
    def update_visual_embedding(self, ros_msg):
        self.visual_embedding_buffer = np.array(ros_msg.data)[None].astype(np.float32)
        self.visual_embedding_get_time = rospy.Time.now()

    def update_base_pose(self, ros_msg):
        """ update robot odometry for position """
        self.base_position_buffer[0, 0] = ros_msg.pose.pose.position.x
        self.base_position_buffer[0, 1] = ros_msg.pose.pose.position.y
        self.base_position_buffer[0, 2] = ros_msg.pose.pose.position.z

    def update_move_cmd(self, ros_msg):
        self.command_buf[0, 0] = ros_msg.linear.x
        self.command_buf[0, 1] = ros_msg.linear.y
        self.command_buf[0, 2] = ros_msg.angular.z

    def update_gamepad(self, ros_msg):
        self.gamepad_buffer = ros_msg
        lin_vel_range = [0, self.cfg["commands"]["ranges"]["lin_vel"][1]]
        ang_vel_range = self.cfg["commands"]["ranges"]["ang_vel_yaw"]

        vel_x = ros_msg.axes[1] * (lin_vel_range[1] - lin_vel_range[0]) + lin_vel_range[0]
        vel_y = -1. * ros_msg.axes[0] * (lin_vel_range[1] - lin_vel_range[0]) + lin_vel_range[0]
        ang_vel = ros_msg.axes[3] * (ang_vel_range[1] - ang_vel_range[0]) + ang_vel_range[0]

        if np.abs(vel_x) < self.cfg["commands"]["small_lin_vel_threshold"]:
            vel_x = 0.
        if np.abs(vel_y) < self.cfg["commands"]["small_lin_vel_threshold"]:
            vel_y = 0.
        if np.abs(ang_vel) < self.cfg["commands"]["small_ang_vel_threshold"]:
            ang_vel = 0.

        self.command_buf[0, 0] = vel_x
        self.command_buf[0, 1] = vel_y
        self.command_buf[0, 2] = ang_vel

    def dummy_handler(self, ros_msg):
        """ To meet the need of teleop-legged-robots requirements """
        pass
