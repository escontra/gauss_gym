#!/home/unitree/agility_ziwenz_venv/bin/python
import os
import os.path as osp
import json
import numpy as np
import torch
from collections import OrderedDict
from functools import partial
from typing import Tuple

import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
import ros_numpy
import yaml
import pickle
import glob

from typing import Dict, List
from deployment.a1.a1_real import UnitreeA1Real, resize2d
from legged_gym.rl.modules import models

@torch.no_grad()
def handle_forward_depth(ros_msg, model, publisher, output_resolution, device):
    """ The callback function to handle the forward depth and send the embedding through ROS topic """
    buf = ros_numpy.numpify(ros_msg).astype(np.float32)
    forward_depth_buf = resize2d(
        torch.from_numpy(buf).unsqueeze(0).unsqueeze(0).to(device),
        output_resolution,
    )
    embedding = model(forward_depth_buf)
    ros_data = embedding.reshape(-1).cpu().numpy().astype(np.float32)
    publisher.publish(Float32MultiArray(data= ros_data.tolist()))

def permute_cnn_obs(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 5:
        return x.permute(0, 1, 4, 2, 3)  # [B, T, H, W, C] -> [B, T, C, H, W]
    elif x.dim() == 4:
        return x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    else:
        return x  # or raise an error if that's not expected

class StandOnlyModel(torch.nn.Module):
    def __init__(self, action_scale, dof_pos_scale, tolerance= 0.2, delta= 0.1):
        rospy.loginfo("Using stand only model, please make sure the proprioception is 48 dim.")
        rospy.loginfo("Using stand only model, -36 to -24 must be joint position.")
        super().__init__()
        if isinstance(action_scale, (tuple, list)):
            self.register_buffer("action_scale", torch.tensor(action_scale))
        else:
            self.action_scale = action_scale
        if isinstance(dof_pos_scale, (tuple, list)):
            self.register_buffer("dof_pos_scale", torch.tensor(dof_pos_scale))
        else:
            self.dof_pos_scale = dof_pos_scale
        self.tolerance = tolerance
        self.delta = delta

    def forward(self, obs):
        joint_positions = obs[..., -36:-24] / self.dof_pos_scale
        diff_large_mask = torch.abs(joint_positions) > self.tolerance
        target_positions = torch.zeros_like(joint_positions)
        target_positions[diff_large_mask] = joint_positions[diff_large_mask] - self.delta * torch.sign(joint_positions[diff_large_mask])
        return torch.clip(
            target_positions / self.action_scale,
            -1.0, 1.0,
        )
    
    def reset(self, *args, **kwargs):
        pass

def load_walk_policy(env, model_dir):
    """ Load the walk policy from the model directory """
    if model_dir == None:
        model = StandOnlyModel(
            action_scale= env.action_scale,
            dof_pos_scale= env.obs_scales["dof_pos"],
        )
        policy = torch.jit.script(model)

    else:
        with open(osp.join(model_dir, "config.json"), "r") as f:
            config_dict = json.load(f, object_pairs_hook= OrderedDict)
        obs_components = config_dict["env"]["obs_components"]
        privileged_obs_components = config_dict["env"].get("privileged_obs_components", obs_components)
        model = getattr(modules, config_dict["runner"]["policy_class_name"])(
            num_actor_obs= env.get_num_obs_from_components(obs_components),
            num_critic_obs= env.get_num_obs_from_components(privileged_obs_components),
            num_actions= 12,
            **config_dict["policy"],
        )
        model_names = [i for i in os.listdir(model_dir) if i.startswith("model_")]
        model_names.sort(key= lambda x: int(x.split("_")[-1].split(".")[0]))
        state_dict = torch.load(osp.join(model_dir, model_names[-1]), map_location= "cpu")
        model.load_state_dict(state_dict["model_state_dict"])
        model_action_scale = torch.tensor(config_dict["control"]["action_scale"]) if isinstance(config_dict["control"]["action_scale"], (tuple, list)) else torch.tensor([config_dict["control"]["action_scale"]])[0]
        if not (torch.is_tensor(model_action_scale) and (model_action_scale == env.action_scale).all()):
            action_rescale_ratio = model_action_scale / env.action_scale
            print("walk_policy action scaling:", action_rescale_ratio.tolist())
        else:
            action_rescale_ratio = 1.0
        memory_module = model.memory_a
        actor_mlp = model.actor
        @torch.jit.script
        def policy_run(obs):
            recurrent_embedding = memory_module(obs)
            actions = actor_mlp(recurrent_embedding.squeeze(0))
            return actions
        if (torch.is_tensor(action_rescale_ratio) and (action_rescale_ratio == 1.).all()) \
            or (not torch.is_tensor(action_rescale_ratio) and action_rescale_ratio == 1.):
            policy = policy_run
        else:
            policy = lambda x: policy_run(x) * action_rescale_ratio
    
    return policy, model

def standup_procedure(env, ros_rate, angle_tolerance= 0.1,
        kp= None,
        kd= None,
        warmup_timesteps= 25,
        device= "cpu",
        policy=None,
    ):
    """
    Args:
        warmup_timesteps: the number of timesteps to linearly increase the target position
    """
    rospy.loginfo("Robot standing up, please wait ...")

    target_pos = torch.zeros((1, 12), device= device, dtype= torch.float32)
    standup_timestep_i = 0
    while not rospy.is_shutdown():
        dof_pos = [env.low_state_buffer.motorState[env.dof_map[i]].q for i in range(12)]
        diff = [env.default_dof_pos[i].item() - dof_pos[i] for i in range(12)]
        direction = [1 if i > 0 else -1 for i in diff]
        if standup_timestep_i < warmup_timesteps:
            direction = [standup_timestep_i / warmup_timesteps * i for i in direction]
        if all([abs(i) < angle_tolerance for i in diff]):
            break
        print("max joint error (rad):", max([abs(i) for i in diff]), end= "\r")
        for i in range(12):
            target_pos[0, i] = dof_pos[i] + direction[i] * angle_tolerance if abs(diff[i]) > angle_tolerance else env.default_dof_pos[i]
        env.publish_legs_cmd(target_pos,
            kp= kp,
            kd= kd,
        )
        if policy is not None:
            _ = policy(env.get_obs()['student_observations'])
        ros_rate.sleep()
        standup_timestep_i += 1

    # TODO(aescontrela): Uncomment this when we find the remote.
    # rospy.loginfo("Robot stood up! press R1 on the remote control to continue ...")
    # while not rospy.is_shutdown():
    #     if env.low_state_buffer.wirelessRemote.btn.components.R1:
    #         break
    #     if env.low_state_buffer.wirelessRemote.btn.components.L2 or env.low_state_buffer.wirelessRemote.btn.components.R2:
    #         env.publish_legs_cmd(env.default_dof_pos.unsqueeze(0), kp= 0, kd= 0.5)
    #         rospy.signal_shutdown("Controller send stop signal, exiting")
    #         exit(0)
    #     env.publish_legs_cmd(env.default_dof_pos.unsqueeze(0), kp= kp, kd= kd)
    #     if policy is not None:
    #         _ = policy(env.get_obs()['student_observations'])
    #     ros_rate.sleep()
    rospy.loginfo("Robot standing up procedure finished!")

class SkilledA1Real(UnitreeA1Real):
    """ Some additional methods to help the execution of skill policy """
    def __init__(self, *args,
            skill_mode_threhold= 0.1,
            skill_vel_range= [0.0, 1.0],
            **kwargs,
        ):
        self.skill_mode_threhold = skill_mode_threhold
        self.skill_vel_range = skill_vel_range
        super().__init__(*args, **kwargs)

    def is_skill_mode(self):
        if self.move_by_wireless_remote:
            return self.low_state_buffer.wirelessRemote.ry > self.skill_mode_threhold
        else:
            # Not implemented yet
            return False

    def update_low_state(self, ros_msg):
        self.low_state_buffer = ros_msg
        if self.move_by_wireless_remote and ros_msg.wirelessRemote.ry > self.skill_mode_threhold:
            skill_vel = (self.low_state_buffer.wirelessRemote.ry - self.skill_mode_threhold) / (1.0 - self.skill_mode_threhold)
            skill_vel *= self.skill_vel_range[1] - self.skill_vel_range[0]
            skill_vel += self.skill_vel_range[0]
            self.command_buf[0, 0] = skill_vel
            self.command_buf[0, 1] = 0.
            self.command_buf[0, 2] = 0.
            return
        return super().update_low_state(ros_msg)

class NoPythonTagLoader(yaml.SafeLoader):
    pass

def ignore_unknown(loader, tag_suffix, node):
    return None  # or raise an error, or log

NoPythonTagLoader.add_multi_constructor("tag:yaml.org,2002:python/", ignore_unknown)

def main(args):
    log_level = rospy.DEBUG if args.debug else rospy.INFO
    rospy.init_node("a1_legged_gym_" + args.mode, log_level= log_level)

    """ Not finished this modification yet """
    # if args.logdir is not None:
    #     rospy.loginfo("Use logdir/config.json to initialize env proxy.")
    #     with open(osp.join(args.logdir, "config.json"), "r") as f:
    #         config_dict = json.load(f, object_pairs_hook= OrderedDict)
    # else:
    #     assert args.walkdir is not None, "You must provide at least a --logdir or --walkdir"
    #     rospy.logwarn("You did not provide logdir, use walkdir/config.json for initializing env proxy.")
    #     with open(osp.join(args.walkdir, "config.json"), "r") as f:
    #         config_dict = json.load(f, object_pairs_hook= OrderedDict)
    assert args.logdir is not None
    # with open(osp.join(args.logdir, "env_config.json"), "r") as f:
    #     config_dict = json.load(f, object_pairs_hook= OrderedDict)
    with open(osp.join(args.logdir, "env_config.pkl"), "rb") as f:
        env_config = pickle.load(f)
    with open(osp.join(args.logdir, "train_config.pkl"), "rb") as f:
        train_config = pickle.load(f)
    with open(osp.join(args.logdir, "obs_group_sizes.pkl"), "rb") as f:
        obs_group_sizes = pickle.load(f)
    print('FROM CONFIG:')
    print(env_config.control.stiffness)
    print(env_config.control.damping)
    
    duration = env_config.sim.dt * env_config.control.decimation # in sec
    # env_config["control"]["stiffness"]["joint"] -= 2.5 # kp

    model_device = torch.device("cpu") if args.mode == "upboard" else torch.device("cuda")

    unitree_real_env = UnitreeA1Real(
        robot_namespace= args.namespace,
        cfg= env_config,
        forward_depth_topic=None,
        forward_depth_embedding_dims=None,
        move_by_wireless_remote= False,
        # skill_vel_range= env_config["commands"]["ranges"]["lin_vel_x"],
        model_device= model_device,
        # extra_cfg= dict(
        #     motor_strength= torch.tensor([
        #         1., 1./0.9, 1./0.9,
        #         1., 1./0.9, 1./0.9,
        #         1., 1., 1.,
        #         1., 1., 1.,
        #     ], dtype= torch.float32, device= model_device, requires_grad= False),
        # ),
    )
    model = getattr(models, train_config.runner.policy_class_name)(
      12,
      obs_group_sizes["student_observations"],
      obs_group_sizes["teacher_observations"],
      train_config.policy.init_noise_std,
      train_config.policy.mu_activation,
    ).to(model_device)
    model_path = sorted(
      glob.glob(osp.join(args.logdir, "nn", "**/*.pth"), recursive=True),
      key=os.path.getmtime,
    )[-1]
    print(f'Loading model from {model_path}')
    model_dict = torch.load(
      model_path, map_location=model_device,
      # weights_only=True
    )
    model.load_state_dict(model_dict["model"], strict=True)

    # Jitted policy.
    memory_module = model.memory_a
    actor = model.actor
    mlp_keys = model.mlp_keys_a
    cnn_keys = model.cnn_keys_a
    cnn_model = model.cnn_a
    @torch.jit.script
    def _policy(observations: Dict[str, torch.Tensor], mlp_keys: List[str], cnn_keys: List[str]) -> torch.Tensor:
        features = torch.cat([observations[k] for k in mlp_keys], dim=-1)
        if cnn_keys:
          cnn_features = []
          for k in cnn_keys:
            cnn_obs = observations[k]
            if cnn_obs.shape[-1] in [1, 3]:
              cnn_obs = permute_cnn_obs(cnn_obs)
            if cnn_obs.dtype == torch.uint8:
              cnn_obs = cnn_obs.float() / 255.0

            orig_batch_size = cnn_obs.shape[0]
            cnn_obs = cnn_obs.reshape(-1, cnn_obs.shape[-3], cnn_obs.shape[-2], cnn_obs.shape[-1])  # Shape: [M*N*L*O, C, H, W]
            cnn_feat = cnn_model(cnn_obs)
            cnn_feat = cnn_feat.reshape(orig_batch_size, cnn_feat.shape[-1])
            cnn_features.append(cnn_feat)

          cnn_features = torch.cat(cnn_features, dim=-1)
          features = torch.cat([features, cnn_features], dim=-1)
        input_a = memory_module(features, None, None)
        actions = actor(input_a.squeeze(0))
        return actions

    import functools
    policy = functools.partial(_policy, mlp_keys=mlp_keys, cnn_keys=cnn_keys)

    unitree_real_env.start_ros()
    unitree_real_env.wait_untill_ros_working()
    rate = rospy.Rate(1 / duration)

    with torch.no_grad():
        standup_procedure(unitree_real_env, rate,
            angle_tolerance= 0.2,
            kp= 50,
            kd= 1.0,
            warmup_timesteps= 50,
            policy=policy,
            device= model_device,
        )
        while not rospy.is_shutdown():
            # inference_start_time = rospy.get_time()
            obs = unitree_real_env.get_obs()
            actions = policy(obs['student_observations'])
            # actions = torch.zeros((1, 12), device=model_device)
            # actions = torch.zeros_like(actions)
            # act_dist = model.act(obs["student_observations"])
            # actions = act_dist.loc
            # actions = policy(obs,
            #     obs_start= visual_obs_slice[0].start.item(),
            #     obs_stop= visual_obs_slice[0].stop.item(),
            #     obs_shape= visual_obs_slice[1],
            # )
            unitree_real_env.send_action(actions)
            # inference_duration = rospy.get_time() - inference_start_time
            # motor_temperatures = [motor_state.temperature for motor_state in unitree_real_env.low_state_buffer.motorState]
            # rospy.loginfo_throttle(10, " ".join(["motor_temperatures:"] + ["{:d},".format(t) for t in motor_temperatures[:12]]))
            rate.sleep()
            # if unitree_real_env.low_state_buffer.wirelessRemote.btn.components.L2 or unitree_real_env.low_state_buffer.wirelessRemote.btn.components.R2:
            #     unitree_real_env.publish_legs_cmd(unitree_real_env.default_dof_pos.unsqueeze(0), kp= 20, kd= 0.5)
            #     rospy.signal_shutdown("Controller send stop signal, exiting")
    return


    # unitree_real_env = SkilledA1Real(
    #     robot_namespace= args.namespace,
    #     cfg= env_config,
    #     forward_depth_topic= "/visual_embedding" if args.mode == "upboard" else "/camera/depth/image_rect_raw",
    #     forward_depth_embedding_dims= env_config["policy"]["visual_latent_size"] if args.mode == "upboard" else None,
    #     move_by_wireless_remote= True,
    #     skill_vel_range= env_config["commands"]["ranges"]["lin_vel_x"],
    #     model_device= model_device,
    #     # extra_cfg= dict(
    #     #     motor_strength= torch.tensor([
    #     #         1., 1./0.9, 1./0.9,
    #     #         1., 1./0.9, 1./0.9,
    #     #         1., 1., 1.,
    #     #         1., 1., 1.,
    #     #     ], dtype= torch.float32, device= model_device, requires_grad= False),
    #     # ),
    # )

    model = getattr(modules, env_config["runner"]["policy_class_name"])(
        num_actor_obs= unitree_real_env.num_obs,
        num_critic_obs= unitree_real_env.num_privileged_obs,
        num_actions= 12,
        obs_segments= unitree_real_env.obs_segments,
        privileged_obs_segments= unitree_real_env.privileged_obs_segments,
        **env_config["policy"],
    )
    env_config["terrain"]["measure_heights"] = False
    # load the model with the latest checkpoint
    model_names = [i for i in os.listdir(args.logdir) if i.startswith("model_")]
    model_names.sort(key= lambda x: int(x.split("_")[-1].split(".")[0]))
    state_dict = torch.load(osp.join(args.logdir, model_names[-1]), map_location= "cpu")
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(model_device)
    model.eval()

    rospy.loginfo("duration: {}, motor Kp: {}, motor Kd: {}".format(
        duration,
        env_config["control"]["stiffness"]["joint"],
        env_config["control"]["damping"]["joint"],
    ))
    # rospy.loginfo("[Env] motor strength: {}".format(unitree_real_env.motor_strength))

    if args.mode == "jetson":
        embeding_publisher = rospy.Publisher(
            args.namespace + "/visual_embedding",
            Float32MultiArray,
            queue_size= 1,
        )
        # extract and build the torch ScriptFunction
        visual_encoder = model.visual_encoder
        visual_encoder = torch.jit.script(visual_encoder)

        forward_depth_subscriber = rospy.Subscriber(
            args.namespace + "/camera/depth/image_rect_raw",
            Image,
            partial(handle_forward_depth,
                model= visual_encoder,
                publisher= embeding_publisher,
                output_resolution= env_config["sensor"]["forward_camera"].get(
                    "output_resolution",
                    env_config["sensor"]["forward_camera"]["resolution"],
                ),
                device= model_device,
            ),
            queue_size= 1,
        )
        rospy.spin()
    elif args.mode == "upboard":
        # extract and build the torch ScriptFunction
        memory_module = model.memory_a
        actor_mlp = model.actor
        @torch.jit.script
        def policy(obs):
            recurrent_embedding = memory_module(obs)
            actions = actor_mlp(recurrent_embedding.squeeze(0))
            return actions
        
        walk_policy, walk_model = load_walk_policy(unitree_real_env, args.walkdir)

        using_walk_policy = True # switch between skill policy and walk policy
        unitree_real_env.start_ros()
        unitree_real_env.wait_untill_ros_working()
        rate = rospy.Rate(1 / duration)
        with torch.no_grad():
            if not args.debug:
                standup_procedure(unitree_real_env, rate,
                    angle_tolerance= 0.2,
                    kp= 40,
                    kd= 0.5,
                    warmup_timesteps= 50,
                    device= model_device,
                )
            while not rospy.is_shutdown():
                # inference_start_time = rospy.get_time()
                # check remote controller and decide which policy to use
                if unitree_real_env.is_skill_mode():
                    if using_walk_policy:
                        rospy.loginfo_throttle(0.1, "switch to skill policy")
                        using_walk_policy = False
                        model.reset()
                else:
                    if not using_walk_policy:
                        rospy.loginfo_throttle(0.1, "switch to walk policy")
                        using_walk_policy = True
                        walk_model.reset()
                if not using_walk_policy:
                    obs = unitree_real_env.get_obs()
                    actions = policy(obs)
                else:
                    walk_obs = unitree_real_env._get_proprioception_obs()
                    actions = walk_policy(walk_obs)
                unitree_real_env.send_action(actions)
                # unitree_real_env.send_action(torch.zeros((1, 12)))
                # inference_duration = rospy.get_time() - inference_start_time
                # rospy.loginfo("inference duration: {:.3f}".format(inference_duration))
                # rospy.loginfo("visual_latency: %f", rospy.get_time() - unitree_real_env.forward_depth_embedding_stamp.to_sec())
                # motor_temperatures = [motor_state.temperature for motor_state in unitree_real_env.low_state_buffer.motorState]
                # rospy.loginfo_throttle(10, " ".join(["motor_temperatures:"] + ["{:d},".format(t) for t in motor_temperatures[:12]]))
                rate.sleep()
                if unitree_real_env.low_state_buffer.wirelessRemote.btn.components.down:
                    rospy.loginfo_throttle(0.1, "model reset")
                    model.reset()
                    walk_model.reset()
                if unitree_real_env.low_state_buffer.wirelessRemote.btn.components.L2 or unitree_real_env.low_state_buffer.wirelessRemote.btn.components.R2:
                    unitree_real_env.publish_legs_cmd(unitree_real_env.default_dof_pos.unsqueeze(0), kp= 2, kd= 0.5)
                    rospy.signal_shutdown("Controller send stop signal, exiting")
    elif args.mode == "full":
        # extract and build the torch ScriptFunction
        visual_obs_slice = get_obs_slice(unitree_real_env.obs_segments, "forward_depth")
        visual_encoder = model.visual_encoder
        memory_module = model.memory_a
        actor_mlp = model.actor
        @torch.jit.script
        def policy(observations: torch.Tensor, obs_start: int, obs_stop: int, obs_shape: Tuple[int, int, int]):
            visual_latent = visual_encoder(
                observations[..., obs_start:obs_stop].reshape(-1, *obs_shape)
            ).reshape(1, -1)
            obs = torch.cat([
                observations[..., :obs_start],
                visual_latent,
                observations[..., obs_stop:],
            ], dim= -1)
            recurrent_embedding = memory_module(obs)
            actions = actor_mlp(recurrent_embedding.squeeze(0))
            return actions

        unitree_real_env.start_ros()
        unitree_real_env.wait_untill_ros_working()
        rate = rospy.Rate(1 / duration)
        with torch.no_grad():
            while not rospy.is_shutdown():
                # inference_start_time = rospy.get_time()
                obs = unitree_real_env.get_obs()
                actions = policy(obs,
                    obs_start= visual_obs_slice[0].start.item(),
                    obs_stop= visual_obs_slice[0].stop.item(),
                    obs_shape= visual_obs_slice[1],
                )
                unitree_real_env.send_action(actions)
                # inference_duration = rospy.get_time() - inference_start_time
                motor_temperatures = [motor_state.temperature for motor_state in unitree_real_env.low_state_buffer.motorState]
                rospy.loginfo_throttle(10, " ".join(["motor_temperatures:"] + ["{:d},".format(t) for t in motor_temperatures[:12]]))
                rate.sleep()
                if unitree_real_env.low_state_buffer.wirelessRemote.btn.components.L2 or unitree_real_env.low_state_buffer.wirelessRemote.btn.components.R2:
                    unitree_real_env.publish_legs_cmd(unitree_real_env.default_dof_pos.unsqueeze(0), kp= 20, kd= 0.5)
                    rospy.signal_shutdown("Controller send stop signal, exiting")
    else:
        rospy.logfatal("Unknown mode, exiting")

if __name__ == "__main__":
    """ The script to run the A1 script in ROS.
    It's designed as a main function and not designed to be a scalable code.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace",
        type= str,
        default= "/a112138",                    
    )
    parser.add_argument("--logdir",
        type= str,
        help= "The log directory of the trained model",
        default= None,
    )
    parser.add_argument("--walkdir",
        type= str,
        help= "The log directory of the walking model, not for the skills.",
        default= None,
    )
    parser.add_argument("--mode",
        type= str,
        help= "The mode to determine which computer to run on.",
        choices= ["jetson", "upboard", "full"],                
    )
    parser.add_argument("--debug",
        action= "store_true",
    )

    args = parser.parse_args()
    main(args)