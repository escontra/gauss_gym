import numpy as np
import pathlib
import pickle
import wandb
import torch
import torch.nn.functional as F
from typing import Dict, Any
import tensorflow as tf
import tensorflow.compat.v1 as tf1

from legged_gym import utils
from legged_gym.utils import config, timer, when


@timer.section('gif')
def _encode_gif(frames, fps):
  from subprocess import Popen, PIPE
  h, w, c = frames[0].shape
  pxfmt = {1: 'gray', 3: 'rgb24'}[c]
  cmd = ' '.join([
      'ffmpeg -y -f rawvideo -vcodec rawvideo',
      f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
      '[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
      f'-r {fps:.02f} -f gif -'])
  proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
  for image in frames:
    proc.stdin.write(image.tobytes())
  out, err = proc.communicate()
  if proc.returncode:
    raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
  del proc
  return out


class Recorder:
  def __init__(
      self,
      log_dir: pathlib.Path,
      cfg: Dict[str, Any],
      deploy_cfg: Dict[str, Any],
      save_dicts: Dict[str, Any]):
    self.cfg = cfg
    self.deploy_cfg = deploy_cfg
    self.log_dir = log_dir
    self.save_dicts = save_dicts
    self.initialized = False
    self.fps = int(1. / (self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]))

  def setup_recorder(self, env):
    from isaacgym import gymapi

    self.mesh_names = []
    self.env_handles = []
    self.camera_handles = []
    self.env_ids = []
    for mesh_id in range(env.scene_manager.num_meshes):
      env_ids = env.scene_manager.env_ids_for_mesh_id(mesh_id)
      env_id = env_ids[0]
      camera_props = gymapi.CameraProperties()
      # camera_props.enable_tensors = True
      camera_props.width = 200
      camera_props.height = 200
      camera_handle = env.gym.create_camera_sensor(env.envs[env_id], camera_props)
      self.camera_handles.append(camera_handle)
      self.env_ids.append(env_id)
      self.env_handles.append(env.envs[env_id])
      self.mesh_names.append(env.scene_manager.mesh_name_from_id(mesh_id))

      # Get center of the mesh.
      cam_trans = env.scene_manager.cam_trans_viz[mesh_id]
      lookat = cam_trans[cam_trans.shape[0] // 2].cpu().numpy()
      pos = (np.array(self.cfg["runner"]["record_distance"]) + lookat).tolist()
      # pos = np.array(self.cfg["viewer"]["pos"] - (self.cfg["viewer"]["lookat"]) + lookat).tolist()
      env.gym.set_camera_location(camera_handle, env.envs[env_id], gymapi.Vec3(*pos), gymapi.Vec3(*lookat))

    self.record_every = when.Clock(self.cfg["runner"]["record_every"])
    self.num_frames = 0
    self.reset_frame_dict()

  def reset_frame_dict(self):
    self.frame_dict = {k: [] for k in self.mesh_names}

  def maybe_record(self, env, image_features={}):
    if self.num_frames == self.cfg["runner"]["record_frames"]:
      # Stop recording.
      utils.print('STOP RECORDING', color='blue')
      stacked_frame_dict = {k: np.stack(v) for k, v in self.frame_dict.items()}
      self.num_frames = 0
      self.reset_frame_dict()
      return stacked_frame_dict
    elif self.num_frames > 0:
      # Recording ongoing.
      self.update_frames(env, image_features)
      self.num_frames += 1
      return {}
    elif self.record_every():
      # Maybe start a new recording.
      utils.print('START RECORDING', color='blue')
      self.update_frames(env, image_features)
      self.num_frames += 1
      return {}
    else:
      return {}

  def update_frames(self, env, image_features={}):
    from isaacgym import gymapi
    env.gym.fetch_results(env.sim, True)
    env.gym.render_all_camera_sensors(env.sim)
    env.gym.step_graphics(env.sim)
    for mesh_name, env_handle, camera_handle, env_id in zip(self.mesh_names, self.env_handles, self.camera_handles, self.env_ids):
      image = env.gym.get_camera_image(env.sim, env_handle, camera_handle, gymapi.IMAGE_COLOR)
      image = image.reshape(image.shape[0], -1, 4)[..., :3]
      for _, im in image_features.items():
        im = im[env_id]
        h, w = im.shape[1:]
        target_h = image.shape[0]
        target_w = int(w * (target_h / h))
        im = im.float() / 255.0
        im = F.interpolate(im.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0)
        im = im.permute(1, 2, 0)
        im = (im * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        image = np.concatenate([image, im], axis=1)

      self.frame_dict[mesh_name].append(image)

  def maybe_init(self):
    if self.initialized:
      return
    utils.print(f"Recording to: {self.log_dir}", color='blue')
    self.log_dir.mkdir(parents=True, exist_ok=True)
    self.model_dir = self.log_dir / "nn"
    self.model_dir.mkdir(parents=True, exist_ok=True)
    self.writer = tf.summary.create_file_writer(str(self.log_dir / "summaries"), max_queue=int(1e9), flush_millis=int(1e9))
    self.writer.set_as_default()
    if self.cfg["runner"]["use_wandb"]:
      wandb.init(
        project=self.cfg["task"],
        dir=self.log_dir,
        name=self.log_dir.name,
        notes=self.cfg["runner"]["description"],
        config=dict(self.cfg),
      )

    self.episode_statistics = {}
    self.last_episode = {}
    self.last_episode["steps"] = []
    self.episode_steps = None

    config.Config(self.cfg).save(self.log_dir / "config.yaml")
    config.Config({'deploy': self.deploy_cfg}).save(self.log_dir / "deploy_config.yaml")

    for key, value in self.save_dicts.items():
      with open(self.log_dir / f"{key}.pkl", "wb") as file:
        pickle.dump(value, file)
    self.initialized = True

  @timer.section("record_episode_statistics")
  def record_episode_statistics(self, done, ep_info, it, discount_factor_dict={}, write_record=False):
    self.maybe_init()
    if self.episode_steps is None:
      self.episode_steps = torch.zeros_like(done, dtype=int)
    else:
      self.episode_steps += 1

    for key, value in ep_info.items():
      if self.episode_statistics.get(key) is None:
        self.episode_statistics[key] = torch.zeros_like(value)
      discount_factor = discount_factor_dict.get(key, 1.0)
      discount_factor = discount_factor ** self.episode_steps
      self.episode_statistics[key] += value * discount_factor
      if self.last_episode.get(key) is None:
        self.last_episode[key] = []
      for done_value in self.episode_statistics[key][done]:
        self.last_episode[key].append(done_value.item())
      self.episode_statistics[key][done] = 0

    for val in self.episode_steps[done]:
      self.last_episode["steps"].append(val.item())
    self.episode_steps[done] = 0

    episode_stats = {}
    for key in self.last_episode.keys():
      episode_stats[key] = self._mean(self.last_episode[key])
      self.last_episode[key].clear()
    return episode_stats

  @timer.section("record_statistics")
  def record_statistics(self, statistics, it):
    self.maybe_init()
    for key, value in statistics.items():
      if isinstance(value, str):
        tf.summary.text(key, value, it)
        if self.cfg["runner"]["use_wandb"]:
          wandb.log({key: value}, step=it)
      elif isinstance(value, (float, int)):
        tf.summary.scalar(key, float(value), it)
        if self.cfg["runner"]["use_wandb"]:
          wandb.log({key: float(value)}, step=it)
      elif isinstance(value, np.ndarray) and value.ndim == 1:
        if len(value) > 1024:
          value = value.copy()
          np.random.shuffle(value)
          value = value[:1024]
        tf.summary.histogram(key, value, it)
      elif isinstance(value, np.ndarray) and value.ndim == 4 and value.dtype == np.uint8:
        gif_bytes = self._video_summary(key, value, it)
        if self.cfg["runner"]["use_wandb"]:
          import io
          wandb.log({key: wandb.Video(io.BytesIO(gif_bytes), format='gif')}, step=it)
      else:
        raise ValueError(f"Unsupported type for {key}: {type(value)}")
    self.writer.flush()

  @timer.section('tensorboard_video')
  def _video_summary(self, name, video, step):
    name = name if isinstance(name, str) else name.decode('utf-8')
    assert video.dtype in (np.float32, np.uint8), (video.shape, video.dtype)
    if np.issubdtype(video.dtype, np.floating):
      video = np.clip(255 * video, 0, 255).astype(np.uint8)
    try:
      T, H, W, C = video.shape
      summary = tf1.Summary()
      image = tf1.Summary.Image(height=H, width=W, colorspace=C)
      gif_bytes = _encode_gif(video, self.fps)
      image.encoded_image_string = gif_bytes
      summary.value.add(tag=name, image=image)
      content = summary.SerializeToString()
      tf.summary.experimental.write_raw_pb(content, step)
    except (IOError, OSError) as e:
      utils.print('GIF summaries require ffmpeg in $PATH.', e, color='red')
      tf.summary.image(name, video, step)
    return gif_bytes

  @timer.section("save")
  def save(self, model_dict, it):
    self.maybe_init()
    path = self.model_dir / f"model_{it:06d}.pth"
    utils.print(f"Saving model to {path}", color='blue')
    torch.save(model_dict, path)

  def _mean(self, data):
    if len(data) == 0:
      return 0.0
    else:
      return sum(data) / len(data)

