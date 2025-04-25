from typing import Dict
import pickle
import torch.utils._pytree as pytree
import pathlib
import onnxruntime


class DeploymentRunner:
  def __init__(self, deploy_cfg: Dict, cfg: Dict):
    self.deploy_cfg = deploy_cfg
    self.cfg = cfg
    self.policy_key = self.cfg["policy"]["obs_key"]

  def load(self, resume_root: pathlib.Path):
    if not self.cfg["runner"]["resume"]:
      return

    load_run = self.cfg["runner"]["load_run"]
    if (load_run == "-1") or (load_run == -1):
      self.resume_path = sorted(
        [item for item in resume_root.iterdir() if item.is_dir()],
        key=lambda path: path.stat().st_mtime,
      )[-1]
    else:
      self.resume_path = resume_root / load_run

    self.obs_space = pickle.load(open(self.resume_path / "obs_space.pkl", "rb"))
    self.action_space = pickle.load(open(self.resume_path / "action_space.pkl", "rb"))

    onnx_dir = self.resume_path / "onnx"
    onnx_path = onnx_dir / "policy.onnx"
    hidden_states_path = onnx_dir / "hidden_states.pkl"
    print(f"Loading ONNX model from: {onnx_path}")
    sess_options = onnxruntime.SessionOptions()
    self.onnx_session = onnxruntime.InferenceSession(str(onnx_path), sess_options=sess_options, providers=['CPUExecutionProvider'])
    self.onnx_input_names = [inp.name for inp in self.onnx_session.get_inputs()]
    self.onnx_output_names = [out.name for out in self.onnx_session.get_outputs()]
    with open(hidden_states_path, 'rb') as f:
      self.onnx_init_hidden_states = pickle.load(f)

  def get_input_dict(self, obs_dict, policy_hidden_states):
    return {
      **obs_dict,
      **{f"hidden_{i}": h for i, h in enumerate(policy_hidden_states)},
    }

  def get_actions_and_hidden_states(self, outputs, act_keys):
    actions = dict(zip(act_keys, outputs[:len(act_keys)]))
    hidden_states = outputs[len(act_keys):]
    return actions, hidden_states

  def act(self, obs):
    input_dict = self.get_input_dict(obs, self.onnx_init_hidden_states)
    outputs = self.onnx_session.run(self.onnx_output_names, input_dict)
    actions, self.onnx_init_hidden_states = self.get_actions_and_hidden_states(outputs, self.action_space.keys())
    return actions
