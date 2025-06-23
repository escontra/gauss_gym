from typing import Dict
import pickle
import pathlib
import onnxruntime


class DeploymentRunner:
  def __init__(self, cfg: Dict, model_name: str, execution_provider: str = 'CPUExecutionProvider'):
    self.cfg = cfg
    self.model_name = model_name
    self.execution_provider = execution_provider

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

    self.obs_space = pickle.load(open(self.resume_path / f"{self.model_name}_obs_space.pkl", "rb"))
    self.action_space = pickle.load(open(self.resume_path / "action_space.pkl", "rb"))

    onnx_dir = self.resume_path / "onnx"
    onnx_path = onnx_dir / f"{self.model_name}.onnx"
    hidden_states_path = onnx_dir / f"{self.model_name}_hidden_states.pkl"
    print(f"Loading ONNX model from: {onnx_path}")
    print(f"Loading hidden states from: {hidden_states_path}")
    sess_options = onnxruntime.SessionOptions()
    self.onnx_session = onnxruntime.InferenceSession(str(onnx_path), sess_options=sess_options, providers=[self.execution_provider])
    self.onnx_input_names = [inp.name for inp in self.onnx_session.get_inputs()]
    self.onnx_output_names = [out.name for out in self.onnx_session.get_outputs()]
    print(f'Input names: {self.onnx_input_names}')
    print(f'Output names: {self.onnx_output_names}')
    self.rnn_state_key = 'out_rnn_state'
    assert self.rnn_state_key in self.onnx_output_names
    self.model_pred_keys = self.onnx_output_names[:self.onnx_output_names.index(self.rnn_state_key)]
    self.hidden_state_keys = self.onnx_output_names[self.onnx_output_names.index(self.rnn_state_key) + 1:]
    with open(hidden_states_path, 'rb') as f:
      self.onnx_init_hidden_states = pickle.load(f)

  def get_input_dict(self, obs_dict, hidden_states):
    return {
      **obs_dict,
      **{f"hidden_{i}": h for i, h in enumerate(hidden_states)},
    }

  def process_outputs(self, outputs, rnn_only: bool):
    if rnn_only:
      model_preds = {}
    else:
      model_preds = {k: outputs[k] for k in self.model_pred_keys}
    rnn_state = outputs[self.rnn_state_key]
    hidden_states = (outputs[k] for k in self.hidden_state_keys)
    return model_preds, rnn_state, hidden_states

  def predict(self, obs, rnn_only: bool = False):
    input_dict = self.get_input_dict(obs, self.onnx_init_hidden_states)
    if rnn_only:
      output_keys = [self.rnn_state_key] + self.hidden_state_keys
    else:
      output_keys = self.onnx_output_names
    outputs = self.onnx_session.run(output_keys, input_dict)
    outputs = dict(zip(output_keys, outputs))
    model_preds, rnn_state, self.onnx_init_hidden_states = self.process_outputs(outputs, rnn_only)
    return model_preds, rnn_state
