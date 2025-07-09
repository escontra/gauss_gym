import types
import os
import isaacgym
import torch
import pathlib
import legged_gym
import torch.onnx

from legged_gym.envs import *
from legged_gym import utils
from legged_gym.utils import flags, config, helpers
from legged_gym.utils.task_registry import task_registry
from legged_gym.rl.runner import Runner


def export_onnx(model_name, model, obs_space, output_names, save_path, test_inputs: bool = False):
    # Prepare dummy inputs
    utils.print(f"Preparing dummy inputs for {model_name}...")
    dummy_obs = {k: torch.ones(v.shape, dtype=v.dtype)[None].to(next(model.parameters()).device) for k, v in obs_space.items()}
    for k, v in dummy_obs.items():
      utils.print(f"\t{k}: {v.shape}, {v.dtype}", color='blue')
    ordered_obs_keys = list(dummy_obs.keys())
    dummy_obs_tuple = tuple(dummy_obs[k] for k in ordered_obs_keys)

    init_hidden_states = model.reset(torch.zeros(1), None)
    ordered_hidden_keys = [f'hidden_{i}' for i in range(len(init_hidden_states))]

    utils.print('Hidden states:')
    for k, v in zip(ordered_hidden_keys, init_hidden_states):
      utils.print(f'\t{k}: {v.shape}', color='blue')

    dummy_input_tuple = dummy_obs_tuple + init_hidden_states
    input_names = ordered_obs_keys + ordered_hidden_keys
    dynamic_axes = {k: {0: 'batch_size'} for k in ordered_obs_keys}
    dynamic_axes.update({k: {1: 'batch_size'} for k in ordered_hidden_keys})

    action_names = [f'out_{k}' for k in output_names]
    rnn_state_names = ['out_rnn_state']
    out_hidden_names = [f'out_{k}' for k in ordered_hidden_keys]
    output_names = action_names + rnn_state_names + out_hidden_names
    dynamic_axes.update({k: {0: 'batch_size'} for k in action_names})
    dynamic_axes.update({k: {0: 'batch_size'} for k in rnn_state_names})
    dynamic_axes.update({k: {1: 'batch_size'} for k in out_hidden_names})

    # Wrap the model to scale actions and return hidden states.
    class OnnxWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.obs_keys = list(self.model.obs_space.keys())

        def forward(self, *obs_tuple):
            obs = obs_tuple[:len(self.obs_keys)]
            hidden_states = obs_tuple[len(self.obs_keys):]
            obs_dict = {k: v for k, v in zip(self.obs_keys, obs)}
            dists, rnn_state, new_hidden_states = self.model(obs_dict, hidden_states)
            preds = []
            for v in dists.values():
              out = v.pred()
              if isinstance(out, tuple):
                preds.extend(out)
              else:
                preds.append(out)
            return tuple(preds) + (rnn_state,) + new_hidden_states

    wrapped_model = OnnxWrapper(model)
    wrapped_model.eval()

    # Ensure ONNX output directory exists
    onnx_dir = save_path / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / f"{model_name}.onnx"
    onnx_hidden_states_path = onnx_dir / f"{model_name}_hidden_states.pkl"

    utils.print(f"Exporting wrapped model to: {onnx_path}")
    torch.onnx.export(
        wrapped_model,
        dummy_input_tuple,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    utils.print(f"Policy successfully exported to: {onnx_path}")

    # Save initial hidden states.
    import pickle
    init_hidden_states_np = tuple([h.detach().cpu().numpy() for h in init_hidden_states])
    with open(onnx_hidden_states_path, 'wb') as f:
      pickle.dump(init_hidden_states_np, f)
    utils.print(f"Hidden states saved to: {onnx_hidden_states_path}")

    if test_inputs:
      utils.print('Testing inputs...')
      import onnxruntime
      sess_options = onnxruntime.SessionOptions()
      # Optional: Configure session options (e.g., providers, optimizations) here
      # sess_options.intra_op_num_threads = 1
      session = onnxruntime.InferenceSession(str(onnx_path), sess_options=sess_options, providers=['CPUExecutionProvider'])
      # 2. Get input/output details
      input_details = session.get_inputs()
      output_details = session.get_outputs()
      input_names = [inp.name for inp in input_details]
      output_names = [out.name for out in output_details]

      utils.print(f"ONNX Model Inputs: {input_names}")
      utils.print(f"ONNX Model Outputs: {output_names}")

      batch_size = 1

      # Create the inference dictionary using the prefixed names expected by the ONNX model
      inference_input_dict = {
        f"{k}": v.repeat(batch_size, *[1 for _ in range(v.ndim - 1)]).cpu().numpy() # Create the prefixed key expected by ONNX
        for k, v in dummy_obs.items() # Iterate through original keys/values
      }
      inference_input_dict.update({
        f"hidden_{i}": v.repeat(1, batch_size, 1).detach().cpu().numpy()
        for i, v in enumerate(init_hidden_states)
      })

      # Pass the flattened dictionary directly to session.run
      outputs = session.run(output_names, inference_input_dict)
      utils.print('INPUTS:')
      for k, v in inference_input_dict.items():
        utils.print(f'\t{k}: {v.shape}', color='blue')
      utils.print('OUTPUTS:')
      for k, v in zip(output_names, outputs):
        utils.print(f"\t{k}: shape={v.shape}, min={v.min():.2f}, max={v.max():.2f}, mean={v.mean():.2f}", color='blue')


def main(argv = None):
    log_root = pathlib.Path(os.path.join(legged_gym.GAUSS_GYM_ROOT_DIR, 'logs'))
    load_run_path = None
    parsed, other = flags.Flags({'runner': {'load_run': ''}, 'model_name': 'all'}).parse_known(argv)
    if parsed.runner.load_run != '':
      load_run_path = log_root / parsed.runner.load_run
    else:
      load_run_path = sorted(
        [item for item in log_root.iterdir() if item.is_dir()],
        key=lambda path: path.stat().st_mtime,
      )[-1]

    utils.print(f'Loading run from: {load_run_path}...')
    cfg = config.Config.load(load_run_path / 'train_config.yaml')
    cfg = cfg.update({'runner.load_run': load_run_path.name})
    cfg = cfg.update({'runner.resume': True})
    cfg = cfg.update({'headless': True})
    cfg = cfg.update({'env.num_envs': 1})
    cfg = cfg.update({'rl_device': 'cuda:0'})
    cfg = cfg.update({'sim_device': 'cuda:0'})
    cfg = flags.Flags(cfg).parse(other)
    cfg = types.MappingProxyType(dict(cfg))

    task_class = task_registry.get_task_class(cfg["task"])
    helpers.set_seed(cfg["seed"])
    env = task_class(cfg=cfg)

    if cfg["logdir"] == "default":
        log_root = pathlib.Path(legged_gym.GAUSS_GYM_ROOT_DIR) / 'logs'
    elif cfg["logdir"] != "":
        log_root = pathlib.Path(cfg["logdir"])
    else:
        raise ValueError("Must specify logdir as 'default' or a path.")

    runner: Runner = eval(cfg["runner"]["class_name"])(env, cfg, device=cfg["rl_device"])
    resume_path = runner.load(log_root)

    utils.print("Exporting policy to ONNX...")
    model_dicts = {
       'policy': (
          runner.policy,
          runner.policy_obs_space,
       ),
       'value': (
          runner.value,
          runner.value_obs_space,
       ),
       'image_encoder': (
          runner.image_encoder,
          runner.image_encoder_obs_space,
       )
    }
    if parsed.model_name == 'all':
      export_models = list(model_dicts.keys())
    else:
      export_models = [parsed.model_name]

    for model_name in export_models:
      model, obs_space = model_dicts[model_name]
      output_names = model.output_dist_names
      model.eval()
      export_onnx(model_name, model, obs_space, output_names, resume_path, test_inputs=True)


if __name__ == "__main__":
    main()
    import sys
    sys.exit(0)
