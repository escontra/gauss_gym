import types
import os
import torch
import pathlib
import legged_gym
import torch.onnx

from legged_gym.utils import flags, config
from legged_gym.rl import deployment_runner

def main(argv = None):
    log_root = pathlib.Path(os.path.join(legged_gym.GAUSS_GYM_ROOT_DIR, 'logs'))
    load_run_path = None
    parsed, other = flags.Flags({'runner': {'load_run': ''}}).parse_known(argv)
    if parsed.runner.load_run != '':
      load_run_path = log_root / parsed.runner.load_run
    else:
      load_run_path = sorted(
        [item for item in log_root.iterdir() if item.is_dir()],
        key=lambda path: path.stat().st_mtime,
      )[-1]

    print(f'Loading run from: {load_run_path}...')
    cfg = config.Config.load(load_run_path / 'config.yaml')
    cfg = cfg.update({'runner.load_run': load_run_path.name})
    cfg = cfg.update({'runner.resume': True})


    cfg = flags.Flags(cfg).parse(other)
    # print(cfg)
    cfg = types.MappingProxyType(dict(cfg))

    deploy_cfg = config.Config.load(load_run_path / 'deploy_config.yaml')
    deploy_cfg = types.MappingProxyType(dict(deploy_cfg))

    if cfg["logdir"] == "default":
        log_root = pathlib.Path(legged_gym.GAUSS_GYM_ROOT_DIR) / 'logs'
    elif cfg["logdir"] != "":
        log_root = pathlib.Path(cfg["logdir"])
    else:
        raise ValueError("Must specify logdir as 'default' or a path.")
    
    runner = deployment_runner.DeploymentRunner(deploy_cfg, cfg, device="cpu")
    runner.load(log_root)

    print("Exporting policy to ONNX...")
    model = runner.policy
    model.eval()

    # Prepare dummy inputs
    print("Preparing dummy inputs...")
    policy_obs_space = runner.obs_space[runner.policy_key]
    policy_dummy_obs = {k: torch.ones(v.shape)[None].to(runner.device) for k, v in policy_obs_space.items()}
    print('POLICY DUMMY OBS:')
    for k, v in policy_dummy_obs.items():
        print(f"\t{k}: {v.shape}")
    ordered_obs_keys = list(policy_dummy_obs.keys())
    dummy_obs_tuple = tuple(policy_dummy_obs[k] for k in ordered_obs_keys)

    init_hidden_states = model.reset(torch.zeros(1), None)
    ordered_hidden_keys = [f'hidden_{i}' for i in range(len(init_hidden_states))]

    print('INIT HIDDEN STATES:')
    for k, v in zip(ordered_hidden_keys, init_hidden_states):
      print(f'\t{k}: {v.shape}')

    dummy_input_tuple = dummy_obs_tuple + init_hidden_states
    input_names = ordered_obs_keys + ordered_hidden_keys
    dynamic_axes = {k: {0: 'batch_size'} for k in ordered_obs_keys}
    dynamic_axes.update({k: {1: 'batch_size'} for k in ordered_hidden_keys})

    action_names = [f'out_{k}' for k in runner.action_space.keys()]
    out_hidden_names = [f'out_{k}' for k in ordered_hidden_keys]
    output_names = action_names + out_hidden_names
    dynamic_axes.update({k: {0: 'batch_size'} for k in action_names})
    dynamic_axes.update({k: {1: 'batch_size'} for k in out_hidden_names})

    # Wrap the model to scale actions and return hidden states.
    class OnnxWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy
            self.obs_keys = list(self.policy.obs_space.keys())
            self.action_keys = list(self.policy.action_space.keys())
            # For scaling actions later on.
            self.lows, self.highs, self.needs_scaling = [], [], []
            for v in self.policy.action_space.values():
               low = torch.tensor(v.low)[None]
               high = torch.tensor(v.high)[None]
               self.needs_scaling.append((torch.isfinite(low).all() and torch.isfinite(high).all()).item())
               self.lows.append(low)
               self.highs.append(high)

        def forward(self, *obs_tuple):
            obs = obs_tuple[:len(self.obs_keys)]
            hidden_states = obs_tuple[len(self.obs_keys):]
            obs_dict = {k: v for k, v in zip(self.obs_keys, obs)}
            preds, new_hidden_states = self.policy(obs_dict, hidden_states, mean_only=True)
            scaled_preds = []
            for pred, need_scale, low, high in zip(preds, self.needs_scaling, self.lows, self.highs):
              if need_scale:
                 scaled_preds.append((pred + 1) / 2 * (high - low) + low)
              else:
                 scaled_preds.append(pred)
            return tuple(scaled_preds) + new_hidden_states

    wrapped_model = OnnxWrapper(model)
    wrapped_model.eval()

    # Ensure ONNX output directory exists
    onnx_dir = runner.resume_path / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "policy.onnx"
    onnx_hidden_states_path = onnx_dir / "hidden_states.pkl"

    print(f"Exporting wrapped model to: {onnx_path}")
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
    print(f"Policy successfully exported to: {onnx_path}")

    # Save initial hidden states.
    import pickle
    init_hidden_states_np = tuple([h.detach().cpu().numpy() for h in init_hidden_states])
    with open(onnx_hidden_states_path, 'wb') as f:
      pickle.dump(init_hidden_states_np, f)

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

    print(f"ONNX Model Inputs: {input_names}")
    print(f"ONNX Model Outputs: {output_names}")

    batch_size = 5

    # Create the inference dictionary using the prefixed names expected by the ONNX model
    inference_input_dict = {
      f"{k}": v.repeat(batch_size, *[1 for _ in range(v.ndim - 1)]).cpu().numpy() # Create the prefixed key expected by ONNX
      for k, v in policy_dummy_obs.items() # Iterate through original keys/values
    }
    inference_input_dict.update({
      f"hidden_{i}": v.repeat(1, batch_size, 1).detach().cpu().numpy()
      for i, v in enumerate(init_hidden_states)
    })

    # Pass the flattened dictionary directly to session.run
    outputs = session.run(output_names, inference_input_dict)
    print('INPUTS:')
    for k, v in inference_input_dict.items():
      print(k, v.shape)
    print('OUTPUTS:')
    for k, v in zip(output_names, outputs):
      print(k, v.shape, v.min(), v.max(), v.mean())


if __name__ == "__main__":
    main()
    import sys
    sys.exit(0)
