import types
import os
import torch
import pathlib
import legged_gym
from legged_gym.utils import flags, config
from legged_gym.rl import deployment_runner
from legged_gym.rl.modules import models

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
    print(cfg)
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

    print(log_root)
    print(cfg["runner"]["load_run"])
    runner.load(log_root)

    print(runner.policy)

    jit_policy: torch.jit.ScriptModule = models.get_policy_jitted(runner.policy, cfg["policy"]["params"])
    print(jit_policy)

if __name__ == "__main__":
    main()
    import sys
    sys.exit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
    parser.add_argument("--checkpoint", type=str, help="Path of model checkpoint to load. Overrides config file if provided.")
    args = parser.parse_args()
    cfg_file = os.path.join("envs", "{}.yaml".format(args.task))
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    if args.checkpoint is not None:
        cfg["basic"]["checkpoint"] = args.checkpoint

    model = ActorCritic(cfg["env"]["num_actions"], cfg["env"]["num_observations"], cfg["env"]["num_privileged_obs"])
    if not cfg["basic"]["checkpoint"] or (cfg["basic"]["checkpoint"] == "-1") or (cfg["basic"]["checkpoint"] == -1):
        cfg["basic"]["checkpoint"] = sorted(glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
    print("Loading model from {}".format(cfg["basic"]["checkpoint"]))
    model_dict = torch.load(cfg["basic"]["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(model_dict["model"])

    model.eval()
    script_module = torch.jit.script(model.actor)
    save_path = os.path.splitext(cfg["basic"]["checkpoint"])[0] + ".pt"
    script_module.save(save_path)
    print(f"Saved model to {save_path}")
