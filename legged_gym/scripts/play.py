import os
import pathlib
import types
import isaacgym
import legged_gym
from legged_gym.envs import *
from legged_gym.utils.task_registry import task_registry
from legged_gym.utils import flags, config
from legged_gym.utils import helpers

from legged_gym.rl.runner import Runner


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
    cfg = cfg.update({'headless': False})
    cfg = cfg.update({'env.num_envs': 50})
    cfg = cfg.update({'domain_rand.apply_domain_rand': False})
    cfg = cfg.update({'curriculum.apply_curriculum': False})
    cfg = cfg.update({'observations.student_observations.add_noise': False})
    cfg = cfg.update({'observations.student_observations.add_latency': False})

    cfg = flags.Flags(cfg).parse(other)
    print(cfg)
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
    
    runner = eval(cfg["runner"]["class_name"])(env, cfg, device=cfg["rl_device"])

    if cfg["runner"]["resume"]:
        assert cfg["runner"]["load_run"] != "", "Must specify load_run when resuming."
        runner.load(log_root)

    runner.play()

if __name__ == '__main__':
    main()
