import types
import os
import pathlib
import time
import isaacgym
import torch.distributed as torch_distributed

from legged_gym.envs import *
import legged_gym
from legged_gym.utils.task_registry import task_registry
from legged_gym.utils import flags
from legged_gym.utils import helpers

from legged_gym.rl.runner import Runner


def main(argv=None):

    parsed, other = flags.Flags(task='').parse_known(argv)
    if parsed.task == '':
        raise ValueError('--task must be specified for training.')
    cfg = task_registry.get_cfgs(parsed.task)
    cfg = cfg.update({'headless': True})
    cfg = flags.Flags(cfg).parse(other)

    if cfg["multi_gpu"]:
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        global_rank = int(os.getenv("RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        print(f"Horovod global rank {global_rank} local rank: {local_rank}")
        cfg = cfg.update({"sim_device": f'cuda:{local_rank}',
                          "rl_device": f'cuda:{local_rank}',
                          "multi_gpu_global_rank": global_rank,
                          "multi_gpu_local_rank": local_rank,
                          "multi_gpu_world_size": world_size,
                          "seed": cfg["seed"] + global_rank}) # need a different seed for each env so that scaling works properly :)
        torch_distributed.init_process_group("nccl", rank=global_rank, world_size=world_size)

    print(cfg)
    cfg = types.MappingProxyType(dict(cfg))
    task_class = task_registry.get_task_class(cfg["task"])
    helpers.set_seed(cfg["seed"])
    env = task_class(cfg=cfg)

    new_run_name = [f'{cfg["task"]}']
    if cfg["runner"]["run_name"] != "":
        new_run_name += [cfg["runner"]["run_name"]]
    new_run_name += [time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())]
    new_run_name = '_'.join(new_run_name)

    if cfg["logdir"] == "default":
        log_root = pathlib.Path(legged_gym.GAUSS_GYM_ROOT_DIR) / 'logs'
        log_dir = pathlib.Path(log_root) / new_run_name
    elif cfg["logdir"] != "":
        log_root = pathlib.Path(cfg["logdir"])
        log_dir = log_root / new_run_name
    else:
        raise ValueError("Must specify logdir as 'default' or a path.")
    
    runner = eval(cfg["runner"]["class_name"])(env, cfg, device=cfg["rl_device"])

    if cfg["runner"]["resume"]:
        assert cfg["runner"]["load_run"] != "", "Must specify load_run when resuming."
        runner.load(log_root)

    runner.learn(
        num_learning_iterations=cfg["runner"]["max_iterations"],
        log_dir=log_dir, init_at_random_ep_len=True)

if __name__ == '__main__':
    main()
