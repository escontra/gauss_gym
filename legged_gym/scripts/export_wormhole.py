import types
import os
import pathlib
import legged_gym
import tempfile
import shutil
import subprocess

from legged_gym import utils
from legged_gym.utils import flags, config

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
    cfg = cfg.update({'sim_device': 'cuda:0'})
    cfg = flags.Flags(cfg).parse(other)
    cfg = types.MappingProxyType(dict(cfg))

    
    checkpoint = cfg["runner"]["checkpoint"]
    utils.print(f'\tNum checkpoints: {len(list((load_run_path / "nn").glob("*.pth")))}', color='blue')
    utils.print(f'\tLoading checkpoint: {checkpoint}', color='blue')

    if (checkpoint == "-1") or (checkpoint == -1):
      model_path = sorted(
        (load_run_path / "nn").glob("*.pth"),
        key=lambda path: path.stat().st_mtime,
      )[-1]
    else:
      model_path = load_run_path / "nn" / f"model_{checkpoint:06d}.pth"
    utils.print(f'\tLoading model weights from: {model_path}', color='blue')
  
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp_path = pathlib.Path(tmpdir)
      utils.print(f"Created temporary dir: {tmp_path}", color='blue')
      subdir = tmp_path / "nn"
      subdir.mkdir(parents=True, exist_ok=True)
      shutil.copy(model_path, subdir / "model.pth")
      shutil.copy(load_run_path / "train_config.yaml", tmp_path / "train_config.yaml")
      subprocess.run(["wormhole", "send", str(tmp_path)])


if __name__ == "__main__":
    main()
    import sys
    sys.exit(0)
