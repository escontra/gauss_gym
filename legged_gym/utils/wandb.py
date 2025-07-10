import os
import wandb

from legged_gym.utils import config

# currently hardcoded
WANDB_PROJECTS=("a1", "t1")


def get_wandb_path(load_run: str, multi_gpu: bool = False, multi_gpu_rank: int = 0) -> str:
    """
    Get the path to a wandb checkpoint.
    Saves in a temp folder in /tmp
    
    Args:
        load_run: Either a wandb run ID or a name prefixed with 'wandb_'
    
    Returns:
        Path to the downloaded checkpoint file
    """
    api = wandb.Api()
    run_id = None
    project = None
    
    # Handle run ID
    # if all(c.isalnum() or c == '-' for c in load_run):
        # run_id = load_run
    if load_run.startswith('wandb_id_'):
        run_id = load_run[9:]
    # Handle run name
    elif load_run.startswith('wandb_'):
        run_name = load_run[6:]  # Remove 'wandb_' prefix

        for project in WANDB_PROJECTS:
            runs = list(api.runs(project, filters={"display_name": run_name}))
            if runs:
                project = project
                break
        if not runs:
            raise ValueError(f"No wandb run found with name: {run_name}")
        if len(runs) > 1:
            print(f"Warning: Multiple runs found with name {run_name}, using the most recent one")
        run_id = runs[0].id
    else:
        return None
    
    print(f"Downloading checkpoint from wandb run ID: {run_id}")
    
    # Get the run and its checkpoints
    run = api.run(f"{project}/{run_id}")
    files = run.files()

    checkpoint_files = [f for f in files if 'model_' in f.name and f.name.endswith('.pth')]
    
    if not checkpoint_files:
        raise ValueError(f"No checkpoints found for wandb run {run_id}")
    
    # Get the latest checkpoint
    # If we upload numerical checkpoints, we can use the following:
    # latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.name.split('model_')[1].split('.')[0]))
    # print(f"Latest checkpoint: {latest_checkpoint.name}")
    # Currently, we assume that we only upload one checkpoint per run (and continually overwrite it)
    assert len(checkpoint_files) == 1, f"Expected 1 checkpoint file, got {len(checkpoint_files)}"
    latest_checkpoint = checkpoint_files[0]
    
    # Create target directory and download
    if multi_gpu:
        target_dir = os.path.join('/tmp/wandb_checkpoints/', run.name, f'rank_{multi_gpu_rank}')
    else:
        target_dir = os.path.join('/tmp/wandb_checkpoints/', run.name)
    os.makedirs(target_dir, exist_ok=True)
    
    target_path = os.path.join(target_dir, latest_checkpoint.name)
    print(f"Downloading checkpoint to {target_path}")
    latest_checkpoint.download(root=target_dir, replace=True)
    print("Download complete!")
    
    return target_path

def get_wandb_config(load_run: str, multi_gpu: bool = False, multi_gpu_rank: int = 0, config_name: str = "train_config.yaml") -> str:
    """
    Get the path to a wandb config YAML file.
    Saves in a temp folder in /tmp
    
    Args:
        load_run: Either a wandb run ID or a name prefixed with 'wandb_'
    
    Returns:
        Path to the downloaded config YAML file
    """
    api = wandb.Api()
    run_id = None
    project = None
    
    # Handle run ID
    if load_run.startswith('wandb_id_'):
        run_id = load_run[9:]
    # Handle run name
    elif load_run.startswith('wandb_'):
        run_name = load_run[6:]  # Remove 'wandb_' prefix
        for project in WANDB_PROJECTS:
            runs = list(api.runs(project, filters={"display_name": run_name}))
            if runs:
                project = project
                break
        if not runs:
            raise ValueError(f"No wandb run found with name: {run_name}")
        if len(runs) > 1:
            print(f"Warning: Multiple runs found with name {run_name}, using the most recent one")
        run_id = runs[0].id
    else:
        return None
    
    print(f"Downloading config from wandb run ID: {run_id}")
    
    # Get the run
    run = api.run(f"{project}/{run_id}")
    files = run.files()

    files = [f for f in files if config_name in f.name]

    assert len(files) == 1, f"Expected 1 config file, got {len(files)}"

    config_file = files[0]


    if multi_gpu:
        target_dir = os.path.join('/tmp/wandb_checkpoints/', run.name, f'rank_{multi_gpu_rank}')
    else:
        target_dir = os.path.join('/tmp/wandb_checkpoints/', run.name)
    os.makedirs(target_dir, exist_ok=True)
    
    target_path = os.path.join(target_dir, config_file.name)
    print(f"Downloading checkpoint to {target_path}")
    config_file.download(root=target_dir, replace=True)
    print("Download complete!")

    cfg = config.Config.load(target_path)

    return cfg