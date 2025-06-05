# Walking from Pixels

---

# Installation

1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
2. Install pytorch:
    - `pip install torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu121`
3. Install Isaac Gym
    - Download and install Isaac Gym Preview 3 (Preview 2 will not work!) from https://developer.nvidia.com/isaac-gym
    - `cd isaacgym/python && pip install -e .`
    - Try running an example `cd examples && python 1080_balls_of_solitude.py`
    - For troubleshooting check docs `isaacgym/docs/index.html`)
4. Install gauss-gym
- Clone this repository
    - `git clone https://github.com/Alescontrela/gauss-gym.git`
    - `cd legged-gym && pip install -e .`


# Training:
1. Download zipped scenes from [here](https://drive.google.com/file/d/1bbmcaEKES6XivAXLE_FAyWAve-Olbhab/view?usp=sharing) and unzip to `<SCENE_PATH>`:
2. Train policies with: `python legged_gym/scripts/train.py --task=a1 --terrain.scenes=<SCENE_PATH>`
  - This will create a folder under `logs/` with checkpoints.

# Evaluation:
1. Evaluate policies with: `python legged_gym/scripts/play.py --runner.load_run=<RUN_NAME>`
  - `<RUN_NAME>` is the name of the run in `logs/`.

# Config structure:
Configs for each environment are located in `legged_gym/envs/*/config.yaml`, and specify setting for both the environment and learning.