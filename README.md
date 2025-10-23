<h1 align="center">GaussGym</h1>

<p align="center">
  <a href="https://arxiv.org/pdf/2510.15352" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="Paper">
  </a>
  <a href="https://escontrela.me/gauss_gym/" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/Website-blue.svg" alt="Website">
  </a>
  <a href="https://huggingface.co/collections/escontra/gauss-gym-datasets-68f1545f33691c8cb43a55ff" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/🤗-Data-yellow.svg" alt="Data">
  </a>
  <a href="https://escontrela.me/gauss_gym/#try-it-yourself" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/Demo-Interactive-green.svg" alt="Demo">
  </a>
</p>
<p align="center">
  <img width="75%" alt="all scenes" src="https://github.com/user-attachments/assets/3b089757-e64f-4ecc-9d23-15588bf93636" />
</p>

---

# Release TODO:

- [ ] Release multi-gpu/node training
- [ ] Provide deployment code
- [ ] Provide pre-trained policies
- [x] [Oct 21, 2025] Initial code and data release

# Installation

Clone `gauss_gym` and install.
```bash
git clone https://github.com/escontra/gauss_gym.git
cd gauss_gym
bash setup_dev.sh
```

This will create a new `gauss_gym` conda environment under `~/.gauss_gym_deps`, which can be activated with:

```bash
source ~/.gauss_gym_deps/miniconda3/bin/activate gauss_gym
```

# Training

Configs have been provided for:
- `a1`
- `a1_vision`
- `go1`
- `go1_vision`
- `t1`
- `t1_vision`
- `anymal_c`
- `anymal_c_vision`

Configs can be created under `gauss_gym/envs` and require updating the registry at `gauss_gym/envs/__init__.py`.

**Note:** Only a1 and t1 configs have been verified on hardware.

Train policies with:

```bash
gauss_train --task=t1 --env.num_envs 2048
```

This will automatically download the scenes from huggingface (requires HF login) and begin training. The number of scenes and their sources is configured in `terrain.scenes`. Use `*_vision` configs to train policy from pixels.

Training will launch a viser server at `localhost:8080` which can be used to visualize various aspects of the policy and scene. Use `--runner.share_url True` to start a tunnel and share this URL with others! Blind policies don't render gaussians by default. To force gaussian visualization, use `--env.force_renderer=True`.

All values in the `config.yaml` can be modified from the command line. Logging to wandb during training can be enabled with: `--runner.use_wandb True`.

# Evaluation

Evaluate policies with: `gauss_play --runner.load_run=<RUN_NAME>`, where `<RUN_NAME>` can be:
  - the name of the run in `logs/`.
  - the wandb run name: `--runner.load_run=wandb_t1_<WANDB_RUN_NAME>` (this will automatically download the latest checkpoint from huggingface)
  - the wandb run id: `--runner.load_run=wandb_id_<WANDB_ID>`
  - a wandb run id pointing to another project: `--runner.load_run=wandb_id_<WANDB_ID>:<WANDB_PROJECT>`
  - a wandb run id pointing to another project/entity: `--runner.load_run=wandb_id_<WANDB_ID>:<WANDB_PROJECT>:<WANDB_ENTITY>`

# Adding your own environment with your iPhone/Android + Polycam:

1. Download [`Polycam`](https://poly.cam/)
2. Use Polycam in "Space" mode to capture scene.
3. Process the data in the Polycam app. Export "Raw Data" and "GLTF" from the app. Place the unzipped contents in <POLYCAM_PATH> and rename the `*.glb` file to `raw.glb`.
  - **Note:** Exporting "Raw Data" may require the upgraded version of the app.
4. Create the nerfstudio conda environment with `bash build/environments/nerfstudio/setup_dev.sh`
  - This will create a nerfstudio environment in `~/.ns_deps`. Which you can activate with: `source ~/.ns_deps/miniconda3/bin/activate ns`
5. With the `ns` conda environment active, train a gaussian splat with `bash scene_generation/iphone/polycam_scenes.sh <POLYCAM_PATH>`
6. Generate environment meshes for the scene with: `python scene_generation/generate_mesh_slices.py --config=scene_generation/configs/polycam.py --config.load_dir=<POLYCAM_PATH>`
7. Train/evaluate a policy in your scene with: `--terrain.scenes.iphone_data.repo_id=local:<POLYCAM_PATH>`. You can also add a new entry to the `terrain.scenes` of the config using your custom local or huggingface path.

# Config structure

Configs for each environment are located in `gauss_gym/envs/*/config.yaml`, and specify setting for both the environment and learning. The main aspects in the config are:
  - `env`: Environment configs.
  - `observations`: Policy, critic, and image encoder observations, latency, noise, and delay.
  - `symmetries`: Observation symmetry as used by [Mittal et al.](https://arxiv.org/abs/2403.04359)
  - `terrain`: Terrain configuration, including `terrain.scenes`, which specifies number of scenes to load and from where (e.g. HF).
  - `init_state`: Robot state initialization.
  - `control`: Robot control parameters, such as stiffness and damping.
  - `asset`: URDF and robot information, including termination links and torque clipping.
  - `domain_rand`: Domain randomization configuration.
  - `termination`: Termination conditions
  - `commands`: Task specification and parameters.
  - `rewards`: Reward configuration.
  - `sim`: Simulator configuration.
  - `image_encoder`, `policy`, `value`: Network parameters.
  - `algorithm`: Learning configuration.
  - `runner`: Training, logging, checkpointing params.
