#!/bin/bash
set -e
set -u

# Set which GPU to use (0 is default, change this number to use a different GPU)
GPU_ID=0

# TODO: switch to xvfb-run.
echo "running docker with Xvfb on GPU $GPU_ID"
docker run -it \
--network=host \
--gpus=all \
-e NVIDIA_VISIBLE_DEVICES=$GPU_ID \
--name=gauss_gym_container \
gauss_gym bash -c "Xvfb :1 -screen 0 1024x768x24 & export DISPLAY=:1 && /bin/bash"