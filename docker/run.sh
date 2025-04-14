#!/bin/bash
set -e
set -u

# Set which GPU to use (0 is default, change this number to use a different GPU)
CONTAINER_NAME="gauss_gym_container"
GPU_ID=5

# Check if the container exists
if docker container inspect "$CONTAINER_NAME" > /dev/null 2>&1; then
    # It exists. Check if it's running.
    if [[ "$(docker container inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" == "true" ]]; then
        echo "Attaching to running container $CONTAINER_NAME..."
        docker exec -it "$CONTAINER_NAME" /bin/bash
        exit 0 # Exit script after attaching
    else
        # It exists but is stopped. Remove it before creating a new one.
        echo "Container $CONTAINER_NAME exists but is stopped. Removing it..."
        docker rm "$CONTAINER_NAME"
    fi
fi

# If we reach here, the container was not running (either never existed or was stopped and removed)
echo "Creating and running new container $CONTAINER_NAME with Xvfb on GPU $GPU_ID"

# Conditionally add WANDB_API_KEY if it's set in the environment
WANDB_ARG=""
if [ ! -z "${WANDB_API_KEY-''}" ]; then
    WANDB_ARG="-e WANDB_API_KEY=$WANDB_API_KEY"
fi

# TODO: switch to xvfb-run.
docker run -it \
--network=host \
--gpus=all \
-v $(pwd)/..:/opt/gauss-gym \
-e NVIDIA_VISIBLE_DEVICES=$GPU_ID \
$WANDB_ARG \
--name=$CONTAINER_NAME \
--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
gauss_gym bash -c "mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix && Xvfb :1 -screen 0 1024x768x24 & export DISPLAY=:1 && /bin/bash"

# gauss_gym xvfb-run --auto-servernum --server-args="-screen 0 1024x768x24" /bin/bash
# gauss_gym bash -c "Xvfb :1 -screen 0 1024x768x24 & export DISPLAY=:1 && /bin/bash"

# gauss_gym xvfb-run --auto-servernum --server-args="-screen 0 1024x768x24" sleep infinity
# docker exec -it "$CONTAINER_NAME" /bin/bash
