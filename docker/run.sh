#!/bin/bash
set -e
set -u

if [ $# -eq 0 ]
then
    echo "running docker without display"
    docker run -it --network=host --gpus=all --name=gauss_gym_container gauss_gym /bin/bash
else
    export DISPLAY=$DISPLAY
    echo "setting display to $DISPLAY"
	# xhost +
	# docker run -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --network=host --gpus=all --name=isaacgym_container isaacgym /bin/bash
	# xhost -
    docker run -it \
    --network=host \
    --gpus=all \
    --name=gauss_gym_container \
    # -e DISPLAY=$DISPLAY \
    # -e XAUTHORITY=/home/gymuser/.Xauthority \
    # -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    # -v $HOME/.Xauthority:/home/gymuser/.Xauthority:rw \
    gauss_gym /bin/bash
fi
20