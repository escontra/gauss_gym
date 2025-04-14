#!/bin/bash
set -e
set -u
SCRIPTROOT="$( cd "$(dirname "$0")" ; pwd -P )"
cd "${SCRIPTROOT}/.."

IMAGE_NAME="gauss_gym" # Define your desired image name
COMPUTE_CAP="7.5"

# Attempt to find nvidia-smi
if ! command -v nvidia-smi &> /dev/null
then
    echo "nvidia-smi could not be found. Using default CUDA_ARCH_LIST=${COMPUTE_CAP}"
else
    # Query the compute capability on the host
    # Extracts major.minor, handles multiple GPUs by comma-separating unique capabilities
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | cut -d'.' -f1,2 | sort -u | paste -sd "," -)

    # Use a default if nvidia-smi ran but didn't return a capability (e.g., no CUDA GPUs)
    if [ -z "$COMPUTE_CAP" ]; then
      echo "WARNING: nvidia-smi found, but could not detect GPU compute capability. Using default CUDA_ARCH_LIST=${COMPUTE_CAP}"
    fi
fi

echo "Building Docker image '$IMAGE_NAME' with TORCH_CUDA_ARCH_LIST=${COMPUTE_CAP}"

docker build \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  --build-arg USERNAME=$(whoami) \
  --build-arg SPECIFIC_CUDA_ARCH="${COMPUTE_CAP}" \
  --network host \
  -t "$IMAGE_NAME" \
  -f docker/Dockerfile \
  .
