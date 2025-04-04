FROM nvcr.io/nvidia/pytorch:23.04-py3
ENV DEBIAN_FRONTEND=noninteractive 

# System dependencies.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
 libxcursor-dev \
 libxrandr-dev \
 libxinerama-dev \
 libxi-dev \
 mesa-common-dev \
 zip \
 unzip \
 make \
#  gcc-8 \
#  g++-8 \
#  vulkan-utils \
 mesa-vulkan-drivers \
 pigz \
 git \
 libegl1 \
 xvfb \
 git-lfs \
 tmux

# Force gcc 8 to avoid CUDA 10 build issues on newer base OS.
# RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
# RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

# WAR for eglReleaseThread shutdown crash in libEGL_mesa.so.0 (ensure it's never detected/loaded).
# Can't remove package libegl-mesa0 directly (because of libegl1 which we need).
RUN rm /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json

COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /opt/gauss-gym

RUN useradd --create-home gymuser
# Ensure the directory is owned by gymuser.
RUN chown -R gymuser:gymuser /opt/gauss-gym
USER gymuser

# Copy gym repo to docker.
COPY --chown=gymuser . .

# Install gym modules.
ENV PATH="/home/gymuser/.local/bin:$PATH"
# Set Cuda arch explicitly with output of `nvidia-smi --query-gpu=gpu_name,compute_cap --format=csv`.
ENV TORCH_CUDA_ARCH_LIST="8.0"
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install git+https://github.com/nerfstudio-project/gsplat.git
RUN pip install jaxtyping
RUN cd isaacgym/python && pip install -q -e .
RUN pip install -q -e .

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
