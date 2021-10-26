FROM nvcr.io/nvidia/pytorch:23.04-py3

ARG USER_ID
ARG GROUP_ID
ARG USERNAME # Or choose another username
ARG SPECIFIC_CUDA_ARCH
# Normally the output of: `nvidia-smi --query-gpu=gpu_name,compute_cap --format=csv`.
RUN echo "SPECIFIC_CUDA_ARCH: ${SPECIFIC_CUDA_ARCH}"
ENV TORCH_CUDA_ARCH_LIST="${SPECIFIC_CUDA_ARCH}"
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
 gcc-8 \
 g++-8 \
 vulkan-utils \
 mesa-vulkan-drivers \
 pigz \
 git \
 libegl1 \
 xvfb \
 git-lfs \
 lsb-release \
 ffmpeg \
 tmux

# Force gcc 8 to avoid CUDA 10 build issues on newer base OS
# RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8
# RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

# WAR for eglReleaseThread shutdown crash in libEGL_mesa.so.0 (ensure it's never detected/loaded).
# Can't remove package libegl-mesa0 directly (because of libegl1 which we need).
RUN rm /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json

COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /opt/gauss-gym


# Add these RUN commands somewhere before the final USER instruction
# Create the group and user, only if USER_ID and GROUP_ID are provided and non-zero
RUN if [ -n "$USER_ID" ] && [ "$USER_ID" != "0" ] && [ -n "$GROUP_ID" ] && [ "$GROUP_ID" != "0" ]; then \
        groupadd -g ${GROUP_ID} ${USERNAME} && \
        useradd --shell /bin/bash -u ${USER_ID} -g ${GROUP_ID} -o -c "" -m ${USERNAME} && \
        # Add any chown commands needed for specific directories inside the image below
        # Example: chown -R ${USERNAME}:${USERNAME} /some/internal/app/path
        echo "Created user ${USERNAME} with UID ${USER_ID} and GID ${GROUP_ID}" \
    ;else \
        echo "Skipping user creation, running as root or default user" \
    ;fi

# Ensure the directory is owned by user.
RUN chown -R ${USERNAME}:${USERNAME} /opt/gauss-gym
# USER gymuser
USER $USERNAME

# Copy gym repo to docker.
COPY --chown=${USERNAME} . .
ENV PATH="/home/${USERNAME}/.local/bin:$PATH"

# Install gym modules.
RUN pip install torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu121
RUN cd isaacgym/python && pip install -q -e .
RUN pip install -q -e .

ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
