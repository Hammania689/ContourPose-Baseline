# 4090 / Ada Lovelace (sm_89) variant — CUDA 12.4, Ubuntu 22.04
# Based on contourpose.Dockerfile; changes:
#   - CUDA base image updated to 12.4.1 (sm_89 support requires >= 11.8)
#   - pip bootstrap URL pinned to the Python 3.9-specific endpoint
#   - DALI variant changed from cuda110 to cuda120
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
SHELL ["/bin/bash", "-c"]

# Hack to not have tzdata cmdline config during build
RUN ln -fs /usr/share/zoneinfo/Europe/Amsterdam /etc/localtime

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

ENV HYDRA_FULL_ERROR=1

# Install prerequisites
RUN apt-get update && apt install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.9 python3.9-dev python3.9-distutils python3.9-venv curl
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Python 3.9-specific pip bootstrap (generic get-pip.py requires >= 3.10 since early 2025)
RUN curl -sS https://bootstrap.pypa.io/pip/3.9/get-pip.py | python3.9
RUN python3.9 -m pip install --upgrade pip setuptools wheel

# Create symlinks so pip/pip3 use python3.9
RUN ln -sf /usr/local/bin/pip3.9 /usr/bin/pip
RUN ln -sf /usr/local/bin/pip3.9 /usr/bin/pip3

RUN apt install -qqy lsb-release git curl unzip tmux wget ranger

RUN pip install jupyterlab glfw wandb
RUN apt install -y fontconfig libglfw3-dev libgles2-mesa-dev libgl1-mesa-glx libglib2.0-0
# DALI for CUDA 12.x
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda120
RUN pip install blenderproc trimesh
RUN blenderproc pip install debugpy tqdm trimesh
RUN apt-get -y install libxi6:amd64 libxkbcommon0 libxkbcommon-x11-0 libsm6
RUN apt-get install libegl1-mesa-dev libgles2-mesa-dev
RUN pip install vispy trimesh pyrender PyOpenGL PyOpenGL_accelerate gdown gin-config transforms3d "numpy<2"
