# This image was constrcuted following instructions outlined: http://wiki.ros.org/docker/Tutorials/Hardware%20Acceleration
# Please refer to the resources above
FROM  nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
SHELL ["/bin/bash", "-c"]

# Hack to not have tzdata cmdline config during build
RUN ln -fs /usr/share/zoneinfo/Europe/Amsterdam /etc/localtime

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
     ${NVIDIA_VISIBLE_DEVICES:-all}

ENV NVIDIA_DRIVER_CAPABILITIES \
     ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

ENV HYDRA_FULL_ERROR=1

# Install prerequisites
RUN apt-get update && apt install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
#RUN apt update
#RUN apt install -y python3.7 python3.7-dev
#RUN apt install -y python3.7-distutils python3.7-venv
#RUN python3.7 -m ensurepip --upgrade
# RUN curl https://bootstrap.pypa.io/pip/3.7/get-pip.py -o get-pip.py && \
# 	python3.7 get-pip.py
# 
# RUN apt install -y python3-dev \
# 	&& update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1 


ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.9 python3.9-dev python3.9-distutils python3.9-venv curl
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Install pip in a way that /usr/bin/python3.9 -m pip works in isolated environments
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9
RUN python3.9 -m pip install --upgrade pip setuptools wheel

# Create symlinks so pip/pip3 use python3.9
RUN ln -sf /usr/local/bin/pip3.9 /usr/bin/pip
RUN ln -sf /usr/local/bin/pip3.9 /usr/bin/pip3

RUN apt install -qqy lsb-release git curl unzip tmux wget ranger


#RUN python -m pip install torch==2.0.1 torchvision==0.15.2 omegaconf torchmetrics
#RUN python -m pip install fvcore iopath  opencv-python pycocotools matplotlib onnxruntime onnx 
RUN pip install jupyterlab glfw wandb
RUN apt install -y fontconfig libglfw3-dev libgles2-mesa-dev libgl1-mesa-glx libglib2.0-0 
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110
RUN pip install blenderproc trimesh
RUN blenderproc pip install debugpy tqdm trimesh
RUN apt-get -y install libxi6:amd64 libxkbcommon0 libxkbcommon-x11-0 libsm6
RUN apt-get install libegl1-mesa-dev libgles2-mesa-dev
RUN pip install vispy trimesh pyrender PyOpenGL PyOpenGL_accelerate gdown gin-config transforms3d "numpy<2"
