#!/bin/bash
set -e  # Exit on error

# pycairo builds from source and needs these system libraries (no pre-built wheel available)
echo "Installing system build dependencies..."
apt-get install -y pkg-config libcairo2-dev

# Install build dependencies first
echo "Installing build dependencies..."
pip install cython==0.29.32 wheel setuptools

# PyTorch for CUDA 12.4, Python 3.9 (2.4.1 is the last release supporting Python 3.9)
echo "Installing PyTorch with CUDA 12.4..."
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies for Python 3.9:
echo "Installing remaining dependencies..."
pip install -r requirements-docker-py39-4090.txt

# Install glumpy separately with --no-build-isolation to avoid pip module issues
echo "Installing glumpy with --no-build-isolation..."
pip install glumpy==1.2.0 --no-build-isolation || echo "Warning: glumpy installation failed, skipping..."
pip install ipywidgets

echo "Installation complete!"



# Linking mounted data to expected location
ROOT=/contourpose
docker_mnt_path="/data/Datasets/ContourPose"
mkdir -p $ROOT/data
ln -s $docker_mnt_path/Train_Scenes/Real $ROOT/data/train/obj_custom
ln -s $docker_mnt_path/Train_Scenes/Synthetic $ROOT/data/train/renders
ln -s $docker_mnt_path/Test_Scenes $ROOT/data/test
ln -s $docker_mnt_path/SUN2012pascalformat $ROOT/data/SUN2012pascalformat
