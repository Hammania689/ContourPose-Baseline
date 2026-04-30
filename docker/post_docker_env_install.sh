#!/bin/bash
set -e  # Exit on error

# pycairo builds from source and needs these system libraries (no pre-built wheel available)
echo "Installing system build dependencies..."
apt-get install -y pkg-config libcairo2-dev

# Install build dependencies first
echo "Installing build dependencies..."
pip install cython==0.29.32 wheel setuptools

# Install PyTorch for CUDA 11.7 with Python 3.9
echo "Installing PyTorch with CUDA 11.7..."
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Install remaining dependencies for Python 3.9:
echo "Installing remaining dependencies..."
pip install -r requirements-docker-py39.txt

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
