# Original image
# FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04
# Test from max
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04
#FROM ubuntu:20.04


# Command added to handle the data on the drive
# ARG USERNAME=vscode
# ARG USER_UID=1002
# ARG USER_GID=$USER_UID
 
# #####################
# # Create a non-root user to use if preferred - see https://aka.ms/vscode-remote/containers/non-root-user.
# #####################
# RUN apt-get update \
# && groupadd --gid $USER_GID $USERNAME \
# && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
#     # [Optional] Add sudo support for non-root user
# && apt-get install -y sudo \
# && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
# && chmod 0440 /etc/sudoers.d/$USERNAME \
#     #
#     # Clean up
# && apt-get autoremove -y \
# && apt-get clean -y
### Xavier's input ends here


ENV TZ=Etc/UTC


RUN DEBIAN_FRONTEND=noninteractive apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3.8

RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    nano \
    git \
    g++ \
    gcc \
    libgl1 \
    libglib2.0-0


# TODO: Add requirements.txt with detailed versions
#RUN python3.8 -m ensurepip --default-pip
#RUN python3.8 -m pip install --upgrade pip

#Pip install of necessary libraries for Pointcept


RUN apt-get -y install python3-pip
RUN pip install matplotlib
RUN pip install scipy
RUN pip install scikit-image
RUN python3.8 -m pip install stonesoup
RUN pip install hdbscan
RUN pip install h5py pyyaml 
RUN pip install nuscenes-devkit
RUN pip install plotly
RUN pip install seaborn
# Version used for their development
#RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 
# Let's try to see if it works better with no specification
RUN pip install torch==2.3.0 torchvision==0.18.0. torchaudio==2.3.0

RUN pip install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm
RUN pip install torchmetrics

# Apparently the argument with the website is necessary for a proper build with CUDA (without it failed)
RUN pip install torch-cluster torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
RUN pip install torch-geometric
RUN pip install wandb
RUN python3.8 -m pip install lightning
#RUN pip install spconv-cu113 Original build
# Test with newer version
RUN pip install spconv-cu120
RUN pip install open3d 

# Necessary for PTv3
RUN pip install flash-attn --no-build-isolation

# The original setuptools installed is too old otherwise
RUN pip install --upgrade pip setuptools

# Had a weird error, this solves it (from https://github.com/Pointcept/Pointcept/issues/92)
RUN pip install yapf==0.40.1


# The command below seems to be working, expect for the fact that the setup.py file cannot locate the nvcc folder.
# (But this is not related to docker i think)
COPY /libs /libs 
WORKDIR /libs/pointops

# The GPU on our machine is NVIDIA GeForce RTX 4090 which is 8.9 according to: https://developer.nvidia.com/cuda-gpus
# The first command in comment is the one suggested by pointcept install, but it install the module as an .egg, which is stange
# The replacement (uncommented) should work
# RUN TORCH_CUDA_ARCH_LIST="8.9" python3.8  setup.py install
RUN TORCH_CUDA_ARCH_LIST="8.9" pip install .
WORKDIR /


# ROS2 installation

# RUN DEBIAN_FRONTEND=noninteractive apt update && DEBIAN_FRONTEND=noninteractive apt install locales
# RUN locale-gen en_US en_US.UTF-8
# RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
# RUN export LANG=en_US.UTF-8

# RUN DEBIAN_FRONTEND=noninteractive apt install -y software-properties-common
# RUN add-apt-repository universe

# RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y curl
# RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# RUN apt update
# RUN apt -y upgrade
# RUN DEBIAN_FRONTEND=noninteractive apt install -y ros-galactic-desktop
# RUN DEBIAN_FRONTEND=noninteractive apt install -y ros-dev-tools
# RUN DEBIAN_FRONTEND=noninteractive apt install -y ros-galactic-sensor-msgs-py


# Realised that if I add change to the dockerfile in the end, it *should* take less time to rebuild
RUN pip install raillabel
RUN pip install wandb
RUN pip install laspy
RUN pip install tzdata
RUN pip install pandas
# Added for PVKD
# RUN pip install tqdm
# RUN pip install numba
# RUN pip install Cython
# RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
# RUN pip install spconv-cu120
# RUN pip install strictyaml
COPY . /baseline
WORKDIR /baseline