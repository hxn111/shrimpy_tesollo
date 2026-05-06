# StackBot: Diffusion Pipeline for Stacking Blocks
This repo contains the pipeline for training a diffusion model to stack blocks on the Franka Emika Panda robot arm with a Tesollo 3-Finger gripper in simulation. The pipelines allows you to collect data using your own hand to teleoperate the robot in simulation.

## Hardware Setup:
These are the hardware requirements:
* Computer with Nvidia Geforce 40XX RTX GPU or higher (this repo has been tested on a 4070 and 4090).
* **If you want to collect data**: Realsense RGB-Depth Camera (this repo has been tested on a d435) connected to the computer via USB C. For collecting data, place the camera on the floor with the lense facing up towards your hand.

## Software Setup

### 1. Setup Docker
On your machine, run the following steps:

1. Install Docker by following the `Install using the apt repository` instruction [here](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository).

2. Install Nvidia Container Toolkit by following [these instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). We recommend version 1.17.8 but other versions may work (although we know for sure that version 1.12 has Vulkan issues). 
    * Make sure you complete the `Installation` section for `With apt: Ubuntu, Debian` and also the `Configuring Docker` section.
    * To check proper installation, please run `sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi`. This should output a table with your Nvidia driver. If you run into `Failed to initialize NVML: Unknown Error`, reference [this post](https://stackoverflow.com/questions/72932940/failed-to-initialize-nvml-unknown-error-in-docker-after-few-hours) for the solution.

3. Install Docker compose by following there `Install using the repository` [instructions here](https://docs.docker.com/compose/install/linux/#install-using-the-repository).

> Note: These instructions are an adapted version of [these](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_container.html) and [these](https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html)


### 2. Compile and Launch Docker Containers
Run this on the specified computer to build and launch the docker container. They will take a while the first time you run them. 
```bash
xhost +local: # Note: This isn't very secure but is th easiest way to do this
sudo docker compose -f compose.isaac.yaml build
sudo docker compose -f compose.isaac.yaml run --rm isaac-base 

# TODO: ADD THIS TO DOCKER:
pip install OneEuroFilter pygame 
pip install -e sensor_interface_py
pip uninstall numpy
conda install dill diffusers zarr "numpy==1.26" "numcodecs<0.16" pandas 
pip install av
pip install -e diffusion_policy
```

To test that isaacsim is working correctly, you can run `. /isaac-sim/isaac-sim.sh`.

NOTE: If you need to start another terminal, once the container is started, run `sudo docker compose -f compose.isaac.yaml exec isaac-base bash`


## Diffusion Low Dimension Training Pipeline
Make sure you run this in the container...


**1. Data Collection**: This will let you teleoperate in simulation. Every time your hand appears in camera frame, a new epoch will start. When you remove your hand from the frame, the epoch recording will stop. This will output data into `/data/isaacsim_demos`. 

```bash
python3 shrimpy_col_data.py
```

**2. Data Conversion**:  Use `one` of the following scripts to convert the data from npz to hdf5 format (which the diffusion policy requires):
```bash
# Use this if you recorded your own data:
python convert_npz_to_hdf5.py --input data/isaacsim_demos --output diffusion_policy/shrimpy_data/isaacsim_demos_converted/low_dim.hdf5

# Use this if you are using the pre-recorded data:
python convert_npz_to_hdf5.py --input data/task2 --output diffusion_policy/shrimpy_data/isaacsim_demos_converted/low_dim.hdf5
```

**3. Policy Training**: This is setup to train 5000 epochs but you will most likely only need to train ~1000 (once loss reaches ~0.002). The training checkpoints will output to `diffusion_policy/data/outputs`
```bash
cd diffusion_policy
python train.py --config-name=train_diffusion_unet_lowdim_workspace task=shrimpy_stack_lowdim
```

**4. Policy Running**: This will run the policy in simulation. The initial block position will reset every so often. Make sure to replace the checkpoint path with your own trained one or one downloaded from our [website](https://hxn111.github.io/shrimpy_tesollo/) (checkpoints are ~1GB so we don't keep them on Github):
```bash
cd /workspace
python shrimpy_eval_isaacsim.py --input <YOUR_CHECKPOINT.ckpt>
```



## Data
The data we recorded and trained the diffusion policy on is located in the [/data](data) folder:
* `task0/`: Teleoperation demonstrations of moving the right robot arm in a square motion.
* `task1/`:  Teleoperation demonstrations of picking up a small blue cube and placing it on a larger flat red cube (fixed initial cube poses).
* `task2/`: Teleoperation demonstrations of picking up a uniform blue cube and placing it on the same-sized red cube (random inital cube poses). In this folder demonstrations that are numbered 00XX or 01XX are demonstrations where the cubes where spawned randomly on the full table workspace ~40x40cm. Alternatively, demonstrations that are numbered 02XX have cubes spawned randomly on a smaller table workspace (~20x20cm).

Note, this repo is currently setup to record and train `task2` data (`task0` should work too). You can look at the repo history to find configurations for the previous tasks.

`task0` has the data format:
* obs     : (T, 19)  [eef_pos(3: x,y,z), eef_quat(4: qx,qy,qz,qw), gripper_qpos(12)]
* actions : (T, 18)  [eef_pos(3: x,y,z), eef_rpy(3: roll,pitch,yaw), gripper_qpos(12)]


`task1` and `task2` have the data format:
* obs     : (T, 33)  [object(14: x,y,z,qx,qy,qz,qw,x,y,z,qx,qy,qz,qw), eef_pos(3: x,y,z), eef_quat(4: qx,qy,qz,qw), gripper_qpos(12)]
* actions : (T, 18)  [eef_pos(3: x,y,z), eef_rpy(3: roll,pitch,yaw), gripper_qpos(12)]

