# shrimpy_tesollo



## Setup
If you want to run Isaacsim (MACHINE A), you will need a computer with a Nvidia 40XX GPU. If you want to run the robots, you will need a computer with the Franka Emika realtime kernel patch (MACHINE B).


### 1. Setup Docker
On **either machine**, follow the below steps.
Note: This allows you to run ros or isaacsim with docker. These instructions are an adapted version of [these](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_container.html) and [these](https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html)

1. Install Docker by following the `Install using the apt repository` instruction [here](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository).

2. [ONLY MACHINE A] Install Nvidia Container Toolkit by following [these instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). We recommend version 1.17.8 but other versions may work (although we know for sure that version 1.12 has Vulkan issues). 
    * Make sure you complete the `Installation` section for `With apt: Ubuntu, Debian` and also the `Configuring Docker` section.
    * To check proper installation, please run `sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi`. This should output a table with your Nvidia driver. If you run into `Failed to initialize NVML: Unknown Error`, reference [this post](https://stackoverflow.com/questions/72932940/failed-to-initialize-nvml-unknown-error-in-docker-after-few-hours) for the solution.

3. Install Docker compose by following there `Install using the repository` [instructions here](https://docs.docker.com/compose/install/linux/#install-using-the-repository).


### 2. Compile and Launch Docker Containers
Run either of these on the specified computer to build and launch the docker container. They will take a while the first time you run them. The reason there are 2 different containers to run is because the Isaacsim one takes A LOT longer to build and is A LOT larger so we also want to give the option of the smaller non-isaacsim container. 

a. On MACHINE A (Docker with Isaacsim and workplace dependencies):

```bash
xhost +local: # Note: This isn't very secure but is th easiest way to do this
sudo docker compose -f compose.isaac.yaml build
sudo docker compose -f compose.isaac.yaml run --rm isaac-base  # Opens TERMINAL 1

# TODO: ADD THIS TO DOCKER:
pip install OneEuroFilter pygame 
pip install -e sensor_interface_py
pip uninstall numpy
conda install numpy==1.26
# pip install "numpy<2"
```

To test that isaacsim is working correctly, you can run `. /isaac-sim/isaac-sim.sh`.

NOTE: If you need to start another terminal, once the container is started, run `sudo docker compose -f compose.isaac.yaml exec isaac-base bash`


## Running

### Sim Testing
```bash
python3 isaacsim_shrimpy.py
```

### Dex Retargeting
```bash
cd dex_retargeting/example/vector_retargeting
```

To generate pkl from the human video (run 1):
```bash
# Right
python detect_from_video.py --robot-name tesollo --video-path data/human_hand_video.mp4 --retargeting-type dexpilot --hand-type right --output-path data/tesollo_joints.pkl
```

To see the video:
```bash
python render_robot_hand.py --pickle-path data/tesollo_joints.pkl --output-video-path data/test.mp4

# Note, if you are rendering a pkl that you did not create, you will need to run:
python render_robot_hand.py --pickle-path data/tesollo_joints.pkl --output-video-path data/test.mp4 --overwritten-pkl-path ../../src/dex_retargeting/configs/teleop
```


To run realtime visualization via webcam:
```bash

python3 show_realtime_retargeting.py --robot-name tesollo --retargeting-type dexpilot --hand-type right 
```


### Diffusion Low Dimension Training
Make sure you run this in the container...

Setup:
```bash

conda install dill diffusers zarr "numpy==1.26" "numcodecs<0.16" pandas 
pip install av
pip install -e diffusion_policy

```

Collecting Data:
```bash
python3 shrimpy_col_data.py

```

npz to hdf5 conversion script:
```bash
# python convert_npz_to_hdf5_objects.py --input data/isaacsim_demos --output diffusion_policy/shrimpy_data/isaacsim_demos_converted/low_dim.hdf5

python convert_npz_to_hdf5_objects.py --input data/isaacsim_demos --output diffusion_policy/shrimpy_data/isaacsim_demos_converted/low_dim.hdf5

```

Training:
```bash
cd diffusion_policy

# HYDRA_FULL_ERROR=1 python train.py --config-name=train_diffusion_unet_lowdim_workspace task=shrimpy_lowdim

HYDRA_FULL_ERROR=1 python train.py --config-name=train_diffusion_unet_lowdim_workspace task=shrimpy_stack_lowdim
```

Evaluation (Make sure to replace input with your actual ckpt file):
```bash
cd /workspace

python shrimpy_eval_isaacsim.py --input epoch=0500-train_loss=0.005.ckpt

python shrimpy_eval_isaacsim.py --input epoch=1100-train_loss=0.002.ckpt
```



## Data
The data we recorded and trained the diffusion policy on is located in the [/data](data) folder:
* `task0/`: Teleoperation demonstrations of moving the right robot arm in a square motion.
* `task1/`:  Teleoperation demonstrations of picking up a small blue cube and placing it on a larger flat red cube (fixed initial cube poses).
* `task2/`: Teleoperation demonstrations of picking up a uniform blue cube and placing it on the same-sized red cube (random inital cube poses). In this folder demonstrations that are numbered 00XX or 01XX are demonstrations where the cubes where spawned randomly on the full table workspace ~40x40cm. Alternatively, demonstrations that are numbered 02XX have cubes spawned randomly on a smaller table workspace (~20x20cm).

Note, this repo is currently setup to record and train `task2` data (`task0` should work too). You can look at the repo history to find configurations for the previous tasks.