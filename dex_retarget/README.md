# Dex Retargeting
Source: https://github.com/dexsuite/dex-retargeting/blob/main/example/vector_retargeting/README.md

## Setup
```bash
# Setup virtual environment (bash):
python3.11 -m venv shrimp-venv
source venv/bin/activate

# Install dependencies
pip install -e dex_retargeting
pip install -r dex_retargeting/example/vector_retargeting/requirements.txt
pip install numpy==1.26.4  # Must be installed after everything in requirements.txt or causes issues (ignore red warning)
```

## Running
```bash
cd dex_retargeting/example/vector_retargeting
```

To generate pkl (training) from the human video:
```bash
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