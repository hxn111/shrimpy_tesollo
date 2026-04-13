# shrimpy_tesollo

Tutorial: https://github.com/dexsuite/dex-retargeting/blob/main/example/vector_retargeting/README.md

# Setup
```bash
# Setup virtual environment (bash):
python3 -m venv shrimp-venv
source shrimp-venv/bin/activate

# Install dependencies
pip install -e .
cd example/vector_retargeting
pip install -r requirements.txt
pip install numpy==1.26.4  # Must be installed after everything in requirements.txt or causes issues (ignore red warning)
```

# Running
```bash
cd example/vector_retargeting
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

# Utils
Visualizing URDF:
```bash
cd assets/robots/hands/tesollo_hand
yourdfpy ./tesollo_hand_left.urdf
```

You can use these keys in the urdf viewer:
* a: Toggle rendered XYZ/RGB axis markers (off, world frame, every frame)
* w: Toggle wireframe mode (good for looking inside meshes, off by default)
