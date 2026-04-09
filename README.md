# shrimpy_tesollo

Tutorial: https://github.com/dexsuite/dex-retargeting/blob/main/example/vector_retargeting/README.md

# Setup
```bash
# Setup virtual environment (bash):
python3 -m venv shrimp-venv
source shrimp-venv/bin/activate

# Install dependencies
pip install dex-retargeting
cd example/vector_retargeting
pip install -r requirements.txt
```

# Running
To generate pkl from the human video:
```bash
cd example/vector_retargeting
python detect_from_video.py --robot-name tesollo --video-path data/human_hand_video.mp4 --retargeting-type dexpilot --hand-type left --output-path data/tesollo_joints.pkl
```

To see the video:
```
python render_robot_hand.py --pickle-path data/tesollo_joints.pkl --output-video-path data/test.mp4
```