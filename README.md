# shrimpy_tesollo

Tutorial: https://github.com/dexsuite/dex-retargeting/blob/main/example/vector_retargeting/README.md

To generate pkl from the human video:
```
python detect_from_video.py --robot-name tesollo --video-path data/human_hand_video.mp4 --retargeting-type dexpilot --hand-type left --output-path data/tesollo_joints.pkl
```

To see the video:
```
python render_robot_hand.py --pickle-path data/tesollo_joints.pkl --output-video-path data/test.mp4
```