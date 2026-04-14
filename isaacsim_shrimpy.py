"""
TODO
"""

from dex_retargeting.example.vector_retargeting.single_hand_detector import SingleHandDetector
# from robot_motion_interface.isaacsim.isaacsim_interface import IsaacsimInterface

from typing import Optional
import time
import multiprocessing

from pathlib import Path
import numpy as np
import cv2
from queue import Empty
from loguru import logger


HAND_TYPE ="Right"
CAMERA_PATH = None

def retargeting(queue: multiprocessing.Queue):
    detector = SingleHandDetector(hand_type=HAND_TYPE, selfie=False)
    while True:
        try:
            bgr = queue.get(timeout=5)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Empty:
            logger.error(
                "Fail to fetch image from camera in 5 secs. Please check your web camera device."
            )
            continue
            # return

        num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot, keypoint_3d_array = detector.detect(rgb)

        # Pass if no keypoints detected
        if keypoint_2d is not None:
            logger.warning("No keypoints detected.")
            bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
        
        cv2.imshow("realtime_retargeting_demo", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

def produce_frame(queue: multiprocessing.Queue, camera_path: Optional[str] = None):
    if camera_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(camera_path)

    while cap.isOpened():
        success, image = cap.read()
        time.sleep(1 / 30.0)
        if not success:
            continue
        queue.put(image)


def start_threading():
    queue = multiprocessing.Queue(maxsize=10)
    producer_process = multiprocessing.Process(
        target=produce_frame, args=(queue, CAMERA_PATH)
    )
    consumer_process = multiprocessing.Process(
        target=retargeting, args=(queue,)
    )

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()

def main():
    """
    Simple example of static bimanual arms in Isaacsim, solved with 
    cartesian IK.
    """
    config_dir = Path(__file__).resolve().parents[0] / "robot_motion_interface" / "config"
    config_path = config_dir / "isaacsim_config.yaml"

    start_threading()

    # isaac = IsaacsimInterface.from_yaml(config_path)

    # wrist_goal_left = np.array([-0.2, 0.2, 1.4, 0.707, 0.707, 0, 0])
    # wrist_goal_right = np.array([0.2, 0.2, 1.4, 0.707, 0.707, 0, 0])
    
    # x = [wrist_goal_left, wrist_goal_right]
    # isaac.set_cartesian_pose(x, ['left_delto_offset_link', 'right_delto_offset_link'])

    # isaac.start_loop()


if __name__ == "__main__":
   
    main()