"""
TODO
"""

from dex_retargeting.example.vector_retargeting.single_hand_detector import SingleHandDetector
from robot_motion_interface.isaacsim.isaacsim_interface import IsaacsimInterface
from robot_motion_interface.interface import Interface
from typing import Optional
import time
import multiprocessing

from pathlib import Path
import numpy as np
import cv2
from queue import Empty
from loguru import logger
from enum import Enum

class Hand(Enum):
    RIGHT = "Right"
    LEFT = "Left"
    

def scale_and_set_wrist_pose(hand:Hand, unscaled_wrist_pose:np.ndarray, robot_interface:Interface):
    pass


def retargeting(queue: multiprocessing.Queue, hand:Hand, robot_interface:Interface):
    detector = SingleHandDetector(hand_type=hand.value, selfie=False)
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

        wrist_pose = np.array(keypoint_3d_array + [0.707, 0.707, 0, 0])
        scale_and_set_wrist_pose(hand, wrist_pose, robot_interface)

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


def start_threading(robot_interface, hand_type, camera_path):
    # queue = multiprocessing.Queue(maxsize=10)
    # producer_process = multiprocessing.Process(
    #     target=produce_frame, args=(queue, camera_path)
    # )
    # consumer_process = multiprocessing.Process(
    #     target=retargeting, args=(queue, hand_type, robot_interface)
    # )


    # producer_process.start()
    # consumer_process.start()

    # Needs to be in "main" thread
    robot_interface.start_loop()

    # producer_process.join()
    # consumer_process.join()

def main():
    """
    TODO
    """
    CONFIG_DIR = Path(__file__).resolve().parents[0] / "robot_motion_interface" / "config"
    CONFIG_PATH = CONFIG_DIR / "isaacsim_config.yaml"

    HAND_TYPE = Hand.RIGHT
    CAMERA_PATH = None

    robot_interface = IsaacsimInterface.from_yaml(CONFIG_PATH)

    start_threading(robot_interface, HAND_TYPE, CAMERA_PATH)

    # 

    # wrist_goal_left = np.array([-0.2, 0.2, 1.4, 0.707, 0.707, 0, 0])
    # wrist_goal_right = np.array([0.2, 0.2, 1.4, 0.707, 0.707, 0, 0])
    
    # x = [wrist_goal_left, wrist_goal_right]
    # isaac.set_cartesian_pose(x, ['left_delto_offset_link', 'right_delto_offset_link'])

    # 


if __name__ == "__main__":
   
    main()