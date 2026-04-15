"""
TODO
"""

from dex_retargeting.example.vector_retargeting.single_hand_detector import SingleHandDetector
from robot_motion_interface.isaacsim.isaacsim_interface import IsaacsimInterface
from robot_motion_interface.interface import Interface
from typing import Optional
import time
import threading
import queue
import signal

from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from queue import Empty
from loguru import logger
from enum import Enum

class Hand(Enum):
    RIGHT = "Right"
    LEFT = "Left"
    
#  Assume camera is on table facing up!
def scale_and_set_wrist_pose(hand:Hand, unscaled_wrist_pose:np.ndarray, robot_interface:Interface):
    # DEFAULT_WRIST_GOAL_LEFT = np.array([-0.2, 0.2, 1.4, 0.707, 0.707, 0, 0])
    # DEFAULT_WRIST_GOAL_RIGHT = np.array([0.2, 0.2, 1.4, 0.707, 0.707, 0, 0])

    print("ORIGINAL", unscaled_wrist_pose)
    scaled_wrist_pose = unscaled_wrist_pose.copy()


    if hand == Hand.RIGHT:
        scaled_wrist_pose[0] = scaled_wrist_pose[0] * 10 + 0.3
        scaled_wrist_pose[1] *= -10
        scaled_wrist_pose[2] = scaled_wrist_pose[2] * 20  - 0.4 # + 0.2 # Add 1.2 meters to z

        print("SCALED", scaled_wrist_pose)
        robot_interface.set_cartesian_pose([scaled_wrist_pose], ['right_delto_offset_link'])
    elif hand == Hand.LEFT:
        robot_interface.set_cartesian_pose([scaled_wrist_pose], ['left_delto_offset_link'])


def retargeting(frame_queue: queue.Queue,stop_event,  hand:Hand, robot_interface:Interface):
    detector = SingleHandDetector(hand_type=hand.value, selfie=False)
    while not stop_event.is_set():
        try:
            bgr = frame_queue.get(timeout=5)
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

            bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
            xyz_cam = keypoint_3d_array[0]
            xyz = np.array([xyz_cam[0], xyz_cam[2], xyz_cam[1]]) # robot frame (x right, y forward, z up)
            wrist_pose = np.concatenate([xyz, np.array([0.707, 0.707, 0, 0]) ])
            
            scale_and_set_wrist_pose(hand, wrist_pose, robot_interface)
        else:
            # logger.warning("No keypoints detected.")
            pass
        
        # This requires uninstalling opencv-python-headless but that is incompatible with Isaacsim
        # cv2.imshow("realtime_retargeting_demo", bgr)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

        # Instead use matplotlib (SUPER SLOW)
        # plt.imshow(bgr)
        # plt.title("realtime_retargeting_demo")
        # plt.axis("off")
        # plt.pause(0.001)


def produce_frame(frame_queue: queue.Queue, stop_event, camera_path: Optional[str] = None):
    if camera_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(camera_path)

    while cap.isOpened() and not stop_event.is_set():
        success, image = cap.read()
        time.sleep(1 / 30.0)
        if not success:
            continue
        frame_queue.put(image)
    
    cap.release()



def start_threading(robot_interface, hand_type, camera_path):
    stop_event = threading.Event()
    
    frame_queue = queue.Queue(maxsize=10)
    producer_process = threading.Thread(
        target=produce_frame, args=(frame_queue, stop_event, camera_path)
    )
    consumer_process = threading.Thread(
        target=retargeting, args=(frame_queue, stop_event, hand_type, robot_interface)
    )


    producer_process.start()
    consumer_process.start()

    def handle_sigint(signum, frame):
        print("Shutting down...")
        stop_event.set()
        robot_interface.stop()
        
    signal.signal(signal.SIGINT, handle_sigint)
    # Needs to be in main thread
    robot_interface.start_loop()


    producer_process.join()
    consumer_process.join()


def main():
    """
    TODO
    """
    CONFIG_DIR = Path(__file__).resolve().parents[0] / "robot_motion_interface" / "config"
    CONFIG_PATH = CONFIG_DIR / "isaacsim_config.yaml"

    HAND_TYPE = Hand.RIGHT
    CAMERA_PATH = 1

    robot_interface = IsaacsimInterface.from_yaml(CONFIG_PATH)
    # robot_interface = None

    start_threading(robot_interface, HAND_TYPE, CAMERA_PATH)

    # 

    # wrist_goal_left = np.array([-0.2, 0.2, 1.4, 0.707, 0.707, 0, 0])
    # wrist_goal_right = np.array([0.2, 0.2, 1.4, 0.707, 0.707, 0, 0])
    
    # x = [wrist_goal_left, wrist_goal_right]
    # isaac.set_cartesian_pose(x, ['left_delto_offset_link', 'right_delto_offset_link'])

    # 


if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn", force=True)
    main()