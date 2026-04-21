"""
TODO
"""


from robot_motion_interface.isaacsim.isaacsim_interface import IsaacsimInterface
from robot_motion_interface.interface import Interface
from sensor_interface.camera.realsense_interface import RealsenseInterface


from dex_retarget.example.vector_retargeting.single_hand_detector import SingleHandDetector
from dex_retargeting.retargeting_config import RetargetingConfig

from typing import Optional
import time
import threading
import queue
import signal

from OneEuroFilter import OneEuroFilter
from pathlib import Path
import numpy as np
import cv2
# import matplotlib.pyplot as plt
import pygame
from queue import Empty
from loguru import logger
from enum import Enum

class Hand(Enum):
    RIGHT = "Right"
    LEFT = "Left"
    
#  Assume camera is on table facing up!
def scale_and_set_poses(hand:Hand, unscaled_wrist_pose:np.ndarray, gripper_pos:np.ndarray, robot_interface:Interface,
                        wrist_filter):
    # DEFAULT_WRIST_GOAL_LEFT = np.array([-0.2, 0.2, 1.4, 0.707, 0.707, 0, 0])
    # DEFAULT_WRIST_GOAL_RIGHT = np.array([0.2, 0.2, 1.4, 0.707, 0.707, 0, 0])

    print("ORIGINAL", unscaled_wrist_pose)
    scaled_wrist_pose = unscaled_wrist_pose.copy()

    # x right, y forward, z up
    if hand == Hand.RIGHT:
        scaled_wrist_pose[0] = -scaled_wrist_pose[0] * 1.5 + 0.1
        scaled_wrist_pose[1] = -scaled_wrist_pose[1] * 2
        scaled_wrist_pose[2] *= 2
        
        filtered_wrist_xyz = np.array([f(v) for f, v in zip(wrist_filter, scaled_wrist_pose[:3])])
        scaled_wrist_pose[:3] = filtered_wrist_xyz
        print("SCALED", scaled_wrist_pose)
        print("Gripper positions", gripper_pos)
        if robot_interface:
            robot_interface.set_cartesian_pose([scaled_wrist_pose], ['right_delto_offset_link'])
            robot_interface.set_joint_positions(gripper_pos, 
                ['right_F1M1','right_F1M2','right_F1M3', 'right_F1M4','right_F2M1','right_F2M2',
                'right_F2M3','right_F2M4','right_F3M1', 'right_F3M2','right_F3M3','right_F3M4'])
    elif hand == Hand.LEFT:
        pass
        # robot_interface.set_cartesian_pose([scaled_wrist_pose], ['left_delto_offset_link'])

def joint_pos_to_robot_pos(joint_pos:np.ndarray, retargeter:RetargetingConfig) -> np.ndarray:
    if joint_pos is None:
        logger.warning(f"Hand is not detected.")
        return
    indices = retargeter.optimizer.target_link_human_indices
    origin_indices = indices[0, :]
    task_indices = indices[1, :]
    ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
    qpos = retargeter.retarget(ref_value)
    return qpos



def retargeting(frame_queue: queue.Queue, stop_event, camera, hand:Hand, robot_interface:Interface, config_path, urdf_path):
    detector = SingleHandDetector(hand_type=hand.value, selfie=False)
    RetargetingConfig.set_default_urdf_dir(str(urdf_path))
    retargeter = RetargetingConfig.load_from_file(config_path).build()
    
    filter_config = {
    'freq': 30,       # Hz (match camera FPS)
    'mincutoff': 0.6,  # Hz
    'beta': 0.3,  
    }

    wrist_filters = [OneEuroFilter(**filter_config) for _ in range(3)]

    pygame.init()
    screen = pygame.display.set_mode((1280, 480))





    ###########################


    while not stop_event.is_set():
        try:
            rgb, depth = frame_queue.get(timeout=5)

        except Empty:
            logger.error(
                "Fail to fetch image from camera in 5 secs. Please check your web camera device."
            )
            continue
            # return


        num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot, keypoint_3d_array = detector.detect(rgb)

        
        # Pass if no keypoints detected
        if keypoint_2d is not None:

            if depth is None:
                # No z coord (hard code to )
                xy_cam = keypoint_2d[0]
                # Scale from 0 -> 1 to -0.2 to 0.2 (mimics depth)
                x = (xy_cam.x - 0.5) * 0.4  
                y = (xy_cam.y - 0.5) * 0.4
                xyz = np.array([x, y, 0.6]) # robot frame (x right, y forward, z up)
            else:
                H, W = depth.shape                                                                                       
                keypoints_px = SingleHandDetector.parse_keypoint_2d(keypoint_2d, (H, W))
                wy, wx = keypoints_px[0].astype(int)  # wrist pixel (x, y)                                               

                print(wx, wy)                                                                                                        
                mask = np.zeros((H, W), dtype=np.int32)
                mask[wx, wy] = 1  # Point cloud "mask" is just one point
                
                point_clouds, mask_ids = camera.depth_to_pointcloud(depth, mask)
                wrist_cloud = point_clouds[mask_ids == 1][0]
                if len(wrist_cloud) == 0:
                    print("WARN: NO DEPTH AT WRIST PIXEL")
                    continue  # no depth at wrist pixel, skip frame
                xyz = wrist_cloud[0]

            wrist_pose = np.concatenate([xyz, np.array([0.707, 0.707, 0, 0]) ])
            gripper_pos = joint_pos_to_robot_pos(joint_pos, retargeter)
            scale_and_set_poses(hand, wrist_pose, gripper_pos, robot_interface, wrist_filters)

            # Prep for drawing
            rgb = detector.draw_skeleton_on_image(rgb, keypoint_2d, style="default")
            if depth is not None:
                depth = detector.draw_skeleton_on_image(depth, keypoint_2d, style="default")
            
        else:
            # logger.warning("No keypoints detected.")
            pass
        
        if depth is not None:
            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_color = cv2.cvtColor(cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        else:
            depth_color = np.zeros_like(rgb)

        combined = np.hstack([rgb, depth_color])
        surf = pygame.surfarray.make_surface(combined.swapaxes(0, 1))
        screen.blit(surf, (0, 0))
        pygame.display.flip()


def produce_frame(frame_queue: queue.Queue, stop_event, camera, use_realsense:bool=True):

    while not stop_event.is_set():
        if not use_realsense:
            success, bgr = camera.read()
            color = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if not success:
                continue
            depth = None
        else:
            try:
                frame = camera.latest()
                color = frame.color
                depth = frame.depth
            except RuntimeError:
                print("WARNING: No frame found from Realsense.")
                continue
       
        time.sleep(1 / 30.0)

        # Put most recent frame (drop others)
        try:                                                                                                     
            frame_queue.get_nowait()  # discard old frame                                                      
        except queue.Empty:                                                                                      
            pass
        frame_queue.put((color, depth))  
        
    if not use_realsense:
        camera.release()



def start_threading(robot_interface, hand_type, camera_path, retarget_config_path, urdf_path, camera_config_path):
    stop_event = threading.Event()

    use_realsense = False
    if camera_config_path:
        use_realsense = True
        camera = RealsenseInterface.from_yaml(camera_config_path)
        camera.start(resolution=(640, 480), fps=30, align="color")
    elif camera_path is None:
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(camera_path)



    
    # frame_queue = queue.Queue(maxsize=10)
    frame_queue = queue.Queue(maxsize=1)
    producer_process = threading.Thread(
        target=produce_frame, args=(frame_queue, stop_event, camera, use_realsense)
    )
    consumer_process = threading.Thread(
        target=retargeting, args=(frame_queue, stop_event, camera, hand_type, robot_interface, retarget_config_path, urdf_path)
    )


    producer_process.start()
    consumer_process.start()

    def handle_sigint(signum, frame):
        print("Shutting down...")
        stop_event.set()
        if robot_interface:
            robot_interface.stop()
        
    signal.signal(signal.SIGINT, handle_sigint)
    # Needs to be in main thread
    if robot_interface:
        robot_interface.start_loop()


    producer_process.join()
    consumer_process.join()


def main():
    """
    TODO
    """
    HAND_TYPE = Hand.RIGHT

    RETARGET_ROOT = Path(__file__).parent / "dex_retarget"
    RETARGET_CONFIG_PATH = RETARGET_ROOT / "src" / "dex_retargeting" / "configs" / "teleop" / f"tesollo_hand_{HAND_TYPE.value.lower()}_dexpilot.yml"
    RETARGET_ROBOT_URDF_DIR =  RETARGET_ROOT / "assets" / "robots" / "hands"
    CONFIG_DIR = Path(__file__).resolve().parents[0] / "robot_motion_interface" / "config"
    CONFIG_PATH = CONFIG_DIR / "isaacsim_config.yaml"

    
    CAMERA_CONFIG_PATH = Path(__file__).parent / "sensor_interface_py" / "src" / "sensor_interface" / "camera" / "config" / "realsense_config.yaml"
    # CAMERA_CONFIG_PATH = None  # None for integrated camera

    CAMERA_PATH = None # None for realsense, 0 for integrated camera

    # robot_interface = None
    robot_interface = IsaacsimInterface.from_yaml(CONFIG_PATH)
    

    start_threading(robot_interface, HAND_TYPE, CAMERA_PATH, RETARGET_CONFIG_PATH, RETARGET_ROBOT_URDF_DIR, CAMERA_CONFIG_PATH)




if __name__ == "__main__":
    main()