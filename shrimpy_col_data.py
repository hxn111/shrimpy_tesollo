"""
collect_demos_isaacsim.py  —  isaacsim_shrimpy.py + demo recording

Recording is automatic:
  hand visible       → episode is being recorded
  hand absent 0.5 s  → episode saved to data/isaacsim_demos/episode_XXXX.npz
  Ctrl-C             → saves in-progress episode and exits

Per-step recording:
  obs    : 19-dim  [eef_pos(3), eef_quat(4), gripper_qpos(12)]  (state BEFORE command)
  action : 18-dim  [eef_pos(3), eef_rpy(3),  gripper_qpos(12)]  (command just sent)
"""

from robot_motion_interface.isaacsim.isaacsim_object_interface import IsaacsimObjectInterface, Object

from robot_motion_interface.interface import Interface
from sensor_interface.camera.realsense_interface import RealsenseInterface

from dex_retarget.example.vector_retargeting.single_hand_detector import SingleHandDetector
from dex_retargeting.retargeting_config import RetargetingConfig

import time
import threading
import queue
import signal
from pathlib import Path

from OneEuroFilter import OneEuroFilter
from scipy.spatial.transform import Rotation
import numpy as np
import cv2
import pygame
from queue import Empty
from loguru import logger
from enum import Enum


HAND_ABSENT_TIMEOUT = 60  # frames (~0.5 s at 30 fps) before episode is saved
DEMO_SAVE_DIR = Path(__file__).parent / "data" / "isaacsim_demos"

EE_FRAME = 'right_delto_offset_link'
GRIPPER_JOINT_NAMES = [
    'right_F1M1', 'right_F1M2', 'right_F1M3', 'right_F1M4',
    'right_F2M1', 'right_F2M2', 'right_F2M3', 'right_F2M4',
    'right_F3M1', 'right_F3M2', 'right_F3M3', 'right_F3M4',
]


class Hand(Enum):
    RIGHT = "Right"
    LEFT = "Left"


# ──────────────────────────────────────────────────────────
# Recording helpers (inline — no separate class needed)
# ──────────────────────────────────────────────────────────

def _episode_count(save_dir: Path) -> int:
    return len(list(save_dir.glob("episode_*.npz")))


def _save_episode(obs_buf: list, act_buf: list, save_dir: Path) -> int:
    print("obs:", obs_buf)
    print("act:", act_buf)
    if not obs_buf:
        return _episode_count(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    idx = _episode_count(save_dir)
    path = save_dir / f"episode_{idx:04d}.npz"
    np.savez(path,
             obs=np.array(obs_buf, dtype=np.float32),      # (T, 33)
             actions=np.array(act_buf, dtype=np.float32))  # (T, 18)
    print(f"[demo] saved {len(obs_buf)} steps → {path}")
    return idx + 1


def _get_obs(robot_interface: Interface) -> np.ndarray:
    """Returns 33-dim obs: [object(14), eef_pos(3), eef_quat(4), gripper_qpos(12)]."""

    cube_0_pose = robot_interface.get_object_pose("cube")
    cube_1_pose = robot_interface.get_object_pose("cube_1")
    objects = np.concatenate([cube_0_pose, cube_1_pose])

    eef_pose = robot_interface.cartesian_pose([EE_FRAME])[0][0]
    eef_pos  = eef_pose[:3]
    eef_quat = eef_pose[3:]
    names = robot_interface.joint_names()
    pos = robot_interface.joint_state()[:len(names)]   
    gripper_qpos = pos[[names.index(n) for n in GRIPPER_JOINT_NAMES]]
    return np.concatenate([objects, eef_pos, eef_quat, gripper_qpos]).astype(np.float32)


# ──────────────────────────────────────────────────────────
# Unchanged from isaacsim_shrimpy.py
# ──────────────────────────────────────────────────────────

# Height-sensitive XY scale thresholds
HEIGHT_FINE_Z   = 0.20   # at or below → fine-control mode (smallest XY scale)
HEIGHT_COARSE_Z = 0.40   # at or above → coarse mode (original XY scale)
X_SCALE_FINE,   X_SCALE_COARSE = 0.75, 1.5
Y_SCALE_FINE,   Y_SCALE_COARSE = 1.00, 2.0

def scale_and_set_poses(hand: Hand, unscaled_wrist_pose: np.ndarray,
                        gripper_pos: np.ndarray, robot_interface: Interface,
                        wrist_filter):
    scaled_wrist_pose = unscaled_wrist_pose.copy()
    if hand == Hand.RIGHT:
        # scaled_wrist_pose[0] = -scaled_wrist_pose[0] * 1.5 + 0.1                                                                                                                   
        # scaled_wrist_pose[1] = -scaled_wrist_pose[1] * 2
         
        # Z fixed at 2× so the arm can reach any height; use it to drive XY gain.
        scaled_wrist_pose[2] *= 2
        blend = float(np.clip(
            (scaled_wrist_pose[2] - HEIGHT_FINE_Z) / (HEIGHT_COARSE_Z - HEIGHT_FINE_Z),
            0.0, 1.0))
        x_scale = X_SCALE_FINE + blend * (X_SCALE_COARSE - X_SCALE_FINE)
        y_scale = Y_SCALE_FINE + blend * (Y_SCALE_COARSE - Y_SCALE_FINE)
        scaled_wrist_pose[0] = -unscaled_wrist_pose[0] * x_scale + 0.1
        scaled_wrist_pose[1] = -unscaled_wrist_pose[1] * y_scale

         
        filtered_wrist_xyz = np.array([f(v) for f, v in zip(wrist_filter, scaled_wrist_pose[:3])])
        scaled_wrist_pose[:3] = filtered_wrist_xyz

        scaled_wrist_pose[2] = max(scaled_wrist_pose[2], 1.05)  # Prevent collision with table

        # "LOCK" x an y position to cube when near
        # LOCK_DISTANCE = 0.05
        # target_pose = robot_interface.get_object_pose("cube") # TODO: PASS THIS INTO FUNCTION
        # target_x = target_pose[0]
        # target_y = target_pose[1]

        # if (np.linalg.norm(np.abs(scaled_wrist_pose[:2] - target_pose[:2])) <= LOCK_DISTANCE):
        #     scaled_wrist_pose[0] = target_x
        #     scaled_wrist_pose[1] = target_y

        # Fix gripper pose to be parallel
        # gripper_pos[0] = 0.0
        # gripper_pos[4] = 0.0
        # gripper_pos[8] = 0.0

        # print("GRIPPER[3] POS:", gripper_pos[3])

        # if  gripper_pos[3] < 0.9:
        #     gripper_pos[2] = 0.7
        #     gripper_pos[6] = 0.7
        #     gripper_pos[10] = 0.7


        if robot_interface:
            robot_interface.set_cartesian_pose([scaled_wrist_pose], ['right_delto_offset_link'])
            robot_interface.set_joint_positions(
                gripper_pos,
                ['right_F1M1', 'right_F1M2', 'right_F1M3', 'right_F1M4',
                 'right_F2M1', 'right_F2M2', 'right_F2M3', 'right_F2M4',
                 'right_F3M1', 'right_F3M2', 'right_F3M3', 'right_F3M4'])
    return scaled_wrist_pose


def joint_pos_to_robot_pos(joint_pos: np.ndarray, retargeter) -> np.ndarray:
    if joint_pos is None:
        return None
    indices = retargeter.optimizer.target_link_human_indices
    ref_value = joint_pos[indices[1], :] - joint_pos[indices[0], :]
    return retargeter.retarget(ref_value)


def produce_frame(frame_queue: queue.Queue, stop_event, camera, use_realsense: bool):
    while not stop_event.is_set():
        if not use_realsense:
            success, bgr = camera.read()
            if not success:
                continue
            color = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = None
        else:
            try:
                frame = camera.latest()
                color, depth = frame.color, frame.depth
            except RuntimeError:
                # print("WARNING: No frame from Realsense.")
                continue
        time.sleep(1 / 30.0)
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
        frame_queue.put((color, depth))
    if not use_realsense:
        camera.release()


# ──────────────────────────────────────────────────────────
# Retargeting + recording (extends isaacsim_shrimpy version)
# ──────────────────────────────────────────────────────────

def retargeting(frame_queue: queue.Queue, stop_event, camera, hand: Hand,
                robot_interface: Interface, config_path, urdf_path):
    
    if robot_interface:
        # Initiate objects
        deadline = time.time() + 120.0
        while not robot_interface.check_loop():
            if time.time() > deadline:
                raise TimeoutError("IsaacSim did not start within timeout")
            time.sleep(0.1)
        cube_0 = Object(handle="cube", pose=[0.1, 0.1, 0.95, 0,0,0,1])
        cube_1 = Object(handle="cube_1", pose=[0.1, -0.1, 0.95, 0,0,0,1])
        robot_interface.place_objects([cube_0, cube_1])

    detector = SingleHandDetector(hand_type=hand.value, selfie=False)
    RetargetingConfig.set_default_urdf_dir(str(urdf_path))
    retargeter = RetargetingConfig.load_from_file(config_path).build()
    wrist_filters = [OneEuroFilter(freq=30, mincutoff=0.6, beta=0.3) for _ in range(3)]

    pygame.init()
    screen = pygame.display.set_mode((1280, 480))
    pygame.display.set_caption("Teleoperation — show hand to record, hide to save episode")
    font = pygame.font.SysFont(None, 36)

    obs_buf, act_buf = [], []
    recording = False
    frames_without_hand = 0
    episode_count = _episode_count(DEMO_SAVE_DIR)

    while not stop_event.is_set():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                episode_count = _save_episode(obs_buf, act_buf, DEMO_SAVE_DIR)
                stop_event.set()

        try:
            rgb, depth = frame_queue.get(timeout=5)
        except Empty:
            logger.error("No frame from camera in 5s.")
            continue

        _, joint_pos, keypoint_2d, _, _ = detector.detect(rgb)

        if keypoint_2d is not None:
            frames_without_hand = 0
            if not recording:
                recording = True
                print(f"[demo] recording episode {episode_count}...")

            if depth is None:
                xy = keypoint_2d[0]
                xyz = np.array([(xy.x - 0.5) * 0.4, (xy.y - 0.5) * 0.4, 0.6])
            else:
                H, W = depth.shape
                px = SingleHandDetector.parse_keypoint_2d(keypoint_2d, (H, W))
                wy, wx = px[0].astype(int)
                mask = np.zeros((H, W), dtype=np.int32)
                mask[wx, wy] = 1
                pts, ids = camera.depth_to_pointcloud(depth, mask)
                cloud = pts[ids == 1][0]
                if len(cloud) == 0:
                    continue
                xyz = cloud[0]

            # wrist_pose = np.concatenate([xyz, [0.707, 0.707, 0, 0]])
            wrist_pose = np.concatenate([xyz, [0.707, -0.707, 0, 0]])
            gripper_pos = joint_pos_to_robot_pos(joint_pos, retargeter)

            # Obs = [eef_pos(3), eef_quat(4), gripper_qpos(12)] BEFORE command
            sim_ready = robot_interface.joint_state() is not None
            if sim_ready:
                obs_before = _get_obs(robot_interface)

            scaled_wrist_pose = scale_and_set_poses(hand, wrist_pose, gripper_pos, robot_interface, wrist_filters)

            # Action = [eef_pos(3), eef_rpy(3), gripper_qpos(12)] — matches IsaacsimLowdimWrapper.step()
            if sim_ready:
                ee_rpy = Rotation.from_quat(scaled_wrist_pose[3:]).as_euler('xyz').astype(np.float32)
                action = np.concatenate([scaled_wrist_pose[:3], ee_rpy, gripper_pos]).astype(np.float32)
                obs_buf.append(obs_before)
                act_buf.append(action)

            rgb = detector.draw_skeleton_on_image(rgb, keypoint_2d, style="default")
            if depth is not None:
                depth = detector.draw_skeleton_on_image(depth, keypoint_2d, style="default")

        else:
            if recording:
                frames_without_hand += 1
                if frames_without_hand >= HAND_ABSENT_TIMEOUT:
                    recording = False
                    episode_count = _save_episode(obs_buf, act_buf, DEMO_SAVE_DIR)
                    obs_buf, act_buf = [], []
                    frames_without_hand = 0

                    # Reset objects
                    robot_interface.move_object("cube", [0.1, 0.1, 0.95, 0,0,0,1])
                    robot_interface.move_object("cube_1", [0.1, -0.1, 0.95, 0,0,0,1])

        # Display
        if depth is not None:
            d8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_color = cv2.cvtColor(cv2.applyColorMap(d8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        else:
            depth_color = np.zeros_like(rgb)

        surf = pygame.surfarray.make_surface(np.hstack([rgb, depth_color]).swapaxes(0, 1))
        screen.blit(surf, (0, 0))
        label = f"{'● REC' if recording else '○ IDLE'}  |  episode {episode_count}  |  steps {len(obs_buf)}"
        screen.blit(font.render(label, True, (220, 50, 50) if recording else (180, 180, 180)), (10, 10))
        pygame.display.flip()


# ──────────────────────────────────────────────────────────

def start_threading(robot_interface, hand_type, camera_path,
                    retarget_config_path, urdf_path, camera_config_path):
    stop_event = threading.Event()
    obs_buf_ref = [[]]  # shared reference for SIGINT handler

    use_realsense = bool(camera_config_path)
    if use_realsense:
        camera = RealsenseInterface.from_yaml(camera_config_path)
        camera.start(resolution=(640, 480), fps=30, align="color")
    else:
        camera = cv2.VideoCapture(0 if camera_path is None else camera_path)


    frame_queue = queue.Queue(maxsize=1)



    threading.Thread(target=produce_frame,
                     args=(frame_queue, stop_event, camera, use_realsense),
                     daemon=True).start()
    threading.Thread(target=retargeting,
                     args=(frame_queue, stop_event, camera, hand_type,
                           robot_interface, retarget_config_path, urdf_path),
                     daemon=True).start()

    def handle_sigint(signum, frame):
        print("\nCtrl-C: saving in-progress episode and exiting...")
        stop_event.set()
        if robot_interface:
            robot_interface.stop()

    signal.signal(signal.SIGINT, handle_sigint)
    if robot_interface:
        robot_interface.start_loop()


def main():
    HAND_TYPE = Hand.RIGHT
    RETARGET_ROOT = Path(__file__).parent / "dex_retarget"
    RETARGET_CONFIG_PATH = (
        RETARGET_ROOT / "src" / "dex_retargeting" / "configs" / "teleop"
        / f"tesollo_hand_{HAND_TYPE.value.lower()}_dexpilot.yml"
    )
    RETARGET_URDF_DIR = RETARGET_ROOT / "assets" / "robots" / "hands"
    CONFIG_PATH = Path(__file__).parent / "robot_motion_interface" / "config" / "isaacsim_config.yaml"
    CAMERA_CONFIG_PATH = (
        Path(__file__).parent / "sensor_interface_py" / "src" / "sensor_interface"
        / "camera" / "config" / "realsense_config.yaml"
    )

    robot_interface = IsaacsimObjectInterface.from_yaml(CONFIG_PATH)
    start_threading(robot_interface, HAND_TYPE, None,
                    RETARGET_CONFIG_PATH, RETARGET_URDF_DIR, CAMERA_CONFIG_PATH)


if __name__ == "__main__":
    main()
