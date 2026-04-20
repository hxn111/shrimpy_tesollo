"""
Renders retargeted robot hand using PyBullet DIRECT mode.
- No SAPIEN, no GPU, no display required — pure CPU software rendering.
- Works on macOS without Vulkan SDK.

Usage:
    python render_pybullet.py \
        --pickle-path data/tesollo_joints.pkl \
        --output-video-path output_3d.mp4
"""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tqdm
import tyro

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    raise ImportError(
        "PyBullet not found. Install with:  pip install pybullet"
    )


def render_robot_pybullet(
    urdf_path: str,
    joint_names_retargeting: list,
    data: list,
    output_video_path: str,
    fps: int = 30,
    width: int = 640,
    height: int = 640,
):
    # ── Connect in DIRECT mode: CPU offscreen, no display/GPU needed ──────
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -9.8, physicsClientId=client)

    # optional ground plane
    p.loadURDF("plane.urdf", physicsClientId=client)

    # ── Load robot URDF ────────────────────────────────────────────────────
    urdf_path = str(urdf_path)
    print(f"Loading URDF: {urdf_path}")
    robot_id = p.loadURDF(
        urdf_path,
        basePosition=[0, 0, 0.1],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True,
        physicsClientId=client,
    )

    # ── Map retargeting joint names → pybullet joint indices ──────────────
    num_joints = p.getNumJoints(robot_id, physicsClientId=client)
    bullet_name_to_idx = {}
    print("\nPyBullet joints:")
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i, physicsClientId=client)
        jname = info[1].decode("utf-8")
        jtype = info[2]
        if jtype != p.JOINT_FIXED:
            bullet_name_to_idx[jname] = i
            print(f"  [{i}] {jname}")

    # Build mapping: retargeting index → bullet joint index
    retargeting_to_bullet = []
    for rname in joint_names_retargeting:
        if rname in bullet_name_to_idx:
            retargeting_to_bullet.append(bullet_name_to_idx[rname])
        else:
            # try fuzzy match (lowercase, strip underscores)
            matched = None
            for bname, bidx in bullet_name_to_idx.items():
                if rname.lower().replace("_", "") == bname.lower().replace("_", ""):
                    matched = bidx
                    break
            retargeting_to_bullet.append(matched)
            if matched is None:
                print(f"  WARNING: joint '{rname}' not found in URDF")

    # ── Camera setup ──────────────────────────────────────────────────────
    view_matrix = p.computeViewMatrix(
        cameraEyePosition    = [0.3, -0.3, 0.35],
        cameraTargetPosition = [0.0,  0.0, 0.1],
        cameraUpVector       = [0,    0,   1],
        physicsClientId=client,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=width / height, nearVal=0.01, farVal=10.0,
        physicsClientId=client,
    )

    # ── Video writer ───────────────────────────────────────────────────────
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )

    # ── Render frames ──────────────────────────────────────────────────────
    for qpos in tqdm.tqdm(data, desc="Rendering 3D"):
        qpos = np.array(qpos)

        for ret_idx, bullet_idx in enumerate(retargeting_to_bullet):
            if bullet_idx is not None and ret_idx < len(qpos):
                p.resetJointState(
                    robot_id, bullet_idx, qpos[ret_idx],
                    physicsClientId=client,
                )

        p.stepSimulation(physicsClientId=client)

        _, _, rgb, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,   # CPU software renderer
            physicsClientId=client,
        )

        frame = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        writer.write(frame_bgr)

    writer.release()
    p.disconnect(client)
    print(f"\n✓ Saved to: {output_video_path}")


def main(
    pickle_path: str,
    output_video_path: Optional[str] = "output_3d.mp4",
    fps: int = 30,
    width: int = 640,
    height: int = 640,
):
    """
    Render retargeted robot hand in 3D using PyBullet (CPU, no GPU needed).

    Args:
        pickle_path: Path to the .pkl file from detect_from_video.py
        output_video_path: Output .mp4 path
        fps: Frames per second
        width: Video width in pixels
        height: Video height in pixels
    """
    from dex_retargeting.retargeting_config import RetargetingConfig

    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))

    pickle_data = np.load(pickle_path, allow_pickle=True)
    meta_data   = pickle_data["meta_data"]
    data        = pickle_data["data"]
    joint_names = list(meta_data["joint_names"])

    # Load config to find URDF path
    config_path = meta_data["config_path"]
    config      = RetargetingConfig.load_from_file(config_path)
    urdf_path   = config.urdf_path

    # If a _glb.urdf exists, prefer the plain one for pybullet
    plain_urdf = str(urdf_path).replace("_glb.urdf", ".urdf")
    if Path(plain_urdf).exists():
        urdf_path = plain_urdf

    print(f"URDF: {urdf_path}")
    print(f"Joints: {joint_names}")
    print(f"Frames: {len(data)}")

    render_robot_pybullet(
        urdf_path=urdf_path,
        joint_names_retargeting=joint_names,
        data=data,
        output_video_path=output_video_path,
        fps=fps,
        width=width,
        height=height,
    )


if __name__ == "__main__":
    tyro.cli(main)
