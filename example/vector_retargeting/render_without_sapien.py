"""
Alternative renderer that does NOT require SAPIEN or GPU.
Uses matplotlib to visualize joint angles as an animated video.
Usage:
    python render_without_sapien.py --pickle-path your_file.pkl --output-video-path out.mp4
"""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tyro
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import tqdm


# ── finger groupings (adjust if your robot has different joint names) ──────────
FINGER_KEYWORDS = {
    "Thumb":  ["thumb", "th"],
    "Index":  ["index",  "ff", "fore"],
    "Middle": ["middle", "mf"],
    "Ring":   ["ring",   "rf"],
    "Pinky":  ["pinky",  "little", "lf", "sf"],
}
FINGER_COLORS = {
    "Thumb":  "#e74c3c",
    "Index":  "#3498db",
    "Middle": "#2ecc71",
    "Ring":   "#f39c12",
    "Pinky":  "#9b59b6",
    "Other":  "#95a5a6",
}


def assign_finger(name: str) -> str:
    name_lower = name.lower()
    for finger, keywords in FINGER_KEYWORDS.items():
        if any(k in name_lower for k in keywords):
            return finger
    return "Other"


def render_frame(
    qpos: np.ndarray,
    joint_names: list,
    frame_idx: int,
    total_frames: int,
    fig,
    axes,
) -> np.ndarray:
    """Draw one frame and return it as an RGB numpy array."""
    for ax in axes:
        ax.cla()

    ax_bar, ax_prog = axes

    # ── assign joints to fingers ───────────────────────────────────────────
    fingers = [assign_finger(n) for n in joint_names]
    colors  = [FINGER_COLORS[f] for f in fingers]

    # ── bar chart of joint angles ──────────────────────────────────────────
    x = np.arange(len(joint_names))
    bars = ax_bar.bar(x, np.degrees(qpos), color=colors, edgecolor="white",
                      linewidth=0.4, zorder=3)
    ax_bar.axhline(0, color="white", linewidth=0.5, linestyle="--", alpha=0.4)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        [n.replace("_", "\n") for n in joint_names],
        fontsize=6, rotation=0, color="white"
    )
    ax_bar.set_ylabel("Joint angle (deg)", color="white", fontsize=9)
    ax_bar.set_title(
        f"Robot Hand Joint Angles — Frame {frame_idx + 1}/{total_frames}",
        color="white", fontsize=11, pad=8
    )
    ax_bar.set_facecolor("#1a1a2e")
    ax_bar.tick_params(colors="white")
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#444")
    ax_bar.grid(axis="y", color="#333", linewidth=0.5, zorder=0)
    ax_bar.set_ylim(-100, 100)

    # legend
    seen = set()
    patches = []
    for f, c in zip(fingers, colors):
        if f not in seen:
            patches.append(mpatches.Patch(color=c, label=f))
            seen.add(f)
    ax_bar.legend(handles=patches, loc="upper right", fontsize=7,
                  facecolor="#16213e", edgecolor="#444", labelcolor="white")

    # ── progress bar ──────────────────────────────────────────────────────
    progress = (frame_idx + 1) / total_frames
    ax_prog.barh([0], [progress], color="#e74c3c", height=0.6)
    ax_prog.barh([0], [1.0],      color="#333",    height=0.6)
    ax_prog.set_xlim(0, 1)
    ax_prog.set_ylim(-0.5, 0.5)
    ax_prog.axis("off")
    ax_prog.text(0.5, 0, f"{int(progress * 100)}%",
                 ha="center", va="center", color="white", fontsize=9,
                 transform=ax_prog.transAxes)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return buf.reshape(h, w, 4)[..., :3]  # RGBA → RGB


def main(
    pickle_path: str,
    output_video_path: Optional[str] = "output_joint_viz.mp4",
    fps: int = 30,
):
    """
    Visualize retargeted robot hand joint angles without SAPIEN/GPU.

    Args:
        pickle_path: Path to the .pickle file from detect_from_video.py
        output_video_path: Where to save the output .mp4
        fps: Frames per second for the output video
    """
    print(f"Loading pickle: {pickle_path}")
    pickle_data  = np.load(pickle_path, allow_pickle=True)
    meta_data    = pickle_data["meta_data"]
    data         = pickle_data["data"]
    joint_names  = list(meta_data["joint_names"])

    print(f"  Joints : {len(joint_names)}")
    print(f"  Frames : {len(data)}")
    print(f"  Names  : {joint_names}")

    # ── figure setup ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 5), facecolor="#16213e")
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[10, 1], hspace=0.05)
    ax_bar  = fig.add_subplot(gs[0])
    ax_prog = fig.add_subplot(gs[1])
    axes    = [ax_bar, ax_prog]

    # ── video writer ──────────────────────────────────────────────────────
    # render one frame to get the size
    sample_frame = render_frame(
        np.array(data[0]), joint_names, 0, len(data), fig, axes
    )
    h, w = sample_frame.shape[:2]

    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )

    # ── render all frames ─────────────────────────────────────────────────
    for i, qpos in enumerate(tqdm.tqdm(data, desc="Rendering")):
        frame_rgb = render_frame(
            np.array(qpos), joint_names, i, len(data), fig, axes
        )
        writer.write(frame_rgb[..., ::-1])  # RGB → BGR for cv2

    writer.release()
    plt.close(fig)
    print(f"\n✓ Video saved to: {output_video_path}")


if __name__ == "__main__":
    tyro.cli(main)
