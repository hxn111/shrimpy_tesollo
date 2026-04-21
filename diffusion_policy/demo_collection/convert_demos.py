"""
convert_demos.py

Converts per-episode .npz files recorded by collect_demos_isaacsim.py into
the padded sequence format that IsaacSimLowdimDataset (and the ReplayBuffer
under it) expects.

Output files written to --output_dir:
  observations_seq.npy  shape (N, T_max, obs_dim)   float32
  actions_seq.npy       shape (N, T_max, action_dim) float32
  existence_mask.npy    shape (N, T_max)              bool

Usage:
  python diffusion_policy/demo_collection/convert_demos.py \\
      --input_dir  data/isaacsim_demos \\
      --output_dir data/isaacsim_demos_converted

Why a separate conversion step
  The raw episode files are independent so a crash during collection never
  corrupts earlier episodes.  This script stitches them into the padded-array
  format that ReplayBuffer / SequenceSampler expects, making it straightforward
  to inspect stats and reproduce the exact training split.
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Directory with episode_XXXX.npz files")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write observations_seq / actions_seq / existence_mask")
    parser.add_argument("--min_episode_len", type=int, default=10,
                        help="Discard episodes shorter than this many steps")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_files = sorted(input_dir.glob("episode_*.npz"))
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.npz files found in {input_dir}")

    episodes = []
    for f in episode_files:
        data = np.load(f)
        obs = data["obs"]       # (T, obs_dim)
        actions = data["actions"]  # (T, action_dim)
        if len(obs) < args.min_episode_len:
            print(f"  skip {f.name} ({len(obs)} steps < min {args.min_episode_len})")
            continue
        episodes.append((obs, actions))
        print(f"  loaded {f.name}  T={len(obs)}")

    if not episodes:
        raise ValueError("No episodes passed the minimum length filter.")

    T_max = max(len(o) for o, _ in episodes)
    obs_dim = episodes[0][0].shape[1]
    action_dim = episodes[0][1].shape[1]
    N = len(episodes)

    print(f"\n{N} episodes  |  T_max={T_max}  obs_dim={obs_dim}  action_dim={action_dim}")

    obs_seq = np.zeros((N, T_max, obs_dim), dtype=np.float32)
    act_seq = np.zeros((N, T_max, action_dim), dtype=np.float32)
    mask = np.zeros((N, T_max), dtype=bool)

    for i, (obs, actions) in enumerate(episodes):
        T = len(obs)
        obs_seq[i, :T] = obs
        act_seq[i, :T] = actions
        mask[i, :T] = True

    np.save(output_dir / "observations_seq.npy", obs_seq)
    np.save(output_dir / "actions_seq.npy", act_seq)
    np.save(output_dir / "existence_mask.npy", mask)

    print(f"\nSaved to {output_dir}")
    print(f"  observations_seq.npy  {obs_seq.shape}")
    print(f"  actions_seq.npy       {act_seq.shape}")
    print(f"  existence_mask.npy    {mask.shape}")


if __name__ == "__main__":
    main()
