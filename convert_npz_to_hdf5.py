"""
convert_npz_to_hdf5.py — convert collected .npz demos to robomimic HDF5 format

Usage:
    python convert_npz_to_hdf5.py
    python convert_npz_to_hdf5.py --input data/task2 --output data/converted/task2.hdf5

Each episode_XXXX.npz must contain:
    obs     : (T, 33)  [object(14),eef_pos(3), eef_quat(4), gripper_qpos(12)]
    actions : (T, 18)  [eef_pos(3), eef_rpy(3), gripper_qpos(12)]

Output HDF5 structure (matches RobomimicReplayLowdimDataset):
    data/demo_N/obs/object               (T, 14)
    data/demo_N/obs/robot0_eef_pos       (T, 3)
    data/demo_N/obs/robot0_eef_quat      (T, 4)
    data/demo_N/obs/robot0_gripper_qpos  (T, 12)
    data/demo_N/actions                  (T, 18)


NOTE: If you uncomment the "NO OBJECTS" section below, your episode_XXXX.npz must contain:
        obs     : (T, 19)  [eef_pos(3), eef_quat(4), gripper_qpos(12)]
        actions : (T, 18)  [eef_pos(3), eef_rpy(3), gripper_qpos(12)]

    And the HDF5 structure  will be (matches RobomimicReplayLowdimDataset):
        data/demo_N/obs/robot0_eef_pos       (T, 3)
        data/demo_N/obs/robot0_eef_quat      (T, 4)
        data/demo_N/obs/robot0_gripper_qpos  (T, 12)
        data/demo_N/actions                  (T, 18)
"""

import argparse
import numpy as np
import h5py
from pathlib import Path


OBS_SPLITS = [
    ('object',               0,  14),
    ('robot0_eef_pos',      14,  17),
    ('robot0_eef_quat',     17,  21),
    ('robot0_gripper_qpos', 21, 33),
]

############ UNCOMMENT IF NO OBJECTS: ############

# OBS_SPLITS = [
#     ('robot0_eef_pos',      0,  3),
#     ('robot0_eef_quat',     3,  7),
#     ('robot0_gripper_qpos', 7, 19),
# ]

################################################

def convert(input_dir: Path, output_path: Path):
    npz_files = sorted(input_dir.glob("episode_*.npz"))
    if not npz_files:
        print(f"No episode_*.npz files found in {input_dir}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        grp = f.create_group('data')

        for i, npz_path in enumerate(npz_files):
            data = np.load(npz_path)
            obs = data['obs']       # (T, 33)
            actions = data['actions']  # (T, 18)

            expected_obs_dim = OBS_SPLITS[-1][2]
            assert obs.shape[1] == expected_obs_dim, f"{npz_path}: expected obs dim {expected_obs_dim}, got {obs.shape[1]}"

            demo = grp.create_group(f'demo_{i}')
            obs_grp = demo.create_group('obs')

            for key, start, end in OBS_SPLITS:
                obs_grp.create_dataset(key, data=obs[:, start:end])

            demo.create_dataset('actions', data=actions)
            print(f"  demo_{i}: {len(obs)} steps  ← {npz_path.name}")

        print(f"\nWrote {len(npz_files)} episodes → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  default='data/isaacsim_demos',
                        help='Directory containing episode_*.npz files')
    parser.add_argument('--output', default='data/isaacsim_demos_converted/low_dim.hdf5',
                        help='Output HDF5 path')
    args = parser.parse_args()

    convert(Path(args.input), Path(args.output))


if __name__ == '__main__':
    main()
