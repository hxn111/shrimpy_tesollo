"""
Shows data inside episode.

Running:
python3 data/check_npz.py
"""
import numpy as np
data = np.load("./data/task2/episode_0002.npz")
print(data)
print(data['obs'])
print(data['actions'])