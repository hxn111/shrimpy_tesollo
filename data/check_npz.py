import numpy as np
data = np.load("./data/isaacsim_demos/episode_0003.npz")
print(data)
print(data['obs'])
print(data['actions'])