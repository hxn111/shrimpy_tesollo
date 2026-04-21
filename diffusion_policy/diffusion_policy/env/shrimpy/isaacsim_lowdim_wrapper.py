from typing import List, Dict, Optional
import numpy as np
import gym
from gym.spaces import Box
from robomimic.envs.env_robosuite import EnvRobosuite
from scipy.spatial.transform import Rotation
from robot_motion_interface.isaacsim.isaacsim_interface import IsaacsimInterface


################################################################# TODO: IMPLEMENT THIS
class IsaacsimLowdimWrapper():
    def __init__(self, 
        # env: EnvRobosuite,
        obs_keys: List[str]=[
            'object', 
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_gripper_qpos'],
        init_state: Optional[np.ndarray]=None,
        render_hw=(256,256),
        render_camera_name='agentview'
        ):

        # TODO: DON'T HARD CODE THIS
        ACTION_DIM = 18
        ROBOT_INTEFACE_CONFIG_DIR = "/workspace/robot_motion_interface/config/isaacsim_config.yaml" 


        self.robot_interface = IsaacsimInterface.from_yaml(ROBOT_INTEFACE_CONFIG_DIR)
        


        # self.env = env
        self.obs_keys = obs_keys
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.seed_state_map = dict()
        self._seed = None
        
        # setup spaces
        low = np.full(ACTION_DIM, fill_value=-1)
        high = np.full(ACTION_DIM, fill_value=1)
        self.action_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype
        )
        obs_example = self.get_observation()
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype
        )

        # TODO: DON'T HARD CODE
        self.EE_FRAME = 'right_delto_offset_link'
        self.GRIPPER_JOINT_NAMES = ['right_F1M1','right_F1M2','right_F1M3', 'right_F1M4','right_F2M1','right_F2M2',
                'right_F2M3','right_F2M4','right_F3M1', 'right_F3M2','right_F3M3','right_F3M4']

        self.robot_interface.start_loop() # TODO: SEE IF THIS NEEDS TO BE THREADED

    def get_observation(self):
        # TODO: UN-HARDCODE
        # ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']

        robot0_eef_pose = self.robot_interface.cartesian_pose([self.EE_FRAME ])[0][0]
        robot0_eef_pos = robot0_eef_pose[:3]
        robot0_eef_quat = robot0_eef_pose[3:]

        
        names = self.robot_interface.joint_names()                 
        pos = self.robot_interface.joint_state()[0::2]                                                                             
        robot0_gripper_qpos = pos[[names.index(n) for n in self.GRIPPER_JOINT_NAMES]]

        
        obs = np.concatenate([robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos], axis=0)
        
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self):
        if self.init_state is not None:
            # always reset to the same state
            # to be compatible with gym
            self.robot_interface.home()
       
        obs = self.get_observation()
        return obs
    
    def step(self, action):

        # 3 for ee pos (x, y, z), 3 for ee rotation (roll, pitch, yaw), 12 for gripper qpos (4 joints each finger)
        ee_pos = action[:3]
        ee_roll_pitch_yaw = action[3:6]
        ee_quat = Rotation.from_euler('xyz', ee_roll_pitch_yaw).as_quat()

        ee_pose = np.concatenate([ee_pos, ee_quat])
        gripper_qpos = action[6:]
        
        self.robot_interface.set_cartesian_pose([ee_pose], [self.EE_FRAME])
        self.robot_interface.set_joint_positions(gripper_qpos, self.GRIPPER_JOINT_NAMES)

        
        obs = self.get_observation()

        # TODO: FIGURE OUT REWARD
        reward = 0
        done = False
        info = None
        return obs, reward, done, info

    
    def render(self, mode='rgb_array'):
        return None


def test():
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    from matplotlib import pyplot as plt

    dataset_path = '/home/cchi/dev/diffusion_policy/data/robomimic/datasets/square/ph/low_dim.hdf5'
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)


    wrapper = IsaacsimLowdimWrapper(
        # env=env,
        obs_keys=[
            'object', 
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_gripper_qpos'
        ]
    )

    states = list()
    for _ in range(2):
        wrapper.seed(0)
        wrapper.reset()
        states.append(wrapper.env.get_state()['states'])
    assert np.allclose(states[0], states[1])

    img = wrapper.render()
    plt.imshow(img)
    # wrapper.seed()
    # states.append(wrapper.env.get_state()['states'])


if __name__ == "__main__":
    test()