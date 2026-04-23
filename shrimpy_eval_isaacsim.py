"""
Usage:
(robodiff)$ python shrimpy_eval_isaacsim.py -i <ckpt_path> 

"""

# %%


from shrimpy_eval_isaacsim import _get_obs, EE_FRAME, GRIPPER_JOINT_NAMES
from robot_motion_interface.isaacsim.isaacsim_interface import IsaacsimInterface

from scipy.spatial.transform import Rotation                                                                                                
import numpy as np        
from pathlib import Path
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
# import pathlib
# import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st
from diffusion_policy.real_world.real_env import RealEnv
# from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
# from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform


OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
# @click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.0.204")
# @click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
# @click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
# @click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
# @click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=30, type=float, help="Control frequency in Hz.")  # Originally 10
# @click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")

def main(input, output,
    steps_per_inference, max_duration,
    frequency):

    
    # load checkpoint
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # hacks for method-specific setup.
    action_offset = 0

    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 16 # DDIM inference iterations
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)

    # setup experiment
    dt = 1/frequency

    # obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)


    ############ ISAACSIM SETUP ############
       
    CONFIG_PATH = Path(__file__).parent / "robot_motion_interface" / "config" / "isaacsim_config.yaml"
    robot_interface = IsaacsimInterface.from_yaml(CONFIG_PATH)
    # TODO: START loop and start below in own thread
    ################################################


    obs = _get_obs(robot_interface)
    with torch.no_grad():
        policy.reset()

        # Format observation
        # TODO: REPLACE get_real_obs_dict
        obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta)
        obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
        
        result = policy.predict_action(obs_dict)
        action = result['action'][0].detach().to('cpu').numpy()
        assert action.shape[-1] == 2
        del result

    print('Ready!')
    while True:

        # ========== policy control loop ==============
        try:
            # start episode
            policy.reset()
            start_delay = 1.0
            eval_t_start = time.time() + start_delay
            t_start = time.monotonic() + start_delay

            # env.start_episode(eval_t_start)

            # wait for 1/30 sec to get the closest frame actually
            # reduces overall latency
            frame_latency = 1/30
            # precise_wait(eval_t_start - frame_latency, time_func=time.time)

            print("Started!")
            iter_idx = 0
            while True:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                # get obs
                print('get_obs')
                obs = _get_obs(robot_interface)

                # obs_timestamps = obs['timestamp']
                # print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                # run inference
                with torch.no_grad():
                    s = time.time()
                    # TODO: REPLACE get_real_obs_dict and add observatio history
                    obs_dict_np = get_real_obs_dict(env_obs=obs, shape_meta=cfg.task.shape_meta)
                    obs_dict = dict_apply(obs_dict_np,  lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                    result = policy.predict_action(obs_dict)
                    # this action starts from the first obs step
                    action = result['action'][0].detach().to('cpu').numpy()
                    print('Inference latency:', time.time() - s)
                

                for i in range(action.shape[0]):                      


                    # action shape: (18,)                                                                                                                     
                    ee_pos        = action[i][:3]          # (3,)
                    ee_rpy        = action[i][3:6]         # (3,) roll, pitch, yaw                                                                                 
                    gripper_qpos  = action[i][6:]          # (12,)                                                                                                 
                                                                                                                                                                
                    ee_quat = Rotation.from_euler('xyz', ee_rpy).as_quat()  # (4,) xyzw                                                                         
                    ee_pose = np.concatenate([ee_pos, ee_quat])              # (7,)
                                                                                                                                                                
                    # Then send to the robot:                               
                    robot_interface.set_cartesian_pose([ee_pose], [EE_FRAME])                                                                                   
                    robot_interface.set_joint_positions(gripper_qpos, GRIPPER_JOINT_NAMES)

                    time.sleep(dt) # TODO: DON'T SLEEP, instead schedule with dt

               
                print(f"Submitted {action.shape[0]} steps of actions.")
                
                iter_idx += steps_per_inference

        except KeyboardInterrupt:
            print("Interrupted!")
            # stop robot.
            # env.end_episode()
        
        print("Stopped.")



# %%
if __name__ == '__main__':
    main()
