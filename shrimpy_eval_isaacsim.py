"""
Usage:
(robodiff)$ python shrimpy_eval_isaacsim.py -i <ckpt_path> 

"""


from shrimpy_col_data import _get_obs, EE_FRAME, GRIPPER_JOINT_NAMES
from robot_motion_interface.isaacsim.isaacsim_object_interface import IsaacsimObjectInterface, Object

from scipy.spatial.transform import Rotation
import threading
import sys
import numpy as np        
from pathlib import Path
import time
from collections import deque
import click
import torch
import dill
import hydra

from omegaconf import OmegaConf

                
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--frequency', '-f', default=30, type=float, help="Control frequency in Hz.")  # Originally 10

def main(input, 
    steps_per_inference,
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
    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 16 # DDIM inference iterations # Use default (8)
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
        # Actions are unnormalized euler angles (up to ±π), so clip_sample would corrupt them
        policy.noise_scheduler.register_to_config(clip_sample=False)
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)

    # setup experiment
    dt = 1/frequency

    # obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)


    ############ ISAACSIM SETUP ############
       
    CONFIG_PATH = Path(__file__).parent / "robot_motion_interface" / "config" / "isaacsim_config.yaml"
      
    sys.argv = sys.argv[:1]  # Click already parsed our args; hide them from Isaac Sim's argparse
    robot_interface = IsaacsimObjectInterface.from_yaml(CONFIG_PATH)

    ################################################


    worker = threading.Thread(target=policy_loop, daemon=True, args=(policy, device, robot_interface, n_obs_steps, steps_per_inference, dt))
    worker.start()
    robot_interface.start_loop()  # blocks on main thread until sim closes


def policy_loop(policy, device, robot_interface, n_obs_steps, steps_per_inference, dt):

    ############ WAIT FOR ISAACSIM TO START ############
    deadline = time.time() + 120.0
    while not robot_interface.check_loop():
         if time.time() > deadline:
             raise TimeoutError("IsaacSim did not start within timeout")
         time.sleep(0.1)
    
    # Initiate objects
    cube_0 = Object(handle="cube", pose=[0.1, 0.1, 0.95, 0,0,0,1])
    cube_1 = Object(handle="cube_1", pose=[0.1, -0.1, 0.95, 0,0,0,1])
    robot_interface.place_objects([cube_0, cube_1])
    
    # Wait for robot state and object poses to be populated
    deadline = time.time() + 30.0
    while (robot_interface.joint_state() is None
           or "cube" not in robot_interface.get_object_poses()
           or "cube_1" not in robot_interface.get_object_poses()):
        if time.time() > deadline:
            raise TimeoutError("Robot state not available within timeout")
        time.sleep(0.1)
    ###################################################


    obs_history = deque(maxlen=n_obs_steps)
    for _ in range(n_obs_steps):
        obs_history.append(_get_obs(robot_interface))  



    print('Ready!')
    while True:

        # ========== policy control loop ==============
        try:
            # Start episode
            policy.reset()

            print("Started!")

            while True:

                # get obs
                print('get_obs')
                obs_history.append(_get_obs(robot_interface))

                # run inference
                with torch.no_grad():
                    s = time.time()

                    obs_seq = np.stack(list(obs_history), axis=0)  # (n_obs_steps, 33)
                    obs_dict = {'obs': torch.from_numpy(obs_seq).unsqueeze(0).to(device)}  # (1, 2, 33)
                    result = policy.predict_action(obs_dict)
                    # This action starts from the first obs step
                    action = result['action'][0].detach().to('cpu').numpy()
                    print('Inference latency:', time.time() - s)
                

                for i in range(action.shape[0]):
                    

                    # Action shape: (18,)
                    ee_pos        = action[i][:3]          # (3,)
                    ee_rpy        = action[i][3:6]         # (3,) roll, pitch, yaw
                    gripper_qpos  = action[i][6:]          # (12,)

                    ee_quat = Rotation.from_euler('xyz', ee_rpy).as_quat()  # (4,) xyzw
                    ee_pose = np.concatenate([ee_pos, ee_quat])              # (7,)
                    

                    print("ee_pos", ee_pos)
                    print("ee_rpy:",  ee_rpy)
                    print("gripper_qpos:",  gripper_qpos)

                    print("ee_quat", ee_quat)
                    print("ee_pose", ee_pose)

                    # Then sent to the Isaacsim:
                    robot_interface.set_cartesian_pose([ee_pose], [EE_FRAME])
                    robot_interface.set_joint_positions(gripper_qpos, GRIPPER_JOINT_NAMES)

                    time.sleep(dt) # TODO: DON'T SLEEP, instead schedule with dt

               
                print(f"Submitted {action.shape[0]} steps of actions.")


        except KeyboardInterrupt:
            print("Interrupted!")

        
        print("Stopped.")


if __name__ == '__main__':
    main()
