import pickle
import os
from robosuite.utils.transform_utils import quat2axisangle, axisangle2quat, quat2mat, mat2quat, mat2euler
from copy import deepcopy
import numpy as np
from multi_task_il.datasets.savers import Trajectory
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm
from multi_task_il.datasets.utils import trasform_from_world_to_bl
import math

'''
This script converts real and sim ur5e dataset in deltas
'''

def convert_to_delta(traj_data, is_sim=True):
    for t in range(traj_data['len']): 
        if t == 0 and is_sim == True: # first step for the sim dataset
            ee_aa = traj_data['traj'].get(t)['obs']['ee_aa']
            gripper = np.array([traj_data['traj'].get(t)['action'][-1]])
            ee_aa = np.concatenate([ee_aa, gripper])
            action_t = apply_transf_ur5e_sim(ee_aa)[:-1]
            # action_t = np.concatenate([action_t, gripper])
            # action_t_minus_1 = traj_data['traj'].get(t)['action']
            action_t_minus_1 = np.concatenate([action_t, gripper]) #obs
            action_t = traj_data['traj'].get(t)['action']
            
        elif t == 0 and is_sim == False:
            eef_pos = traj_data['traj'].get(t)['obs']['eef_pos']
            eef_rpy = mat2euler(quat2mat(traj_data['traj'].get(t)['obs']['eef_quat']))
            gripper = np.array([traj_data['traj'].get(t)['action'][-1]])
            action_t_minus_1 = np.concatenate([eef_pos, eef_rpy, gripper])
            action_t = traj_data['traj'].get(t)['action']
            
        elif t == (traj_data['len']-1): # last step
            action_t = traj_data['traj'].get(t)['action']
            action_t_minus_1 = action_t
            
        else:
            action_t = traj_data['traj'].get(t)['action']
            # action_t_minus_1 is the one saved at the previous step
            
        delta_t = np.array([0.0] * 7)
        delta_t[:-1] = action_t[:-1] - action_t_minus_1[:-1]
        delta_t[-1] = action_t[-1] # gripper
        
        # check for angles for which value is > pi
        for angle_idx, delta_angle in enumerate(delta_t[3:6]):
            if delta_angle > math.pi or delta_angle < -math.pi:
                ang_t_minus_1 = action_t_minus_1[angle_idx+3]
                ang_t = action_t[angle_idx+3]
                
                if ang_t > 0.0 and ang_t_minus_1 < 0.0:
                    ang_t_minus_1 = 2*math.pi - abs(ang_t_minus_1)
                elif ang_t < 0.0 and ang_t_minus_1 > 0.0:
                    ang_t = 2*math.pi - abs(ang_t)
                else:
                    print(f'[WARNING] unexpected situation when computing angle:\nang_t:{ang_t}, ang_t_minus_1:{ang_t_minus_1}')
                    
                delta_angle = ang_t - ang_t_minus_1
                delta_t[angle_idx+3] = delta_angle
                
                
                    
        action_t_minus_1 = action_t # save before overwrite it
        change_action(traj_data['traj'], t, delta_t)
        
def convert_quat_RPY(traj_data):
    for t in range(traj_data['len']):
        step_t = traj_data['traj'].get(t)
        action_t = step_t['action']
        
        # TCP position wrt BL, TCP quat wrt BL, gripper
        pos_t, quat_t, gripper_t = action_t[0:3], action_t[3:-1], action_t[-1]
        step_t_rpy = mat2euler(quat2mat(quat_t)) # TCP rpy wrt BL
        gripper_t = np.array([0.0]) if gripper_t == 1.0 else np.array([1.0]) # invert to 1.0 open, 0.0 closed
        
        action_t = np.concatenate([pos_t, step_t_rpy, gripper_t])
        
        change_action(traj_data['traj'], t, action_t)
        
def apply_transf_ur5e_sim(action_t):
    action_t_conv = trasform_from_world_to_bl(action_t)
    # from axisangle to rpy
    axis_angle_rot = action_t_conv[3:6]
    euler_rot = mat2euler(quat2mat(axisangle2quat(axis_angle_rot)))
    action_t_conv[3:6] = euler_rot
    action_t_conv[-1] = 1.0 if action_t_conv[-1] == 0.0 else 0.0
    return action_t_conv

def change_action(trajectory, t, new_action):
    obs_t, reward_t, done_t, info_t, action_t = trajectory._data[t]
    trajectory._data[t] = obs_t, reward_t, done_t, info_t, new_action

def plot_action(traj, title, save_name):
    
    action_len = traj.get(1)['action'].shape[0]
    # print(f"plotting {action_len} actions.")
    
    fig = plt.figure()
    fig.set_figheight(8.5)
    fig.set_figwidth(10.5) 
    fig.suptitle(title)
    
    #grid specifications
    gs0 = gridspec.GridSpec(3,3, figure=fig)
    
    # gs00 = gridspec.GridSpecFromSubplotSpec(5,5, subplot_spec=gs0)

    ax00 = fig.add_subplot(gs0[0,0])
    ax01 = fig.add_subplot(gs0[0,1])
    ax02 = fig.add_subplot(gs0[0,2])
        
    ax03 = fig.add_subplot(gs0[1,0])
    ax04 = fig.add_subplot(gs0[1,1])   
    ax05 = fig.add_subplot(gs0[1,2])
    
    ax06 = fig.add_subplot(gs0[2,0])
    ax07 = fig.add_subplot(gs0[2,1])

    if action_len == 7:
        axes = [ax00, ax01, ax02, ax03, ax04, ax05, ax06]
    elif action_len == 8:
        axes = [ax00, ax01, ax02, ax03, ax04, ax05, ax06, ax07]
    else:
        raise NotImplementedError
    
    
    actions_to_plot = np.stack([traj_data['traj'].get(t)['action'] for t in range(len(traj))])
    
    for idx, ax in enumerate(axes):
        ax.plot(actions_to_plot[:, idx])
    
    plt.savefig(f'{save_name}.png')
    

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action='store_true', help="whether or not attach the debugger")
    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    real_ur5_dataset_path = '/raid/home/frosa_Loc/opt_dataset/pick_place/real_new_ur5e_pick_place'
    sim_ur5_dataset_path = '/user/frosa/multi_task_lfd/ur_multitask_dataset/opt_dataset/pick_place/ur5e_pick_place'
    
    root_save_real_ur5_conv_dataset_path = '/user/frosa/multi_task_lfd/datasets/real_new_ur5e_pick_place_converted'
    root_save_sim_ur5_conv_dataset_path = '/user/frosa/multi_task_lfd/datasets/sim_new_ur5e_pick_place_converted'
    
    if not os.path.exists(root_save_real_ur5_conv_dataset_path):
        os.mkdir(root_save_real_ur5_conv_dataset_path)
        
    if not os.path.exists(root_save_sim_ur5_conv_dataset_path):
        os.mkdir(root_save_sim_ur5_conv_dataset_path)
    
    
    # with open('/user/frosa/multi_task_lfd/ur_multitask_dataset/opt_dataset/pick_place/ur5e_pick_place/task_13/traj034.pkl', "rb") as f:
    #     a = pickle.load(f)

    
    ##----------- CONVERT REAL DATASET
    # real:
    # frame is BL
    # quat -> RPY -> deltas
    
    # save this for plotting
    print('\n Converting real dataset...')
    
    saved_orig_traj = False
    saved_conv_traj = False
    saved_conv_delta_traj = False    
    orig_traj = None
    conv_traj = None
    conv_delta_traj = None
    
    for task_dir in sorted(os.listdir(real_ur5_dataset_path)):
        if 'task_' in task_dir:
            root_dir = real_ur5_dataset_path + '/' + task_dir
            for root, dirs, files in os.walk(root_dir):
                for f in tqdm(sorted(files), desc=f'converting {task_dir}'):
                    
                    # create the new trajectory object
                    # sub_traj = Trajectory()
                    
                    traj_path = root + '/' + f
                    with open(traj_path, "rb") as f:
                        traj_data = pickle.load(f)
                        
                    if not saved_orig_traj:
                        orig_traj = deepcopy(traj_data) # pass by value
                        plot_action(orig_traj['traj'], 'original traj real', 'delta_script_original_traj_real')
                        saved_orig_traj = True
                        
                    ## convert quat -> RPY
                    convert_quat_RPY(traj_data)
                        
                    if not saved_conv_traj:
                        conv_traj = deepcopy(traj_data)
                        plot_action(conv_traj['traj'], 'conv traj real', 'delta_script_conv_traj_real')
                        saved_conv_traj = True
                        
                    ## convert to deltas
                    convert_to_delta(traj_data, is_sim=False)
                        
                    if not saved_conv_delta_traj:
                        conv_delta_traj = deepcopy(traj_data)
                        plot_action(conv_delta_traj['traj'], 'conv traj delta real', 'delta_script_conv_traj_delta_real')
                        saved_conv_delta_traj = True
                        
                    # exit()
                          
                    # save the converted trajectory
                    traj_pkl_save_path = root_save_real_ur5_conv_dataset_path + '/' + task_dir + '/' + traj_path.split('/')[-1]
                    
                    try:
                        pickle.dump({
                            'traj': traj_data['traj'],
                            'len': len(traj_data['traj']),
                            'env_type': traj_data['env_type'],
                            'task_id': traj_data['task_id']}, open(traj_pkl_save_path, 'wb'))
                    except Exception:
                        task_path_dir = root_save_real_ur5_conv_dataset_path + '/' + task_dir 
                        if not os.path.exists(task_path_dir):
                            os.mkdir(task_path_dir)
                        pickle.dump({
                            'traj': traj_data['traj'],
                            'len': len(traj_data['traj']),
                            'env_type': traj_data['env_type'],
                            'task_id': traj_data['task_id']}, open(traj_pkl_save_path, 'wb'))
          
          
    ##----------- CONVERT SIM DATASET
    
    # save this for plotting
    
    saved_orig_traj = False
    saved_conv_traj = False
    saved_conv_delta_traj = False    
    orig_traj = None
    conv_traj = None
    conv_delta_traj = None
    
    print('\n Converting sim dataset...')
    
    for task_dir in sorted(os.listdir(sim_ur5_dataset_path)):
        if 'task_' in task_dir:
            root_dir = sim_ur5_dataset_path + '/' + task_dir
            for root, dirs, files in os.walk(root_dir):
                for f in tqdm(sorted(files), desc=f'converting {task_dir}'):
                    
                    traj_path = root + '/' + f
                    with open(traj_path, "rb") as f:
                        traj_data = pickle.load(f)
                                                
                    ####------ shift
                    for t in range(traj_data['len']):
                        
                        try:
                            step_t1 = traj_data['traj'].get(t+1)
                            action_t1 = step_t1['action']
                        except AssertionError:
                            step_t1 = traj_data['traj'].get(t)
                            action_t1 = deepcopy(step_t1['action'])
                            action_t1[:-1] = action_t1[:-1] - action_t1[:-1] # all 0. except for gripper
                        
                        change_action(traj_data['traj'], t, action_t1)
                    
                    if not saved_orig_traj:
                        orig_traj = deepcopy(traj_data) # pass by value
                        plot_action(orig_traj['traj'], 'original traj sim', 'delta_script_original_traj_sim')
                        saved_orig_traj = True
                        
                    ####----conversion
                    for t in range(traj_data['len']):
                        action_t = traj_data['traj'].get(t)['action']
                        
                        action_t_conv = apply_transf_ur5e_sim(action_t)
                        
                        change_action(traj_data['traj'], t, action_t_conv)
                        
                    if not saved_conv_traj:
                        conv_traj = deepcopy(traj_data) # pass by value
                        plot_action(conv_traj['traj'], 'conv traj sim', 'delta_script_conv_traj_sim')
                        saved_conv_traj = True     
                        
                    ####----- convert to deltas
                    convert_to_delta(traj_data)
                        
                    if not saved_conv_delta_traj:
                        conv_delta_traj = deepcopy(traj_data)
                        plot_action(conv_delta_traj['traj'], 'conv traj delta sim', 'delta_script_conv_traj_delta_sim')
                        saved_conv_delta_traj = True
                        
                    # exit()
                    
                    # save the converted trajectory
                    traj_pkl_save_path = root_save_sim_ur5_conv_dataset_path + '/' + task_dir + '/' + traj_path.split('/')[-1]
                    
                    try:
                        pickle.dump({
                            'traj': traj_data['traj'],
                            'len': len(traj_data['traj']),
                            'env_type': traj_data['env_type'],
                            'task_id': traj_data['task_id']}, open(traj_pkl_save_path, 'wb'))
                    except Exception:
                        task_path_dir = root_save_sim_ur5_conv_dataset_path + '/' + task_dir 
                        if not os.path.exists(task_path_dir):
                            os.mkdir(task_path_dir)
                        pickle.dump({
                            'traj': traj_data['traj'],
                            'len': len(traj_data['traj']),
                            'env_type': traj_data['env_type'],
                            'task_id': traj_data['task_id']}, open(traj_pkl_save_path, 'wb'))
                        
                    
                    
    

    


# SAVE_PATH = '/user/frosa/multi_task_lfd/datasets'


# plot conversione prima e dopo

# plot originale

# plot conversione

# plot conversione + delta

# open trajectories
# traj_datas = []
# for traj_path in traj_paths:
#     with open(traj_path, "rb") as f:
#         traj_data = pickle.load(f)
#     traj_datas.append(traj_data)