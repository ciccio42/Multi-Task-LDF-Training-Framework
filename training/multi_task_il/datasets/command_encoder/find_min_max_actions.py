
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
import json
import pickle as pkl

def compute_min_max_for_traj(traj_path):
    with open(traj_path, "rb") as f:
        agent_file_data = pkl.load(f)
    
    # array of action for all t
    all_t_action = np.array([ t['action'][:-1] for t in agent_file_data['traj'] ])
    max_all_t_action = np.max(all_t_action, 0)
    min_all_t_action = np.min(all_t_action, 0)
    
    return max_all_t_action, min_all_t_action

def check_if_min_max(min_all_t_action, max_all_t_action, min_max_actions_per_dataset):
    for i, i_min in enumerate(min_all_t_action):
        if i_min < min_max_actions_per_dataset[dataset_name]['min'][i]:
            min_max_actions_per_dataset[dataset_name]['min'][i] = i_min
    
    for i, i_max in enumerate(max_all_t_action):
        if i_max > min_max_actions_per_dataset[dataset_name]['max'][i]:
            min_max_actions_per_dataset[dataset_name]['max'][i] = i_max


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
        
        
    BLACK_LIST = ['asu_table_top_converted', 'berkeley_autolab_ur5_converted', 'iamlab_cmu_pickup_insert_converted', 'taco_play_converted', 'droid_converted_old', 'droid_converted', 'panda_pick_place']
    
    min_max_actions_per_dataset = {}
        
    all_traj_path = '/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes/all_pkl_paths.json'
    
    with open(all_traj_path, 'r') as file:
        all_pkl_paths_dict = json.load(file)
        
    all_file_count = 0
    for dataset_name in all_pkl_paths_dict.keys():
        min_max_actions_per_dataset[dataset_name] = {
            'min' : [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], # we exclude gripper
            'max' : [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
        }
        if dataset_name not in BLACK_LIST:
            for task in tqdm(all_pkl_paths_dict[dataset_name].keys(), desc=f'analyzing {dataset_name}'):
                if type(all_pkl_paths_dict[dataset_name][task]) == list:
                    for t_path in all_pkl_paths_dict[dataset_name][task]: # for all task in the list
                        max_all_t_action, min_all_t_action = compute_min_max_for_traj(t_path)
                        check_if_min_max(min_all_t_action, max_all_t_action, min_max_actions_per_dataset)
                    
                elif type(all_pkl_paths_dict[dataset_name][task]) == dict:
                    for subtask in all_pkl_paths_dict[dataset_name][task].keys():
                        for t_path in all_pkl_paths_dict[dataset_name][task][subtask]:
                            max_all_t_action, min_all_t_action = compute_min_max_for_traj(t_path)
                            check_if_min_max(min_all_t_action, max_all_t_action, min_max_actions_per_dataset)
  
    
    
    with open("min_max_delta_datasets_2.json", "w") as outfile: 
        json.dump(min_max_actions_per_dataset,outfile,indent=2) 
    
    print(min_max_actions_per_dataset)
        
    