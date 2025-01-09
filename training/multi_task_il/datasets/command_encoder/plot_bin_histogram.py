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


class DatasetIterator():
    
    def __init__(self, dataset_dict):
        self.dataset_dict = dataset_dict
    
    def __iter__(self):
        for task_str in self.dataset_dict.keys():
            if type(self.dataset_dict[task_str]) == list:
                for el_path in self.dataset_dict[task_str]:
                    with open(el_path, "rb") as f:
                        el = pkl.load(f)
                    yield {'traj_el': el,
                           'task_str': task_str,
                           'sub_task_str': None}
            elif type(self.dataset_dict[task_str]) == dict:
                for sub_task_str in self.dataset_dict[task_str].keys():
                    if type(self.dataset_dict[task_str][sub_task_str]) == list:
                        for el_path in self.dataset_dict[task_str][sub_task_str]:
                            with open(el_path, "rb") as f:
                                el = pkl.load(f)
                            yield {'traj_el': el,
                                   'task_str': task_str,
                                   'sub_task_str': sub_task_str}
                            
    def __len__(self):
        count = 0
        
        for task_str in self.dataset_dict.keys():
            if type(self.dataset_dict[task_str]) == list:
                for el_path in self.dataset_dict[task_str]:
                    count+=1
            elif type(self.dataset_dict[task_str]) == dict:
                for sub_task_str in self.dataset_dict[task_str].keys():
                    if type(self.dataset_dict[task_str][sub_task_str]) == list:
                        for el_path in self.dataset_dict[task_str][sub_task_str]:
                            count+=1
                            
        return count
    
# class Rescaler():
    
#     def __init__(self, mu_old, sigma_old, mu_new, sigma_new):
#         self.mu_old = mu_old,
#         self.sigma_old = sigma_old,
#         self.mu_new = mu_new,
#         self.sigma_new = sigma_new
    
#     def rescale_value(self,value):
        
#         if value != 0.0:
#             value_norm = (value - self.mu_old) / self.sigma_old
#             return value_norm * self.sigma_new + self.mu_new
#         else:
#             return 0.0

class Rescaler():
    
    def __init__(self, min_old, max_old, min_new, max_new):
        self.min_old = min_old
        self.max_old = max_old
        self.min_new = min_new
        self.max_new = max_new
    
    def rescale_value(self,value):
        value_scaled = (value - self.min_old) / (self.max_old - self.min_old) # [0,1]
        return value_scaled * (self.max_new - self.min_new) + self.min_new


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
        
    min_max_traj_path = '/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes/min_max_delta_datasets.json'
    min_max_traj_path_2 = '/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes/min_max_delta_datasets_2.json'
    
    
    with open(min_max_traj_path, 'r') as file:
        min_max_dict = json.load(file)
        
    with open(min_max_traj_path_2, 'r') as file:
        min_max_dict_2 = json.load(file)
        
    min_max_dict['real_new_ur5e_pick_place_converted'] = min_max_dict_2['real_new_ur5e_pick_place_converted']
    min_max_dict['sim_new_ur5e_pick_place_converted'] = min_max_dict_2['sim_new_ur5e_pick_place_converted']
        
    mins_array = np.stack([min_max_dict[i]['min'][:3] for i in min_max_dict.keys()])
    maxs_array = np.stack([min_max_dict[i]['max'][:3] for i in min_max_dict.keys()])
    min_min = np.min(mins_array, axis=0)
    max_max = np.max(maxs_array, axis=0)
    arg_mins = np.argmin(mins_array, axis=0)
    arg_maxs = np.argmax(maxs_array, axis=0)
    
    print('**POSITIONS**')
    print(f"min_min: {min_min}\nmax_max: {max_max}")
    print(f"argmins: {arg_mins}, argmaxs: {arg_maxs}")
    
    
    ## scalare rispetto al min_min e max_max
    
    all_pkl_paths_path = '/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes/all_pkl_paths.json'
    with open(all_pkl_paths_path, 'r') as file:
        all_pkl_dict = json.load(file)
    
    # BLACK_LIST = ['asu_table_top_converted',
    #             'berkeley_autolab_ur5_converted',
    #             'iamlab_cmu_pickup_insert_converted',
    #             'taco_play_converted',
    #             'droid_converted_old',
    #             'droid_converted',
    #             'real_new_ur5e_pick_place_converted',
    #             'sim_new_ur5e_pick_place_converted',
    #             'panda_pick_place']
    
    # BLACK_LIST = ['asu_table_top_converted',
    #             'berkeley_autolab_ur5_converted',
    #             'iamlab_cmu_pickup_insert_converted',
    #             'taco_play_converted',
    #             'droid_converted_old',
    #             'droid_converted',
    #             'real_new_ur5e_pick_place_converted',
    #             'panda_pick_place']
    
    BLACK_LIST = [
                'taco_play_converted',
                'droid_converted_old',
                'droid_converted',
                'panda_pick_place']

    # min_new = min_min[0]
    # max_new = max_max[0]
    min_new = -0.09
    max_new = 0.09
    # mu_new = (x_range_min + x_range_max)/2
    # sigma_new = x_range_max - x_range_min
    
    # print(f"rescaling to new parameters: mu_new {mu_new}, sigma_new: {sigma_new}")s
    
    for dataset_str in min_max_dict.keys():
        if dataset_str not in BLACK_LIST:
            # sigma_old = min_max_dict[dataset_str]['max'][0] - min_max_dict[dataset_str]['min'][0]
            # mu_old = (min_max_dict[dataset_str]['max'][0] + min_max_dict[dataset_str]['min'][0]) / 2
            # print(f"{dataset_str}: mu_old {mu_old}, sigma_old: {sigma_old}")
            
            min_old = min_max_dict[dataset_str]['min'][0]
            max_old = min_max_dict[dataset_str]['max'][0]
            
            # all_pkl_dict[dataset_str]
            
            dataset_iterator = DatasetIterator(all_pkl_dict[dataset_str])
            # rescaler = Rescaler(, sigma_old, mu_new, sigma_new)
            rescaler = Rescaler(min_old, max_old, min_new, max_new)
            
            for traj_dict in tqdm(dataset_iterator, desc=f'{dataset_str}', total=(len(dataset_iterator))):
                # print(traj_dict)
                traj = traj_dict['traj_el']['traj']
                old_actions = []
                new_actions = []
                for step_t in traj:
                    action_t = step_t['action']
                    
                    # mean normalization
                    x_scaled = rescaler.rescale_value(action_t[0])
                    print(f'{action_t[0]} -> {x_scaled}')
                    old_actions.append(action_t[0])
                    new_actions.append(x_scaled)
                    
                fig = plt.figure()
                plt.plot(old_actions, '--o', label='old_actions', markersize=1)
                plt.plot(new_actions, '--o', label='new_actions', markersize=1)
                plt.legend(['old_actions', 'new_actions'])
                
                plt.axhline(y = 0, color = 'b', linestyle = ':') 
                
                plt.title(f'scaling from [{min_old:.2f},{max_old:.2f}] to [{min_new:.2f},{max_new:.2f}]')
                if not os.path.exists('test_scalings'):
                    os.mkdir('test_scalings')
                
                plt.savefig(f'test_scalings/scaling_old_actions_{dataset_str}.png')
                    
                    
                break  # to plot only 1 traj for each dataset (outer loop)
                
                    
                    
                    
                    
            
        
        
        
        
    
    
    
    
    
    
    
    
    
    
        
    
    
    
    
    
    
    
    ## istogramma per ogni valore del vettore azione
    
    
    
    
    
    
    
    
    
