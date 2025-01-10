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
    
    
class SimTokenizer():
    
    def __init__(self, min, max, vocabsize):
        self.min = min
        self.max = max
        self.vocabsize = vocabsize

    def tokenize(self, value):
        
        norm_value = (value - self.min) / (self.max - self.min)
        # discretize
        return int(norm_value * (self.vocabsize - 1))
        
        
def plot_scaling(old_actions, new_actions, dataset_str, min_old, max_old, min_new, max_new, last_rt1=False):
    fig = plt.figure()
    plt.plot(old_actions, '--o', label='old_actions', markersize=1)
    plt.plot(new_actions, '--o', label='new_actions', markersize=1)
    plt.legend(['old_actions', 'new_actions'])
    
    plt.axhline(y = 0, color = 'b', linestyle = ':') 
    
    plt.title(f'scaling from [{min_old:.2f},{max_old:.2f}] to [{min_new:.2f},{max_new:.2f}]')
    
    if not last_rt1:
        if not os.path.exists('test_scalings'):
            os.mkdir('test_scalings')
        plt.savefig(f'test_scalings/scaling_old_actions_{dataset_str}.png')
    else:
        if not os.path.exists('test_scalings_toRT1'):
            os.mkdir('test_scalings_toRT1')
        plt.savefig(f'test_scalings_toRT1/scaling_old_actions_{dataset_str}.png')
    
def plot_hist(tokens_actions, bucket_size, dataset_str, min_new, max_new, action_el_str, ax):
    
    N, bins, patches = ax.hist(tokens_actions,bins=[i for i in range(bucket_size)], label='freq of bin')    
    # plt.title(f'{dataset_str}: frequency of bins in [{min_new:.2f},{max_new:.2f}]')
    ax.set_ylabel(f'number of bin')
    ax.set_xlabel(f'bins for {action_el_str}')
    ax.legend(['freq of bin'])
    

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
    
    ACTIONS_ELS = ['dx', 'dy', 'dz']
    
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

    MIN_RT1 = -1
    MAX_RT1 = 1
    
    for dataset_str in min_max_dict.keys():
        if dataset_str not in BLACK_LIST:
            
            fig = plt.figure()
            fig.set_figheight(4)
            fig.set_figwidth(10) 
            fig.suptitle(f'{dataset_str}')
            
            #grid specifications
            gs0 = gridspec.GridSpec(1,3, figure=fig)
            
            ax00 = fig.add_subplot(gs0[0,0])
            ax01 = fig.add_subplot(gs0[0,1])
            ax02 = fig.add_subplot(gs0[0,2])
            
            hist_dataset_axes = [ax00, ax01, ax02]
            
            for act_id, action_el_str in enumerate(ACTIONS_ELS): # plot each coord
                
                # sigma_old = min_max_dict[dataset_str]['max'][0] - min_max_dict[dataset_str]['min'][0]
                # mu_old = (min_max_dict[dataset_str]['max'][0] + min_max_dict[dataset_str]['min'][0]) / 2
                # print(f"{dataset_str}: mu_old {mu_old}, sigma_old: {sigma_old}")
                
                min_old = min_max_dict[dataset_str]['min'][act_id]
                max_old = min_max_dict[dataset_str]['max'][act_id]
                
                min_new = min_min[act_id]
                max_new = max_max[act_id]
                
                # all_pkl_dict[dataset_str]
                
                dataset_iterator = DatasetIterator(all_pkl_dict[dataset_str])
                # rescaler = Rescaler(, sigma_old, mu_new, sigma_new)
                rescaler = Rescaler(min_old, max_old, min_new, max_new)
                rescaler_to_RT1 = Rescaler(min_new, max_new, MIN_RT1, MAX_RT1)
                BUCKET_SIZE = 256
                tokenizer = SimTokenizer(min_new, max_new, BUCKET_SIZE)
                
                tokens_actions_per_dataset = []
                for traj_dict in tqdm(dataset_iterator, desc=f'{dataset_str}, action_el: {ACTIONS_ELS[act_id]}', total=(len(dataset_iterator))):
                    # print(traj_dict)
                    traj = traj_dict['traj_el']['traj']
                    old_actions = []
                    new_actions = []
                    new_actions_rt1_scale = []
                    # for every step in trajectory
                    for step_t in traj:
                        action_t = step_t['action']
                        
                        # mean normalization
                        act_value = rescaler.rescale_value(action_t[act_id]) # scale to min_min max_max scale
                        act_value_rt1 = rescaler_to_RT1.rescale_value(act_value) # scale to -1 1 from min_min max_max
                        act_value_token = tokenizer.tokenize(act_value_rt1) # tokenize
                        # print(f'{action_t[act_id]} -> {act_value} -> {act_value_token}')
                        old_actions.append(action_t[act_id])
                        new_actions.append(act_value)
                        new_actions_rt1_scale.append(act_value_rt1)
                        tokens_actions_per_dataset.append(act_value_token)
                    
                    
                    # plot_scaling(old_actions, new_actions, dataset_str, min_old, max_old, min_new, max_new) # to plot values across the old and new dataset range
                    # plot_scaling(new_actions, new_actions_rt1_scale, dataset_str, min_new, max_new, MIN_RT1, MAX_RT1, last_rt1=True) # to plot values from new dat range to RT1 input tokenization range
                    
                    # break  # to plot only 1 traj for each dataset (outer loop)
                    
                plot_hist(tokens_actions_per_dataset, BUCKET_SIZE, dataset_str, MIN_RT1, MAX_RT1, action_el_str, hist_dataset_axes[act_id])
            
            if not os.path.exists('test_hists_dx_dy_dz'):
                    os.mkdir('test_hists_dx_dy_dz')
            plt.savefig(f'test_hists_dx_dy_dz/bin_freq_{dataset_str}.png')
            
            # exit() # to plot all traj for first dataset     
    
