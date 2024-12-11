
import torch
from torch.utils.data import Dataset, BatchSampler
import glob
import pickle as pkl
import json
from collections import defaultdict, OrderedDict
from multi_task_il.datasets.command_encoder.utils import *


class CommandEncoderFinetuningDataset(Dataset):
    
    def __init__(self,
                 mode='train',
                 jsons_folder='',
                 demo_T=4,
                 width=180,
                 height=100,
                 aug_twice=True,
                 aux_pose=True,
                 use_strong_augs=True,
                 data_augs=None,
                 black_list=[],
                 demo_crop=[0, 0, 0, 0] #TODO
                 ):
        super().__init__()
        
        # processing video demo
        self.demo_crop = OrderedDict()
        self.mode = mode
        self._demo_T = demo_T
        self.width, self.height = width, height
        self.aug_twice = aug_twice
        self.aux_pose = aux_pose
        self.select_random_frames = False
        self.black_list = black_list # dataset to exclude
        self.use_strong_augs = use_strong_augs
        self.data_augs = data_augs
        self.frame_aug = create_data_aug(self)
        
        assert jsons_folder != '', 'you must specify a location for the json folder'
        if self.mode == 'train':
            with open(f'{jsons_folder}/train_pkl_paths.json', 'r') as file:
                self.pkl_paths_dict = json.load(file)
        elif self.mode == 'val':
            with open(f'{jsons_folder}/val_pkl_paths.json', 'r') as file:
                self.pkl_paths_dict = json.load(file)
                
        self.all_pkl_paths = defaultdict() # store all paths
        self.map_tasks_to_idxs = defaultdict()
        
        all_file_count = 0
        for dataset_name in self.pkl_paths_dict.keys():
            if dataset_name not in self.black_list:
                self.map_tasks_to_idxs[dataset_name] = defaultdict()
                for task in self.pkl_paths_dict[dataset_name].keys():
                    if type(self.pkl_paths_dict[dataset_name][task]) == list:
                        self.map_tasks_to_idxs[dataset_name][task] = []
                        self.demo_crop[task] = demo_crop  # same crop
                        for t in self.pkl_paths_dict[dataset_name][task]: # for all task in the list
                            self.all_pkl_paths[all_file_count] = (t, task) #add to all_pkl_paths
                            self.map_tasks_to_idxs[dataset_name][task].append(all_file_count) #memorize mapping
                            all_file_count+=1
                        
                    elif type(self.pkl_paths_dict[dataset_name][task]) == dict:
                        self.map_tasks_to_idxs[dataset_name][task] = defaultdict()
                        for subtask in self.pkl_paths_dict[dataset_name][task].keys():
                            self.map_tasks_to_idxs[dataset_name][task][subtask] = []
                            self.demo_crop[subtask] = demo_crop
                            for t in self.pkl_paths_dict[dataset_name][task][subtask]:
                                self.all_pkl_paths[all_file_count] = (t, subtask) #add to all_pkl_paths
                                self.map_tasks_to_idxs[dataset_name][task][subtask].append(all_file_count) #memorize mapping
                                all_file_count+=1
            
                
        
        self.all_file_count = all_file_count
        print(f'total file count: {all_file_count}')
        
        # with open("map_tasks_to_idxs.json", "w") as outfile: 
        #     json.dump(self.map_tasks_to_idxs,outfile,indent=2) 

        
        print('cdcnsdocndocn')
        
        
    def __getitem__(self, index):
        traj_path, task_name = self.all_pkl_paths[index]

        demo_traj = load_traj(traj_path) # loading traiettoria
        # embedding_data = self.load_embedding(embedding_file)
        
        demo_data = make_demo(self, demo_traj[0], task_name)  # solo video di 4 frame del task
        
        for t,frame in enumerate(demo_data['demo']):
            img_debug = np.moveaxis(frame.detach().cpu().numpy()*255, 0, -1)
            cv2.imwrite(f"debug_demo_{t}.png", img_debug) #images are already rgb
    
        return demo_data
    
    def __len__(self):
        return self.all_file_count
    
    
class CommandEncoderSampler(BatchSampler):
    
    def __init__(self, dataset, batch_size, shuffle=False):
        pass
    
    def __iter__(self):
        pass
    
    def __len__(self):
        return len(self.dataset)

        

class ResultsDisplayer():
    
    def __init__(self):
        pass
    
    def display_results(self):
        pass
    
    
if __name__ == '__main__':

    import debugpy
    debugpy.listen(('0.0.0.0', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    
    BLACK_LIST = ['taco_play_converted',
                  'droid_converted'
                  ]
    
    

    DATA_AUGS = {
                "old_aug": False,
                "brightness": [0.9, 1.1],
                "contrast": [0.9, 1.1],
                "saturation": [0.9, 1.1],
                "hue": [0.0, 0.0],
                "p": 0.1,
                "horizontal_flip_p": 0.1,
                "brightness_strong": [0.875, 1.125],
                "contrast_strong": [0.5, 1.5],
                "saturation_strong": [0.5, 1.5],
                "hue_strong": [-0.05, 0.05],
                "p_strong": 0.5,
                "horizontal_flip_p_strong": 0.5,
                "null_bb": False,
            }
    
    finetuning_dataset = CommandEncoderFinetuningDataset(mode='train',
                                                         jsons_folder='/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes',
                                                         black_list=BLACK_LIST,
                                                         data_augs=DATA_AUGS)
    
    # for i in finetuning_dataset:
    #     a = 1
    
    for _idx in [1000, 1200]:
        el = finetuning_dataset[_idx]
    
    
    print('hello')

