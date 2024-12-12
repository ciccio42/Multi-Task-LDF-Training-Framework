
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
                 black_list=[], #datasets to exclude
                 demo_crop=[0, 0, 0, 0], #TODO,
                 select_random_frames=False
                 ):
        super().__init__()
        
        # processing video demo
        self.demo_crop = OrderedDict()
        self.mode = mode
        self._demo_T = demo_T
        self.width, self.height = width, height
        self.aug_twice = aug_twice
        self.aux_pose = aux_pose
        self.select_random_frames = select_random_frames
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
                
        #load embedding json paths
        with open(f'{jsons_folder}/embeddings_data.json', 'r') as file:
            self.embeddings_paths_dict = json.load(file)
                
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
        
    def __getitem__(self, index):
        traj_path, task_name = self.all_pkl_paths[index]
        
        #this is in order to search for embedding
        embedding_file = None
        start = False
        found_dataset = False
        for i, name in enumerate(traj_path.split('/')):
            if start:
                if not found_dataset:
                    temp = self.embeddings_paths_dict[name]
                    found_dataset = True
                else:
                    if '.pkl' not in name:
                        temp = temp[name]
                    else: # we found the embedding file
                        embedding_file = temp[0]
                        break
                    
            if name == 'datasets':
                start = True                

        demo_traj = load_traj(traj_path) # loading traiettoria
        demo_data = make_demo(self, demo_traj[0], task_name)  # solo video di 4 frame del task
        
        # for t,frame in enumerate(demo_data['demo']):
        #     img_debug = np.moveaxis(frame.detach().cpu().numpy()*255, 0, -1)
        #     cv2.imwrite(f"debug_demo_{t}.png", img_debug) #images are already rgb
        
        embedding_data = pkl.load(open(embedding_file, 'rb'))
    
        return {'demo_data': demo_data, 'embedding_data': torch.from_numpy(embedding_data)}
    
    def __len__(self):
        return self.all_file_count
    
    
class CommandEncoderSampler(BatchSampler):
    
    def __init__(self, dataset, shuffle=False):
        self.dataset = dataset
        # self.batch_size = batch_size # no, ci fermiamo quando abbiamo campionato un indice per ogni task
        self.shuffle = shuffle
        
        self.max_len = 0
        # save the longest idxs lenght
        for dataset_str in self.dataset.map_tasks_to_idxs.keys():
            for task_str in self.dataset.map_tasks_to_idxs[dataset_str].keys():
                if type(self.dataset.map_tasks_to_idxs[dataset_str][task_str]) == list: # if we found idxs
                    idx_len = len(self.dataset.map_tasks_to_idxs[dataset_str][task_str])
                    self.max_len = idx_len if idx_len > self.max_len else self.max_len
                else:
                    for subtask_str in self.dataset.map_tasks_to_idxs[dataset_str][task_str].keys():
                        if type(self.dataset.map_tasks_to_idxs[dataset_str][task_str][subtask_str]) == list: # if we found idxs
                            idx_len = len(self.dataset.map_tasks_to_idxs[dataset_str][task_str][subtask_str])
                            self.max_len = idx_len if idx_len > self.max_len else self.max_len
                        else:
                            raise NotImplementedError
                        
        print(f"max_len: {self.max_len}")
        
        # sampler per ogni task (random sampler)
        self.task_idx_samplers = {} # store all samplers here
        self.task_iterators = {}
        for dataset_str in self.dataset.map_tasks_to_idxs.keys():
            self.task_idx_samplers[dataset_str] = {}
            self.task_iterators[dataset_str] = {}
            for task_str in self.dataset.map_tasks_to_idxs[dataset_str].keys():
                if type(self.dataset.map_tasks_to_idxs[dataset_str][task_str]) == list: # if we found idxs
                    self.task_idx_samplers[dataset_str][task_str] = RandomSampler(self.dataset.map_tasks_to_idxs[dataset_str][task_str])
                    self.task_iterators[dataset_str][task_str] = iter(self.task_idx_samplers[dataset_str][task_str])
                    # self.task_idx_samplers.append(RandomSampler(self.dataset.map_tasks_to_idxs[dataset_str][task_str]))
                    # print(self.dataset.map_tasks_to_idxs[dataset_str][task_str])
                else:
                    self.task_idx_samplers[dataset_str][task_str] = {}
                    self.task_iterators[dataset_str][task_str] = {}
                    for subtask_str in self.dataset.map_tasks_to_idxs[dataset_str][task_str].keys():
                        self.task_idx_samplers[dataset_str][task_str][subtask_str] = {}
                        self.task_iterators[dataset_str][task_str][subtask_str] = {}
                        if type(self.dataset.map_tasks_to_idxs[dataset_str][task_str][subtask_str]) == list: # if we found idxs
                            self.task_idx_samplers[dataset_str][task_str][subtask_str] = RandomSampler(self.dataset.map_tasks_to_idxs[dataset_str][task_str][subtask_str])
                            self.task_iterators[dataset_str][task_str][subtask_str] = iter(self.task_idx_samplers[dataset_str][task_str][subtask_str])
                            # self.task_idx_samplers.append(RandomSampler(self.dataset.map_tasks_to_idxs[dataset_str][task_str][subtask_str]))  
                            # print(self.dataset.map_tasks_to_idxs[dataset_str][task_str][subtask_str])
                        else:
                            raise NotImplementedError
        
        print(f"created {len(self.task_idx_samplers)} samplers")
        
    
    def __iter__(self):
        
        # batch = []
        # # call next of the iterators
        # for iter_idx, iterator in enumerate(self.task_iterators):
        #     try:
        #         sample = next(iterator)
        #     except StopIteration:
        #         print('reset sampler')
        #         self.task_iterators[iter_idx] = iter(self.task_idx_samplers[iter_idx])
        #     batch.append(sample)
        
        for i in range(self.max_len): # quante volte farlo? fino alla lunghezza del task più lungo
            batch = []
            for dataset_str in self.task_iterators.keys(): # per come è ora si ferma appena ha letto tutti i task
                for task_str in self.task_iterators[dataset_str].keys():
                    if type(self.task_iterators[dataset_str][task_str]) == dict:
                        for subtask_str in self.task_iterators[dataset_str][task_str].keys():
                            try:
                                batch.append(
                                    self.dataset.map_tasks_to_idxs[dataset_str][task_str][subtask_str][next(
                                        self.task_iterators[dataset_str][task_str][subtask_str]
                                    )]
                                )
                            except StopIteration:
                                self.task_iterators[dataset_str][task_str][subtask_str] = iter(self.task_idx_samplers[dataset_str][task_str][subtask_str])
                                batch.append(
                                    self.dataset.map_tasks_to_idxs[dataset_str][task_str][subtask_str][next(
                                        self.task_iterators[dataset_str][task_str][subtask_str]
                                    )]
                                )    
                    else:
                        try:
                            batch.append(
                                self.dataset.map_tasks_to_idxs[dataset_str][task_str][next(
                                    self.task_iterators[dataset_str][task_str]
                                )]
                            )
                        except StopIteration:
                            self.task_iterators[dataset_str][task_str] = iter(self.task_idx_samplers[dataset_str][task_str])
                            batch.append(
                                self.dataset.map_tasks_to_idxs[dataset_str][task_str][next(
                                    self.task_iterators[dataset_str][task_str]
                                )]
                            )
            print(f"batch: {batch}")
            yield batch
        
        
    
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
    import torch
    from multi_task_il.datasets.command_encoder.cond_module import CondModule
    from torch.utils.data import DataLoader, BatchSampler, RandomSampler
    
    finetuning_dataset = CommandEncoderFinetuningDataset(mode='train',
                                                         jsons_folder='/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes',
                                                         black_list=BLACK_LIST,
                                                         data_augs=DATA_AUGS)
    
    command_encoder_batch_sampler = CommandEncoderSampler(finetuning_dataset, shuffle=True)
    # sampler = RandomSampler(finetuning_dataset)
    # batch_sampler = BatchSampler(sampler, batch_size=32, drop_last=True)
    
    train_loader = DataLoader(finetuning_dataset, batch_sampler=command_encoder_batch_sampler)
    
    # for i in finetuning_dataset:
    #     a = 1

    MODEL_PATH = '/user/frosa/multi_task_lfd/checkpoint_save_folder/1Task-pick_place-cond_module_lr_1e-4_good_split-Batch32/model_save-225.pt'
    cond_module = CondModule(model_name='r2plus1d_18', demo_linear_dim=[512, 512, 512], pretrained=True)
    weights = torch.load(MODEL_PATH, weights_only=True)
    cond_module.load_state_dict(weights)
    cond_module.eval()
    
    for i in train_loader:
        print('prova')
        
        
        
    
    
    print('hello')

