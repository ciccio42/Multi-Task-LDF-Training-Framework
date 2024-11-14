import random
import torch
from multi_task_il.datasets import load_traj
import cv2
from torch.utils.data import Dataset, BatchSampler


import pickle as pkl
from collections import defaultdict, OrderedDict
import glob
import numpy as np
import matplotlib.pyplot as plt
import copy

from multi_task_il.utils import normalize_action
from multi_task_il.datasets.command_encoder.utils import *

from torch.utils.data import DataLoader

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


class CommandEncoderSampler(BatchSampler):
    
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset, self.batch_size = dataset, batch_size
        self.shuffle = shuffle # if we want to shuffle the elements in the batch which come ordered by subtask idx 
        
        self.demo_task_samplers = OrderedDict()
        self.demo_task_iterators = OrderedDict()
        
        self.demo_task_samplers['pick_place'] = OrderedDict()
        self.demo_task_iterators['pick_place'] = OrderedDict()
        
        task_name = 'pick_place'
        first_id = list(self.dataset.demo_subtask_to_idx[task_name].keys())[0]

        sub_task_size = len(self.dataset.demo_subtask_to_idx[task_name].get(first_id))
        
        for sub_task, sub_idxs in self.dataset.demo_subtask_to_idx[task_name].items():

            self.demo_task_samplers[task_name][sub_task] = RandomSampler(
                data_source=sub_idxs)
            assert len(sub_idxs) == sub_task_size, \
                'Got uneven data sizes for sub-{} under the task {}!'.format(
                    sub_task, task_name)
            self.demo_task_iterators[task_name][sub_task] = iter(
                RandomSampler(sub_idxs))
    
    def __iter__(self):
        
        # iteratore per ogni sottotask
        batch = []
        
        _n_subtask = len(self.dataset.demo_subtask_to_idx['pick_place'].keys())
        if self.batch_size < _n_subtask:
            print("[WARNING] you're using a batch_size which does not guarantee a sample for every subtask! \
                batch_size: {}, n_subtask: {}".format(self.batch_size, _n_subtask))
            
        first_task = list(self.dataset.demo_subtask_to_idx.keys())[0]
        first_subtask = list(self.dataset.demo_subtask_to_idx[first_task].keys())[0]
        _n_samples_subtask = len(self.dataset.demo_subtask_to_idx[first_task][first_subtask])
        
        number_of_iterations = len(self.dataset) // self.batch_size
        # assert number_of_iterations == _n_samples_subtask, "{} and {} are of different size!".format(number_of_iterations, _n_samples_subtask)
        
        # for i in range(number_of_iterations):
        for i in range(_n_samples_subtask):
            for _task in self.dataset.demo_subtask_to_idx.keys():
                for _subtask in self.dataset.demo_subtask_to_idx[_task].keys(): 
                    demo_sampler = self.demo_task_samplers[_task][_subtask]
                    demo_iterator = self.demo_task_iterators[_task][_subtask]
                    # new agent_indx in epoch
                    # sample demo for current
                    try:
                        demo_indx = self.dataset.demo_subtask_to_idx[_task][_subtask][next(
                            demo_iterator)]
                    except StopIteration: # in questo caso siamo in una nuova epoch, sono finiti gli indici e dobbiamo re-settare il sampler
                        # print(f"reset sampler")
                        self.demo_task_iterators[_task][_subtask] = iter(demo_sampler)
                        demo_iterator = self.demo_task_iterators[_task][_subtask]
                        demo_indx = self.dataset.demo_subtask_to_idx[_task][_subtask][next(
                            demo_iterator)]
                        
                    batch.append(demo_indx)
                    if len(batch) == self.batch_size:
                        # print(f"return batch: {batch}") # per debug, ha il comportamento desiderato
                        if self.shuffle: # if we choose to shuffle the dataset
                            random.shuffle(batch)
                        yield batch
                        batch = []
        
        # counter = 0
        # for _task in self.dataset.demo_subtask_to_idx.keys():
        #     for _subtask in self.dataset.demo_subtask_to_idx[_task].keys():
        #         _subtask_idx = self.dataset.demo_subtask_to_idx[_task][_subtask]
        #         for _idx, _sample_id in enumerate(_subtask_idx):
        #             batch.append(self.dataset.demo_subtask_to_idx[_task][_subtask][_idx])
        #             break
        #         if len(batch) == self.batch_size:
        #             yield batch
                
    
    def __len__(self):
        return len(self.dataset)
    


class CommandEncoderDataset(Dataset):
    
    def __init__(self,
                 mode='',
                 robot='',
                 n_train_and_val_samples_per_subtask=None,
                 n_test_samples_per_subtask = None,
                 demo_T=4,
                 width=180,
                 height=100,
                 aug_twice=True,
                 aux_pose=True,
                 use_strong_augs=True,
                 data_augs=None,
                 use_embedding_centroids=False) -> None:   # ricorda di mettere DATA_AUGS
        
        self.available_robots = ['ur5e' , 'panda']
        
        assert mode == 'train' or mode == 'val' or mode == 'test', f'{mode} is not a valid modality, choose btw \'train\', \'val\' or \'test\''
        assert data_augs != None, f'choose some data augmentations!'
        if mode == 'train' or mode == 'val':
            assert n_train_and_val_samples_per_subtask != None, 'you must specify the number of train and val samples'
        else: # if test
            assert n_train_and_val_samples_per_subtask != None and n_test_samples_per_subtask != None, 'you must specify train,val and test samples number'
        assert robot != '', 'you must choose a robot'
        assert robot in self.available_robots, f'{robot} is not available. Choose one from the list: {self.available_robots}' 
            
        # self.embedding_dir = embedding_dir
        self.mode = mode
        # dizionari per video dimostrazione
        self.demo_files = dict() # path video delle dimostazioni
        self.all_demo_files = OrderedDict()
        self.demo_task_to_idx = defaultdict(list)
        self.demo_subtask_to_idx = OrderedDict()
        # dizionari per embeddings
        self.embedding_files = dict()
        self.all_embedding_files = OrderedDict()
        self.embedding_task_to_idx = defaultdict(list)
        self.embedding_subtask_to_idx = OrderedDict()
        # processing video demo
        self.demo_crop = OrderedDict()
        self._demo_T = demo_T
        self.width, self.height = width, height
        self.aug_twice = aug_twice
        self.aux_pose = aux_pose
        self.select_random_frames = True
        self.n_test_samples_per_subtask = n_test_samples_per_subtask
        
        root_dir = '/raid/home/frosa_Loc/opt_dataset/'
        name = 'pick_place'
        n_tasks = 16
        
        split = [0.8,0.2]
        
        # directory dimostrazioni ed embeddings
        demo_dir = join(
            root_dir, name, '{}_{}'.format(robot, name))
        if not use_embedding_centroids:
            embedding_dir = join(
                root_dir, name, '{}'.format('command_embs')
            )
        else: # se vogliamo caricare i centroidi per ogni sottotask come gt
            embedding_dir = join(
                root_dir, name, '{}'.format('centroids_commands_embs')
            )
                       
        count = 0
        demo_file_cnt = 0
        embedding_file_cnt = 0
        
        # abbiamo solo pick_place
        self.demo_files['pick_place'] = {}
        self.demo_subtask_to_idx['pick_place'] = {}
        self.embedding_files['pick_place'] = {}
        self.embedding_subtask_to_idx['pick_place'] = {}
        
        # transformations
        DEMO_CROP = [20, 25, 80, 75] 
        self.demo_crop[name] = DEMO_CROP
        
        for _id in range(n_tasks):
            
            task_id = 'task_{:02d}'.format(_id)
            task_dir = expanduser(join(demo_dir,  task_id, '*.pkl')) # '/raid/home/frosa_Loc/opt_dataset/pick_place/ur5e_pick_place/task_00/*.pkl'
            demo_files = sorted(glob.glob(task_dir))
            
            embedding_id = task_id
            embedding_task_dir = expanduser(join(embedding_dir, embedding_id, '*.pkl')) # '/raid/home/frosa_Loc/opt_dataset/pick_place/command_embs/task_00/*.pkl'
            embedding_files = sorted(glob.glob(embedding_task_dir))
        
            subtask_size = n_train_and_val_samples_per_subtask
            assert len(demo_files) >= subtask_size, "Doesn't have enough demonstration data "+str(len(demo_files))
            assert len(embedding_files) >= subtask_size, "Doesn't have enough embedding data "+str(len(embedding_files))
            if self.mode != 'test':
                demo_files = demo_files[:subtask_size]
                embedding_files = embedding_files[:subtask_size]
            else: # in test mode we want to take other samples from train/val
                end_file_idx = subtask_size + self.n_test_samples_per_subtask # where we want to stop with training examples
                assert end_file_idx <= len(demo_files), 'you don\'t have enough demo data for test'
                demo_files = demo_files[subtask_size:end_file_idx]
                embedding_files = embedding_files[:self.n_test_samples_per_subtask]
                
            if self.mode != 'test':
                idxs = split_files(len(demo_files), split, self.mode) # train/test split
                demo_files = [demo_files[i] for i in idxs] # file di train
                embedding_files = [embedding_files[i] for i in idxs] # train embeddings (task0 e emb0 non necessariamente devono essere accoppiati)
                
            self.demo_files[name][_id] = deepcopy(demo_files)   # pick_place -> 00/01/../15 -> paths
            self.embedding_files[name][_id] = deepcopy(embedding_files)
            
            print(f"Loading task {name} - sub-task {_id} in mod {self.mode}")
        
            # assegnare indici ai 
            for demo_indx, demo in enumerate(demo_files):   # demo: path al file .pkl, _id: id della variazione, name: nome task
                self.all_demo_files[demo_file_cnt] = (
                    name, _id, demo)
                self.demo_task_to_idx[name].append( # indici
                    demo_file_cnt)
                try:
                    self.demo_subtask_to_idx[name][task_id].append(
                        demo_file_cnt)
                except KeyError:
                    self.demo_subtask_to_idx[name][task_id] = []
                    self.demo_subtask_to_idx[name][task_id].append(
                        demo_file_cnt)
                    
                demo_file_cnt += 1
                
            # dizionari embeddings
            for embedding_indx, embedding in enumerate(embedding_files):   # embedding: path al file .pkl, _id: id della variazione, name: nome task
                self.all_embedding_files[embedding_file_cnt] = (
                    name, _id, embedding)
                self.embedding_task_to_idx[name].append( # indici
                    embedding_file_cnt)
                try:
                    self.embedding_subtask_to_idx[name][task_id].append(
                        embedding_file_cnt)
                except KeyError:
                    self.embedding_subtask_to_idx[name][task_id] = []
                    self.embedding_subtask_to_idx[name][task_id].append(
                        embedding_file_cnt)
                    
                embedding_file_cnt += 1
        
        print('Done loading Task {}, demo trajctores and embeddings pairs reach a count of: {}'.format(name, demo_file_cnt))
        
        assert demo_file_cnt == embedding_file_cnt, f'demo videos and embeddings should have the same lenght! {demo_file_cnt, embedding_file_cnt}'
        self.file_count = demo_file_cnt # in questo caso abbiamo solo video del dimostratore
        self.use_strong_augs = use_strong_augs
        self.data_augs = data_augs
        self.frame_aug = create_data_aug(self)
        
    def __getitem__(self, index):
        
        # if type(index) == int:
        #     task_name, sub_task_id, demo_file = self.all_demo_files[index]
        # else:
        #     pass
        # task_name, sub_task_id, demo_file, embedding = self.all_demo_files[index]
        task_name, sub_task_id, demo_file = self.all_demo_files[index]
        _, _, embedding_file = self.all_embedding_files[index]
        
        start = time.time()
        demo_traj = load_traj(demo_file) # loading traiettoria
        
        embedding_data = self.load_embedding(embedding_file)
        
        end = time.time()
        logger.debug(f"Loading time {end-start}")
        
        demo_data = make_demo(self, demo_traj[0], task_name)    # solo video di 4 frame del task
        
        # return {'demo_data' : demo_data, 'label' : embedding}
        return {'demo_data' : demo_data, 'embedding' : embedding_data}
    
        # return traj_video, label    # label sarebbe l'embedding
    
    def __len__(self):
        """NOTE: we should count total possible demo-agent pairs, not just single-file counts
        total pairs should sum over all possible sub-task pairs"""
        return self.file_count
    
            
    def load_embedding(self, embedding_file):
        embedding_dict = pkl.load(open(embedding_file, 'rb'))
        return embedding_dict
    

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime
    
    
    dataset = CommandEncoderDataset(data_augs=DATA_AUGS)
    dataloader = DataLoader(dataset)
    
    for count, i in enumerate(dataloader):
        frames = i['demo_data']['demo'][0]
        
        # non perché in questo ciclo la frase dell'embedding viene racchiusa in [' e ']
        print('skipping {}'.format(count))
        
        if count != 310:
            continue
        
        fig, axs = plt.subplots(1,4, figsize=(10,3))
        fig.suptitle('sample {} : {}'.format(count, i['embedding']['sentence'][0]))
        for _frame_idx, ax in enumerate(axs):
            ax.imshow(frames[_frame_idx].permute(1,2,0))
        
        ts = datetime.now().strftime("%m-%d_%H:%M")
        plt.savefig(f'{os.getcwd()}/training/multi_task_il/datasets/command_encoder/sample_{ts}.png')
        
        exit()
    

    
    
    # # check comandi
    # sample = pkl.load(open('/raid/home/frosa_Loc/opt_dataset/pick_place/ur5e_pick_place/task_12/traj000.pkl', 'rb'))
    # command = sample['command']
    # print(command)
    
    