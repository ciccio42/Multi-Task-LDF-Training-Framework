
import torch
from torch.utils.data import Dataset, BatchSampler
import glob
import pickle as pkl
import json
from collections import defaultdict, OrderedDict
import random
from multi_task_il.datasets.command_encoder.utils import * ###########

class FinetuningPairedDataset(Dataset):
    
    def __init__(self,
                 mode='train',
                 jsons_folder='',
                 obs_T=7,
                 demo_T=4,
                 action_T=1,
                 width=180,
                 height=100,
                 aug_twice=True,
                 aux_pose=True,
                 use_strong_augs=True,
                 data_augs=None,
                 black_list=[], #datasets to exclude
                 demo_crop=[0, 0, 0, 0], #TODO,
                 select_random_frames=False,
                 convert_action = False, #TODO
                 take_first_frame = False, #TODO
                 non_sequential=False,
                 split_pick_place = False,
                 state_spec=('ee_aa', 'gripper_qpos'),
                 load_eef_point=False,
                 bbs_T = 1,
                 perform_augs=True,
                 perform_scale_resize=False,
                 agent_name='ur5',
                 pick_next=False,
                 normalize_action = False
                 ):
        super().__init__()
        
        # processing video demo
        self.task_crops = OrderedDict()
        self.demo_crop = OrderedDict()
        self.agent_crop = OrderedDict()
        self.mode = mode
        self._demo_T = demo_T
        self._obs_T = obs_T
        self._bbs_T = bbs_T
        self._action_T = action_T
        self.width, self.height = width, height
        self.aug_twice = aug_twice
        self.aux_pose = aux_pose
        self.select_random_frames = select_random_frames
        self.black_list = black_list # dataset to exclude
        self.use_strong_augs = use_strong_augs
        self.data_augs = data_augs
        self.frame_aug = create_data_aug(self)
        self._convert_action = convert_action
        self._take_first_frame = take_first_frame
        self.split_pick_place = split_pick_place
        self.non_sequential = non_sequential
        self._state_spec = state_spec
        self._load_state_spec = True if state_spec is not None else False
        self._load_eef_point = load_eef_point
        self._perform_augs = perform_augs
        self.perform_scale_resize=perform_scale_resize
        self.agent_name = agent_name
        self.pick_next = pick_next
        self._normalize_action = normalize_action
        
        if non_sequential:
            print("Warning! The agent observations are not sampled in neighboring timesteps, make sure inverse dynamics loss is NOT used in training \n ")
        
        assert jsons_folder != '', 'you must specify a location for the json folder'
        if self.mode == 'train':
            with open(f'{jsons_folder}/train_pkl_paths_couples.json', 'r') as file:
                self.pkl_paths_dict = json.load(file)
        elif self.mode == 'val':
            with open(f'{jsons_folder}/val_pkl_paths_couples.json', 'r') as file:
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
        print(f'[{self.mode.capitalize()}] total file count: {all_file_count}')
        
        # with open("map_tasks_to_idxs.json", "w") as outfile: 
        #     json.dump(self.map_tasks_to_idxs,outfile,indent=2) 
        
    def __getitem__(self, index):
        couple_path, task_name = self.all_pkl_paths[index]
        
        # for convention, in the couple the first element is the demonstration, the second one is the trajectory
        demo_path = couple_path[0]
        traj_path = couple_path[1]
                 
        sim_crop = False #TODO
        sub_task_id = 0 #TODO
        demo_traj, agent_traj = load_traj(demo_path), load_traj(traj_path) # loading traiettoria
        demo_data = make_demo(self, demo_traj[0], task_name)  # solo video di 4 frame del task
        traj = self._make_traj(
            agent_traj[0], # traj object
            agent_traj[1], # command
            task_name,
            sub_task_id,
            sim_crop,
            self._convert_action)
        
        # for t,frame in enumerate(demo_data['demo']):
        #     img_debug = np.moveaxis(frame.detach().cpu().numpy()*255, 0, -1)
        #     cv2.imwrite(f"debug_demo_{t}.png", img_debug) #images are already rgb
    
        return {'demo_data': demo_data, 'traj': traj, 'task_name': 'finetuning'} # task_name key is for the collate_fn, loss grouping...
    
    def __len__(self):
        return self.all_file_count
    
    def _make_traj(self, traj, command, task_name, sub_task_id, sim_crop, convert_action):

        ret_dict = {}

        end = len(traj)
        start = torch.randint(low=1, high=max(
            1, end - self._obs_T + 1), size=(1,))

        if self._take_first_frame:
            first_frame = [torch.tensor(1)]
            chosen_t = first_frame + [j + start for j in range(self._obs_T)]
        else:
            chosen_t = [j + start for j in range(self._obs_T)]

        if self.non_sequential:
            chosen_t = torch.randperm(end)
            chosen_t = chosen_t[chosen_t != 0][:self._obs_T]

        first_phase = None
        if self.split_pick_place:
            first_t = chosen_t[0].item()
            last_t = chosen_t[-1].item()
            if task_name == 'nut_assembly' or task_name == 'pick_place' or 'button' in task_name or 'stack_block' in task_name:
                first_step_gripper_state = traj.get(first_t)['action'][-1]
                first_phase = True if first_step_gripper_state == -1.0 or first_step_gripper_state == 0.0 else False
                last_step_gripper_state = traj.get(last_t)['action'][-1]

                # if first_step_gripper_state == 1.0 and last_step_gripper_state == -1.0:
                #     print("Last with placing")
                if (first_step_gripper_state != last_step_gripper_state) and not (first_step_gripper_state == 1.0 and (last_step_gripper_state == -1.0 or last_step_gripper_state == 0.0)):
                    # change in task phase
                    for indx, step in enumerate(range(first_t, last_t+1)):
                        action_t = traj.get(step)['action'][-1]
                        if first_step_gripper_state != action_t:
                            step_change = step
                            break
                    for indx, step in enumerate(range(step_change+1-self._obs_T, step_change+1)):
                        chosen_t[indx] = torch.tensor(step)


        ############################## TODO
        self._load_state_spec = False
        images, images_cp, bb, obj_classes, action, states, points = create_sample(
            dataset_loader=self,
            traj=traj,
            chosen_t=chosen_t,
            task_name=task_name,
            command=command,
            load_action=True,
            load_state=self._load_state_spec,
            load_eef_point=self._load_eef_point,
            agent_task_id=sub_task_id,
            sim_crop=sim_crop,
            convert_action=convert_action)

        ret_dict['images'] = torch.stack(images)

        if self.aug_twice:
            ret_dict['images_cp'] = torch.stack(images_cp)

        ret_dict['gt_bb'] = torch.stack(bb)
        ret_dict['gt_classes'] = torch.stack(obj_classes)

        ret_dict['states'] = []
        ret_dict['states'] = np.array(states)

        ret_dict['actions'] = []
        ret_dict['actions'] = np.array(action)

        ret_dict['points'] = []
        ret_dict['points'] = np.array(points)

        if self.split_pick_place:
            ret_dict['first_phase'] = torch.tensor(first_phase)

        if self.aux_pose:
            grip_close = np.array(
                [traj.get(i, False)['action'][-1] > 0 for i in range(1, len(traj))])
            grip_t = np.argmax(grip_close)
            drop_t = len(traj) - 1 - \
                np.argmax(np.logical_not(grip_close)[::-1])
            aux_pose = [traj.get(t, False)['obs']['ee_aa'][:3]
                        for t in (grip_t, drop_t)]
            ret_dict['aux_pose'] = np.concatenate(aux_pose).astype(np.float32)
        return ret_dict
    
    
class FinetuningPairedDatasetSampler(BatchSampler):
    
    def __init__(self, dataset, shuffle=True):
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
                        
        print(f"[{self.dataset.mode.capitalize()}] max_len: {self.max_len}")
        
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
        
        print(f"[{self.dataset.mode.capitalize()}] created {len(self.task_idx_samplers)} samplers")
        
    
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
        
        for i in range(self.max_len): # quante volte farlo? fino alla lunghezza del task con più traiettorie
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
            
            
            if self.shuffle:
                random.shuffle(batch)
                print(f"batch: {batch}")
                yield batch
            else:   
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
    
    finetuning_dataset = FinetuningPairedDataset(mode='train',
                                                         jsons_folder='/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes/traj_couples',
                                                         black_list=BLACK_LIST,
                                                         data_augs=DATA_AUGS)
    val_finetuning_dataset = FinetuningPairedDataset(mode='val',
                                                            jsons_folder='/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes/traj_couples',
                                                            black_list=BLACK_LIST,
                                                            data_augs=DATA_AUGS)
        
    command_encoder_batch_sampler = FinetuningPairedDatasetSampler(finetuning_dataset, shuffle=True)
    val_command_encoder_batch_sampler = FinetuningPairedDatasetSampler(val_finetuning_dataset, shuffle=True)
    # sampler = RandomSampler(finetuning_dataset)
    # batch_sampler = BatchSampler(sampler, batch_size=32, drop_last=True)
    
    train_loader = DataLoader(finetuning_dataset, batch_sampler=command_encoder_batch_sampler)
    val_loader = DataLoader(val_finetuning_dataset, batch_sampler=val_command_encoder_batch_sampler)
    
    
    # for i in finetuning_dataset:
    #     a = 1

    MODEL_PATH = '/user/frosa/multi_task_lfd/checkpoint_save_folder/1Task-pick_place-cond_module_lr_1e-4_good_split-Batch32/model_save-225.pt'
    cond_module = CondModule(model_name='r2plus1d_18', demo_linear_dim=[512, 512, 512], pretrained=True)
    weights = torch.load(MODEL_PATH, weights_only=True)
    cond_module.load_state_dict(weights)
    cond_module.eval()
    
    # for i in train_loader:
    #     print('prova')
    #     break
    
    for i in val_loader:
        print('prova')    
        
        
    
    
    print('hello')

