
import torch
from torch.utils.data import Dataset
import glob
import pickle
import json

class TrajectoryCommandsDataset(Dataset):
    
    def __init__(self,
                 root_path=None,
                 info_path='/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes/dataset_info.json'):
        assert root_path is not None, 'you MUST specify a root folder'
        self.root_path = root_path
        self.all_traj_paths = sorted(glob.glob(f'{self.root_path}/**/*.pkl'))
        # load precomputed info on datasets (command, env type...)
        self.info_path = info_path
        with open(info_path, 'r') as file:
            self.dataset_info = json.load(file)
        # command info
        self._command_num_classes = len(self.dataset_info['commands'])
        self._command_classes = self.dataset_info['commands']
        
        # log info
        print(f'loaded {len(self.all_traj_paths)} trajectories')
        print(f'found {self._command_num_classes} classes: {self._command_classes}')
        
    
    def __len__(self):
        return len(self.all_traj_paths)
    
    def __getitem__(self, idx):
        with open(self.all_traj_paths[idx] , 'rb') as f:
            _data = pickle.load(f)
            
        with open('traj000.pkl' , 'rb') as f:
            _data = pickle.load(f)
        
        # return_data = {}
        # return_data['traj'] = _data['traj']
        
        return _data
    
    
class CommandClassDefiner():
    
    def __init__(self,
                 root_path=None):
        assert root_path is not None, 'you must specify a root directory'
        self.root_path = root_path
        self.all_traj_paths = sorted(glob.glob(f'{self.root_path}/**/*.pkl'))
        print(f'loaded {len(self.all_traj_paths)} trajectories')
    
    def produce_info(self, output_file_path):
    
        commands_set = set()
        env_type_set = set()
        
        for traj_path in self.all_traj_paths:
            with open(traj_path , 'rb') as f:
                traj = pickle.load(f)
            commands_set.update([traj['command']])
            env_type_set.update([traj['env_type']])
            
        _info = {'commands': list(commands_set),
                'env_type' : list(env_type_set)}
        
        _info_json_path = f'{output_file_path}/dataset_info.json'
        
        with open(_info_json_path, 'w') as output_file:
            print(json.dumps(_info, indent=2), file=output_file)    
    
    
class ResultsDisplayer():
    
    def __init__(self):
        pass
    
    def display_results(self):
        pass