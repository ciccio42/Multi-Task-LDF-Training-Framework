
from pytorch_robotics_transformer.transformer_network import TransformerNetwork
from typing import Optional, Tuple, Union, List, Dict
from gym import spaces
import numpy as np
from collections import OrderedDict
import torch
from pytorch_robotics_transformer.tokenizers.utils import batched_space_sampler, np_to_tensor

HEIGHT = 256
WIDTH = 320
TIME_SEQUENCE_LENGHT = 4
TRAIN_BATCH_SIZE = 4    # 4*4 = 16. 512/16 = 32

def define_spaces():
    state_space = spaces.Dict(
    {
        'image': spaces.Box(low=0.0, high=1.0, 
                        shape=(3, HEIGHT, WIDTH), dtype=np.float32),
        'natural_language_embedding': spaces.Box(low=-np.inf, high=np.inf, 
                        shape=[512], dtype=np.float32)
    }
    )
    
    action_space = spaces.Dict(
    OrderedDict([
        ('terminate_episode', spaces.Discrete(2)), 
        ('world_vector', spaces.Box(low= -1.0, high= 1.0, shape=(3,), dtype=np.float32)),
        ('rotation_delta', spaces.Box(low= -np.pi / 2, high= np.pi / 2, shape=(3,), dtype=np.float32)),
        ('gripper_closedness_action', spaces.Box(low= -1.0  , high= 1.0, shape=(1,), dtype=np.float32))
        ])
    )
    
    return state_space, action_space


if __name__ == '__main__':
    
    
    # traj_path = '/raid/home/frosa_Loc/opt_dataset/pick_place/ur5e_pick_place'
    # import pickle
    # import os
    # import glob
    # tasks = sorted(os.listdir(traj_path))
    # for t in tasks:
    #     task_path = traj_path + f'/{t}'
    #     traj_files_path = sorted(glob.glob(f'{task_path}/*.pkl'))
    #     break
    
    # for traj_path in traj_files_path:
    #     with open(traj_path, 'rb') as f:
    #         traj = pickle.load(f)
    #     break
    
    # traj = traj['traj']
    # for t in traj:
    #     obs, info = t['obs'], t['info'] # dallo stato del gripper (info) posso capire se il gripper è aperto o chiuso
    #     print(info)
    #     eef_pos, eef_quat, camera_front_image = obs['eef_pos'], obs['eef_quat'], obs['camera_front_image']

    # print(t)
    
        
    # pickle.load()
    
    
    train_action = {
        'world_vector':
            torch.full([TRAIN_BATCH_SIZE, TIME_SEQUENCE_LENGHT, 3], 0.5),
        'rotation_delta':
            torch.full([TRAIN_BATCH_SIZE, TIME_SEQUENCE_LENGHT, 3], 0.5),
        'terminate_episode':
            torch.full([TRAIN_BATCH_SIZE, TIME_SEQUENCE_LENGHT], 1),
        'gripper_closedness_action':
            torch.full([TRAIN_BATCH_SIZE, TIME_SEQUENCE_LENGHT, 1], 0.5),
    }

    # usare hydra
    
    state_space, action_space = define_spaces()
    
    model = TransformerNetwork(
        input_tensor_space=state_space,
        output_tensor_space=action_space,
        time_sequence_length=TIME_SEQUENCE_LENGHT
    )
    
    # se non le setto, verranno settate a zero quando vengono creati gli action_token
    
    model.set_actions(train_action)  # setto le azioni di gt (verifica)
    
    # model._state_space
    
    model.eval()
    
    sample_obs = {
        'image': torch.randn((TRAIN_BATCH_SIZE,TIME_SEQUENCE_LENGHT,3,256,320)),
        'natural_language_embedding': torch.randn((TRAIN_BATCH_SIZE,TIME_SEQUENCE_LENGHT,512))
    }
    
    # sample_net_state = {
    #     'action_tokens': torch.randn((1,3,8)),
    #     'context_image_tokens': torch.randn((1,3,8,512)),
    #     'seq_idx': torch.randn((1))
    # }
    
    network_state = batched_space_sampler(model._state_space, TRAIN_BATCH_SIZE) # campionamento a caso
    network_state = np_to_tensor(network_state)
    
    # observation: Dict[str, torch.Tensor] ('image', 'natural_language_embedding')
        # image: torch.Size([1, 3, 256, 320])
        # natural_language_embedding: torch.Size([1, 512])
    # network_state: Dict[str, .Tensor] ('action_tokens', 'context_image_tokens', 'seq_idx')
        # action_tokens: torch.Size(torch[1, 3, 8])
        # context_image_tokens: torch.Size([1, 3, 8, 512])
        # seq_idx: torch.Size([1])
    
    with torch.no_grad():
        output = model(sample_obs,
                       network_state)
        print(output)