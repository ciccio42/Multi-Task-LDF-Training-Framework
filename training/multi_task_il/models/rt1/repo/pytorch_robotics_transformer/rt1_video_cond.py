import torch
from torch import nn
from multi_task_il.datasets.command_encoder.cond_module import CondModule
from multi_task_il.models.rt1.repo.pytorch_robotics_transformer.transformer_network import TransformerNetwork
from typing import Optional, Tuple, Union, Any, Dict, List
from gym import spaces
from collections import OrderedDict
from multi_task_il.models.rt1.repo.pytorch_robotics_transformer.tokenizers.utils import *

#TODO implement RT1 + cond_module (video conditioned)
class RT1_video_cond(nn.Module):
    def __init__(
            self,
            
            ### RT1 parameters
            input_tensor_space: spaces.Dict, # observatioin space like dict. keys are image, natural_language_embedding
            output_tensor_space: spaces.Dict, # action space like dict. keys are world_vector, rotation_delta, gripper_closedness_action, terminate_episode
            train_step_counter: int = 0,
            vocab_size: int = 256, # Dimensionality of tokens from the output layer. This is also dimensionality of tokens from the input layer.
            token_embedding_size: int = 512, # RT1ImageTokenizer outputs(=context_image_tokens) has embedding dimension of token_embedding_size. This will finally be utilized in 1x1 Conv in EfficientNetEncoder class.
            num_layers: int = 1,
            layer_size: int = 4096, # This corresponds to key_dim which is the size of each attention head for query, key and values.
            num_heads: int = 8,
            feed_forward_size: int = 512, # This corresponds to d_model which is embedding dimension of each token in transformer part.
            dropout_rate: float = 0.1,
            time_sequence_length: int = 1,
            crop_size: int = 236,
            # action_order: Optional[List[str]] = None,
            use_token_learner: Optional[bool] = True,
            return_attention_scores: bool = False,
            img_height: int = 224,
            img_width: int = 224,
            concat_target_obj_embedding: bool = False,
            
            ### cond_module parameters
            height=120,
            width=160,
            demo_T=4,
            model_name="slow_r50",
            pretrained=False,
            cond_video=True,
            n_layers=3,
            demo_W=7,
            demo_H=7,
            demo_ff_dim=[128, 64, 32],
            demo_linear_dim=[512, 256, 128],
            conv_drop_dim=3
        
        ) -> None:
        super().__init__()
        self.rt1 = TransformerNetwork(
            input_tensor_space=input_tensor_space,
            output_tensor_space=output_tensor_space,
            train_step_counter=train_step_counter,
            vocab_size=vocab_size,
            token_embedding_size=token_embedding_size,
            num_layers=num_layers,
            layer_size=layer_size,
            num_heads=num_heads,
            feed_forward_size=feed_forward_size,
            dropout_rate=dropout_rate,
            time_sequence_length=time_sequence_length,
            crop_size=crop_size,
            use_token_learner=use_token_learner,
            return_attention_scores=return_attention_scores,
            img_height=img_height,
            img_width=img_width,
            concat_target_obj_embedding=concat_target_obj_embedding
        )
        self.cond_module = CondModule(
            height=height,
            width=width,
            demo_T=demo_T,
            model_name=model_name,
            pretrained=pretrained,
            cond_video=cond_video,
            n_layers=n_layers,
            demo_W=demo_W,
            demo_H=demo_H,
            demo_ff_dim=demo_ff_dim,
            demo_linear_dim=demo_linear_dim,
            conv_drop_dim= conv_drop_dim      
        ).eval() # in evaluation because already training and for memory consumption purposes
        
        
    def forward(self,
                images,
                states,
                demo,
                actions, # actions are normalized in [-1.0, 1.0] with normalizations ranges defined in .yaml
                bsize):
        
        # create embedding from video demonstration that will be use to condition
        # conv function activations via film layers
        
        # TODO: load the weights of pretrained cond_module
        with torch.no_grad():
            cond_embedding = self.cond_module(demo) # 15GB for the computation graph -> 4GB with torch no grad
        if actions is not None:
            # not inference: there is more than one time step
            t = actions.shape[1]
            cond_embedding = cond_embedding.tile((t,1,1)).permute(1,0,2)

            rt1_obs = {
                "image": images[:,:-1,:,:,:], # we exclude the last one
                "natural_language_embedding": cond_embedding
            }
            
            rt1_actions = {
                'world_vector' : actions[:,:,:,0:3].squeeze(),
                'rotation_delta' : actions[:,:,:,3:6].squeeze(),
                'gripper_closedness_action' : actions[:,:,:,-1].squeeze()
            }
            self.rt1.set_actions(rt1_actions) # gt_action for ce loss
        else:
            # we are in inference, the model expects (b,c,h,w) observations ???
            rt1_obs = {
                "image": images.squeeze(1), # remove time dimension
                "natural_language_embedding": cond_embedding
            }
        
            # add something if needed here

        
        #TODO: [TEST] output -> azione dell'ultimo istante -> ERRORE 
        # last_action = actions[:, -1]
        # rt1_action = {
        #     'world_vector' : last_action[:,:,0:3].squeeze(),
        #     'rotation_delta' : last_action[:,:,3:6].squeeze(),
        #     'gripper_closedness_action' : last_action[:,:,-1].squeeze()
        # }
        # self.rt1.set_actions(rt1_action)
        
        # set action that will be used in RT1
        
        # 2 metodi per training:
            # 1 [SCELTO]
            # Prendere stack di 7 immagini e for di 7 iterazioni dove l'ultima immagine è t_i e azione a_i
            # 2
            # Usare stack di 7 immagini una sola volta con azione quella finale
        
        #TODO: inferenza: self.rt1._state_space (1x7) no (tx7) -> all'inizio tutti zero?
        #TODO: comprendere a cosa serve in inferenza e perché viene usato solo in quel caso
        rt1_network_state = batched_space_sampler(self.rt1._state_space, bsize) # campionamento a caso
        rt1_network_state = np_to_tensor(rt1_network_state)
        rt1_network_state = tensor_from_cpu_to_cuda(rt1_network_state, next(self.cond_module.parameters()).device)
        
        # test for actions
        # test_action = batched_space_sampler(self.rt1._output_tensor_space, bsize)
        # test_action = np_to_tensor(test_action)
        # self.rt1.set_actions(test_action)
        
        # actions[:,:,:,0] -> un tipo particolare di azione
        out = self.rt1(rt1_obs, rt1_network_state) # usa tutte le 6 azioni per calcolo loss ma l'output è data dall'ultima azione
        
        if actions is None: # inference
            # return out, self.rt1._aux_info['action_labels'], self.rt1._aux_info['action_predictions_logits']
            return out
        else: # training
            return out, self.rt1._aux_info['action_loss']
        
if __name__ == '__main__':
    pass