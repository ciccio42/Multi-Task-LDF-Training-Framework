import torch
from torch import nn
from multi_task_il.datasets.command_encoder.cond_module import CondModule
from multi_task_il.models.rt1.repo.pytorch_robotics_transformer.transformer_network import TransformerNetwork
from typing import Optional, Tuple, Union, Any, Dict, List
from gym import spaces
from collections import OrderedDict
from multi_task_il.models.rt1.repo.pytorch_robotics_transformer.tokenizers.utils import *
import cv2

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
            conv_drop_dim=3,
            
            cond_module_model_path = None
        
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
        ) # in evaluation because already training and for memory consumption purposes
        
        # load pretrained weights of cond_module
        # self.cond_module = CondModule(model_name='r2plus1d_18', demo_linear_dim=[512, 512, 512], pretrained=True)
        weights = torch.load(cond_module_model_path, weights_only=True)
        self.cond_module.load_state_dict(weights)
        self.cond_module.eval()
        # used to store network_state
        # this is used for inference in order to remember the previous tokens up to _time_sequence_length steps
        
        model_parameters = filter(lambda p: p.requires_grad, self.cond_module.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(self.cond_module)
        print('Total params in cond module before freezing:', params)

        # freeze cond module
        for p in self.cond_module.parameters():
            p.requires_grad = False
            
        model_parameters = filter(lambda p: p.requires_grad, self.cond_module.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(self.cond_module)
        print('Total params in cond module after freezing:', params)

        self.rt1_memory = None
        self.base_net_state_sampler = RT1_SpaceSampler() # random sampler class
        self.inference_first_state_sampler = FirstStep_RT1_SpaceSampler(self.base_net_state_sampler) # sampler for first state at inference
        
    def compute_bin_accuracy(self, actor_bin, gt_bin):
        
        # add accuracy
        pass
        
    def forward(self,
                images,
                states,
                demo,
                actions, # actions are normalized in [-1.0, 1.0] with normalizations ranges defined in .yaml
                bsize):
        
        # create embedding from video demonstration that will be use to condition
        # conv function activations via film layers
        debug = False
        
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
            
            # rt1_network_state = batched_space_sampler(self.rt1._state_space, bsize) # campionamento a caso
            rt1_network_state = self.base_net_state_sampler.batched_space_sampler(self.rt1._state_space, bsize) # campionamento a caso
            rt1_network_state = np_to_tensor(rt1_network_state)
            rt1_network_state = tensor_from_cpu_to_cuda(rt1_network_state, next(self.cond_module.parameters()).device)
        
        else:
            # we are in inference, the model expects (b,c,h,w) observations
            rt1_obs = {
                "image": images.squeeze(1), # remove time dimension
                "natural_language_embedding": cond_embedding
            }
            
            if debug:
                img_debug = np.moveaxis(rt1_obs['image'][0].detach().cpu().numpy()*255, 0, -1)
                cv2.imwrite(f"debug_rt1_obs.png", img_debug) #images are already rgb
            # cv2.imwrite(f"debug_rt1.png", img_tensor[:, :, ::-1]) #BGR -> RGB
            
            if self.rt1_memory == None: # if this is the initial step
                rt1_network_state = self.inference_first_state_sampler.batched_space_sampler(self.rt1._state_space, bsize)
                rt1_network_state = np_to_tensor(rt1_network_state)
                rt1_network_state = tensor_from_cpu_to_cuda(rt1_network_state, next(self.cond_module.parameters()).device)
            else: # if at least one step has been executed
                rt1_network_state = self.rt1_memory # retrieve the network state of the previous timestep
               
        if actions is not None: # training-eval: compute also accuracy on action bins
            out, rt1_network_state, bin_acc = self.rt1(rt1_obs, rt1_network_state)
        else: # inference: don't compute accuracy
            out, rt1_network_state = self.rt1(rt1_obs, rt1_network_state)
            
        
        if actions is None: # inference
            # return out, self.rt1._aux_info['action_labels'], self.rt1._aux_info['action_predictions_logits']
            self.rt1_memory = rt1_network_state # save new network state
            return out, rt1_network_state
        else: # training
            return out, self.rt1._aux_info['action_loss'], bin_acc
        
if __name__ == '__main__':
    pass