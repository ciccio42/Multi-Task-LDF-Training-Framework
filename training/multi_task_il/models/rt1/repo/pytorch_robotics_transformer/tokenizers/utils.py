from typing import Dict, Union
import numpy as np
from gym import spaces
import torch


class Component_RT1_SpaceSampler():
    def batched_space_sampler(self):
        raise NotImplementedError
    
class RT1_SpaceSampler(Component_RT1_SpaceSampler):
    # This function will turn the space into the batched space and return a batched action sample.
    # The output format is compatible with OpenAI gym's Vectorized Environments.
    def batched_space_sampler(self, space: spaces.Dict, batch_size: int):
        batched_sample : Dict[str, np.ndarray] = {}
        samples = [space.sample() for _ in range(batch_size)] # 辞書のリスト
        for key in samples[0].keys():
            value_list = []
            for i in range(batch_size):
                value_list.append(samples[i][key])
            value_list = np.stack(value_list, axis=0)
            batched_sample[key] = value_list
        return batched_sample
        

class Decorator_RT1_SpaceSampler(Component_RT1_SpaceSampler):
    _rt1_space_sampler: Component_RT1_SpaceSampler = None
    
    def __init__(self, rt1_space_sampler: Component_RT1_SpaceSampler):
        self._rt1_space_sampler = rt1_space_sampler
    
    @property
    def rt1_space_sampler(self) -> Component_RT1_SpaceSampler:
        """The Decorator delegates all work to the wrapped component."""
        return self._rt1_space_sampler
    
    def batched_space_sampler(self):
        return self._rt1_space_sampler.batched_space_sampler()
    
    
class FirstStep_RT1_SpaceSampler(Decorator_RT1_SpaceSampler):
    
    def batched_space_sampler(self, space: spaces.Dict, batch_size: int):
        res = self.rt1_space_sampler.batched_space_sampler(space, batch_size)
        # make tokens in memory zeros for the first step
        for k in res.keys():
            res[k] = np.zeros(res[k].shape, dtype=res[k].dtype)
        return res
        


# This function will turn the space into the batched space and return a batched action sample.
# The output format is compatible with OpenAI gym's Vectorized Environments.
def batched_space_sampler(space: spaces.Dict, batch_size: int):
    batched_sample : Dict[str, np.ndarray] = {}
    samples = [space.sample() for _ in range(batch_size)] # 辞書のリスト
    for key in samples[0].keys():
        value_list = []
        for i in range(batch_size):
            value_list.append(samples[i][key])
        value_list = np.stack(value_list, axis=0)
        batched_sample[key] = value_list
    return batched_sample

# This function turn all dict values into tensor.
def np_to_tensor(sample_dict: Dict[str, Union[int,np.ndarray]]) -> Dict[str, torch.Tensor]:
    new_dict = {}
    for key, value in sample_dict.items():
        value = torch.tensor(value)
        new_dict[key] = value

    return new_dict

def tensor_from_cpu_to_cuda(sample_dict: Dict[str, Union[int,np.ndarray]], device) -> Dict[str, torch.Tensor]:
    
    new_dict = {}
    for key, value in sample_dict.items():
        value = value.to(device)
        new_dict[key] = value
        
    return new_dict