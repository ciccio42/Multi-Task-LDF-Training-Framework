from mmdet.models.backbones.swin import SwinTransformer
from mmengine.registry import init_default_scope
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from multi_task_il.models import get_model
from multi_task_il.models.discrete_logistic import DiscreteMixLogistic
from multi_task_il.models.rep_modules import BYOLModule, ContrastiveModule
from multi_task_il.models.basic_embedding import TemporalPositionalEncoding
from einops import rearrange, repeat, parse_shape
from collections import OrderedDict
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from torchvision import models
import pickle
from multi_task_il.datasets.multi_task_datasets import MultiTaskPairedDataset
# mmdet moduls
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple, Union)
from torchsummary import summary
from torchvision import models
from multi_task_il.models.basic_embedding import ResNetFeats
import torch
import json
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from typing import Dict
from einops import rearrange, repeat, parse_shape


def get_backbone(backbone_name="slow_r50", video_backbone=True, pretrained=False):
    if video_backbone:
        print(f"Loading video backbone {backbone_name}.....")
        return torch.hub.load("facebookresearch/pytorchvideo",
                              model=backbone_name,
                              pretrained=pretrained)
    else:
        print(f"Loading  backbone {backbone_name}.....")
        if backbone_name == "resnet18":
            return ResNetFeats(use_resnet18=True,
                               pretrained=pretrained,
                               output_raw=True,
                               drop_dim=2)


class CondModule(nn.Module):

    def __init__(self, height=120, width=160, demo_T=4, model_name="slow_r50", pretrained=False, cond_video=True, n_layers=3, demo_W=7, demo_H=12, demo_ff_dim=[128, 64, 32], demo_linear_dim=[512, 256, 128],):
        super().__init__()
        self._demo_T = demo_T
        self._cond_video = cond_video

        self._backbone = get_backbone(backbone_name=model_name,
                                      video_backbone=cond_video,
                                      pretrained=pretrained)

        if not cond_video:
            conv_layer = []
            input_dim = [512, demo_ff_dim[0], demo_ff_dim[1]]
            output_dim = demo_ff_dim
            for i in range(n_layers):
                if i == 0:
                    depth_dim = self._demo_T
                else:
                    depth_dim = 1
                conv_layer.append(
                    nn.Conv3d(input_dim[i], output_dim[i], (depth_dim, 1, 1), bias=False))

            self._3d_conv = nn.Sequential(*conv_layer)
        else:
            # [TODO] Implement ff for video backbone
            pass

        # MLP encoder
        linear_input = demo_ff_dim[-1] * demo_W * demo_H
        mlp_encoder = []
        for indx, layer_dim in enumerate(demo_linear_dim):
            if indx == 0:
                input_dim = linear_input
            else:
                input_dim = demo_linear_dim[indx-1]
            mlp_encoder.append(nn.Linear(in_features=input_dim,
                                         out_features=layer_dim))
            mlp_encoder.append(nn.ReLU())

        self._mlp_encoder = nn.Sequential(*mlp_encoder)

    def forward(self, input):
        backbone_out = self._backbone(input)
        backbone_out = rearrange(backbone_out, 'B C W H -> C B W H')
        temp_conv_out = self._3d_conv(backbone_out)
        temp_conv_out = rearrange(temp_conv_out, 'C B W H-> B C W H')
        linear_input = torch.flatten(temp_conv_out, start_dim=1)
        task_embedding = self._mlp_encoder(linear_input)
        print(task_embedding.shape)
        return temp_conv_out


class AgentModule(nn.Module):

    def __init__(self, height=120, width=160, obs_T=4, model_name="resnet18", pretrained=False):
        super().__init__()
        self._module = get_backbone(backbone_name=model_name,
                                    video_backbone=False,
                                    pretrained=pretrained)

    def forward(self, input):
        return self._module(input)


class CondTargetObjectDetector(nn.Module):

    def __init__(self,
                 height=120,
                 width=160,
                 demo_T=4,
                 obs_T=6,
                 cond_backbone_name="slow_r50",
                 agent_backbone_name="resnet18",
                 cond_video=False,
                 pretrained=False):
        super().__init__()

        self._cond_backbone = CondModule(height=height,
                                         width=width,
                                         demo_T=demo_T,
                                         model_name=cond_backbone_name,
                                         pretrained=pretrained,
                                         cond_video=cond_video)

        self._agent_backone = AgentModule(height=height,
                                          width=width,
                                          obs_T=obs_T,
                                          model_name=agent_backbone_name,
                                          pretrained=pretrained)

        summary(self)

    def forward(self, inputs: dict):
        cond_video = inputs['demo']
        agent_obs = inputs['images']

        cond_emb = self._cond_backbone(cond_video)
        print(cond_emb.shape)
        agent_emb = self._agent_backone(agent_obs)
        print(agent_emb.shape)


@hydra.main(
    version_base=None,
    config_path="../../experiments",
    config_name="target_object_detector_config.yaml")
def main(cfg):
    import debugpy
    debugpy.listen(('0.0.0.0', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    inputs = dict()

    width = 360
    height = 200
    demo_T = 4
    inputs['demo'] = torch.rand(
        (4, 3, 200, 360),  dtype=torch.float).to('cuda:0')

    inputs['images'] = torch.rand(
        (1, 3, 200, 360),  dtype=torch.float).to('cuda:0')

    module = CondTargetObjectDetector(
        width=width,
        height=height,
        demo_T=demo_T,
        cond_backbone_name="resnet18",
        cond_video=False)

    module.to('cuda:0')
    module(inputs)


if __name__ == '__main__':
    main()
