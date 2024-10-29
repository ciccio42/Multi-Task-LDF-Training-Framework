import torch.multiprocessing as mp
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange, parse_shape
import os
import hydra
# mmdet moduls
from torchsummary import summary
from multi_task_il.models.basic_embedding import ResNetFeats
import torch
from torch.autograd import Variable
from torchvision import ops
from multi_task_il.models.cond_target_obj_detector.utils import *
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
import cv2
import matplotlib.pyplot as plt
import time
DEBUG = False

def get_backbone(backbone_name="slow_r50", video_backbone=True, pretrained=False, conv_drop_dim=3):
    if video_backbone:
        print(f"Loading video backbone {backbone_name}.....")
        if backbone_name == "r2plus1d_18":
            if not pretrained:
                weights = None
            else:
                weights = R2Plus1D_18_Weights

            return nn.Sequential(*list(r2plus1d_18(weights=weights).children())[:-1])
    else:
        print(f"Loading  backbone {backbone_name}.....")
        if backbone_name == "resnet18":
            return ResNetFeats(use_resnet18=True,
                               pretrained=pretrained,
                               output_raw=True,
                               drop_dim=conv_drop_dim)

class CondModule(nn.Module):

    def __init__(self, height=120, width=160, demo_T=4, model_name="slow_r50", pretrained=False, cond_video=True, n_layers=3, demo_W=7, demo_H=7, demo_ff_dim=[128, 64, 32], demo_linear_dim=[512, 256, 128], conv_drop_dim=3):
        super().__init__()
        self._demo_T = demo_T
        self._cond_video = cond_video

        self._backbone = get_backbone(backbone_name=model_name,
                                      video_backbone=cond_video,
                                      pretrained=pretrained,
                                      conv_drop_dim=conv_drop_dim)

        if not cond_video:
            conv_layer = []
            if conv_drop_dim == 2:
                input_dim_0 = 512
            elif conv_drop_dim == 3:
                input_dim_0 = 256
            input_dim = [input_dim_0, demo_ff_dim[0], demo_ff_dim[1]]
            output_dim = demo_ff_dim
            for i in range(n_layers):
                if i == 0:
                    depth_dim = self._demo_T
                else:
                    depth_dim = 1
                conv_layer.append(
                    nn.Conv3d(input_dim[i], output_dim[i], (depth_dim, 1, 1), bias=False))

            self._3d_conv = nn.Sequential(*conv_layer)
            linear_input = demo_ff_dim[-1] * demo_W * demo_H
        else:
            # [TODO] Implement ff for video backbone
            linear_input = 512

        # MLP encoder
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
        # 1. Compute features for each frame in the batch
        sizes = parse_shape(input, 'B T _ _ _')
        if not self._cond_video:
            backbone_input = rearrange(input, 'B T C H W -> (B T) C H W')
            backbone_out = self._backbone(backbone_input)
            backbone_out = rearrange(
                backbone_out, '(B T) C H W -> B T C H W', **sizes)
            backbone_out = rearrange(backbone_out, 'B T C H W -> B C T H W')
            temp_conv_out = self._3d_conv(backbone_out)
            temp_conv_out = rearrange(temp_conv_out, 'B C T H W -> B T C H W')
            linear_input = torch.flatten(temp_conv_out, start_dim=1)
            task_embedding = self._mlp_encoder(linear_input)
        else:
            backbone_input = rearrange(input, 'B T C H W -> B C T H W')
            backbone_out = rearrange(self._backbone(
                backbone_input), 'B C T H W -> B (C T H W)')
            task_embedding = self._mlp_encoder(backbone_out)

        # print(task_embedding.shape)
        return task_embedding
