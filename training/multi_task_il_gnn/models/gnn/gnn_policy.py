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
from multi_task_il_gnn.models.gnn.lcg import LCGNnet
from multi_task_il_gnn.datasets.read_scene_graph import read_pkl

from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger object
logger = logging.getLogger('GNN-Policy')


colorama_init()


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


class GNNPolicy(nn.Module):

    def __init__(self,
                 cfg) -> None:
        super().__init__()

        self._cond_backbone = CondModule(height=cfg.height,
                                         width=cfg.width,
                                         demo_T=cfg.demo_T,
                                         model_name=cfg.cond_backbone_name,
                                         pretrained=cfg.pretrained,
                                         cond_video=cfg.cond_video,
                                         demo_H=cfg.dim_H,
                                         demo_W=cfg.dim_W,
                                         conv_drop_dim=cfg.conv_drop_dim,
                                         demo_ff_dim=cfg.demo_ff_dim,
                                         demo_linear_dim=cfg.demo_linear_dim
                                         )
        print(f"{Fore.YELLOW}Build conditioning backbone{Style.RESET_ALL}")

        self._cgn = LCGNnet(cfg.lcgnet_conf)
        print(f"{Fore.YELLOW}Build CGN Network{Style.RESET_ALL}")

    def forward(self, inputs: dict, inference: bool = False):
        cond_video = inputs['demo']
        scene_graph = inputs['node_features']
        # geometric_graph = inputs['agent_graph']
        cond_emb = self._cond_backbone(cond_video)
        # print(
        #     f"{Fore.YELLOW}Generated cond-embedding with shape {cond_emb.shape}{Style.RESET_ALL}")

        logits = self._cgn(input=scene_graph,
                           c_vect=cond_emb,
                           run_vqa=self._cgn.cfg.BUILD_VQA,
                           run_ref=self._cgn.cfg.BUILD_REF,
                           run_node_classifier=self._cgn.cfg.BUILD_NODE_CLASSIFIER)
        return logits


@hydra.main(
    version_base=None,
    config_path="../../../experiments",
    config_name="config_gnn.yaml")
def main(cfg):
    import debugpy
    debugpy.listen(('0.0.0.0', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    inputs = dict()

    width = 100
    height = 180
    demo_T = 4
    inputs['demo'] = torch.rand(
        (4, 3, height, width),  dtype=torch.float).to('cuda:0')[None]  # B, T, C, W, H

    # read example graphs
    pkl_file_path = "/user/frosa/multi_task_lfd/ur_multitask_dataset/geometric_graphs/pick_place/ur5e_pick_place/task_00/traj000.pkl"
    scene_graphs = read_pkl(file_path=pkl_file_path)[0].to('cuda:0')
    inputs['scene_graph'] = scene_graphs

    cfg.gnn_policy.gnn_policy_cfg.height = 100
    cfg.gnn_policy.gnn_policy_cfg.width = 180
    module = GNNPolicy(
        cfg=cfg.gnn_policy.gnn_policy_cfg)

    module.to('cuda:0')
    module(inputs, inference=True)


if __name__ == '__main__':
    main()
