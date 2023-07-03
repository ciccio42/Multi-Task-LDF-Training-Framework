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
from torch.autograd import Variable
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


def get_backbone(backbone_name="slow_r50", video_backbone=True, pretrained=False, drop_dim=2):
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
                               drop_dim=drop_dim)


def conv(ic, oc, k, s, p):
    return nn.Sequential(
        nn.Conv2d(ic, oc, k, s, p),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(oc),
    )


def coord_map(shape, start=-1, end=1):
    """
    Gives, a 2d shape tuple, returns two mxn coordinate maps,
    Ranging min-max in the x and y directions, respectively.
    """
    m, n = shape
    x_coord_row = torch.linspace(
        start, end, steps=n).type(torch.cuda.FloatTensor)
    y_coord_row = torch.linspace(
        start, end, steps=m).type(torch.cuda.FloatTensor)
    x_coords = x_coord_row.unsqueeze(0).expand(torch.Size((m, n))).unsqueeze(0)
    y_coords = y_coord_row.unsqueeze(1).expand(torch.Size((m, n))).unsqueeze(0)
    return Variable(torch.cat([x_coords, y_coords], 0))


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.model = nn.Sequential(
            # conv(3, 128, 5, 1, 2),
            # conv(128, 128, 3, 1, 1),
            # conv(128, 128, 4, 2, 1),
            # conv(128, 128, 4, 2, 1),
            # conv(128, 128, 4, 2, 1),
            conv(3, 128, 5, 2, 2),
            conv(128, 128, 3, 2, 1),
            conv(128, 128, 3, 2, 1),
            conv(128, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 1),
            # conv(3, 128, 4, 2, 2),
            # conv(128, 128, 4, 2, 1),
            # conv(128, 128, 4, 2, 1),
            # conv(128, 128, 4, 2, 1),
        )

    def forward(self, x):
        return self.model(x)


class FiLMBlock(nn.Module):
    def __init__(self):
        super(FiLMBlock, self).__init__()

    def forward(self, x, gamma, beta):
        beta = beta.view(x.size(0), x.size(1), 1, 1)
        gamma = gamma.view(x.size(0), x.size(1), 1, 1)

        x = gamma * x + beta

        return x


class ResBlock(nn.Module):
    def __init__(self, in_place, out_place):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_place, out_place, 1, 1, 0)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_place, out_place, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(out_place)
        self.film = FiLMBlock()
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, beta, gamma):
        x = self.conv1(x)
        x = self.relu1(x)
        identity = x

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.film(x, beta, gamma)
        x = self.relu2(x)

        x = x + identity

        return x


class Classifier(nn.Module):
    def __init__(self, prev_channels, n_classes):
        super(Classifier, self).__init__()

        self.conv = nn.Conv2d(prev_channels, 512, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.model = nn.Sequential(nn.Linear(512, 1024),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(1024, 1024),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(1024, n_classes))

    def forward(self, x):
        x = self.conv(x)
        feature = x
        x = self.global_max_pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.model(x)

        return x, feature


class FiLM(nn.Module):
    def __init__(self, backbone_name="resnet18", n_res_blocks=18, n_classes=1, n_channels=128, task_embedding_dim=128):
        super(FiLM, self).__init__()

        self.task_embedding_dim = task_embedding_dim

        self.film_generator = nn.Linear(
            task_embedding_dim, 2 * n_res_blocks * n_channels)
        self.feature_extractor = get_backbone(backbone_name=backbone_name,
                                              video_backbone=False,
                                              pretrained=False,
                                              drop_dim=2)
        self.res_blocks = nn.ModuleList()

        for _ in range(n_res_blocks):
            self.res_blocks.append(ResBlock(n_channels + 2, n_channels))

        self.classifier = Classifier(n_channels, n_classes)

        self.n_res_blocks = n_res_blocks
        self.n_channels = n_channels

    def forward(self, agent_obs, task_emb):

        sizes = parse_shape(agent_obs, 'B T _ _ _')

        agent_obs = rearrange(agent_obs, "B T C H W -> (B T) C H W")
        agent_obs_feat = self.feature_extractor(agent_obs)  # B*T, C, H, W
        # agent_obs_feat = rearrange(
        #     agent_obs_feat, "(B T) C H W -> B T C H W", **sizes)

        film_vector = self.film_generator(task_emb).view(
            sizes['B'], self.n_res_blocks, 2, self.n_channels)  # B N_RES 2(alpha, beta) N_CHANNELS

        h = agent_obs_feat.size(2)
        w = agent_obs_feat.size(3)
        coords = coord_map((h, w))[None]  # B 2 h w

        for i, res_block in enumerate(self.res_blocks):
            beta = film_vector[:, i, 0, :]
            gamma = film_vector[:, i, 1, :]

            if i == 0:
                x = agent_obs_feat
            x = torch.cat([x, coords], 1)
            x = res_block(x, beta, gamma)

        cond_feature = x
        # x = self.classifier(x)

        return cond_feature


def make_model(model_dict, backbone_name="resnet18", task_embedding_dim=128):
    return FiLM(
        backbone_name=backbone_name,
        n_res_blocks=model_dict['n_res_blocks'],
        n_classes=model_dict['n_classes'],
        n_channels=model_dict['n_channels'],
        task_embedding_dim=task_embedding_dim)


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
        # 1. Compute features for each frame in the batch
        sizes = parse_shape(input, 'B T _ _ _')
        backbone_input = rearrange(input, 'B T C H W -> (B T) C H W')
        backbone_out = self._backbone(backbone_input)
        backbone_out = rearrange(
            backbone_out, '(B T) C H W -> B T C H W', **sizes)
        backbone_out = rearrange(backbone_out, 'B T C H W -> B C T H W')
        temp_conv_out = self._3d_conv(backbone_out)
        temp_conv_out = rearrange(temp_conv_out, 'B C T H W -> B T C H W')
        linear_input = torch.flatten(temp_conv_out, start_dim=1)
        task_embedding = self._mlp_encoder(linear_input)
        # print(task_embedding.shape)
        return task_embedding


class AgentModule(nn.Module):

    def __init__(self, height=120, width=160, obs_T=4, model_name="resnet18", pretrained=False, load_film=True, n_res_blocks=6, n_classes=1, n_channels=512, task_embedding_dim=128):
        super().__init__()
        if not load_film:
            self._module = get_backbone(backbone_name=model_name,
                                        video_backbone=False,
                                        pretrained=pretrained)
        else:
            # Create model with backbone + film
            model_dict = dict()
            model_dict['n_res_blocks'] = n_res_blocks
            model_dict['n_classes'] = n_classes
            model_dict['n_channels'] = n_channels
            backbone = make_model(model_dict=model_dict,
                                  task_embedding_dim=task_embedding_dim)
            backbone.out_channels = n_channels
            summary(backbone)
            # Add FasterRCNN head
            # We create a total of 9 anchors
            anchor_generator = AnchorGenerator(
                sizes=((4, 8, 16, 32),),
                aspect_ratios=((0.5, 1.0, 2.0),)
            )

            # Feature maps to perform RoI cropping.
            # If backbone returns a Tensor, `featmap_names` is expected to
            # be [0]. We can choose which feature maps to use.
            roi_pooler = MultiScaleRoIAlign(
                featmap_names=['0'],
                output_size=4,
                sampling_ratio=2
            )

            # Final Faster RCNN model.
            self._model = FasterRCNN(
                backbone=backbone,
                num_classes=n_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler
            )

            self._model.train()

        self.load_film = load_film

    def forward(self, agent_obs, task_embedding, gt_bb=None):
        if self.load_film:
            # # 1. Compute conditioned embedding
            # cond_feature = self._backbone(
            #     agent_obs=agent_obs, task_emb=task_embedding)
            # 2. Predict bounding boxes given conditioned embedding and input image
            predictions = self._model(agent_obs, gt_bb)
            return predictions
        else:
            scene_embegging = self._module(input)
            # print(scene_embegging.shape)
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
        # print(f"Cond embedding shape: {cond_emb.shape}")
        agent_emb = self._agent_backone(agent_obs, cond_emb)
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
        (4, 3, 200, 360),  dtype=torch.float).to('cuda:0')[None]  # B, T, C, W, H

    inputs['images'] = torch.rand(
        (1, 3, 200, 360),  dtype=torch.float).to('cuda:0')[None]  # B, T, C, W, H

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
