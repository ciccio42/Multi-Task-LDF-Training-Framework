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
from torchvision import ops
from utils import *


def get_backbone(backbone_name="slow_r50", video_backbone=True, pretrained=False, drop_dim=3):
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
    def __init__(self, backbone_name="resnet18", drop_dim=3, n_res_blocks=18, n_classes=1, n_channels=128, task_embedding_dim=128):
        super(FiLM, self).__init__()

        self.task_embedding_dim = task_embedding_dim

        self.film_generator = nn.Linear(
            task_embedding_dim, 2 * n_res_blocks * n_channels)
        self.feature_extractor = get_backbone(backbone_name=backbone_name,
                                              video_backbone=False,
                                              pretrained=False,
                                              drop_dim=drop_dim)
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


class ProposalModule(nn.Module):
    def __init__(self, in_features, hidden_dim=512, n_anchors=9, p_dropout=0.3):
        super().__init__()
        self.n_anchors = n_anchors
        self.conv1 = nn.Conv2d(in_features, hidden_dim,
                               kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p_dropout)
        self.conf_head = nn.Conv2d(hidden_dim, n_anchors, kernel_size=1)
        self.reg_head = nn.Conv2d(hidden_dim, n_anchors * 4, kernel_size=1)

    def forward(self, feature_map, pos_anc_ind=None, neg_anc_ind=None, pos_anc_coords=None):
        # determine mode
        if pos_anc_ind is None or neg_anc_ind is None or pos_anc_coords is None:
            mode = 'eval'
        else:
            mode = 'train'

        out = self.conv1(feature_map)
        out = F.relu(self.dropout(out))

        reg_offsets_pred = self.reg_head(out)  # (B, A*4, hmap, wmap)
        conf_scores_pred = self.conf_head(out)  # (B, A, hmap, wmap)

        if mode == 'train':
            # get conf scores
            conf_scores_pos = conf_scores_pred.flatten()[pos_anc_ind]
            conf_scores_neg = conf_scores_pred.flatten()[neg_anc_ind]
            # get offsets for +ve anchors
            offsets_pos = reg_offsets_pred.contiguous().view(-1,
                                                             4)[pos_anc_ind]
            # generate proposals using offsets
            proposals = generate_proposals(pos_anc_coords, offsets_pos)

            return conf_scores_pos, conf_scores_neg, offsets_pos, proposals

        elif mode == 'eval':
            return conf_scores_pred, reg_offsets_pred


def make_model(model_dict, backbone_name="resnet18", task_embedding_dim=128, drop_dim=3):
    return FiLM(
        backbone_name=backbone_name,
        drop_dim=drop_dim,
        n_res_blocks=model_dict['n_res_blocks'],
        n_classes=model_dict['n_classes'],
        n_channels=model_dict['n_channels'],
        task_embedding_dim=task_embedding_dim)


class CondModule(nn.Module):

    def __init__(self, height=120, width=160, demo_T=4, model_name="slow_r50", pretrained=False, cond_video=True, n_layers=3, demo_W=14, demo_H=14, demo_ff_dim=[128, 64, 32], demo_linear_dim=[512, 256, 128], drop_dim=3):
        super().__init__()
        self._demo_T = demo_T
        self._cond_video = cond_video

        self._backbone = get_backbone(backbone_name=model_name,
                                      video_backbone=cond_video,
                                      pretrained=pretrained,
                                      drop_dim=drop_dim)

        if not cond_video:
            conv_layer = []
            if drop_dim == 2:
                input_dim_0 = 512
            elif drop_dim == 3:
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

    def __init__(self, height=120, width=160, obs_T=4, model_name="resnet18", pretrained=False, load_film=True, n_res_blocks=6, n_classes=1, n_channels=512, task_embedding_dim=128, dim_H=14, dim_W=14, drop_dim=3):
        super().__init__()
        if not load_film:
            self._module = get_backbone(backbone_name=model_name,
                                        video_backbone=False,
                                        pretrained=pretrained,
                                        drop_dim=drop_dim)
        else:
            #### Create model with backbone + film ####
            model_dict = dict()
            model_dict['n_res_blocks'] = n_res_blocks
            model_dict['n_classes'] = n_classes
            if drop_dim == 2:
                n_channels = 512
            elif drop_dim == 3:
                n_channels = 256
            model_dict['n_channels'] = n_channels
            backbone = make_model(model_dict=model_dict,
                                  task_embedding_dim=task_embedding_dim)
            backbone.out_channels = n_channels
            self.out_channels_backbone = n_channels
            self._backbone = backbone
            print("---- Summary backbone with FiLM layer ----")
            summary(self._backbone)

            #### Create Region Proposal Network ####
            self.img_height, self.img_width = height, width
            self.out_h, self.out_w = dim_H, dim_W

            # downsampling scale factor
            self.width_scale_factor = self.img_width // self.out_w
            self.height_scale_factor = self.img_height // self.out_h

            # scales and ratios for anchor boxes
            self.anc_scales = [4, 8, 16, 32]
            self.anc_ratios = [0.5, 1.0, 2.0]
            self.n_anc_boxes = len(self.anc_scales) * len(self.anc_ratios)

            # IoU thresholds for +ve and -ve anchors
            self.pos_thresh = 0.7
            self.neg_thresh = 0.3

            # weights for loss
            self.w_conf = 1
            self.w_reg = 5

            self.conf_thresh = 0.5
            self.nms_thresh = 0.7

            self.proposal_module = ProposalModule(
                self.out_channels_backbone, n_anchors=self.n_anc_boxes)

        self.load_film = load_film

    def forward(self, agent_obs, task_embedding, gt_bb=None, gt_classes=None, validation=False):

        ret_dict = dict()

        # 1. Compute conditioned embedding
        feature_map = self._backbone(
            agent_obs=agent_obs, task_emb=task_embedding)

        # 2. Predict bounding boxes given conditioned embedding and input image
        B, T, C, H, W = agent_obs.shape

        if self.load_film:
            # generate anchors
            anc_pts_x, anc_pts_y = gen_anc_centers(
                out_size=(self.out_h, self.out_w))
            anc_base = gen_anc_base(
                anc_pts_x, anc_pts_y, self.anc_scales, self.anc_ratios, (self.out_h, self.out_w))
            anc_boxes_all = anc_base.repeat(B, 1, 1, 1, 1)

            if not validation:
                # if the model is training
                # get positive and negative anchors amongst other things
                gt_bboxes_proj = project_bboxes(
                    gt_bb,
                    self.width_scale_factor,
                    self.height_scale_factor,
                    mode='p2a')

                positive_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class_pos, positive_anc_coords, negative_anc_coords, positive_anc_ind_sep = get_req_anchors(
                    anc_boxes_all,
                    gt_bboxes_proj,
                    gt_classes)

                # pass through the proposal module
                conf_scores_pos, conf_scores_neg, offsets_pos, proposals = self.proposal_module(
                    feature_map,
                    positive_anc_ind,
                    negative_anc_ind,
                    positive_anc_coords)

                ret_dict['feature_map'] = feature_map
                ret_dict['proposals'] = proposals

                return ret_dict
            else:
                # model is in validation/inference mode
                with torch.no_grad():
                    # generate anchors
                    anc_pts_x, anc_pts_y = gen_anc_centers(
                        out_size=(self.out_h, self.out_w))
                    anc_base = gen_anc_base(
                        anc_pts_x, anc_pts_y, self.anc_scales, self.anc_ratios, (self.out_h, self.out_w))
                    anc_boxes_all = anc_base.repeat(B, 1, 1, 1, 1)
                    anc_boxes_flat = anc_boxes_all.reshape(B, -1, 4)

                    # get conf scores and offsets
                    conf_scores_pred, offsets_pred = self.proposal_module(
                        feature_map)
                    conf_scores_pred = conf_scores_pred.reshape(B, -1)
                    offsets_pred = offsets_pred.reshape(B, -1, 4)

                    # filter out proposals based on conf threshold and nms threshold for each image
                    proposals_final = []
                    conf_scores_final = []
                    for i in range(B):
                        conf_scores = torch.sigmoid(conf_scores_pred[i])
                        offsets = offsets_pred[i]
                        anc_boxes = anc_boxes_flat[i]
                        proposals = generate_proposals(anc_boxes, offsets)
                        # filter based on confidence threshold
                        conf_idx = torch.where(
                            conf_scores >= self.conf_thresh)[0]
                        conf_scores_pos = conf_scores[conf_idx]
                        proposals_pos = proposals[conf_idx]
                        # filter based on nms threshold
                        nms_idx = ops.nms(
                            proposals_pos, conf_scores_pos, self.nms_thresh)
                        conf_scores_pos = conf_scores_pos[nms_idx]
                        proposals_pos = proposals_pos[nms_idx]

                        proposals_final.append(proposals_pos)
                        conf_scores_final.append(conf_scores_pos)

                        ret_dict['proposals'] = proposals_final
                        ret_dict['conf_scores_final'] = conf_scores_final

                        return ret_dict

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

        # summary(self)

    def forward(self, inputs: dict, validation=False):
        cond_video = inputs['demo']
        agent_obs = inputs['images']

        cond_emb = self._cond_backbone(cond_video)
        # print(f"Cond embedding shape: {cond_emb.shape}")
        ret_dict = self._agent_backone(
            agent_obs, cond_emb, validation=validation)
        # print(agent_emb.shape)


@hydra.main(
    version_base=None,
    config_path="../../../experiments",
    config_name="target_object_detector_config.yaml")
def main(cfg):
    import debugpy
    debugpy.listen(('0.0.0.0', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    inputs = dict()

    width = 224
    height = 224
    demo_T = 4
    inputs['demo'] = torch.rand(
        (4, 3, height, width),  dtype=torch.float).to('cuda:0')[None]  # B, T, C, W, H

    inputs['images'] = torch.rand(
        (1, 3, height, width),  dtype=torch.float).to('cuda:0')[None]  # B, T, C, W, H

    module = CondTargetObjectDetector(
        width=width,
        height=height,
        demo_T=demo_T,
        cond_backbone_name="resnet18",
        cond_video=False)

    module.to('cuda:0')
    module(inputs, validation=True)


if __name__ == '__main__':
    main()