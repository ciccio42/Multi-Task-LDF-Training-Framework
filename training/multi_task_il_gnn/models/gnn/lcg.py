from colorama import Style
from colorama import Fore
from colorama import init as colorama_init
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import multi_task_il_gnn.models.gnn.ops as ops
from multi_task_il_gnn.models.gnn.utils import SingleHop, Classifier, NodeClassifier
from torch_geometric.data import Data

colorama_init()


class LCGN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.build_loc_ctx_init()
        self.build_extract_textual_command()
        self.build_propagate_message()

    def build_loc_ctx_init(self):
        assert self.cfg.STEM_LINEAR != self.cfg.STEM_CNN
        if self.cfg.STEM_LINEAR:
            self.initKB = ops.Linear(self.cfg.D_FEAT, self.cfg.CTX_DIM)
            self.x_loc_drop = nn.Dropout(1 - self.cfg.stemDropout)
        elif self.cfg.STEM_CNN:
            self.cnn = nn.Sequential(
                nn.Dropout(1 - self.cfg.stemDropout),
                ops.Conv(self.cfg.D_FEAT,
                         self.cfg.STEM_CNN_DIM,
                         (3, 3),
                         padding=1),
                nn.ELU(),
                nn.Dropout(1 - self.cfg.stemDropout),
                ops.Conv(self.cfg.STEM_CNN_DIM,
                         self.cfg.CTX_DIM,
                         (3, 3),
                         padding=1),
                nn.ELU())

        self.initMem = nn.Parameter(torch.randn(1, 1, self.cfg.CTX_DIM))

    def build_extract_textual_command(self):
        self.qInput = ops.Linear(self.cfg.CMD_DIM, self.cfg.CMD_DIM)
        for t in range(self.cfg.MSG_ITER_NUM):
            qInput_layer2 = ops.Linear(self.cfg.CMD_DIM, self.cfg.CMD_DIM)
            setattr(self, "qInput%d" % t, qInput_layer2)
        self.cmd_inter2logits = ops.Linear(self.cfg.CMD_DIM, 1)

    def build_propagate_message(self):
        self.read_drop = nn.Dropout(1 - self.cfg.readDropout)
        self.project_x_loc = ops.Linear(self.cfg.CTX_DIM, self.cfg.CTX_DIM)
        self.project_x_ctx = ops.Linear(self.cfg.CTX_DIM, self.cfg.CTX_DIM)
        self.queries = ops.Linear(3*self.cfg.CTX_DIM, self.cfg.CTX_DIM)
        self.keys = ops.Linear(3*self.cfg.CTX_DIM, self.cfg.CTX_DIM)
        self.vals = ops.Linear(3*self.cfg.CTX_DIM, self.cfg.CTX_DIM)
        self.proj_keys = ops.Linear(self.cfg.CMD_DIM, self.cfg.CTX_DIM)
        self.proj_vals = ops.Linear(self.cfg.CMD_DIM, self.cfg.CTX_DIM)
        self.mem_update = ops.Linear(2*self.cfg.CTX_DIM, self.cfg.CTX_DIM)
        self.combine_kb = ops.Linear(2*self.cfg.CTX_DIM, self.cfg.CTX_DIM)

    def forward(self, input, c_vect, batch_size):
        if isinstance(input, Data):
            local_features = input.x.to(torch.float32)[
                None]  # B, N_nodes, Features
            B, N, F = local_features.shape
            entity_num = torch.from_numpy(
                np.array([N]).astype(np.int64)).cuda()
        else:
            B, N, F = input.shape
            local_features = input.to(torch.float32)
            entity_num = torch.from_numpy(
                np.array([N]).astype(np.int64)).cuda()
        # initialize vector embeddings
        x_loc, x_ctx, x_ctx_var_drop = self.loc_ctx_init(local_features)
        for t in range(self.cfg.MSG_ITER_NUM):
            x_ctx = self.run_message_passing_iter(
                x_loc=x_loc,
                x_ctx=x_ctx,
                x_ctx_var_drop=x_ctx_var_drop,
                t=t,
                cmd=c_vect,
                entity_num=entity_num)
        x_out = self.combine_kb(torch.cat([x_loc, x_ctx], dim=-1))
        return x_out

    def extract_textual_command(self, q_encoding, lstm_outputs, q_length, t):
        qInput_layer2 = getattr(self, "qInput%d" % t)
        act_fun = ops.activations[self.cfg.CMD_INPUT_ACT]
        q_cmd = qInput_layer2(act_fun(self.qInput(q_encoding)))
        raw_att = self.cmd_inter2logits(
            q_cmd[:, None, :] * lstm_outputs).squeeze(-1)
        raw_att = ops.apply_mask1d(raw_att, q_length)
        att = F.softmax(raw_att, dim=-1)
        cmd = torch.bmm(att[:, None, :], lstm_outputs).squeeze(1)
        return cmd

    def propagate_message(self, cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num):
        x_ctx = x_ctx * x_ctx_var_drop
        proj_x_loc = self.project_x_loc(self.read_drop(x_loc))
        proj_x_ctx = self.project_x_ctx(self.read_drop(x_ctx))
        x_joint = torch.cat(
            [x_loc, x_ctx, proj_x_loc * proj_x_ctx], dim=-1)

        queries = self.queries(x_joint)
        keys = self.keys(x_joint) * self.proj_keys(cmd)[:, None, :]
        vals = self.vals(x_joint) * self.proj_vals(cmd)[:, None, :]
        edge_score = (
            torch.bmm(queries, torch.transpose(keys, 1, 2)) /
            np.sqrt(self.cfg.CTX_DIM))
        edge_score = ops.apply_mask2d(edge_score, entity_num)
        edge_prob = F.softmax(edge_score, dim=-1)
        message = torch.bmm(edge_prob, vals)

        x_ctx_new = self.mem_update(torch.cat([x_ctx, message], dim=-1))
        return x_ctx_new

    def run_message_passing_iter(
            self, x_loc, x_ctx, x_ctx_var_drop, t, cmd, entity_num):
        # cmd = self.extract_textual_command(
        #     q_encoding, lstm_outputs, q_length, t)
        x_ctx = self.propagate_message(
            cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num)
        return x_ctx

    def loc_ctx_init(self, images):
        if self.cfg.STEM_NORMALIZE:
            images = F.normalize(images, dim=-1)
        if self.cfg.STEM_LINEAR:
            x_loc = self.initKB(images)
            x_loc = self.x_loc_drop(x_loc)
        elif self.cfg.STEM_CNN:
            images = torch.transpose(images, 1, 2)  # N(HW)C => NC(HW)
            x_loc = images.view(-1, self.cfg.D_FEAT,
                                self.cfg.H_FEAT, self.cfg.W_FEAT)
            x_loc = self.cnn(x_loc)
            x_loc = x_loc.view(-1, self.cfg.CTX_DIM,
                               self.cfg.H_FEAT * self.cfg.W_FEAT)
            x_loc = torch.transpose(x_loc, 1, 2)  # NC(HW) => N(HW)C
        if self.cfg.STEM_RENORMALIZE:
            x_loc = F.normalize(x_loc, dim=-1)

        x_ctx = self.initMem.expand(x_loc.size())
        x_ctx_var_drop = ops.generate_scaled_var_drop_mask(
            x_ctx.size(),
            keep_prob=(self.cfg.memoryDropout if self.training else 1.))
        return x_loc, x_ctx, x_ctx_var_drop


class LCGNnet(nn.Module):
    def __init__(self, cfg):

        super().__init__()
        self.cfg = cfg
        self.lcgn = LCGN(cfg=cfg)
        if cfg.BUILD_VQA:
            self.single_hop = SingleHop(cfg=cfg.single_hop_cfg)
            print(f"{Fore.YELLOW} Build SingleHop module{Style.RESET_ALL}")
            self.classifier = Classifier(cfg=cfg.classifier_cfg)
            print(f"{Fore.YELLOW} Build Classifier{Style.RESET_ALL}")
        if cfg.BUILD_NODE_CLASSIFIER:
            self.object_classifier = NodeClassifier(
                cfg=cfg.node_classifier_cfg)
            self.target_classifier = NodeClassifier(
                cfg=cfg.node_classifier_cfg)
            print(f"{Fore.YELLOW} Build Node Classifier{Style.RESET_ALL}")

        # if cfg.BUILD_REF:
        #     self.grounder = GroundeR()
        #     self.bbox_regression = BboxRegression()

    def forward(self, input, c_vect, run_vqa, run_ref, run_node_classifier):
        B, D = c_vect.shape
        if isinstance(input, Data):
            local_features = input.x.to(torch.float32)[
                None]  # B, N_nodes, Features
        else:
            local_features = input.to(torch.float32)

        B, N, F = local_features.shape
        entity_num = torch.from_numpy(
            np.array([N]).astype(np.int64)).cuda()
        # LCGN
        x_out = self.lcgn(
            input=local_features,
            c_vect=c_vect,
            batch_size=B
        )
        if run_vqa:
            x_att = self.single_hop(x_out, c_vect, entity_num)
            logits = self.classifier(x_att, c_vect)
            return logits
        if run_ref:
            assert "Not Implemented Error"
            # def add_pred_op(self, logits, answers):
            #     if cfg.MASK_PADUNK_IN_LOGITS:
            #         logits = logits.clone()
            #         logits[..., :2] += -1e30  # mask <pad> and <unk>

            #     preds = torch.argmax(logits, dim=-1).detach()
            #     corrects = (preds == answers)
            #     correctNum = torch.sum(corrects).item()
            #     preds = preds.cpu().numpy()

            #     return preds, correctNum

            # def add_answer_loss_op(self, logits, answers):
            #     if cfg.TRAIN.LOSS_TYPE == "softmax":
            #         loss = F.cross_entropy(logits, answers)
            #     elif cfg.TRAIN.LOSS_TYPE == "sigmoid":
            #         answerDist = F.one_hot(answers, self.num_choices).float()
            #         loss = F.binary_cross_entropy_with_logits(
            #             logits, answerDist) * self.num_choices
            #     else:
            #         raise Exception("non-identified loss")
            #     return loss

            # def add_bbox_loss_op(self, ref_scores, bbox_offset_fcn, bbox_ind_gt,
            #                      bbox_offset_gt):
            #     # bounding box selection loss
            #     bbox_ind_loss = torch.mean(
            #         F.cross_entropy(ref_scores, bbox_ind_gt))

            #     # bounding box regression loss
            #     N = bbox_offset_fcn.size(0)
            #     M = bbox_offset_fcn.size(1)
            #     bbox_offset_flat = bbox_offset_fcn.view(-1, 4)
            #     slice_inds = (
            #         torch.arange(N, device=ref_scores.device) * M + bbox_ind_gt)
            #     bbox_offset_sliced = bbox_offset_flat[slice_inds]
            #     bbox_offset_loss = F.mse_loss(bbox_offset_sliced, bbox_offset_gt)

            #     return bbox_ind_loss, bbox_offset_loss
        if run_node_classifier:
            obj_logits = self.object_classifier(x_out, c_vect)
            target_logits = self.target_classifier(x_out, c_vect)
            return obj_logits, target_logits
