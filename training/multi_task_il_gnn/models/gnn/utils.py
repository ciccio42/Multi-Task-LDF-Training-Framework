import torch
import torch.nn as nn
import multi_task_il_gnn.models.gnn.ops as ops
import torch.nn.functional as F


class SingleHop(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.proj_q = ops.Linear(cfg.ENC_DIM, cfg.CTX_DIM)
        self.inter2att = ops.Linear(cfg.CTX_DIM, 1)

    def forward(self, kb, c_vect, entitiesnum):
        proj_q = self.proj_q(c_vect)
        interactions = F.normalize(kb * proj_q[:, None, :], dim=-1)
        raw_att = self.inter2att(interactions).squeeze(-1)
        raw_att = ops.apply_mask1d(raw_att, entitiesnum)
        att = F.softmax(raw_att, dim=-1)

        x_att = torch.bmm(att[:, None, :], kb).squeeze(1)
        return x_att


class Classifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.outQuestion = ops.Linear(cfg.CMD_DIM, cfg.CTX_DIM)
        in_dim = 3 * cfg.CTX_DIM if cfg.OUT_QUESTION_MUL else 2 * cfg.CTX_DIM
        self.classifier_layer = nn.Sequential(
            nn.Dropout(1 - cfg.outputDropout),
            ops.Linear(in_dim, cfg.OUT_CLASSIFIER_DIM),
            nn.ELU(),
            nn.Dropout(1 - cfg.outputDropout),
            ops.Linear(cfg.OUT_CLASSIFIER_DIM, self.cfg.NUM_CLASSES))

    def forward(self, x_att, vecQuestions):
        eQ = self.outQuestion(vecQuestions)
        if self.cfg.OUT_QUESTION_MUL:
            features = torch.cat([x_att, eQ, x_att*eQ], dim=-1)
        else:
            features = torch.cat([x_att, eQ], dim=-1)
        logits = self.classifier_layer(features)
        return logits


class NodeClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        input_dim = 2*cfg.INPUT_DIM if cfg.CONCAT_C_VEC else cfg.INPUT_DIM

        mlp_classifier = []
        for indx, layer_dim in enumerate(cfg.INTERMEDIATE_LAYER_DIM):
            if indx == 0:
                input_dim = input_dim
            else:
                input_dim = cfg.INTERMEDIATE_LAYER_DIM[indx-1]
            mlp_classifier.append(nn.Linear(in_features=input_dim,
                                            out_features=layer_dim))
            mlp_classifier.append(nn.ReLU())

        mlp_classifier.append(nn.Linear(in_features=cfg.INTERMEDIATE_LAYER_DIM[-1],
                                        out_features=cfg.OUT_CLASSES))

        self.mlp_classifier = nn.Sequential(*mlp_classifier)

    def forward(self, x_nodes, c_vect):

        if self.cfg.CONCAT_C_VEC:
            input = torch.cat(
                [x_nodes, c_vect[:, None, :].repeat(1, x_nodes.shape[1], 1)], dim=-1)
        else:
            input = x_nodes

        nodes_logits = self.mlp_classifier(input)
        return nodes_logits
