import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat
from collections import OrderedDict
import hydra

from torchsummary import summary
import torch

from multi_task_il.models.cond_target_obj_detector.utils import *
from multi_task_il.models.cond_target_obj_detector.cond_target_obj_detector import *
from omegaconf import OmegaConf
from multi_task_il.models.discrete_logistic import DiscreteMixLogistic

DEBUG = False


class _DiscreteLogHead(nn.Module):
    def __init__(self, in_dim, out_dim, n_mixtures, const_var=True, sep_var=False):
        super().__init__()
        assert n_mixtures >= 1, "must predict at least one mixture!"
        self._n_mixtures = n_mixtures
        self._dist_size = torch.Size((out_dim, n_mixtures))
        self._mu = nn.Linear(in_dim, out_dim * n_mixtures)
        self._logit_prob = nn.Linear(
            in_dim, out_dim * n_mixtures) if n_mixtures > 1 else None
        if const_var:
            ln_scale = torch.randn(
                out_dim, dtype=torch.float32) / np.sqrt(out_dim)
            self.register_parameter(
                '_ln_scale', nn.Parameter(ln_scale, requires_grad=True))
        if sep_var:
            ln_scale = torch.randn((out_dim, n_mixtures),
                                   dtype=torch.float32) / np.sqrt(out_dim)
            self.register_parameter(
                '_ln_scale', nn.Parameter(ln_scale, requires_grad=True))
        if not (const_var or sep_var):
            self._ln_scale = nn.Linear(in_dim, out_dim * n_mixtures)

    def forward(self, x):  #  x has shape B T d
        mu = self._mu(x).reshape((x.shape[:-1] + self._dist_size))

        if isinstance(self._ln_scale, nn.Linear):
            ln_scale = self._ln_scale(x).reshape(
                (x.shape[:-1] + self._dist_size))
        else:
            ln_scale = self._ln_scale if self.training else self._ln_scale.detach()
            if len(ln_scale.shape) == 1:
                ln_scale = ln_scale.reshape((1, 1, -1, 1)).expand_as(mu)
                # (1, 1, 8, 1) -> (B T, dist_size[0], dist_size[1]) i.e. each mixture has the **same** constant variance
            else:  # the sep_val case:
                ln_scale = repeat(
                    ln_scale, 'out_d n_mix -> B T out_d n_mix', B=x.shape[0], T=x.shape[1])

        logit_prob = self._logit_prob(x).reshape(
            mu.shape) if self._n_mixtures > 1 else torch.ones_like(mu)
        return (mu, ln_scale, logit_prob)


class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width):
        super(SpatialSoftmax, self).__init__()
        self.height = height
        self.width = width

    def forward(self, x):
        b, c, h, w = x.size()
        # reshape input tensor to (b, c, h * w)
        x = x.view(b, c, -1)
        # apply softmax along the last dimension
        x = F.softmax(x, dim=2)
        # reshape back to original shape
        x = x.view(b, c, h, w)
        return x


class CondPolicy(nn.Module):

    def __init__(self,
                 freeze_target_detector=True,
                 concat_state=True,
                 cond_target_obj_detector_pretrained=True,
                 cond_target_obj_detector_weights=None,
                 cond_target_obj_detector_step=4050,
                 action_cfg=None,
                 concat_bb=True,
                 mlp_layers=[512, 256, 128],
                 pooling=False,
                 avg_pooling=False
                 ):

        super().__init__()

        # 1. Istantiate cond_target_obj_detector

        # Load configuration file
        conf_file = OmegaConf.load(os.path.join(
            cond_target_obj_detector_weights, "config.yaml"))

        self.cond_target_obj_detector = hydra.utils.instantiate(
            conf_file.cond_target_obj_detector)

        # 2. Load cond_target_obj_detector weights
        if cond_target_obj_detector_pretrained:
            weights_path = os.path.join(
                cond_target_obj_detector_weights, f"model_save-{cond_target_obj_detector_step}.pt")
            print(
                f"Loading Cond-Target-Obj-Detector from {weights_path}")
            cond_target_obj_detector_state_dict = torch.load(
                weights_path, map_location=torch.device('cuda:0'))
            self.cond_target_obj_detector.load_state_dict(
                cond_target_obj_detector_state_dict)

        if freeze_target_detector:
            # Freeze Conditioned Target Object Detector
            for param in self.cond_target_obj_detector.parameters():
                if param.requires_grad:
                    param.requires_grad = False

        # Create Policy NN
        self.pooling_layer = None
        self.spatial_softmax = None
        latent_dim = 0
        if pooling and not avg_pooling:
            pass
        elif pooling and avg_pooling:
            # for each feature-point (i,j) perform average over channels
            # Get a tensor of shape B, T, 1, dim_H, dim_W
            self.avg_pooling = torch.mean
            # apply spatial softmax
            self.spatial_softmax = SpatialSoftmax(height=conf_file.cond_target_obj_detector_cfg.dim_H,
                                                  width=conf_file.cond_target_obj_detector_cfg.dim_W)
            latent_dim = conf_file.cond_target_obj_detector_cfg.dim_W * \
                conf_file.cond_target_obj_detector_cfg.dim_H

        # Compute action_module_input
        action_module_input = int(
            latent_dim + float(concat_state) * action_cfg.sdim + float(concat_bb) * 4)
        # action module
        self.action_module = None
        action_module_mlp = []
        for indx in range(len(mlp_layers)):
            if indx == 0:
                action_module_mlp.append(
                    nn.Linear(action_module_input, mlp_layers[indx]))
                action_module_mlp.append(nn.ReLU())
            else:
                action_module_mlp.append(
                    nn.Linear(mlp_layers[indx-1], mlp_layers[indx]))
                action_module_mlp.append(nn.ReLU())

        self.action_module = nn.Sequential(
            *action_module_mlp)

        head_in_dim = mlp_layers[-1]
        self._action_dist = _DiscreteLogHead(
            in_dim=head_in_dim,
            out_dim=action_cfg.adim,
            n_mixtures=action_cfg.n_mixtures,
            const_var=action_cfg.const_var,
            sep_var=action_cfg.sep_var
        )

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Total params in Imitation module:', params)
        print("\n\n---- Complete model ----\n")
        summary(self)

    def forward(self, inputs: dict, inference: bool = False):
        out = dict()

        # get target object prediction
        prediction_target_obj_detector = self.cond_target_obj_detector(inputs=inputs,
                                                                       inference=True)

        # Reshape BBs B*T, 4 to B,T,4
        bbs = torch.stack(prediction_target_obj_detector['proposals'])
        B, T, _, _, _ = inputs['images'].shape
        bbs = rearrange(bbs, '(B T) N -> B T N', B=B, T=T, N=4)

        # 1. Get last layer feature maps
        # B*T, 512, 7, 7
        last_layer_feature = prediction_target_obj_detector['feature_map']

        # 2. Compute average pooling channels wise
        # B*T, 1, 7, 7
        average_pooling = self.avg_pooling(last_layer_feature,
                                           dim=1, keepdim=True)
        # 3. Compute SpatialSoftmax
        spatial_softmax_out = self.spatial_softmax(average_pooling)

        # 4. Flat the vector
        # B*T, 1, 7, 7 -> B*T, 49
        spatial_softmax_out_flat = torch.flatten(
            spatial_softmax_out, start_dim=1)
        spatial_softmax_out_flat = rearrange(spatial_softmax_out_flat,
                                             '(B T) N -> B T N', B=B, T=T)

        # 5. Create action_in vector
        # reshape states
        states = inputs['states']
        # get the bb with the highest conf score
        action_in = torch.concat(
            [spatial_softmax_out_flat, states, bbs], dim=2).to(torch.float32)
        action_in = F.normalize(action_in, dim=2).to(torch.float32)
        # 6. Infer action embedding
        ac_pred = self.action_module(action_in)
        mu_bc, scale_bc, logit_bc = self._action_dist(ac_pred)

        out['bc_distrib'] = (mu_bc, scale_bc, logit_bc) \
            if not inference else DiscreteMixLogistic(mu_bc, scale_bc, logit_bc)
        out['prediction_target_obj_detector'] = prediction_target_obj_detector
        return out


@hydra.main(
    version_base=None,
    config_path="../../../experiments",
    config_name="config_cond_target_obj_detector.yaml")
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

    inputs['states'] = torch.rand(
        (1, 6),  dtype=torch.float).to('cuda:0')[None]

    inputs['gt_bb'] = torch.rand(
        (1, 1, 4),  dtype=torch.float).to('cuda:0')[None]

    inputs['gt_classes'] = torch.rand(
        (1, 1),  dtype=torch.float).to('cuda:0')[None]

    # module = CondPolicy(freeze_target_detector=cfg.policy.freeze_target_detector,
    #                     concat_state=cfg.policy.concat_state,
    #                     cond_target_obj_detector_cfg=cfg.policy.cond_target_obj_detector_cfg,
    #                     action_cfg=cfg.policy.action_cfg,
    #                     concat_bb=cfg.policy.concat_bb,
    #                     mlp_layers=cfg.policy.mlp_layers,
    #                     pooling=cfg.policy.pooling,
    #                     avg_pooling=cfg.policy.avg_pooling,
    #                     )

    module = hydra.utils.instantiate(cfg.policy)

    module.to('cuda:0')
    module(inputs, inference=False)


if __name__ == '__main__':
    main()
