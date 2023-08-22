import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat
from collections import OrderedDict
import hydra

from torchsummary import summary
import torch
import torch.nn.functional as F
import torch.distributions as D

from multi_task_il.models.cond_target_obj_detector.utils import *
from multi_task_il.models.cond_target_obj_detector.cond_target_obj_detector import *
from omegaconf import OmegaConf
from multi_task_il.models.discrete_logistic import DiscreteMixLogistic
from multi_task_il.models import get_class_activation_map

DEBUG = False


class TanhWrappedDistribution(D.Distribution):
    """
    Class that wraps another valid torch distribution, such that sampled values from the base distribution are
    passed through a tanh layer. The corresponding (log) probabilities are also modified accordingly.
    Tanh Normal distribution - adapted from rlkit and CQL codebase
    (https://github.com/aviralkumar2907/CQL/blob/d67dbe9cf5d2b96e3b462b6146f249b3d6569796/d4rl/rlkit/torch/distributions.py#L6).
    """

    def __init__(self, base_dist, scale=1.0, epsilon=1e-6):
        """
        Args:
            base_dist (Distribution): Distribution to wrap with tanh output
            scale (float): Scale of output
            epsilon (float): Numerical stability epsilon when computing log-prob.
        """
        self.base_dist = base_dist
        self.scale = scale
        self.tanh_epsilon = epsilon
        super(TanhWrappedDistribution, self).__init__()

    def log_prob(self, value, pre_tanh_value=None):
        """
        Args:
            value (torch.Tensor): some tensor to compute log probabilities for
            pre_tanh_value: If specified, will not calculate atanh manually from @value. More numerically stable
        """
        value = value / self.scale
        if pre_tanh_value is None:
            one_plus_x = (1. + value).clamp(min=self.tanh_epsilon)
            one_minus_x = (1. - value).clamp(min=self.tanh_epsilon)
            pre_tanh_value = 0.5 * torch.log(one_plus_x / one_minus_x)
        lp = self.base_dist.log_prob(pre_tanh_value)
        tanh_lp = torch.log(1 - value * value + self.tanh_epsilon)
        # In case the base dist already sums up the log probs, make sure we do the same
        return lp - tanh_lp if len(lp.shape) == len(tanh_lp.shape) else lp - tanh_lp.sum(-1)

    def sample(self, sample_shape=torch.Size(), return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.base_dist.sample(sample_shape=sample_shape).detach()

        if return_pretanh_value:
            return torch.tanh(z) * self.scale, z
        else:
            return torch.tanh(z) * self.scale

    def rsample(self, sample_shape=torch.Size(), return_pretanh_value=False):
        """
        Sampling in the reparameterization case - for differentiable samples.
        """
        z = self.base_dist.rsample(sample_shape=sample_shape)

        if return_pretanh_value:
            return torch.tanh(z) * self.scale, z
        else:
            return torch.tanh(z) * self.scale

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def stddev(self):
        return self.base_dist.stddev


class MLP(torch.nn.Module):
    """
    Base class for simple Multi-Layer Perceptrons.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        layer_dims=(),
        layer_func=nn.Linear,
        layer_func_kwargs=None,
        activation=nn.ReLU,
        dropouts=None,
        normalization=False,
        output_activation=None,
    ):
        """
        Args:
            input_dim (int): dimension of inputs

            output_dim (int): dimension of outputs

            layer_dims ([int]): sequence of integers for the hidden layers sizes

            layer_func: mapping per layer - defaults to Linear

            layer_func_kwargs (dict): kwargs for @layer_func

            activation: non-linearity per layer - defaults to ReLU

            dropouts ([float]): if not None, adds dropout layers with the corresponding probabilities
                after every layer. Must be same size as @layer_dims.

            normalization (bool): if True, apply layer normalization after each layer

            output_activation: if provided, applies the provided non-linearity to the output layer
        """
        super(MLP, self).__init__()
        layers = []
        dim = input_dim
        if layer_func_kwargs is None:
            layer_func_kwargs = dict()
        if dropouts is not None:
            assert (len(dropouts) == len(layer_dims))
        for i, l in enumerate(layer_dims):
            layers.append(layer_func(dim, l, **layer_func_kwargs))
            if normalization:
                layers.append(nn.LayerNorm(l))
            layers.append(activation())
            if dropouts is not None and dropouts[i] > 0.:
                layers.append(nn.Dropout(dropouts[i]))
            dim = l
        layers.append(layer_func(dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())
        self._layer_func = layer_func
        self.nets = layers
        self._model = nn.Sequential(*layers)

        self._layer_dims = layer_dims
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._dropouts = dropouts
        self._act = activation
        self._output_act = output_activation


class GMM(nn.Module):
    def __init__(self,
                 input_dim,
                 ac_dim,
                 mlp_layer_dims=[128, 64, 32],
                 num_modes=5,
                 min_std=0.01,
                 std_activation="softplus",
                 low_noise_eval=True,
                 use_tanh=False,
                 goal_shapes=None,
                 encoder_kwargs=None,):
        super().__init__()

        # parameters specific to GMM actor
        self.num_modes = num_modes
        self.ac_dim = ac_dim
        self.min_std = torch.from_numpy(np.array(min_std)).to("cuda:0")
        self.low_noise_eval = low_noise_eval
        self.use_tanh = use_tanh

        # Define activations to use
        self.activations = {
            "softplus": F.softplus,
            "exp": torch.exp,
        }
        assert std_activation in self.activations, \
            "std_activation must be one of: {}; instead got: {}".format(
                self.activations.keys(), std_activation)
        self.std_activation = std_activation

        # action module
        self.action_module = None
        action_module_mlp = []
        for indx in range(len(mlp_layer_dims)):
            if indx == 0:
                action_module_mlp.append(
                    nn.Linear(input_dim, mlp_layer_dims[indx]))
                action_module_mlp.append(nn.ReLU())
            else:
                action_module_mlp.append(
                    nn.Linear(mlp_layer_dims[indx-1], mlp_layer_dims[indx]))
                action_module_mlp.append(nn.ReLU())
        self.action_module = nn.Sequential(*action_module_mlp)

        # Distributions
        self.nets = nn.ModuleDict()
        for k in ["mean", "scale", "logits"]:
            if k != "logits":
                self.nets[k] = nn.Linear(mlp_layer_dims[-1], num_modes*ac_dim)
            else:
                self.nets[k] = nn.Linear(mlp_layer_dims[-1], num_modes)

    def forward(self, inputs):

        # Input: B*T, 48, Out; B*T, 32
        action_module_out = self.action_module(inputs)

        out = OrderedDict()
        for k in ["mean", "scale", "logits"]:
            out[k] = self.nets[k](action_module_out)

        means = rearrange(out["mean"], 'BT (M A) -> BT M A',
                          M=self.num_modes, A=self.ac_dim)
        scales = rearrange(out["scale"], 'BT (M A) -> BT M A',
                           M=self.num_modes, A=self.ac_dim)
        logits = out["logits"]

        # apply tanh squashing to means if not using tanh-GMM to ensure means are in [-1, 1]
        if not self.use_tanh:
            means = torch.tanh(means)

        # Calculate scale
        if self.low_noise_eval and (not self.training):
            # low-noise for all Gaussian dists
            scales = torch.ones_like(means) * 1e-4
        else:
            # post-process the scale accordingly
            scales = self.activations[self.std_activation](
                scales) + self.min_std

        # mixture components - make sure that `batch_shape` for the distribution is equal
        # to (batch_size, num_modes) since MixtureSameFamily expects this shape
        component_distribution = D.Normal(loc=means, scale=scales)
        component_distribution = D.Independent(component_distribution, 1)

        # unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = D.Categorical(logits=logits)

        dist = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        if self.use_tanh:
            # Wrap distribution with Tanh
            dist = TanhWrappedDistribution(base_dist=dist, scale=1.)

        return dist


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Layer.

    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """

    def __init__(
        self,
        input_shape,
        num_kp=32,
        temperature=1.,
        learnable_temperature=False,
        output_variance=False,
        noise_std=0.0,
    ):
        """
        Args:
            input_shape (list): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not using spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution, and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape  # (C, H, W)

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        if self.learnable_temperature:
            # temperature will be learned
            temperature = torch.nn.Parameter(
                torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter('temperature', temperature)
        else:
            # temperature held constant after initialization
            temperature = torch.nn.Parameter(
                torch.ones(1) * temperature, requires_grad=False)
            self.register_buffer('temperature', temperature)

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self._in_w),
            np.linspace(-1., 1., self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(
            1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(
            1, self._in_h * self._in_w)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        self.kps = None

    def __repr__(self):
        """Pretty print network."""
        header = format(str(self.__class__.__name__))
        return header + '(num_kp={}, temperature={}, noise={})'.format(
            self._num_kp, self.temperature.item(), self.noise_std)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert (len(input_shape) == 3)
        assert (input_shape[0] == self._in_c)
        return [self._num_kp, 2]

    def forward(self, feature):
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial 
        probability distribution is created using a softmax, where the support is the 
        pixel locations. This distribution is used to compute the expected value of 
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.

        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """
        assert (feature.shape[1] == self._in_c)
        assert (feature.shape[2] == self._in_h)
        assert (feature.shape[3] == self._in_w)
        if self.nets is not None:
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise

        if self.output_variance:
            # treat attention as a distribution, and compute second-order statistics to return
            expected_xx = torch.sum(
                self.pos_x * self.pos_x * attention, dim=1, keepdim=True)
            expected_yy = torch.sum(
                self.pos_y * self.pos_y * attention, dim=1, keepdim=True)
            expected_xy = torch.sum(
                self.pos_x * self.pos_y * attention, dim=1, keepdim=True)
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2] where last 2 dims are covariance matrix
            feature_covar = torch.cat(
                [var_x, var_xy, var_xy, var_y], 1).reshape(-1, self._num_kp, 2, 2)
            feature_keypoints = (feature_keypoints, feature_covar)

        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(),
                        feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()
        return feature_keypoints


class CondPolicy(nn.Module):

    def __init__(self,
                 freeze_target_detector=True,
                 concat_state=True,
                 cond_target_obj_detector_pretrained=True,
                 cond_target_obj_detector_weights=None,
                 cond_target_obj_detector_step=4050,
                 action_cfg=None,
                 concat_bb=True,
                 mlp_layers=[128, 64, 32],
                 pooling=False,
                 spatial_softmax=False,
                 min_std=[0.3236332, 0.41879308, 0.14637606,
                          0.00677641, 0.06317256, 0.00703644, 0.976691],
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
        if spatial_softmax:
            self.spatial_softmax = SpatialSoftmax(input_shape=(512, 7, 7),
                                                  num_kp=16,
                                                  temperature=1.0,
                                                  learnable_temperature=False,
                                                  output_variance=False,
                                                  noise_std=0.0)
            latent_dim = self.spatial_softmax.output_shape(
                input_shape=(512, 7, 7))

        # Create action module
        action_module_input = int(
            np.prod(latent_dim) + float(concat_state) * action_cfg.sdim + float(concat_bb) * 4)

        self.action_module = GMM(input_dim=action_module_input,
                                 ac_dim=action_cfg.adim,
                                 num_modes=action_cfg.n_mixtures,
                                 mlp_layer_dims=mlp_layers,
                                 min_std=min_std,
                                 std_activation="softplus",
                                 use_tanh=False,
                                 low_noise_eval=True)

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('Total params in Imitation module:', params)
        print("\n\n---- Complete model ----\n")
        summary(self)

    def get_scale_factors(self):
        return self.cond_target_obj_detector.get_scale_factors()

    def forward(self, inputs: dict, inference: bool = False, oracle: bool = False, bb_queue: list = [], plot_activation=False):
        out = dict()

        # get target object prediction
        prediction_target_obj_detector = self.cond_target_obj_detector(inputs=inputs,
                                                                       inference=True)

        # Reshape BBs B*T, 4 to B,T,4
        bbs = torch.stack(prediction_target_obj_detector['proposals'])
        B, T, _, _, _ = inputs['images'].shape
        if inference == True and prediction_target_obj_detector['conf_scores_final'][0] == -1:
            bbs = bb_queue[0][None]
            prediction_target_obj_detector['proposals'] = [bb_queue[0]]
        bbs = rearrange(bbs, '(B T) N -> B T N', B=B, T=T, N=4)

        # 1. Get last layer feature maps
        # B*T, 512, 7, 7
        last_layer_feature = prediction_target_obj_detector['feature_map']

        if not self.training and plot_activation:
            # convert from tensor to numpy
            last_layer_feature_np = rearrange(
                last_layer_feature, '(B T) C H W -> B T H W C', B=B, T=T).cpu().numpy()
            input_image_np = np.array(rearrange(
                inputs['images'], 'B T C H W -> B T H W C', B=B, T=T).cpu().numpy()*255, dtype=np.uint8)
            cam_img, output_image = get_class_activation_map(feature_map=last_layer_feature_np[0][0],
                                                             input_image=input_image_np[0][0])
            out['cam_img'] = cam_img
            out['output_image'] = output_image

        # 2. Compute spatial softmax
        spatial_softmax_out = self.spatial_softmax(last_layer_feature)

        if not self.training and plot_activation:
            # convert from tensor to numpy
            spatial_softmax_out_np = rearrange(
                spatial_softmax_out, '(B T) C H W -> B T H W C', B=B, T=T).cpu().numpy()
            input_image_np = np.array(rearrange(
                inputs['images'], 'B T C H W -> B T H W C', B=B, T=T).cpu().numpy()*255, dtype=np.uint8)
            cam_img, output_image = get_class_activation_map(feature_map=spatial_softmax_out_np[0][0],
                                                             input_image=input_image_np[0][0])
            out['cam_img'] = cam_img
            out['output_image'] = output_image

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
        bb = None
        if oracle:
            bb = project_bboxes(bboxes=inputs['gt_bb'].float(),
                                width_scale_factor=self.get_scale_factors()[0],
                                height_scale_factor=self.get_scale_factors()[
                1],
                mode='p2a')[:, :, 0, :]
            prediction_target_obj_detector['proposals'] = [bb[0, 0, :]]
        else:
            bb = bbs

        action_in = torch.concat(
            [spatial_softmax_out_flat, states, bb], dim=2).to(torch.float32)
        # 6. Infer action embedding
        action_in = rearrange(action_in, "B T D -> (B T) D")
        ac_dist = self.action_module(action_in)

        out['bc_distrib'] = ac_dist
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
        (1, 12),  dtype=torch.float).to('cuda:0')[None]

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
    #                     spatial_softmax=cfg.policy.spatial_softmax,
    #                     )

    module = hydra.utils.instantiate(cfg.policy)

    module.to('cuda:0')
    module.train()
    module(inputs, inference=False)


if __name__ == '__main__':
    main()
