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
from mmengine.registry import MODELS
from mmengine.config import Config, ConfigDict
from mmengine.runner.checkpoint import (_load_checkpoint,
                                        _load_checkpoint_to_model)
from mmengine.dataset import Compose
ConfigType = Union[Config, ConfigDict]


class TargetObjDetector(nn.Module):

    def __init__(self,
                 out_dim: int = 256,
                 output_raw: bool = False,
                 drom_dim: int = 1,
                 model: str = None,
                 weights: str = None,
                 device: int = 0
                 ) -> None:

        super().__init__()

        self._device = device
        # 1. Load Swin-Transformer
        init_default_scope('mmdet')
        self.agent_emb = self._load_model_from_config(model=model,
                                                      weights=weights)
        self.agent_emb.to(f'cuda:{device}')

    def _load_model_from_config(self, model: str = None, weights: str = None):
        """_summary_

        Args:
            model (str, optional): . Defaults to None.
            weights (str, optional): . Defaults to None.
        """
        # 1. Load configuration from model file
        assert ".py" in model, f"{model} must be a path to a .py file"
        cfg: ConfigType
        cfg = Config.fromfile(model)

        # 2. Init model
        # 2.1 Load model weights from checkpoint
        assert weights is not None, "Weights must be not None"
        checkpoint = _load_checkpoint(weights, map_location='cpu')
        # 2.2 Initialise model
        model = MODELS.build(cfg.model)
        model.cfg = cfg
        model.cfg.test_dataloader.dataset.pipeline[0].type = 'LoadImageFromNDArray'
        print(
            f"Model test pipeline {model.cfg.test_dataloader.dataset.pipeline}")
        _load_checkpoint_to_model(model, checkpoint)
        return model

    def preprocess(self, inputs, batch_size, **kwargs):
        """Process the inputs into a model-feedable format.

        Customize your preprocess by overriding this method. Preprocess should
        return an iterable object, of which each item will be used as the
        input of ``model.test_step``.

        ``BaseInferencer.preprocess`` will return an iterable chunked data,
        which will be used in __call__ like this:

        .. code-block:: python

            def __call__(self, inputs, batch_size=1, **kwargs):
                chunked_data = self.preprocess(inputs, batch_size, **kwargs)
                for batch in chunked_data:
                    preds = self.forward(batch, **kwargs)

        Args:
            inputs (InputsType): Inputs given by user.
            batch_size (int): batch size. Defaults to 1.

        Yields:
            Any: Data processed by the ``pipeline`` and ``collate_fn``.
        """
        chunked_data = self._get_chunk_data(inputs, batch_size)
        yield from map(self.collate_fn, chunked_data)

    def _get_chunk_data(self, inputs: Iterable, chunk_size: int):
        """Get batch data from inputs.

        Args:
            inputs (Iterable): An iterable dataset.
            chunk_size (int): Equivalent to batch size.

        Yields:
            list: batch data.
        """
        inputs_iter = iter(inputs)
        while True:
            try:
                chunk_data = []
                for _ in range(chunk_size):
                    inputs_ = next(inputs_iter)
                    chunk_data.append((inputs_, self.pipeline(inputs_)))
                yield chunk_data
            except StopIteration:
                if chunk_data:
                    yield chunk_data
                break

    def forward(self, obs, context):
        if not self.model.training:
            obs.to(f'cuda:{self._device}')

            print(self.demo_emb(obs))


if __name__ == '__main__':

    def _make_demo(traj, task_name):
        """
        Do a near-uniform sampling of the demonstration trajectory
        """
        def clip(x): return int(max(0, min(x, len(traj) - 1)))
        per_bracket = max(len(traj) / 4, 1)
        frames = []
        cp_frames = []
        for i in range(4):
            # fix to using uniform + 'sample_side' now
            if i == 4 - 1:
                n = len(traj) - 1
            elif i == 0:
                n = 0
            else:
                n = clip(np.random.randint(
                    int(i * per_bracket), int((i + 1) * per_bracket)))
            # frames.append(_make_frame(n))
            # convert from BGR to RGB and scale to 0-1 range
            obs = copy.copy(
                traj.get(n)['obs']['camera_front_image'][:, :, ::-1])
            processed = obs
            frames.append(processed)

        return torch.from_numpy(np.array(frames))

    import debugpy
    debugpy.listen(('0.0.0.0', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()

    model_path = "/home/ciccio/Desktop/multi_task_lfd/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/repo/mmdetection/configs/swin/mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco.py"
    checkpoint_path = "/home/ciccio/Desktop/multi_task_lfd/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/models/checkpoints/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth"
    target_obj_detector = TargetObjDetector(model=model_path,
                                            weights=checkpoint_path)

    # 1. Inference test
    # 1.1 Load inputs
    demo_path = "/home/ciccio/Desktop/multi_task_lfd/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/models/checkpoints/test/agent/traj000.pkl"
    agent_path = "/home/ciccio/Desktop/multi_task_lfd/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/models/checkpoints/test/agent/traj000.pkl"

    # Select 4 random frames from demo
    with open(demo_path, "rb") as f:
        context_data = pickle.load(f)
    context = _make_demo(context_data['traj'], 'pick_place')

    # get first frame from agent
    with open(agent_path, "rb") as f:
        agent_data = pickle.load(f)
    agent_traj = agent_data['traj']
    obs_t = torch.from_numpy(copy.copy(
        agent_traj.get(1)['obs']['camera_front_image'][:, :, ::-1])).cuda().float()
    obs_t = torch.swapaxes(torch.swapaxes(obs_t, 0, 2), 2, 1)[None]
    target_obj_detector(obs=obs_t,
                        context=context)
