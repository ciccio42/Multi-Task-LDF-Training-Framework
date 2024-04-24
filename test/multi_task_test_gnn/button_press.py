from collections import deque
import torch
import numpy as np
from multi_task_il.datasets import Trajectory
import cv2
from multi_task_il.utils import denormalize_action_vima
from einops import rearrange
from multi_task_il.models.vima.utils import *
import robosuite.utils.transform_utils as T
from multi_task_test.primitive import *
from multi_task_test.utils import *
from multi_task_il.models.cond_target_obj_detector.utils import project_bboxes


def press_button_eval(model, env, gt_env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False, action_ranges=[], model_name=None, task_name="pick_place", config=None, gt_file=None, gt_bb=False, sub_action=False, gt_action=4, real=True):

    pass
