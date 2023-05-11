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
#### ------ #####
# ToDo: Task Classification network
# Input: 4 frames from demonstrator, first agent observation
# Output: Target object position distribution probability [dx, c-dx. c-sx, dx]
# The GT is taken from the

#### ------ #####


