import random
import torch
from multi_task_il.datasets import load_traj
import cv2
from torch.utils.data import Dataset


import pickle as pkl
from collections import defaultdict, OrderedDict
import glob
import numpy as np
import matplotlib.pyplot as plt
import copy

from multi_task_il.utils import normalize_action
from multi_task_il.datasets.utils import *
import robosuite.utils.transform_utils as T
from multiprocessing import Pool, cpu_count
import functools


DEBUG = False


class CondTargetObjDetectorDataset(Dataset):
    def __init__(
            self,
            tasks_spec,
            root_dir='mosaic_multitask_dataset',
            mode='train',
            split=[0.9, 0.1],
            aug_twice=True,
            width=180,
            height=100,
            demo_T=4,
            obs_T=1,
            data_augs=None,
            non_sequential=False,
            allow_val_skip=False,
            allow_train_skip=False,
            use_strong_augs=False,
            aux_pose=False,
            select_random_frames=True,
            compute_obj_distribution=False,
            agent_name='ur5e',
            demo_name='panda',
            load_action=False,
            load_state=False,
            state_spec=[''],
            action_spec='action',
            normalize_action=True,
            normalization_ranges=[],
            n_action_bin=256,
            first_frames=False,
            only_first_frame=True,
            task_id=False,
            tasks={},
            n_tasks=16,
            perform_augs=False,
            ** params):

        self.task_crops = OrderedDict()
        # each idx i maps to a unique tuple of (task_name, sub_task_id, agent.pkl, demo.pkl)
        self.all_file_pairs = OrderedDict()
        count = 0
        self.task_to_idx = defaultdict(list)
        self.subtask_to_idx = OrderedDict()
        self.agent_files = dict()
        self.demo_files = dict()
        self.mode = mode
        self._demo_T = demo_T
        self._obs_T = obs_T
        self._load_action = load_action
        self._load_state = load_state
        self._state_spec = state_spec
        self._action_spec = action_spec
        self._normalize_action = normalize_action
        self._normalization_ranges = np.array(normalization_ranges)
        self._n_action_bin = n_action_bin
        self._first_frames = first_frames
        self._only_first_frame = only_first_frame
        self._task_id = task_id
        self._tasks = tasks
        self._n_tasks = n_tasks
        self._perform_augs = perform_augs

        self.select_random_frames = select_random_frames
        self.compute_obj_distribution = compute_obj_distribution
        self.object_distribution = OrderedDict()
        self.object_distribution_to_indx = OrderedDict()
        self.index_to_slot = OrderedDict()

        create_train_val_dict(self,
                              agent_name,
                              demo_name,
                              root_dir,
                              tasks_spec,
                              split,
                              allow_train_skip,
                              allow_val_skip)

        self.pairs_count = count
        self.task_count = len(tasks_spec)

        self.width, self.height = width, height
        self.aug_twice = aug_twice
        self.aux_pose = aux_pose

        self._state_action_spec = (state_spec, action_spec)
        self.non_sequential = non_sequential
        if non_sequential:
            print("Warning! The agent observations are not sampled in neighboring timesteps, make sure inverse dynamics loss is NOT used in training \n ")

        self.use_strong_augs = use_strong_augs
        self.data_augs = data_augs
        self.frame_aug = create_data_aug(self)

    def __len__(self):
        """NOTE: we should count total possible demo-agent pairs, not just single-file counts
        total pairs should sum over all possible sub-task pairs"""
        return self.pairs_count

    def __getitem__(self, idx):
        """since the data is organized by task, use a mapping here to convert
        an index to a proper sub-task index """
        if self.mode == 'train':
            pass
        (task_name, sub_task_id, demo_file,
         agent_file) = self.all_file_pairs[idx]
        demo_traj, agent_traj = load_traj(demo_file), load_traj(agent_file)
        demo_data = make_demo(self, demo_traj[0], task_name)
        traj = self._make_traj(
            agent_traj[0], agent_traj[1], task_name, sub_task_id)
        return {'demo_data': demo_data, 'traj': traj, 'task_name': task_name, 'task_id': sub_task_id}

    def _make_traj(self, traj, command, task_name, sub_task_id):
        # get the first frame from the trajectory
        ret_dict = {}
        # print(f"Command {command}")
        take_first_frames = False
        assert not self._first_frames or not self._only_first_frame, f"First frames and only first frames cannot be both True"
        if self._first_frames and not self._only_first_frame:
            take_first_frames = random.choices(
                [True, False], weights=[0.5, 0.50])

        if not take_first_frames:
            if self.non_sequential and self._obs_T > 1:
                end = len(traj)
                chosen_t = torch.randperm(end)
                chosen_t = chosen_t[chosen_t != 0][:self._obs_T]

            elif not self.non_sequential and not self._only_first_frame:
                end = len(traj)
                start = self._obs_T if self._first_frames else 1
                start = torch.randint(low=1, high=max(
                    1, end - self._obs_T - 1), size=(1,))
                chosen_t = [j + start for j in range(self._obs_T+1)]

            elif self._only_first_frame:
                end = self._obs_T
                start = torch.Tensor([1]).int()
                chosen_t = [j + start for j in range(self._obs_T+1)]
        else:
            end = self._obs_T
            start = torch.Tensor([1]).int()
            chosen_t = [j + start for j in range(self._obs_T+1)]

        images, images_cp, bb, obj_classes, actions, states, points = create_sample(
            dataset_loader=self,
            traj=traj,
            chosen_t=chosen_t,
            task_name=task_name,
            command=command,
            load_action=self._load_action,
            load_state=self._load_state)

        if self._perform_augs:
            ret_dict['images'] = torch.stack(images)
        if self.aug_twice:
            ret_dict['images_cp'] = torch.stack(images_cp)

        ret_dict['gt_bb'] = torch.stack(bb)
        ret_dict['gt_classes'] = torch.stack(obj_classes)
        if self._load_state:
            ret_dict['states'] = []
            ret_dict['states'] = np.array(states)
        if self._load_action:
            ret_dict['actions'] = []
            ret_dict['actions'] = np.array(actions)
        if self._task_id:
            task_one_hot = np.zeros((1, self._n_tasks))
            task_one_hot[0][self._tasks[task_name]
                            [0]+sub_task_id] = 1
            ret_dict['task_id'] = np.array(task_one_hot)
        return ret_dict
