import random
import torch
from os.path import join, expanduser
from multi_task_il.datasets import load_traj, split_files
import cv2
from torch.utils.data import Dataset, Sampler, SubsetRandomSampler, RandomSampler, WeightedRandomSampler
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from torchvision.transforms import RandomAffine, ToTensor, Normalize, \
    RandomGrayscale, ColorJitter, RandomApply, RandomHorizontalFlip, GaussianBlur, RandomResizedCrop
from torchvision.transforms.functional import resized_crop

import pickle as pkl
from collections import defaultdict, OrderedDict
import glob
import numpy as np
import matplotlib.pyplot as plt
import copy
from copy import deepcopy
from functools import reduce
from operator import concat
from multi_task_il.utils import normalize_action
from multi_task_il.datasets.utils import *


class MultiTaskPairedDataset(Dataset):
    def __init__(
            self,
            tasks_spec,
            root_dir='mosaic_multitask_dataset',
            mode='train',
            split=[0.9, 0.1],
            demo_T=4,
            obs_T=7,
            take_first_frame=False,
            aug_twice=True,
            width=180,
            height=100,
            data_augs=None,
            non_sequential=False,
            state_spec=('ee_aa', 'gripper_qpos'),
            action_spec=('action',),
            allow_val_skip=False,
            allow_train_skip=False,
            use_strong_augs=False,
            aux_pose=False,
            select_random_frames=True,
            balance_target_obj_pos=True,
            compute_obj_distribution=False,
            agent_name='ur5e',
            demo_name='panda',
            normalize_action=True,
            normalization_ranges=[],
            n_action_bin=256,
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

        self.select_random_frames = select_random_frames
        self.balance_target_obj_pos = balance_target_obj_pos
        self.compute_obj_distribution = compute_obj_distribution
        self.object_distribution = OrderedDict()
        self.object_distribution_to_indx = OrderedDict()

        self._take_first_frame = take_first_frame

        self._selected_target_frame_distribution_task_object_target_position = OrderedDict()

        self._compute_frame_distribution = False
        self._normalize_action = normalize_action
        self._normalization_ranges = np.array(normalization_ranges)
        self._n_action_bin = n_action_bin

        # Frame distribution for each trajectory
        self._frame_distribution = OrderedDict()

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

        self._demo_T, self._obs_T = demo_T, obs_T
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

        if agent_file not in self.all_file_pairs:
            self._frame_distribution[agent_file] = np.zeros((1, 250))

        demo_traj, agent_traj = load_traj(demo_file), load_traj(agent_file)
        demo_data = self._make_demo(demo_traj[0], task_name)
        traj = self._make_traj(agent_traj[0], task_name, agent_file)
        return {'demo_data': demo_data, 'traj': traj, 'task_name': task_name, 'task_id': sub_task_id}

    def _make_demo(self, traj, task_name):
        """
        Do a near-uniform sampling of the demonstration trajectory
        """
        if self.select_random_frames:
            def clip(x): return int(max(1, min(x, len(traj) - 1)))
            per_bracket = max(len(traj) / self._demo_T, 1)
            frames = []
            cp_frames = []
            for i in range(self._demo_T):
                # fix to using uniform + 'sample_side' now
                if i == self._demo_T - 1:
                    n = len(traj) - 1
                elif i == 0:
                    n = 1
                else:
                    n = clip(np.random.randint(
                        int(i * per_bracket), int((i + 1) * per_bracket)))
                # frames.append(_make_frame(n))
                # convert from BGR to RGB and scale to 0-1 range
                obs = copy.copy(
                    traj.get(n)['obs']['camera_front_image'][:, :, ::-1])
                processed = self.frame_aug(task_name, obs)
                frames.append(processed)
                if self.aug_twice:
                    cp_frames.append(self.frame_aug(task_name, obs, True))
        else:
            frames = []
            cp_frames = []
            for i in range(self._demo_T):
                # get first frame
                if i == 0:
                    n = 1
                # get the last frame
                elif i == self._demo_T - 1:
                    n = len(traj) - 1
                elif i == 1:
                    obj_in_hand = 0
                    # get the first frame with obj_in_hand and the gripper is closed
                    for t in range(1, len(traj)):
                        state = traj.get(t)['info']['status']
                        trj_t = traj.get(t)
                        gripper_act = trj_t['action'][-1]
                        if state == 'obj_in_hand' and gripper_act == 1:
                            obj_in_hand = t
                            n = t
                            break
                elif i == 2:
                    # get the middle moving frame
                    start_moving = 0
                    end_moving = 0
                    for t in range(obj_in_hand, len(traj)):
                        state = traj.get(t)['info']['status']
                        if state == 'moving' and start_moving == 0:
                            start_moving = t
                        elif state != 'moving' and start_moving != 0 and end_moving == 0:
                            end_moving = t
                            break
                    n = start_moving + int((end_moving-start_moving)/2)

                # convert from BGR to RGB and scale to 0-1 range
                obs = copy.copy(
                    traj.get(n)['obs']['camera_front_image'][:, :, ::-1])

                processed = self.frame_aug(task_name, obs)
                frames.append(processed)
                if self.aug_twice:
                    cp_frames.append(self.frame_aug(task_name, obs, True))

        ret_dict = dict()
        ret_dict['demo'] = torch.stack(frames)
        ret_dict['demo_cp'] = torch.stack(cp_frames)
        return ret_dict

    def _make_traj(self, traj, task_name, agent_file):
        crop_params = self.task_crops.get(task_name, [0, 0, 0, 0])

        def _adjust_points(points, frame_dims):
            h = np.clip(points[0] - crop_params[0], 0,
                        frame_dims[0] - crop_params[1])
            w = np.clip(points[1] - crop_params[2], 0,
                        frame_dims[1] - crop_params[3])
            h = float(
                h) / (frame_dims[0] - crop_params[0] - crop_params[1]) * self.height
            w = float(
                w) / (frame_dims[1] - crop_params[2] - crop_params[3]) * self.width
            return tuple([int(min(x, d - 1)) for x, d in zip([h, w], (self.height, self.width))])

        def _get_tensor(k, step_t):
            if k == 'action':
                return step_t['action']
            elif k == 'grip_action':
                return [step_t['action'][-1]]
            o = step_t['obs']
            if k == 'ee_aa' and 'ee_aa' not in o:
                ee, axis_angle = o['ee_pos'][:3], o['axis_angle']
                if axis_angle[0] < 0:
                    axis_angle[0] += 2
                o = np.concatenate((ee, axis_angle)).astype(np.float32)
            else:
                o = o[k]
            return o

        state_keys, action_keys = self._state_action_spec
        ret_dict = {'states': [], 'actions': []}
        has_eef_point = 'eef_point' in traj.get(0, False)['obs']
        if has_eef_point:
            ret_dict['points'] = []
        end = len(traj)
        start = torch.randint(low=1, high=max(
            1, end - self._obs_T + 1), size=(1,))

        if self._take_first_frame:
            first_frame = [torch.tensor(1)]
            chosen_t = first_frame + [j + start for j in range(self._obs_T)]
        else:
            chosen_t = [j + start for j in range(self._obs_T)]

        if self.non_sequential:
            chosen_t = torch.randperm(end)
            chosen_t = chosen_t[chosen_t != 0][:self._obs_T]
        images = []
        images_cp = []

        for j, t in enumerate(chosen_t):

            # self._frame_distribution[agent_file][t] = self._frame_distribution[agent_file][t]

            t = t.item()

            step_t = traj.get(t)
            image = copy.copy(
                step_t['obs']['camera_front_image'][:, :, ::-1])
            processed = self.frame_aug(task_name, image)
            images.append(processed)
            if self.aug_twice:
                images_cp.append(self.frame_aug(task_name, image, True))
            if has_eef_point:
                ret_dict['points'].append(np.array(
                    _adjust_points(step_t['obs']['eef_point'], image.shape[:2]))[None])

            state = []
            for k in state_keys:
                state.append(_get_tensor(k, step_t))
            ret_dict['states'].append(
                np.concatenate(state).astype(np.float32)[None])

            if (j >= 1 and not self._take_first_frame) or (self._take_first_frame and j >= 2):
                action = []
                for k in action_keys:
                    action.append(_get_tensor(k, step_t))

                if self._normalize_action:
                    action = normalize_action(
                        action=action[0],
                        n_action_bin=self._n_action_bin,
                        action_ranges=self._normalization_ranges
                    )[None]

                ret_dict['actions'].append(
                    np.concatenate(action).astype(np.float32)[None])

        for k, v in ret_dict.items():
            ret_dict[k] = np.concatenate(v, 0).astype(np.float32)

        ret_dict['images'] = torch.stack(images)
        if self.aug_twice:
            ret_dict['images_cp'] = torch.stack(images_cp)

        if self.aux_pose:
            grip_close = np.array(
                [traj.get(i, False)['action'][-1] > 0 for i in range(1, len(traj))])
            grip_t = np.argmax(grip_close)
            drop_t = len(traj) - 1 - \
                np.argmax(np.logical_not(grip_close)[::-1])
            aux_pose = [traj.get(t, False)['obs']['ee_aa'][:3]
                        for t in (grip_t, drop_t)]
            ret_dict['aux_pose'] = np.concatenate(aux_pose).astype(np.float32)
        return ret_dict
