import random
import torch
from os.path import join, expanduser
from multi_task_il.datasets import load_traj, split_files
import cv2
from torch.utils.data import Dataset, Sampler, SubsetRandomSampler
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

from utils import *


class MultiTaskPairedTargetObjDataset(Dataset):
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
            **params):

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

        print('Done loading Task {}, agent/demo trajctores pairs reach a count of: {}'.format(name, count))
        self.pairs_count = count
        self.task_count = len(tasks_spec)

        self.width, self.height = width, height
        self.aug_twice = aug_twice
        self.aux_pose = aux_pose

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
        demo_data = self._make_demo(demo_traj[0], task_name)
        traj = self._make_traj(agent_traj[0], task_name, idx)
        return {'demo_data': demo_data, 'traj': traj, 'task_name': task_name, 'task_id': sub_task_id}

    def _make_demo(self, traj, task_name):
        """
        Do a near-uniform sampling of the demonstration trajectory
        """
        if self.select_random_frames:
            def clip(x): return int(max(0, min(x, len(traj) - 1)))
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
                    n = 0
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

    def _make_traj(self, traj, task_name, indx):
        # get the first frame from the trajectory
        ret_dict = {}
        images = []
        images_cp = []
        bb = []

        if self.non_sequential and self._obs_T > 1:
            end = len(traj)
            chosen_t = torch.randperm(end)
            chosen_t = chosen_t[chosen_t != 0][:self._obs_T]
            for j, t in enumerate(chosen_t):
                t = t.item()
                step_t = traj.get(t)
                image = copy.copy(
                    step_t['obs']['camera_front_image'][:, :, ::-1])
                processed = self.frame_aug(task_name, image)
                images.append(processed)
                if self.aug_twice:
                    images_cp.append(self.frame_aug(task_name, image, True))
                # Create GT BB
                bb.append(self._create_gt_bb(traj=traj,
                                             t=1,
                                             task_name=task_name))
        elif not self.non_sequential and self._obs_T > 1:
            end = len(traj)
            start = torch.randint(low=1, high=max(
                1, end - self._obs_T + 1), size=(1,))
            chosen_t = [j + start for j in range(self._obs_T)]
            images = []
            images_cp = []

            for j, t in enumerate(chosen_t):
                t = t.item()
                step_t = traj.get(t)
                image = copy.copy(
                    step_t['obs']['camera_front_image'][:, :, ::-1])
                processed = self.frame_aug(task_name, image)
                images.append(processed)
                if self.aug_twice:
                    images_cp.append(self.frame_aug(task_name, image, True))
                # Create GT BB
                bb.append(self._create_gt_bb(traj=traj,
                                             t=t,
                                             task_name=task_name))
        else:
            image = copy.copy(
                traj.get(1)['obs']['camera_front_image'][:, :, ::-1])
            images.append(self.frame_aug(task_name, image))
            images_cp.append(self.frame_aug(task_name, image, True))
            # Create GT BB
            bb.append(self._create_gt_bb(traj=traj,
                                         t=1,
                                         task_name=task_name))

        ret_dict['images'] = torch.stack(images)
        if self.aug_twice:
            ret_dict['images_cp'] = torch.stack(images_cp)

        ret_dict['gt_bb'] = torch.stack(bb)

        return ret_dict

    def _create_gt_bb(self, traj, t, task_name):
        # 1. Get Target Object
        target_obj_id = traj.get(
            t)['obs']['target-object']
        # 2. Get stored BB
        top_left_x = traj.get(
            t)['obs']['obj_bb']["camera_front_img"][ENV_OBJECTS[task_name]['obj_names'][target_obj_id]]['bottom_right_corner'][0]
        top_left_y = traj.get(
            t)['obs']['obj_bb']["camera_front_img"][ENV_OBJECTS[task_name]['obj_names'][target_obj_id]]['bottom_right_corner'][1]
        # print(f"Top-Left X {top_left_x} - Top-Left Y {top_left_y}")
        bottom_right_x = traj.get(
            t)['obs']['obj_bb']["camera_front_img"][ENV_OBJECTS[task_name]['obj_names'][target_obj_id]]['upper_left_corner'][0]
        bottom_right_y = traj.get(
            t)['obs']['obj_bb']["camera_front_img"][ENV_OBJECTS[task_name]['obj_names'][target_obj_id]]['upper_left_corner'][1]

        # bounding-box
        # right_x - left_x
        width = bottom_right_x - top_left_x
        # left_y - right_y
        height = bottom_right_y - top_left_y

        return np.array([top_left_x, top_left_y, width, height])
