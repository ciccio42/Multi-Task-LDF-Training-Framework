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
        traj = self._make_traj(agent_traj[0], agent_traj[1], task_name, idx)
        return {'demo_data': demo_data, 'traj': traj, 'task_name': task_name, 'task_id': sub_task_id}

    def _create_sample(self, traj, chosen_t, task_name, command, load_action=False, load_state=False):
        images = []
        images_cp = []
        bb = []
        obj_classes = []
        actions = []
        states = []

        for j, t in enumerate(chosen_t):
            t = t.item()
            step_t = traj.get(t)
            image = copy.copy(
                step_t['obs']['camera_front_image'][:, :, ::-1])

            # Create GT BB
            bb_frame, class_frame = self._create_gt_bb(traj=traj,
                                                       t=t,
                                                       task_name=task_name)
            # Append bb, obj classes and images
            processed, bb_aug, class_frame = self.frame_aug(
                task_name, image, False, bb_frame, class_frame)
            bb.append(torch.from_numpy(bb_aug))
            obj_classes.append((torch.from_numpy(class_frame)))
            images.append(processed)

            if self.aug_twice:
                image_cp = self.frame_aug(task_name, image, True)
                images_cp.append(image_cp)

            if load_action:
                # Load action
                action = step_t['action'] if not self._normalize_action else normalize_action(
                    action=step_t['action'], n_action_bin=self._n_action_bin, action_ranges=self._normalization_ranges)
                actions.append(action)

            if load_state:
                state = []
                # Load states
                for k in self._state_spec:
                    state.append(step_t['obs'][k])
                states.append(np.concatenate(state))

            # test GT
            if DEBUG:
                image = np.array(np.moveaxis(
                    images[j].cpu().numpy()*255, 0, -1), dtype=np.uint8)
                image = cv2.rectangle(np.ascontiguousarray(image),
                                      (int(bb_frame[0][0]),
                                       int(bb_frame[0][1])),
                                      (int(bb_frame[0][2]),
                                       int(bb_frame[0][3])),
                                      color=(0, 0, 255),
                                      thickness=1)
                # print(f"Command {command}")
                cv2.imwrite("GT_bb_after_aug.png", image)

        return images, images_cp, bb, obj_classes, actions, states

    def _make_traj(self, traj, command, task_name, indx):
        # get the first frame from the trajectory
        ret_dict = {}
        # print(f"Command {command}")

        if self.non_sequential and self._obs_T > 1:
            end = len(traj)
            chosen_t = torch.randperm(end)
            chosen_t = chosen_t[chosen_t != 0][:self._obs_T]

        elif not self.non_sequential and self._obs_T > 1:
            end = len(traj)
            start = torch.randint(low=1, high=max(
                1, end - self._obs_T + 1), size=(1,))
            chosen_t = [j + start for j in range(self._obs_T)]

        else:
            chosen_t = [1]

        images, images_cp, bb, obj_classes, action, states = self._create_sample(traj=traj,
                                                                                 chosen_t=chosen_t,
                                                                                 task_name=task_name,
                                                                                 command=command,
                                                                                 load_action=self._load_action,
                                                                                 load_state=self._load_state)

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
            ret_dict['actions'] = np.array(action)
        return ret_dict

    def _create_gt_bb(self, traj, t, task_name):
        bb = []
        cl = []
        image_size = traj.get(
            t)['obs']['camera_front_image'].shape

        # 1. Get Target Object
        target_obj_id = traj.get(
            t)['obs']['target-object']
        for obj_id, object_name in enumerate(traj.get(t)['obs']['obj_bb']['camera_front'].keys()):
            if object_name != 'bin' and obj_id == target_obj_id:
                # 2. Get stored BB
                top_left_x = traj.get(
                    t)['obs']['obj_bb']["camera_front"][object_name]['bottom_right_corner'][0]
                top_left_y = traj.get(
                    t)['obs']['obj_bb']["camera_front"][object_name]['bottom_right_corner'][1]
                # print(f"Top-Left X {top_left_x} - Top-Left Y {top_left_y}")
                bottom_right_x = traj.get(
                    t)['obs']['obj_bb']["camera_front"][object_name]['upper_left_corner'][0]
                bottom_right_y = traj.get(
                    t)['obs']['obj_bb']["camera_front"][object_name]['upper_left_corner'][1]

                # center_x
                center_x = traj.get(
                    t)['obs']['obj_bb']["camera_front"][object_name]['center'][0]
                # center_y
                center_y = traj.get(
                    t)['obs']['obj_bb']["camera_front"][object_name]['center'][1]

                # bounding-box
                # right_x - left_x
                width = bottom_right_x - top_left_x
                # left_y - right_y
                height = bottom_right_y - top_left_y
                # test GT
                if DEBUG:
                    image = cv2.rectangle(np.array(traj.get(
                        t)['obs']['camera_front_image'][:, :, ::-1]),
                        (int(top_left_x),
                         int(top_left_y)),
                        (int(bottom_right_x),
                         int(bottom_right_y)),
                        color=(0, 0, 255),
                        thickness=1)
                    cv2.imwrite("GT_bb.png", image)

                bb.append([top_left_x, top_left_y,
                          bottom_right_x, bottom_right_y])
                # [1] - Target object
                # [0] - No target
                if obj_id == target_obj_id:
                    cl.append(1)
                else:
                    cl.append(0)
        return np.array(bb), np.array(cl)
