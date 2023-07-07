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

ENV_OBJECTS = {
    'pick_place': {
        'obj_names': ['greenbox', 'yellowbox', 'bluebox', 'redbox'],
        'ranges': [[-0.255, -0.195], [-0.105, -0.045], [0.045, 0.105], [0.195, 0.255]],
    },
    'nut_assembly': {
        'obj_names': ['nut0', 'nut1', 'nut2'],
        'ranges': [[0.10, 0.31], [-0.10, 0.10], [-0.31, -0.10]]
    }
}

JITTER_FACTORS = {'brightness': 0.4,
                  'contrast': 0.4, 'saturation': 0.4, 'hue': 0.1}


def collate_by_task(batch):
    """ Use this for validation: groups data by task names to compute per-task losses """
    per_task_data = defaultdict(list)
    for b in batch:
        per_task_data[b['task_name']].append(
            {k: v for k, v in b.items() if k != 'task_name' and k != 'task_id'}
        )

    for name, data in per_task_data.items():
        per_task_data[name] = default_collate(data)
    return per_task_data


def create_train_val_dict(dataset_loader=object, agent_name: str = "ur5e", demo_name: str = "panda", root_dir: str = "", task_spec=None, split: list = [0.9, 0.1], allow_train_skip: bool = False, allow_val_skip: bool = False):

    count = 0
    for spec in task_spec:
        name, date = spec.get('name', None), spec.get('date', None)
        assert name, 'need to specify the task name for data generated, for easier tracking'
        dataset_loader.agent_files[name] = dict()
        dataset_loader.demo_files[name] = dict()

        dataset_loader.object_distribution[name] = OrderedDict()
        dataset_loader.object_distribution_to_indx[name] = OrderedDict()

        if dataset_loader.mode == 'train':
            print(
                "Loading task [{:<9}] saved on date {}".format(name, date))
        if date is None:
            agent_dir = join(
                root_dir, name, '{}_{}'.format(agent_name, name))
            demo_dir = join(
                root_dir, name, '{}_{}'.format(demo_name, name))
        else:
            agent_dir = join(
                root_dir, name, '{}_{}_{}'.format(date, agent_name, name))
            demo_dir = join(
                root_dir, name, '{}_{}_{}'.format(date, demo_name, name))
        dataset_loader.subtask_to_idx[name] = defaultdict(list)
        for _id in range(spec.get('n_tasks')):
            if _id in spec.get('skip_ids', []):
                if (allow_train_skip and dataset_loader.mode == 'train') or (allow_val_skip and dataset_loader.mode == 'val'):
                    print(
                        'Warning! Excluding subtask id {} from loaded **{}** dataset for task {}'.format(_id, dataset_loader.mode, name))
                    continue
            task_id = 'task_{:02d}'.format(_id)
            task_dir = expanduser(join(agent_dir,  task_id, '*.pkl'))
            agent_files = sorted(glob.glob(task_dir))
            assert len(agent_files) != 0, "Can't find dataset for task {}, subtask {} in dir {}".format(
                name, _id, task_dir)
            subtask_size = spec.get('traj_per_subtask', 100)
            assert len(
                agent_files) >= subtask_size, "Doesn't have enough data "+str(len(agent_files))
            agent_files = agent_files[:subtask_size]

            # prev. version does split randomly, here we strictly split each subtask in the same split ratio:
            idxs = split_files(len(agent_files), split, dataset_loader.mode)
            agent_files = [agent_files[i] for i in idxs]

            task_dir = expanduser(join(demo_dir, task_id, '*.pkl'))

            demo_files = sorted(glob.glob(task_dir))
            subtask_size = spec.get('demo_per_subtask', 100)
            assert len(
                demo_files) >= subtask_size, "Doesn't have enough data "+str(len(demo_files))
            demo_files = demo_files[:subtask_size]
            idxs = split_files(len(demo_files), split, dataset_loader.mode)
            demo_files = [demo_files[i] for i in idxs]
            # assert len(agent_files) == len(demo_files), \
            #     'data for task {}, subtask #{} is not matched'.format(name, task_id)

            dataset_loader.agent_files[name][_id] = deepcopy(agent_files)
            dataset_loader.demo_files[name][_id] = deepcopy(demo_files)

            dataset_loader.object_distribution[name][task_id] = OrderedDict()
            dataset_loader.object_distribution_to_indx[name][task_id] = [
                [] for i in range(len(ENV_OBJECTS[name]['ranges']))]
            if dataset_loader.compute_obj_distribution:
                # for each subtask, create a dict with the object name
                # assign the slot at each file
                for agent in agent_files:
                    # compute object distribution if requested
                    if dataset_loader.compute_obj_distribution:
                        # load pickle file
                        with open(agent, "rb") as f:
                            agent_file_data = pkl.load(f)
                        # take trj
                        trj = agent_file_data['traj']
                        # take target object id
                        target_obj_id = trj[1]['obs']['target-object']
                        for id, obj_name in enumerate(ENV_OBJECTS[name]['obj_names']):
                            if id == target_obj_id:
                                if obj_name not in dataset_loader.object_distribution[name][task_id]:
                                    dataset_loader.object_distribution[name][task_id][obj_name] = OrderedDict(
                                    )
                                # get object position
                                if name == 'nut_assembly':
                                    if id == 0:
                                        pos = trj[1]['obs']['round-nut_pos']
                                    else:
                                        pos = trj[1]['obs'][f'round-nut-{id+1}_pos']
                                else:
                                    pos = trj[1]['obs'][f'{obj_name}_pos']
                                for i, pos_range in enumerate(ENV_OBJECTS[name]["ranges"]):
                                    if pos[1] >= pos_range[0] and pos[1] <= pos_range[1]:
                                        dataset_loader.object_distribution[name][task_id][obj_name][agent] = i
                                        break
                                break

            for demo in demo_files:
                for agent in agent_files:
                    dataset_loader.all_file_pairs[count] = (
                        name, _id, demo, agent)
                    dataset_loader.task_to_idx[name].append(count)
                    dataset_loader.subtask_to_idx[name][task_id].append(count)
                    if dataset_loader.compute_obj_distribution:
                        # take objs for the current task_id
                        for obj in dataset_loader.object_distribution[name][task_id].keys():
                            # take the slot for the given agent file
                            if agent in dataset_loader.object_distribution[name][task_id][obj]:
                                slot_indx = dataset_loader.object_distribution[name][task_id][obj][agent]
                                # assign the slot for the given agent file
                                dataset_loader.object_distribution_to_indx[name][task_id][slot_indx].append(
                                    count)
                                dataset_loader.index_to_slot[count] = slot_indx
                    count += 1

        print('Done loading Task {}, agent/demo trajctores pairs reach a count of: {}'.format(name, count))
        dataset_loader.task_crops[name] = spec.get('crop', [0, 0, 0, 0])


def create_data_aug(dataset_loader=object):

    assert dataset_loader.data_augs, 'Must give some basic data-aug parameters'
    if dataset_loader.mode == 'train':
        print('Data aug parameters:', dataset_loader.data_augs)
    # self.randAffine = RandomAffine(degrees=0, translate=(dataset_loader.data_augs.get('rand_trans', 0.1), dataset_loader.data_augs.get('rand_trans', 0.1)))
    dataset_loader.toTensor = ToTensor()
    dataset_loader.normalize = Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    jitters = {k: v * dataset_loader.data_augs.get('weak_jitter', 0)
               for k, v in JITTER_FACTORS.items()}
    weak_jitter = ColorJitter(**jitters)

    weak_scale = dataset_loader.data_augs.get('weak_crop_scale', (0.8, 1.0))
    weak_ratio = dataset_loader.data_augs.get('weak_crop_ratio', (1.6, 1.8))
    randcrop = RandomResizedCrop(
        size=(dataset_loader.height, dataset_loader.width), scale=weak_scale, ratio=weak_ratio)
    if dataset_loader.data_augs.use_affine:
        randcrop = RandomAffine(degrees=0, translate=(dataset_loader.data_augs.get(
            'rand_trans', 0.1), dataset_loader.data_augs.get('rand_trans', 0.1)))
    dataset_loader.transforms = transforms.Compose([
        RandomApply([weak_jitter], p=0.1),
        RandomApply(
            [GaussianBlur(kernel_size=5, sigma=dataset_loader.data_augs.get('blur', (0.1, 2.0)))], p=0.1),
        randcrop,
        # dataset_loader.normalize
    ])

    print("Using strong augmentations?", dataset_loader.use_strong_augs)
    jitters = {k: v * dataset_loader.data_augs.get('strong_jitter', 0)
               for k, v in JITTER_FACTORS.items()}
    strong_jitter = ColorJitter(**jitters)
    dataset_loader.grayscale = RandomGrayscale(
        dataset_loader.data_augs.get("grayscale", 0))
    strong_scale = dataset_loader.data_augs.get(
        'strong_crop_scale', (0.2, 0.76))
    strong_ratio = dataset_loader.data_augs.get(
        'strong_crop_ratio', (1.2, 1.8))
    dataset_loader.strong_augs = transforms.Compose([
        RandomApply([strong_jitter], p=0.05),
        dataset_loader.grayscale,
        RandomHorizontalFlip(p=dataset_loader.data_augs.get('flip', 0)),
        RandomApply(
            [GaussianBlur(kernel_size=5, sigma=dataset_loader.data_augs.get('blur', (0.1, 2.0)))], p=0.01),
        RandomResizedCrop(
            size=(dataset_loader.height, dataset_loader.width), scale=strong_scale, ratio=strong_ratio),
        # dataset_loader.normalize,
    ])

    def frame_aug(task_name, obs, second=False):
        """applies to every timestep's RGB obs['camera_front_image']"""
        crop_params = dataset_loader.task_crops.get(task_name, [0, 0, 0, 0])
        top, left = crop_params[0], crop_params[2]
        img_height, img_width = obs.shape[0], obs.shape[1]
        box_h, box_w = img_height - top - \
            crop_params[1], img_width - left - crop_params[3]

        obs = dataset_loader.toTensor(obs)
        # only this resize+crop is task-specific
        obs = resized_crop(obs, top=top, left=left, height=box_h,
                           width=box_w, size=(dataset_loader.height, dataset_loader.width))
        # cv2.imwrite("resized_target_obj.png", np.moveaxis(
        #     obs.numpy()*255, 0, -1))
        if dataset_loader.use_strong_augs and second:
            augmented = dataset_loader.strong_augs(obs)
        else:
            augmented = dataset_loader.transforms(obs)
        assert augmented.shape == obs.shape
        # if dataset_loader.mode == 'val':
        #     cv2.imwrite(f"augment_val_target_obj.png", np.moveaxis(
        #         augmented.numpy()*255, 0, -1))
        # if dataset_loader.mode == 'train':
        #     cv2.imwrite(f"augment_train_target_obj.png", np.moveaxis(
        #         augmented.numpy()*255, 0, -1))
        return augmented

    return frame_aug


class DIYBatchSampler(Sampler):
    """
    Customize any possible combination of both task families and sub-tasks in a batch of data.
    """

    def __init__(
        self,
        task_to_idx,
        subtask_to_idx,
        object_distribution_to_indx,
        sampler_spec=dict(),
        tasks_spec=dict(),
        n_step=0,
    ):
        """
        Args:
        - batch_size:
            total number of samples draw at each yield step
        - task_to_idx: {
            task_name: [all_idxs_for this task]}
        - sub_task_to_idx: {
            task_name: {
                {sub_task_id: [all_idxs_for this sub-task]}}
           all indics in both these dict()'s should sum to the total dataset size,
        - tasks_spec:
            should additionally contain batch-constructon guide:
            explicitly specify how to contruct the batch, use this spec we should be
            able to construct a mapping from each batch index to a fixed pair
            of [task_name, subtask_id] to sample from,
            but if set shuffle=true, the sampled batch would lose this ordering,
            e.g. give a _list_: ['${place}', '${nut_hard}']
            batch spec is extracted from:
                {'place':
                        {'task_ids':     [0,1,2],
                        'n_per_task':    [5, 10, 5]}
                'nut_hard':
                        {'task_ids':     [4],
                        'n_per_task':    [6]}
                'stack':
                        {...}
                }
                will yield a batch of 36 points, where first 5 comes from pickplace subtask#0, last 6 comes from nut-assembly task#4
        - shuffle:
            if true, we lose control over how each batch is distributed to gpus
        """
        batch_size = sampler_spec.get('batch_size', 30)
        drop_last = sampler_spec.get('drop_last', False)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.shuffle = sampler_spec.get('shuffle', False)

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.task_samplers = OrderedDict()
        self.task_iterators = OrderedDict()
        self.task_info = OrderedDict()
        self.balancing_policy = sampler_spec.get('balancing_policy', 0)
        self.object_distribution_to_indx = object_distribution_to_indx
        self.num_step = n_step
        for spec in tasks_spec:
            task_name = spec.name
            idxs = task_to_idx.get(task_name)
            self.task_samplers[task_name] = OrderedDict(
                {'all_sub_tasks': SubsetRandomSampler(idxs)})  # uniformly draw from union of all sub-tasks
            self.task_iterators[task_name] = OrderedDict(
                {'all_sub_tasks': iter(SubsetRandomSampler(idxs))})
            assert task_name in subtask_to_idx.keys(), \
                'Mismatch between {} task idxs and subtasks!'.format(task_name)
            num_loaded_sub_tasks = len(subtask_to_idx[task_name].keys())
            first_id = list(subtask_to_idx[task_name].keys())[0]

            sub_task_size = len(subtask_to_idx[task_name].get(first_id))
            print("Task {} loaded {} subtasks, starting from {}, should all have sizes {}".format(
                task_name, num_loaded_sub_tasks, first_id, sub_task_size))
            for sub_task, sub_idxs in subtask_to_idx[task_name].items():

                # the balancing has been requested
                if self.balancing_policy == 1 and self.object_distribution_to_indx != None:
                    self.task_samplers[task_name][sub_task] = [SubsetRandomSampler(
                        sample_list) for sample_list in object_distribution_to_indx[task_name][sub_task]]
                    self.task_iterators[task_name][sub_task] = [iter(SubsetRandomSampler(
                        sample_list)) for sample_list in object_distribution_to_indx[task_name][sub_task]]
                    for i, sample_list in enumerate(object_distribution_to_indx[task_name][sub_task]):
                        if len(sample_list) == 0:
                            print(
                                f"Task {task_name} - Sub-task {sub_task} - Position {i}")

                else:
                    self.task_samplers[task_name][sub_task] = SubsetRandomSampler(
                        sub_idxs)
                    assert len(sub_idxs) == sub_task_size, \
                        'Got uneven data sizes for sub-{} under the task {}!'.format(
                            sub_task, task_name)
                    self.task_iterators[task_name][sub_task] = iter(
                        SubsetRandomSampler(sub_idxs))
                    # print('subtask indexs:', sub_task, max(sub_idxs))
            curr_task_info = {
                'size':         len(idxs),
                'n_tasks':      len(subtask_to_idx[task_name].keys()),
                'sub_id_to_name': {i: name for i, name in enumerate(subtask_to_idx[task_name].keys())},
                'traj_per_subtask': sub_task_size,
                'sampler_len': -1  # to be decided below
            }
            self.task_info[task_name] = curr_task_info

        n_tasks = len(self.task_samplers.keys())
        n_total = sum([info['size'] for info in self.task_info.values()])

        self.idx_map = OrderedDict()
        idx = 0
        for spec in tasks_spec:
            name = spec.name
            _ids = spec.get('task_ids', None)
            n = spec.get('n_per_task', None)
            assert (
                _ids and n), 'Must specify which subtask ids to use and how many is contained in each batch'
            info = self.task_info[name]
            subtask_names = info.get('sub_id_to_name')
            for _id in _ids:
                subtask = subtask_names[_id]
                for _ in range(n):
                    self.idx_map[idx] = (name, subtask)
                    idx += 1
                sub_length = int(info['traj_per_subtask'] / n)
                self.task_info[name]['sampler_len'] = max(
                    sub_length, self.task_info[name]['sampler_len'])
        # print("Index map:", self.idx_map)

        self.max_len = max([info['sampler_len']
                           for info in self.task_info.values()])
        print('Max length for sampler iterator:', self.max_len)
        self.n_tasks = n_tasks

        assert idx == batch_size, "The constructed batch size {} doesn't match desired {}".format(
            idx, batch_size)
        self.batch_size = idx
        self.drop_last = drop_last

        print("Shuffling to break the task ordering in each batch? ", self.shuffle)

    def __iter__(self):
        """Given task families A,B,C, each has sub-tasks A00, A01,...
        Fix a total self.batch_size, sample different numbers of datapoints from
        each task"""
        batch = []
        for i in range(self.max_len):
            for idx in range(self.batch_size):
                (name, sub_task) = self.idx_map[idx]
                if self.balancing_policy == 1 and self.object_distribution_to_indx != None:
                    slot_indx = idx % len(self.task_samplers[name][sub_task])
                    # take one sample for the current task, sub_task, and slot
                    sampler = self.task_samplers[name][sub_task][slot_indx]
                    iterator = self.task_iterators[name][sub_task][slot_indx]
                    try:
                        batch.append(next(iterator))
                    except StopIteration:  # print('early sstop:', i, name)
                        # re-start the smaller-sized tasks
                        iterator = iter(sampler)
                        batch.append(next(iterator))
                        self.task_iterators[name][sub_task][slot_indx] = iterator
                else:
                    # print(name, sub_task)
                    sampler = self.task_samplers[name][sub_task]
                    iterator = self.task_iterators[name][sub_task]
                    try:
                        batch.append(next(iterator))
                    except StopIteration:  # print('early sstop:', i, name)
                        # re-start the smaller-sized tasks
                        iterator = iter(sampler)
                        batch.append(next(iterator))
                        self.task_iterators[name][sub_task] = iterator

            if len(batch) == self.batch_size:
                if self.shuffle:
                    random.shuffle(batch)
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            if self.shuffle:
                random.shuffle(batch)
            yield batch

    def __len__(self):
        # Since different task may have different data sizes,
        # define total length of sampler as number of iterations to
        # exhaust the last task
        return self.max_len
