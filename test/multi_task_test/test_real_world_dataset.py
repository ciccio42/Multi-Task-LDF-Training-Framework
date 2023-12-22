"""
Evaluate each task for the same number of --eval_each_task times. 
"""
import warnings
from robosuite import load_controller_config
from multi_task_robosuite_env.controllers.controllers.expert_nut_assembly import \
    get_expert_trajectory as nut_expert
from multi_task_robosuite_env.controllers.controllers.expert_pick_place import \
    get_expert_trajectory as place_expert
import cv2
import random
import os
from os.path import join
from collections import defaultdict
import torch
from multi_task_il.datasets import Trajectory
import numpy as np
import pickle as pkl
import functools
from torch.multiprocessing import Pool, set_start_method
import json
import wandb
from collections import OrderedDict
import hydra
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import resized_crop
import learn2learn as l2l
from torchvision.transforms import ToTensor
from multi_task_test.nut_assembly import nut_assembly_eval
from multi_task_test.pick_place import pick_place_eval
from multi_task_test import select_random_frames
import re


set_start_method('forkserver', force=True)
LOG_PATH = None
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def extract_last_number(path):
    # Use regular expression to find the last number in the path
    check_point_number = path.split(
        '/')[-1].split('_')[-1].split('-')[-1].split('.')[0]
    return int(check_point_number)


def build_tvf_formatter(config, env_name='stack'):
    """Use this for torchvision.transforms in multi-task dataset, 
    note eval_fn always feeds in traj['obs']['images'], i.e. shape (h,w,3)
    """
    dataset_cfg = config.train_cfg.dataset
    height, width = dataset_cfg.get(
        'height', 100), dataset_cfg.get('width', 180)
    task_spec = config.tasks_cfgs.get(env_name, dict())

    crop_params = task_spec.get('crop', [0, 0, 0, 0])
    # print(crop_params)
    top, left = crop_params[0], crop_params[2]

    def resize_crop(img):
        if len(img.shape) == 4:
            img = img[0]
        img_h, img_w = img.shape[0], img.shape[1]
        assert img_h != 3 and img_w != 3, img.shape
        box_h, box_w = img_h - top - \
            crop_params[1], img_w - left - crop_params[3]
        # cv2.imwrite("obs.png", np.array(img))
        obs = ToTensor()(img.copy())
        obs = resized_crop(obs, top=top, left=left, height=box_h, width=box_w,
                           size=(height, width))
        cv2.imwrite("resized_test.png",
                    np.moveaxis(obs.numpy(), 0, -1)*255)

        # weak_scale = config.augs.get('weak_crop_scale', (0.8, 1.0))
        # weak_ratio = [1.0, 1.0]
        # randcrop = RandomResizedCrop(
        #     size=(height, width), scale=weak_scale, ratio=weak_ratio)
        # cv2.imwrite("obs_cropped.png", np.moveaxis(obs.numpy(), 0, -1)*255)
        # # obs = Normalize(mean=[0.485, 0.456, 0.406],
        # #                 std=[0.229, 0.224, 0.225])(obs)
        # obs = randcrop(obs)
        cv2.imwrite("random_resized_crop_test.png",
                    np.moveaxis(obs.numpy(), 0, -1)*255)
        return obs
    return resize_crop


def build_tvf_formatter_obj_detector(config, env_name):
    """Use this for torchvision.transforms in multi-task dataset, 
    note eval_fn always feeds in traj['obs']['images'], i.e. shape (h,w,3)
    """

    def resize_crop(img, bb=None):
        img_height, img_width = img.shape[:2]
        """applies to every timestep's RGB obs['camera_front_image']"""
        task_spec = config.tasks_cfgs.get(env_name, dict())
        crop_params = task_spec.get('crop', [0, 0, 0, 0])
        top, left = crop_params[0], crop_params[2]
        img_height, img_width = img.shape[0], img.shape[1]
        box_h, box_w = img_height - top - \
            crop_params[1], img_width - left - crop_params[3]

        img = transforms.ToTensor()(img.copy())
        # ---- Resized crop ----#
        img = resized_crop(img, top=top, left=left, height=box_h,
                           width=box_w, size=(config.dataset_cfg.height, config.dataset_cfg.width))
        # transforms_pipe = transforms.Compose([
        #     transforms.ColorJitter(
        #         brightness=list(config.augs.get(
        #             "brightness", [0.875, 1.125])),
        #         contrast=list(config.augs.get(
        #             "contrast", [0.5, 1.5])),
        #         saturation=list(config.augs.get(
        #             "contrast", [0.5, 1.5])),
        #         hue=list(config.augs.get("hue", [-0.05, 0.05]))
        #     ),
        # ])
        # img = transforms_pipe(img)

        # cv2.imwrite("resized_target_obj.png", np.moveaxis(
        #     img.numpy()*255, 0, -1))

        if bb is not None:
            from multi_task_il.datasets.utils import adjust_bb
            bb = adjust_bb(dataset_loader=config.dataset_cfg,
                           bb=bb,
                           obs=img,
                           img_height=img_height,
                           img_width=img_width,
                           top=top,
                           left=left,
                           box_w=box_w,
                           box_h=box_h)

            # image = cv2.rectangle(np.ascontiguousarray(np.array(np.moveaxis(
            #     img.numpy()*255, 0, -1), dtype=np.uint8)),
            #     (bb[0][0],
            #      bb[0][1]),
            #     (bb[0][2],
            #      bb[0][3]),
            #     color=(0, 0, 255),
            #     thickness=1)
            # cv2.imwrite("bb_cropped.png", image)
            return img, bb

        return img

    return resize_crop


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--wandb_log', action='store_true')
    parser.add_argument('--project_name', '-pn', default="mosaic", type=str)
    parser.add_argument('--config', default='')
    parser.add_argument('--N', default=-1, type=int)
    parser.add_argument('--use_h', default=-1, type=int)
    parser.add_argument('--use_w', default=-1, type=int)
    parser.add_argument('--num_workers', default=3, type=int)
    # for block stacking only!
    parser.add_argument('--size', action='store_true')
    parser.add_argument('--shape', action='store_true')
    parser.add_argument('--color', action='store_true')
    parser.add_argument('--env', '-e', default='door', type=str)
    parser.add_argument('--eval_each_task',  default=30, type=int)
    parser.add_argument('--eval_subsets',  default=0, type=int)
    parser.add_argument('--saved_step', '-s', default=1000, type=int)
    parser.add_argument('--baseline', '-bline', default=None, type=str,
                        help='baseline uses more frames at each test-time step')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--results_name', default=None, type=str)
    parser.add_argument('--variation', default=None, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--controller_path', default=None, type=str)
    parser.add_argument('--gpu_id', default=-1, type=int)
    parser.add_argument('--save_path', default=None, type=str)
    parser.add_argument('--test_gt', action='store_true')
    parser.add_argument('--save_files', action='store_true')
    parser.add_argument('--gt_bb', action='store_true')

    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    try_path = args.model
    # if 'log' not in args.model and 'mosaic' not in args.model:
    #     print("Appending dir to given exp_name: ", args.model)
    #     try_path = join(LOG_PATH, args.model)
    #     assert os.path.exists(try_path), f"Cannot find {try_path} anywhere"
    try_path_list = []
    if 'model_save' not in args.model:
        print(args.saved_step)
        if args.saved_step != -1:
            print("Appending saved step {}".format(args.saved_step))
            try_path = join(
                try_path, 'model_save-{}.pt'.format(args.saved_step))
            try_path_list.append(try_path)
            assert os.path.exists(
                try_path), "Cannot find anywhere: " + str(try_path)
        else:
            import glob
            print(f"Finding checkpoints in {try_path}")
            check_point_list = glob.glob(
                os.path.join(try_path, "model_save-*.pt"))
            exclude_pattern = r'model_save-optim-\d+\.pt'
            check_point_list = [
                path for path in check_point_list if not re.search(exclude_pattern, path)]
            check_point_list = sorted(
                check_point_list, key=extract_last_number)
            # take the last check point
            try_paths = check_point_list
            epoch_numbers = len(try_paths)
            try_path_list = try_paths[-10:]

    # 1. Create dataset

    for try_path in try_path_list:
        pass
