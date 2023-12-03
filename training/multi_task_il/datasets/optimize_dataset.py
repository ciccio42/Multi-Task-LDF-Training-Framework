"""\
This file loads the trajectories in pkl format from the specified folder and add the bounding-box related to the objects in the scene
The bounding box is defined as follow: (center_x, center_y, width, height)
"""

from multi_task_il.datasets.savers import _compress_obs
import os
import sys
import pickle
import cv2
import numpy as np
import logging
import copy
import robosuite.utils.transform_utils as T
import functools
from multiprocessing import Pool, cpu_count
import glob
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import resized_crop
import yaml

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger("BB-Creator")

KEY_INTEREST = ["joint_pos", "joint_vel", "eef_pos",
                "eef_quat", "gripper_qpos", "gripper_qvel", "camera_front_image",
                "target-box-id", "target-object", "obj_bb",
                "extent", "zfar", "znear", "eef_point", "ee_aa", "target-peg"]


def overwrite_pkl_file(pkl_file_path, sample, traj_obj_bb):
    # get trajectory from sample
    new_sample = copy.deepcopy(sample)

    traj = new_sample['traj']

    # modify trajectory observation
    for t in range(len(traj)):
        try:
            obs = traj.get(t)['obs']
        except:
            _img = traj._data[t][0]['camera_front_image']
            okay, im_string = cv2.imencode(
                '.jpg', _img)
            traj._data[t][0]['camera_front_image'] = im_string
            obs = traj.get(t)['obs']

        obs['obj_bb'] = traj_obj_bb[t]
        obs = _compress_obs(obs)
        traj.change_obs(t, obs)
        logger.debug(obs.keys())

    pickle.dump({
        'traj': traj,
        'len': len(traj),
        'env_type': sample['env_type'],
        'task_id': sample['task_id']}, open(pkl_file_path, 'wb'))


def opt_traj(task_name, task_spec, out_path, pkl_file_path):
    # pkl_file_path = os.path.join(task_path, pkl_file_path)
    # logger.info(f"Task id {dir} - Trajectory {pkl_file_path}")
    # 2. Load pickle file
    with open(pkl_file_path, "rb") as f:
        sample = pickle.load(f)

    keys = list(sample['traj']._data[0][0].keys())
    keys_to_remove = []
    for key in keys:
        if key not in KEY_INTEREST:
            keys_to_remove.append(key)

    # remove data not of interest for training
    for t in range(len(sample['traj'])):
        for key in keys_to_remove:
            sample['traj']._data[t][0].pop(key)

    trj_name = pkl_file_path.split('/')[-1]
    out_pkl_file_path = os.path.join(out_path, trj_name)
    with open(out_pkl_file_path, "wb") as f:
        pickle.dump(sample, f)


if __name__ == '__main__':
    import debugpy
    # debugpy.listen(('0.0.0.0', 5678))
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="/", help="Path to task")
    parser.add_argument('--task_name', default="/", help="Name of the task")
    parser.add_argument('--robot_name', default="/", help="Name of the robot")
    parser.add_argument('--out_path', default=None,)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    # 1. Load the dataset
    folder_path = os.path.join(
        args.dataset_path, args.task_name, f"{args.robot_name}_{args.task_name}")

    if args.out_path is None:
        out_path = os.path.join(args.dataset_path,
                                f"{args.task_name}_opt",
                                f"{args.robot_name}_{args.task_name}")
    else:
        out_path = os.path.join(args.out_path,
                                f"{args.task_name}_opt",
                                f"{args.robot_name}_{args.task_name}")

    os.makedirs(name=out_path, exist_ok=True)

    # load task configuration file
    conf_file_path = "/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/experiments/tasks_cfgs/7_tasks.yaml"
    with open(conf_file_path, 'r') as file:
        task_conf = yaml.safe_load(file)

    for dir in os.listdir(folder_path):
        # print(dir)
        if "task_" in dir:
            print(f"Considering task {dir}")
            out_task = os.path.join(out_path, dir)

            os.makedirs(name=out_task, exist_ok=True)

            task_path = os.path.join(folder_path, dir)

            i = 0
            trj_list = glob.glob(f"{task_path}/*.pkl")

            with Pool(10) as p:
                f = functools.partial(opt_traj,
                                      args.task_name,
                                      task_conf,
                                      out_task)
                p.map(f, trj_list)
