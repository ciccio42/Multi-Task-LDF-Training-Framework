"""\
This file loads the trajectories in pkl format from the specified folder and add the bounding-box related to the objects in the scene
The bounding box is defined as follow: (center_x, center_y, width, height)
"""
import yaml
from torchvision.transforms.functional import resized_crop
from torchvision.transforms import ToTensor
import glob
from multiprocessing import Pool, cpu_count
import functools
import robosuite.utils.transform_utils as T
import copy
import logging
import numpy as np
import cv2
import pickle
import sys
from multi_task_il.datasets.savers import _compress_obs
import os

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger("BB-Creator")

KEY_INTEREST = ["joint_pos", "joint_vel", "eef_pos",
                "eef_quat", "gripper_qpos", "gripper_qvel", "camera_front_image",
                "target-box-id", "target-object", "obj_bb",
                "extent", "zfar", "znear", "eef_point", "ee_aa", "target-peg"]


def crop_resize_img(task_cfg, task_name, obs, bb):
    """applies to every timestep's RGB obs['camera_front_image']"""
    crop_params = task_cfg[task_name].get('crop', [0, 0, 0, 0])
    top, left = crop_params[0], crop_params[2]
    img_height, img_width = obs.shape[0], obs.shape[1]
    box_h, box_w = img_height - top - \
        crop_params[1], img_width - left - crop_params[3]

    cropped_img = obs[top:box_h, left:box_w]
    cv2.imwrite("cropped.jpg", cropped_img)

    img_res = cv2.resize(cropped_img, (180, 100))
    adj_bb = None
    if bb is not None:
        adj_bb = adjust_bb(bb,
                           obs,
                           cropped_img,
                           img_res,
                           img_width=img_width,
                           img_height=img_height,
                           top=top,
                           left=left,
                           box_w=box_w,
                           box_h=box_h)

    return img_res, adj_bb


def adjust_bb(bb, original, cropped_img, obs, img_width=360, img_height=200, top=0, left=0, box_w=360, box_h=200):
    # For each bounding box
    for obj_indx, obj_bb_name in enumerate(bb):
        obj_bb = np.concatenate(
            (bb[obj_bb_name]['bottom_right_corner'], bb[obj_bb_name]['upper_left_corner']))
        # Convert normalized bounding box coordinates to actual coordinates
        x1_old, y1_old, x2_old, y2_old = obj_bb
        x1_old = int(x1_old)
        y1_old = int(y1_old)
        x2_old = int(x2_old)
        y2_old = int(y2_old)

        # Modify bb based on computed resized-crop
        # 1. Take into account crop and resize
        x_scale = obs.shape[1]/cropped_img.shape[1]
        y_scale = obs.shape[0]/cropped_img.shape[0]
        x1 = int(np.round((x1_old - left) * x_scale))
        x2 = int(np.round((x2_old - left) * x_scale))
        y1 = int(np.round((y1_old - top) * y_scale))
        y2 = int(np.round((y2_old - top) * y_scale))

        # image = cv2.rectangle(original,
        #                       (x1_old,
        #                        y1_old),
        #                       (x2_old,
        #                        y2_old),
        #                       color=(0, 0, 255),
        #                       thickness=1)
        # cv2.imwrite("bb_original.png", image)

        # image = cv2.rectangle(cropped_img,
        #                       (int((x1_old - left)),
        #                        int((y1_old - top))),
        #                       (int((x2_old - left)),
        #                        int((y2_old - top))),
        #                       color=(0, 0, 255),
        #                       thickness=1)
        # cv2.imwrite("bb_cropped.png", image)

        # image = cv2.rectangle(obs,
        #                       (x1,
        #                        y1),
        #                       (x2,
        #                        y2),
        #                       color=(0, 0, 255),
        #                       thickness=1)
        # cv2.imwrite("bb_cropped_resize.png", image)

        # replace with new bb
        bb[obj_bb_name]['bottom_right_corner'] = np.array([x2, y2])
        bb[obj_bb_name]['upper_left_corner'] = np.array([x1, y1])
        bb[obj_bb_name]['center'] = np.array([int((x2-x1)/2), int((y2-y1)/2)])
    return bb


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
            try:
                sample['traj']._data[t][0].pop(key)
            except:
                pass

    if "real" in pkl_file_path:
        # perform reshape a priori
        for t in range(len(sample['traj'])):
            for camera_name in ["camera_front", "camera_lateral_left", "camera_lateral_right", "eye_in_hand"]:
                img = sample['traj'].get(t)['obs'].get(
                    f"{camera_name}_image", None)
                if img is not None:
                    cv2.imwrite("original.png", img)
                    bb_dict = sample['traj'].get(
                        t)['obs'].get("obj_bb", None)
                    bb = None
                    if bb_dict is not None:
                        bb = bb_dict.get(camera_name, None)
                    img_res, adj_bb = crop_resize_img(task_cfg=task_spec,
                                                      task_name=task_name,
                                                      obs=img,
                                                      bb=bb
                                                      )
                    sample['traj'].get(
                        t)['obs'][f"{camera_name}_image"] = img_res
                    sample['traj'].get(
                        t)['obs']['obj_bb'][camera_name] = adj_bb
                    sample['traj'].get(
                        t)['obs']["target-object"] = int(int(sample['task_id'])/4)
                    img = copy.deepcopy(img_res)
                    for obj_name in adj_bb:
                        img = cv2.rectangle(img, adj_bb[obj_name]['upper_left_corner'],
                                            adj_bb[obj_name]['bottom_right_corner'],
                                            (0, 255, 0),
                                            1)
                    cv2.imwrite("prova.png", img)
                    print("prova image")

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
    parser.add_argument('--real', action='store_true')

    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    # 1. Load the dataset
    folder_path = os.path.join(
        args.dataset_path, args.task_name, f"{args.robot_name}_{args.task_name}")
    # folder_path = "/user/frosa/multi_task_lfd/ur_multitask_dataset/pick_place/real_ur5e_pick_place/only_frontal/"
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
    if args.real:
        conf_file_path = "/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/experiments/tasks_cfgs/7_tasks_real.yaml"
    else:
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

            with Pool(1) as p:
                f = functools.partial(opt_traj,
                                      args.task_name,
                                      task_conf,
                                      out_task)
                p.map(f, trj_list)
