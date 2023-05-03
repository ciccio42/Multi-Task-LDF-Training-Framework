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
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger("BB-Creator")

ENV_OBJECTS = {
    'pick_place': {
        'obj_names': ['milk', 'bread', 'cereal', 'can'],
        'obj_dim': {'milk': [0.05, 0.12, 0.045],
                    'bread': [0.045, 0.055, 0.045],
                    'cereal': [0.045, 0.055, 0.045],
                    'can': [0.045, 0.065, 0.045]},
        'world_to_camera_rot': np.array([[0.0, -0.70, 0.707852],
                                         [0.9999999999999953, 0.0, 0.0],
                                         [0.0, 0.70, 0.707852]]),
        'world_to_camera_pos': np.array([[0.5],
                                         [0.0],
                                         [1.35]]),
        'camera_fovy': 45,
        'img_dim': [100, 180]},
    'nut_assembly': {
        'obj_names': ['nut0', 'nut1', 'nut2'],
        'ranges': [[0.10, 0.31], [-0.10, 0.10], [-0.31, -0.10]]
    }
}


def plot_bb(img, obj_bb):

    # draw bb
    for obj_name in obj_bb.keys():
        center = obj_bb[obj_name]['center']
        upper_left_corner = obj_bb[obj_name]['upper_left_corner']
        bottom_right_corner = obj_bb[obj_name]['bottom_right_corner']
        img = cv2.circle(
            img, center, radius=1, color=(0, 0, 255), thickness=-1)
        img = cv2.rectangle(
            img, upper_left_corner,
            bottom_right_corner, (255, 0, 0), 1)

    cv2.imshow("Test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def overwrite_pkl_file(pkl_file_path, sample, traj_obj_bb):
    # get trajectory from sample
    new_sample = copy.deepcopy(sample)

    traj = new_sample['traj']

    # modify trajectory observation
    for t in range(len(traj)):
        try:
            obs = traj.get(t)['obs']
        except:
            _img = traj._data[t][0]['image']
            okay, im_string = cv2.imencode(
                '.jpg', _img)
            traj._data[t][0]['image'] = im_string
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


if __name__ == '__main__':
    import debugpy
    debugpy.listen(('0.0.0.0', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_path', default="/", help="Path to task")
    parser.add_argument('--task_name', default="/", help="Name of the task")
    parser.add_argument('--robot_name', default="/", help="Name of the robot")
    args = parser.parse_args()

    # 1. Load the dataset
    folder_path = os.path.join(
        args.task_path, args.task_name, f"{args.robot_name}_{args.task_name}")

    for dir in os.listdir(folder_path):
        if "task" in dir:
            task_path = os.path.join(folder_path, dir)
            i = 0
            for pkl in os.listdir(task_path):
                pkl_file_path = os.path.join(task_path, pkl)
                logger.info(f"Task id {dir} - Trajectory {pkl}")
                # 2. Load pickle file
                with open(pkl_file_path, "rb") as f:
                    sample = pickle.load(f)
                # 3. Identify objects positions
                object_name_list = ENV_OBJECTS[args.task_name]['obj_names']
                traj = sample['traj']
                traj_bb = []
                for t in range(len(sample['traj'])):
                    # for each object in the observation get the position
                    obj_positions = dict()
                    obj_bb = dict()
                    if args.task_name == 'pick_place':
                        try:
                            obs = traj.get(t)['obs']
                        except:
                            _img = traj._data[t][0]['image']
                            okay, im_string = cv2.imencode(
                                '.jpg', _img)
                            traj._data[t][0]['image'] = im_string
                            obs = traj.get(t)['obs']

                        logger.debug(obs.keys())
                        for obj_name in object_name_list:
                            obj_positions[obj_name] = obs[f"{obj_name}_pos"]

                    # for each object create bb
                    for obj_name in object_name_list:
                        logger.debug(f"\nObject: {obj_name}")
                        # convert obj pos in camera coordinate
                        obj_pos = obj_positions[obj_name]
                        obj_pos = np.array([obj_pos])
                        # 1. Compute rotation_camera_to_world
                        r_camera_world = ENV_OBJECTS[args.task_name]['world_to_camera_rot'].T
                        p_camera_world = - \
                            r_camera_world @ ENV_OBJECTS[args.task_name]['world_to_camera_pos']
                        # 2. Create transformation matrix
                        T_camera_world = np.concatenate(
                            (r_camera_world, p_camera_world), axis=1)
                        T_camera_world = np.concatenate(
                            (T_camera_world, np.array([[0, 0, 0, 1]])), axis=0)
                        # logger.debug(T_camera_world)
                        p_world_object = np.expand_dims(
                            np.insert(obj_pos, 3, 1), 0).T
                        p_camera_object = T_camera_world @ p_world_object
                        logger.debug(
                            f"\nP_world_object:\n{p_world_object} - \nP_camera_object:\n {p_camera_object}")

                        # 2.1 Compute the position of the upper-left and bottom-right corner to compute width and height of bb
                        p_world_object_upper_left_corner = p_world_object + \
                            np.array(
                                [[0.0],
                                 [-ENV_OBJECTS[args.task_name]
                                     ['obj_dim'][obj_name][0]/2],
                                 [ENV_OBJECTS[args.task_name]
                                     ['obj_dim'][obj_name][1]/2],
                                 [0]])
                        p_camera_object_upper_left_corner = T_camera_world @               p_world_object_upper_left_corner
                        logger.debug(
                            f"\nP_world_object_upper_left:\n{p_world_object_upper_left_corner} -   \nP_camera_object_upper_left:\n {p_camera_object_upper_left_corner}")

                        p_world_object_bottom_right_corner = p_world_object + \
                            np.array(
                                [[0.0],
                                 [ENV_OBJECTS[args.task_name]
                                     ['obj_dim'][obj_name][0]/2],
                                 [-ENV_OBJECTS[args.task_name]
                                     ['obj_dim'][obj_name][1]/2],
                                 [0]])
                        p_camera_object_bottom_right_corner = T_camera_world @               p_world_object_bottom_right_corner
                        logger.debug(
                            f"\nP_world_object_bottom_right:\n{p_world_object_bottom_right_corner} -   \nP_camera_object_bottom_right:\n {p_camera_object_bottom_right_corner}")

                        # 3. Cnversion into pixel coordinates
                        f = 0.5 * ENV_OBJECTS[args.task_name]['img_dim'][0] / \
                            np.tan(ENV_OBJECTS[args.task_name]
                                   ['camera_fovy'] * np.pi / 360)
                        p_x = int(
                            (p_camera_object[0][0] / - p_camera_object[2][0]) * f + ENV_OBJECTS[args.task_name]['img_dim'][1] / 2)

                        p_y = int(
                            (- p_camera_object[1][0] / - p_camera_object[2][0]) * f + ENV_OBJECTS[args.task_name]['img_dim'][0] / 2)
                        logger.debug(
                            f"\nImage coordinate: px {p_x}, py {p_y}")

                        # 3.1 Upper-left corner and bottom right corner in pixel coordinate
                        p_x_upper_left = int(
                            (p_camera_object_upper_left_corner[0][0] / - p_camera_object_upper_left_corner[2][0]) * f + ENV_OBJECTS[args.task_name]['img_dim'][1] / 2) - 3

                        p_y_upper_left = int(
                            (- p_camera_object_upper_left_corner[1][0] / - p_camera_object_upper_left_corner[2][0]) * f + ENV_OBJECTS[args.task_name]['img_dim'][0] / 2) - 3
                        logger.debug(
                            f"\nImage coordinate upper_left corner: px {p_x_upper_left}, py {p_y_upper_left}")

                        p_x_bottom_right = int(
                            (p_camera_object_bottom_right_corner[0][0] / - p_camera_object_bottom_right_corner[2][0]) * f + ENV_OBJECTS[args.task_name]['img_dim'][1] / 2) + 3

                        p_y_bottom_right = int(
                            (- p_camera_object_bottom_right_corner[1][0] / - p_camera_object_bottom_right_corner[2][0]) * f + ENV_OBJECTS[args.task_name]['img_dim'][0] / 2) + 3
                        logger.debug(
                            f"\nImage coordinate bottom_right corner: px {p_x_bottom_right}, py {p_y_bottom_right}")

                        # save bb
                        obj_bb[obj_name] = dict()
                        obj_bb[obj_name]['center'] = [p_x, p_y]
                        obj_bb[obj_name]['upper_left_corner'] = [
                            p_x_upper_left, p_y_upper_left]
                        obj_bb[obj_name]['bottom_right_corner'] = [
                            p_x_bottom_right, p_y_bottom_right]

                    traj_bb.append(obj_bb)
                    # # draw center
                    # img = np.array(traj.get(
                    #     t)['obs']['image'][:, :, ::-1])
                    # plot_bb(img, obj_bb)

                # save sample with objects bb
                overwrite_pkl_file(pkl_file_path=pkl_file_path,
                                   sample=sample,
                                   traj_obj_bb=traj_bb)
