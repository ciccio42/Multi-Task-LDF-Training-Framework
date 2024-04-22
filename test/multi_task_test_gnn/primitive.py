import copy
import numpy as np
import cv2
from multi_task_il.models.vima.utils import *
import robosuite.utils.transform_utils as T
from pyquaternion import Quaternion


TARGET_GRIPPER_WRT_OBJ = {'pick_place':  np.array(
    [[1, 0, 0.], [0, -1, 0.], [0., 0., -1.]])}


def compute_target_orientation(obj_rot, task_name):

    obj_rot = T.quat2mat(T.axisangle2quat(obj_rot))
    world_ee_rot = np.matmul(obj_rot, TARGET_GRIPPER_WRT_OBJ[task_name])

    return Quaternion(matrix=world_ee_rot)


def reaching_primitive(env: object, desired_action: np.array, trajectory: object, task_name: str):

    # Action Logic

    # 1. Align with the object
    # get the current gripper position
    current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]

    target_orientation_quat = compute_target_orientation(
        obj_rot=desired_action[3:],
        task_name=task_name)

    action_alignment = copy.copy(desired_action)
    # change z value with the current value
    action_alignment[2] = current_gripper_position[2]
    t = 0
    while np.linalg.norm(current_gripper_position[:2]-action_alignment[:2]) > 0.005 and t < 100:
        current_gripper_orientation_quat = Quaternion(matrix=np.reshape(
            env.sim.data.site_xmat[env.robots[0].eef_site_id], (3, 3)))
        quat_t = Quaternion.slerp(
            current_gripper_orientation_quat,
            target_orientation_quat,
            min(1, float(t) / 20))

        delta_position = np.round(
            action_alignment[:3] - current_gripper_position, 2)
        delta_position = np.clip(delta_position, -0.02, 0.02)
        action_to_perform = copy.copy(action_alignment)
        action_to_perform[:3] = current_gripper_position + delta_position
        action_to_perform[3:] = T.quat2axisangle(
            np.array([quat_t.x, quat_t.y, quat_t.z, quat_t.w]))
        action_to_perform = np.concatenate(
            (action_to_perform, np.array([-1])), axis=-1)
        obs, reward, env_done, info = env.step(action_to_perform)
        cv2.imwrite("debug.png", obs['camera_front_image'][:, :, ::-1])
        trajectory.append(obs, reward, False, info, desired_action)
        current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]
        t += 1

    if t == 100:
        print("Failed in alignement")
        return obs, reward, env_done, info, False

    # 2. Move toward the object
    current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]
    action_pick_object = copy.copy(desired_action)
    t = 0
    while abs(current_gripper_position[2]-action_pick_object[2]) > 0.005 and t < 100:
        current_gripper_orientation_quat = Quaternion(matrix=np.reshape(
            env.sim.data.site_xmat[env.robots[0].eef_site_id], (3, 3)))
        quat_t = Quaternion.slerp(
            current_gripper_orientation_quat,
            target_orientation_quat,
            min(1, float(t) / 20))

        delta_position = np.round(
            action_pick_object[:3] - current_gripper_position, 2)
        delta_position = np.clip(delta_position, -0.02, 0.02)
        action_to_perform = copy.copy(action_pick_object)
        action_to_perform[:3] = current_gripper_position + delta_position
        action_to_perform[3:] = T.quat2axisangle(
            np.array([quat_t.x, quat_t.y, quat_t.z, quat_t.w]))
        action_to_perform = np.concatenate(
            (action_to_perform, np.array([-1])), axis=-1)
        obs, reward, env_done, info = env.step(action_to_perform)
        cv2.imwrite("debug.png", obs['camera_front_image'][:, :, ::-1])
        trajectory.append(obs, reward, False, info, desired_action)
        current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]
        t += 1

    if t == 100:
        print("Failed in moving toward the object")
        return obs, reward, env_done, info, False

    # 3. Close the gripper
    action_pick_object[3:] = T.quat2axisangle(
        np.array([target_orientation_quat.x, target_orientation_quat.y, target_orientation_quat.z, target_orientation_quat.w]))
    action_pick_object = np.concatenate(
        (action_pick_object, np.array([1])), axis=-1)
    obs, reward, env_done, info = env.step(action_pick_object)
    cv2.imwrite("debug.png", obs['camera_front_image'][:, :, ::-1])
    trajectory.append(obs, reward, False, info, desired_action)

    return obs, reward, env_done, info, True


def placing_primitive(env: object, desired_action: np.array, trajectory: object, task_name: str):
    # Action Logic

    # 1. Move up from current position
    current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]
    current_gripper_orientation = T.quat2axisangle(T.mat2quat(np.reshape(
        env.sim.data.site_xmat[env.robots[0].eef_site_id], (3, 3))))
    current_gripper_pose = np.concatenate(
        (current_gripper_position, current_gripper_orientation, np.array([1])), axis=-1)

    target_orientation_quat = compute_target_orientation(
        obj_rot=desired_action[3:],
        task_name=task_name)

    start_gripper_pose = current_gripper_pose
    t = 0
    while abs(current_gripper_pose[2] - start_gripper_pose[2]) < 0.10 and t < 100:
        current_gripper_orientation_quat = Quaternion(matrix=np.reshape(
            env.sim.data.site_xmat[env.robots[0].eef_site_id], (3, 3)))
        quat_t = Quaternion.slerp(
            current_gripper_orientation_quat,
            target_orientation_quat,
            min(1, float(t) / 20))

        delta_position = np.array([0, 0, 0.02])
        action_to_perform = copy.copy(current_gripper_pose)
        action_to_perform[:3] = action_to_perform[:3] + delta_position
        action_to_perform[3:6] = T.quat2axisangle(
            np.array([quat_t.x, quat_t.y, quat_t.z, quat_t.w]))
        obs, reward, env_done, info = env.step(action_to_perform)
        cv2.imwrite("debug.png", obs['camera_front_image'][:, :, ::-1])
        trajectory.append(obs, reward, False, info, desired_action)
        current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]
        current_gripper_orientation = T.quat2axisangle(T.mat2quat(np.reshape(
            env.sim.data.site_xmat[env.robots[0].eef_site_id], (3, 3))))
        current_gripper_pose = np.concatenate(
            (current_gripper_position, current_gripper_orientation, np.array([1])), axis=-1)
        t += 1

    if t == 100:
        print("Failed in moving up")
        return obs, reward, env_done, info, False

    # 2. Move toward bin
    target_action = copy.copy(desired_action)
    target_action[2] = current_gripper_pose[2]
    t = 0
    while np.linalg.norm(current_gripper_position[:2]-target_action[:2]) > 0.005 and t < 100:
        current_gripper_orientation_quat = Quaternion(matrix=np.reshape(
            env.sim.data.site_xmat[env.robots[0].eef_site_id], (3, 3)))
        quat_t = Quaternion.slerp(
            current_gripper_orientation_quat,
            target_orientation_quat,
            min(1, float(t) / 20))

        delta_position = np.round(
            target_action[:3] - current_gripper_position, 2)
        delta_position = np.clip(delta_position, -0.02, 0.02)
        action_to_perform = copy.copy(target_action)
        action_to_perform[:3] = current_gripper_position + delta_position
        action_to_perform[3:6] = T.quat2axisangle(
            np.array([quat_t.x, quat_t.y, quat_t.z, quat_t.w]))
        action_to_perform = np.concatenate(
            (action_to_perform, np.array([1])), axis=-1)
        obs, reward, env_done, info = env.step(action_to_perform)
        cv2.imwrite("debug.png", obs['camera_front_image'][:, :, ::-1])
        trajectory.append(obs, reward, False, info, desired_action)
        current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]
        current_gripper_orientation = T.quat2axisangle(T.mat2quat(np.reshape(
            env.sim.data.site_xmat[env.robots[0].eef_site_id], (3, 3))))
        current_gripper_pose = np.concatenate(
            (current_gripper_position, current_gripper_orientation, np.array([1])), axis=-1)
        t += 1
    if t == 100:
        print("Failed in moving up")
        return obs, reward, env_done, info, False

    # 4. Placing object
    current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]
    target_action[2] = desired_action[2]
    t = 0
    while abs(current_gripper_position[2]-target_action[2]) > 0.005 and t < 100:
        current_gripper_orientation_quat = Quaternion(matrix=np.reshape(
            env.sim.data.site_xmat[env.robots[0].eef_site_id], (3, 3)))
        quat_t = Quaternion.slerp(
            current_gripper_orientation_quat,
            target_orientation_quat,
            min(1, float(t) / 20))

        delta_position = np.round(
            target_action[:3] - current_gripper_position, 2)
        delta_position = np.clip(delta_position, -0.02, 0.02)
        action_to_perform = target_action
        action_to_perform[:3] = current_gripper_position + delta_position
        action_to_perform[3:6] = T.quat2axisangle(
            np.array([quat_t.x, quat_t.y, quat_t.z, quat_t.w]))
        action_to_perform = np.concatenate(
            (action_to_perform, np.array([1])), axis=-1)
        obs, reward, env_done, info = env.step(action_to_perform)
        cv2.imwrite("debug.png", obs['camera_front_image'][:, :, ::-1])
        trajectory.append(obs, reward, False, info, desired_action)
        current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]
        t += 1
    if t == 100:
        print("Failed in placing ")
        return obs, reward, env_done, info, False

    # 5. Open the gripper
    desired_action[3] = T.quat2axisangle(
        np.array([target_orientation_quat.x, target_orientation_quat.y, target_orientation_quat.z, target_orientation_quat.w]))
    desired_action = np.concatenate((desired_action, np.array([-1])), axis=-1)
    obs, reward, env_done, info = env.step(desired_action)
    cv2.imwrite("debug.png", obs['camera_front_image'][:, :, ::-1])
    trajectory.append(obs, reward, False, info, desired_action)

    return obs, reward, env_done, info, True


def assembly_primitive(env: object, desired_action: np.array, trajectory: object):
    # Action Logic

    # 1. Move up from current position
    current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]
    current_gripper_orientation = T.quat2axisangle(T.mat2quat(np.reshape(
        env.sim.data.site_xmat[env.robots[0].eef_site_id], (3, 3))))
    current_gripper_pose = np.concatenate(
        (current_gripper_position, current_gripper_orientation, np.array([1])), axis=-1)

    start_gripper_pose = current_gripper_pose
    t = 0
    while abs(current_gripper_pose[2] - start_gripper_pose[2]) < 0.15 and t < 100:
        delta_position = np.array([0, 0, 0.02])
        action_to_perform = copy.copy(current_gripper_pose)
        action_to_perform[:3] = action_to_perform[:3] + delta_position
        obs, reward, env_done, info = env.step(action_to_perform)
        cv2.imwrite("debug.png", obs['camera_front_image'][:, :, ::-1])
        trajectory.append(obs, reward, False, info, action_to_perform)
        current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]
        current_gripper_orientation = T.quat2axisangle(T.mat2quat(np.reshape(
            env.sim.data.site_xmat[env.robots[0].eef_site_id], (3, 3))))
        current_gripper_pose = np.concatenate(
            (current_gripper_position, current_gripper_orientation, np.array([1])), axis=-1)
        t += 1

    if t == 100:
        print("Failed in moving up")
        return obs, reward, env_done, info, False

    # 2. Move toward the peg
    target_action = copy.copy(desired_action)
    target_action[2] = current_gripper_pose[2]
    t = 0
    while np.linalg.norm(current_gripper_position[:2]-target_action[:2]) > 0.005 and t < 100:
        delta_position = np.round(
            target_action[:3] - current_gripper_position, 2)
        delta_position = np.clip(delta_position, -0.02, 0.02)
        action_to_perform = copy.copy(target_action)
        action_to_perform[:3] = current_gripper_position + delta_position
        action_to_perform = np.concatenate(
            (action_to_perform, np.array([1])), axis=-1)
        obs, reward, env_done, info = env.step(action_to_perform)
        cv2.imwrite("debug.png", obs['camera_front_image'][:, :, ::-1])
        trajectory.append(obs, reward, False, info, desired_action)
        current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]
        current_gripper_orientation = T.quat2axisangle(T.mat2quat(np.reshape(
            env.sim.data.site_xmat[env.robots[0].eef_site_id], (3, 3))))
        current_gripper_pose = np.concatenate(
            (current_gripper_position, current_gripper_orientation, np.array([1])), axis=-1)
        t += 1
    if t == 100:
        print("Failed in moving up")
        return obs, reward, env_done, info, False

    # 4. Assembly object
    current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]
    target_action[2] = desired_action[2]
    t = 0
    while abs(current_gripper_position[2]-target_action[2]) > 0.005 and t < 100:
        delta_position = np.round(
            target_action[:3] - current_gripper_position, 2)
        delta_position = np.clip(delta_position, -0.02, 0.02)
        action_to_perform = target_action
        action_to_perform[:3] = current_gripper_position + delta_position
        action_to_perform = np.concatenate(
            (action_to_perform, np.array([1])), axis=-1)
        obs, reward, env_done, info = env.step(action_to_perform)
        cv2.imwrite("debug.png", obs['camera_front_image'][:, :, ::-1])
        trajectory.append(obs, reward, False, info, desired_action)
        current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]
        t += 1
    if t == 100:
        print("Failed in placing ")
        return obs, reward, env_done, info, False

    # 5. Open the gripper
    desired_action = np.concatenate((desired_action, np.array([-1])), axis=-1)
    obs, reward, env_done, info = env.step(desired_action)
    cv2.imwrite("debug.png", obs['camera_front_image'][:, :, ::-1])
    trajectory.append(obs, reward, False, info, desired_action)

    return obs, reward, env_done, info, True


def perform_primitive(env: object, action: dict() = None, primitive: str = "reaching", trajectory: object = None):

    if primitive == "reaching":
        print("Reaching object")
        return reaching_primitive(env, action, trajectory)
    elif primitive == "placing":
        print("Placing object")
        return placing_primitive(env, action, trajectory)
    elif primitive == "assembly":
        print("Assembly")
        return assembly_primitive(env, action, trajectory)
    else:
        print(f"Primitive {primitive} not implemented")


def perform_pick_place_primitive(env: object, picking_loc: np.array, placing_loc: np.array, trajectory: object = None):

    if picking_loc.shape[0] == 6:
        pass
    _, _, _, _, success = reaching_primitive(env=env,
                                             desired_action=picking_loc,
                                             trajectory=trajectory,
                                             task_name='pick_place')
    if success:
        _, _, _, _, success = placing_primitive(env=env,
                                                desired_action=placing_loc,
                                                trajectory=trajectory,
                                                task_name='pick_place')
        if success:
            return _, True
        else:
            return "placing", False

    else:
        return "reaching", False

    return False
