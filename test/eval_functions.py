"""
New(0501): baseline models use different length inputs
Define how to evaluate the (new, harder) Nut assembly task
Note using only one frame of observation per timestep
"""
import random
import copy
import os
from collections import defaultdict, deque
from pyquaternion import Quaternion
import torch
import pickle as pkl
import numpy as np
from multi_task_il.datasets import Trajectory
import cv2
from multi_task_il.utils import denormalize_action
# Frezzes at this line if torchvision is imported
# cv2.imshow("debug", np.zeros((128, 128, 3), dtype=np.uint8))
# cv2.waitKey(1)
# cv2.destroyAllWindows()

ENV_OBJECTS = {
    'pick_place': {
        'obj_names': ['milk', 'bread', 'cereal', 'can'],
        'ranges':  [[0.16, 0.19], [0.05, 0.09], [-0.08, -0.03], [-0.19, -0.15]]
    },
    'nut_assembly': {
        'obj_names': ['nut0', 'nut1', 'nut2'],
        'ranges': [[0.10, 0.31], [-0.10, 0.10], [-0.31, -0.10]]
    }
}


def get_action(model, target_obj_dec, states, images, context, gpu_id, n_steps, max_T=80, baseline=None, action_ranges=[]):
    s_t = torch.from_numpy(np.concatenate(states, 0).astype(np.float32))[None]
    if isinstance(images[-1], np.ndarray):
        i_t = torch.from_numpy(np.concatenate(
            images, 0).astype(np.float32))[None]
    else:
        i_t = images[0][None]
    s_t, i_t = s_t.float().cuda(gpu_id), i_t.float().cuda(gpu_id)

    if baseline == 'maml':
        learner = model.clone()
        learner.adapt(
            learner(None, context[0], learned_loss=True)['learned_loss'], allow_nograd=True, allow_unused=True)
        out = learner(states=s_t[0], images=i_t[0], ret_dist=True)
        action = out['action_dist'].sample()[-1].cpu().detach().numpy()

    else:
        target_obj_embedding = None
        with torch.no_grad():
            out = model(states=s_t, images=i_t, context=context, eval=True,
                        target_obj_embedding=target_obj_embedding)  # to avoid computing ATC loss
            try:
                target_obj_embedding = out['target_obj_embedding']
            except:
                pass

            action = out['bc_distrib'].sample()[0, -1].cpu().numpy()
            if target_obj_dec is not None:
                target_obj_position = target_obj_dec(i_t, context, eval=True)
                predicted_prob = torch.nn.Softmax(dim=2)(
                    target_obj_position['target_obj_pred']).to('cpu').tolist()
            else:
                predicted_prob = None
    # action[3:7] = [1.0, 1.0, 0.0, 0.0]
    action = denormalize_action(action, action_ranges)
    action[-1] = 1 if action[-1] > 0 and n_steps < max_T - 1 else -1
    return action, predicted_prob


def startup_env(model, env, context, gpu_id, variation_id, baseline=None):
    done, states, images = False, [], []
    if baseline is None:
        states = deque(states, maxlen=1)
        images = deque(images, maxlen=1)  # NOTE: always use only one frame
    context = context.cuda(gpu_id).float()
    np.random.seed(None)
    while True:
        try:
            obs = env.reset()
            break
        except:
            pass
    traj = Trajectory()
    traj.append(obs)
    tasks = {'success': False, 'reached': False,
             'picked': False, 'variation_id': variation_id}
    return done, states, images, context, obs, traj, tasks


def nut_assembly_eval(model, env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False):

    done, states, images, context, obs, traj, tasks = \
        startup_env(model, env, context, gpu_id,
                    variation_id, baseline=baseline)

    object_name = env.nuts[env.nut_id].name
    if env.nut_id == 0:
        handle_loc = env.sim.data.site_xpos[env.sim.model.site_name2id(
            'round-nut_handle_site')]
    elif env.nut_id == 1:
        handle_loc = env.sim.data.site_xpos[env.sim.model.site_name2id(
            'round-nut-2_handle_site')]
    else:
        handle_loc = env.sim.data.site_xpos[env.sim.model.site_name2id(
            'round-nut-3_handle_site')]

    obj_key = object_name + '_pos'
    start_z = obs[obj_key][2]
    n_steps = 0
    while not done:
        tasks['reached'] = tasks['reached'] or np.linalg.norm(
            handle_loc - obs['eef_pos']) < 0.045
        tasks['picked'] = tasks['picked'] or (
            tasks['reached'] and obs[obj_key][2] - start_z > 0.05)
        if baseline and len(states) >= 5:
            states, images = [], []
        states.append(np.concatenate(
            (obs['ee_aa'], obs['gripper_qpos'])).astype(np.float32)[None])
        images.append(img_formatter(obs['camera_front_image'])[None])
        action = get_action(model, states, images, context,
                            gpu_id, n_steps, max_T, baseline)

        obs, reward, env_done, info = env.step(action)
        traj.append(obs, reward, done, info, action)
        tasks['success'] = (reward and tasks['reached']) or tasks['success']
        n_steps += 1
        if env_done or reward or n_steps > max_T:
            done = True
    env.close()
    del env
    del states
    del images
    del model
    # print("Done evaluating traj #{}, task#{}, success? {} ".format(ctr, variation_id, tasks['success']))
    return traj, tasks


def basketball_eval(model, env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False):
    done, states, images, context, obs, traj, tasks = \
        startup_env(model, env, context, gpu_id,
                    variation_id, baseline=baseline)

    obj_delta_key = 'gripper_to_target_obj'
    obj_key = 'target_obj_pos'
    start_z = obs[obj_key][2]
    n_steps = 0
    while not done:
        tasks['reached'] = tasks['reached'] or np.linalg.norm(
            obs[obj_delta_key][:2]) < 0.03
        tasks['picked'] = tasks['picked'] or (
            tasks['reached'] and obs[obj_key][2] - start_z > 0.05)
        if baseline and len(states) >= 5:
            states, images = [], []
        states.append(np.concatenate(
            (obs['ee_aa'], obs['gripper_qpos'])).astype(np.float32)[None])
        images.append(img_formatter(obs['camera_front_image'])[None])
        action = get_action(model, states, images, context,
                            gpu_id, n_steps, max_T, baseline)

        obs, reward, env_done, info = env.step(action)
        traj.append(obs, reward, done, info, action)

        tasks['success'] = reward or tasks['success']
        n_steps += 1
        if env_done or reward or n_steps > max_T:
            done = True
    env.close()
    del env
    del states
    del images
    del model

    return traj, tasks


def block_stack_eval(model, env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False):
    done, states, images, context, obs, traj, tasks = \
        startup_env(model, env, context, gpu_id,
                    variation_id, baseline=baseline)
    n_steps = 0
    obj_loc = env.sim.data.body_xpos[env.cubeA_body_id]
    obj_key = 'cubeA_pos'
    start_z = obs[obj_key][2]
    while not done:
        tasks['reached'] = tasks['reached'] or np.linalg.norm(
            obj_loc - obs['eef_pos']) < 0.045
        tasks['picked'] = tasks['picked'] or (
            tasks['reached'] and obs[obj_key][2] - start_z > 0.05)
        if baseline and len(states) >= 5:
            states, images = [], []
        states.append(np.concatenate(
            (obs['ee_aa'], obs['gripper_qpos'])).astype(np.float32)[None])
        images.append(img_formatter(obs['camera_front_image'])[None])

        action = get_action(model, states, images, context,
                            gpu_id, n_steps, max_T, baseline)

        obs, reward, env_done, info = env.step(action)
        traj.append(obs, reward, done, info, action)

        tasks['success'] = reward or tasks['success']
        n_steps += 1
        if env_done or reward or n_steps > max_T:
            done = True
    env.close()
    del env
    del states
    del images
    del model

    return traj, tasks


def press_button_eval(model, env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False):
    done, states, images, context, obs, traj, tasks = \
        startup_env(model, env, context, gpu_id,
                    variation_id, baseline=baseline)
    n_steps = 0

    button_loc = np.array(env.sim.data.site_xpos[env.target_button_id])
    dist = 0.015
    while not done:
        tasks['reached'] = tasks['reached'] or \
            np.linalg.norm(obs['eef_pos'] - button_loc) < dist
        tasks['picked'] = tasks['picked'] or \
            (tasks['reached'])  # and obs[obj_key][2] - start_z > 0.05)
        if baseline and len(states) >= 5:
            states, images = [], []
        states.append(np.concatenate(
            (obs['ee_aa'], obs['gripper_qpos'])).astype(np.float32)[None])
        images.append(img_formatter(obs['camera_front_image'])[None])

        action = get_action(model, states, images, context,
                            gpu_id, n_steps, max_T, baseline)
        action[3:7] = [1, 0.70710678, 0.70710678, 0]  # for Press only!

        obs, reward, env_done, info = env.step(action)
        traj.append(obs, reward, done, info, action)

        tasks['success'] = reward or tasks['success']
        if tasks['success']:
            tasks['reached'] = True
        n_steps += 1
        if env_done or reward or n_steps > max_T:
            done = True
    env.close()
    del env
    del states
    del images
    del model

    return traj, tasks


def pick_place_eval(model, target_obj_dec, env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False, action_ranges=[]):

    done, states, images, context, obs, traj, tasks = \
        startup_env(model, env, context, gpu_id,
                    variation_id, baseline=baseline)
    n_steps = 0

    object_name = env.objects[env.object_id].name
    obj_delta_key = object_name + '_to_robot0_eef_pos'
    obj_key = object_name + '_pos'

    start_z = obs[obj_key][2]

    # Compute the target obj-slot
    if target_obj_dec != None:
        agent_target_obj_position = -1
        agent_target_obj_id = traj.get(0)['obs']['target-object']
        for id, obj_name in enumerate(ENV_OBJECTS['pick_place']['obj_names']):
            if id == agent_target_obj_id:
                # get object position
                pos = traj.get(0)['obs'][f'{obj_name}_pos']
                for i, pos_range in enumerate(ENV_OBJECTS['pick_place']["ranges"]):
                    if pos[1] >= pos_range[0] and pos[1] <= pos_range[1]:
                        agent_target_obj_position = i
                break
    # compute the average prediction over the whole trajectory
    avg_prediction = 0
    print(f"Max-t {max_T}")
    while not done:
        tasks['reached'] = tasks['reached'] or np.linalg.norm(
            obs[obj_delta_key][:2]) < 0.03
        tasks['picked'] = tasks['picked'] or (
            tasks['reached'] and obs[obj_key][2] - start_z > 0.05)
        if baseline and len(states) >= 5:
            states, images = [], []
        states.append(np.concatenate(
            (obs['ee_aa'], obs['gripper_qpos'])).astype(np.float32)[None])
        # convert observation from BGR to RGB and scale to 0-1
        images.append(img_formatter(
            obs['camera_front_image'][:, :, ::-1]/255)[None])
        action, target_pred = get_action(
            model,
            target_obj_dec,
            states,
            images,
            context,
            gpu_id,
            n_steps,
            max_T,
            baseline,
            action_ranges)
        try:
            obs, reward, env_done, info = env.step(action)
            # cv2.imwrite(
            #     f"{n_steps}.png", obs['camera_front_image'][:, :, ::-1])
        except:
            print("Exception during step")
        if target_obj_dec is not None:
            info['target_pred'] = target_pred
            info['target_gt'] = agent_target_obj_position
            if np.argmax(target_pred) == agent_target_obj_position:
                avg_prediction += 1
        traj.append(obs, reward, done, info, action)

        tasks['success'] = reward or tasks['success']
        n_steps += 1
        if env_done or reward or n_steps > max_T:
            done = True
    env.close()
    tasks['avg_pred'] = avg_prediction/len(traj)
    del env
    del states
    del images
    del model

    return traj, tasks


def draw_eval(model, env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False):
    done, states, images, context, obs, traj, tasks = \
        startup_env(model, env, context, gpu_id,
                    variation_id, baseline=baseline)
    n_steps = 0
    handle_loc = np.array(env.sim.data.body_xpos[env.target_handle_body_id])

    while not done:
        tasks['reached'] = tasks['reached'] or np.linalg.norm(
            handle_loc - obs['eef_pos']) < 0.0175
        # no 'pick'
        if baseline and len(states) >= 5:
            states, images = [], []
        states.append(np.concatenate(
            (obs['ee_aa'], obs['gripper_qpos'])).astype(np.float32)[None])
        images.append(img_formatter(obs['camera_front_image'])[None])

        action = get_action(model, states, images, context,
                            gpu_id, n_steps, max_T, baseline)
        obs, reward, env_done, info = env.step(action)
        traj.append(obs, reward, done, info, action)

        tasks['success'] = reward or tasks['success']
        if tasks['success']:
            tasks['reached'] = True
        n_steps += 1
        if env_done or reward or n_steps > max_T:
            done = True
    env.close()
    del env
    del states
    del images
    del model
    return traj, tasks


def open_door_eval(model, env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False):
    done, states, images, context, obs, traj, tasks = \
        startup_env(model, env, context, gpu_id,
                    variation_id, baseline=baseline)
    n_steps = 0
    handle_loc = np.array(env.sim.data.site_xpos[env.door_handle_site_id])
    dist = 0.016
    while not done:
        tasks['reached'] = tasks['reached'] or np.linalg.norm(
            handle_loc - obs['eef_pos']) < dist
        # no 'pick'
        if baseline and len(states) >= 5:
            states, images = [], []
        states.append(np.concatenate(
            (obs['ee_aa'], obs['gripper_qpos'])).astype(np.float32)[None])
        images.append(img_formatter(obs['camera_front_image'])[None])

        action = get_action(model, states, images, context,
                            gpu_id, n_steps, max_T, baseline)
        obs, reward, env_done, info = env.step(action)
        traj.append(obs, reward, done, info, action)

        tasks['success'] = reward or tasks['success']
        if tasks['success']:
            tasks['reached'] = True
        n_steps += 1
        if env_done or reward or n_steps > max_T:
            done = True
    env.close()
    del env
    del states
    del images
    del model
    return traj, tasks