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
from __init__ import make_prompt, prepare_obs
from einops import rearrange, repeat
from vima.utils import *


# Frezzes at this line if torchvision is imported
# cv2.imshow("debug", np.zeros((128, 128, 3), dtype=np.uint8))
# cv2.waitKey(1)
# cv2.destroyAllWindows()

# open command json file
import json
with open("/home/frosa_loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/tasks/collect_data/command.json") as f:
    TASK_COMMAND = json.load(f)

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


def nut_assembly_eval(model, target_obj_dec, env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False, action_ranges=[]):

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

    # Compute the target obj-slot
    if target_obj_dec != None:
        agent_target_obj_position = -1
        agent_target_obj_id = traj.get(0)['obs']['target-object']
        for id, obj_name in enumerate(ENV_OBJECTS['nut_assembly']['obj_names']):
            if id == agent_target_obj_id:
                # get object position
                pos = traj.get(0)['obs'][f'{obj_name}_pos']
                for i, pos_range in enumerate(ENV_OBJECTS['nut_assembly']["ranges"]):
                    if pos[1] >= pos_range[0] and pos[1] <= pos_range[1]:
                        agent_target_obj_position = i
                break
    # compute the average prediction over the whole trajectory
    avg_prediction = 0
    print(f"Max-t {max_T}")

    while not done:
        tasks['reached'] = tasks['reached'] or np.linalg.norm(
            handle_loc - obs['eef_pos']) < 0.045
        tasks['picked'] = tasks['picked'] or (
            tasks['reached'] and obs[obj_key][2] - start_z > 0.05)
        if baseline and len(states) >= 5:
            states, images = [], []
        states.append(np.concatenate(
            (obs['ee_aa'], obs['gripper_qpos'])).astype(np.float32)[None])
        images.append(img_formatter(
            obs['camera_front_image'][:, :, ::-1])[None])
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

        obs, reward, env_done, info = env.step(action)
        traj.append(obs, reward, done, info, action)

        if target_obj_dec is not None:
            info['target_pred'] = target_pred
            info['target_gt'] = agent_target_obj_position
            if np.argmax(target_pred) == agent_target_obj_position:
                avg_prediction += 1

        tasks['success'] = (reward and tasks['reached']) or tasks['success']
        n_steps += 1
        if env_done or reward or n_steps > max_T:
            done = True
    env.close()
    tasks['avg_pred'] = avg_prediction/len(traj)
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


def pick_place_eval_vima(model, env, gpu_id, variation_id, img_formatter=None, max_T=85, baseline=False, action_ranges=[]):
    print(f"Max-t {max_T}")

    done, states, images = False, [], []
    if baseline is None:
        states = deque(states, maxlen=10)
        images = deque(images, maxlen=10)  # NOTE: always use only one frame
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

    n_steps = 0

    object_name = env.objects[env.object_id].name
    obj_delta_key = object_name + '_to_robot0_eef_pos'
    obj_key = object_name + '_pos'

    start_z = obs[obj_key][2]

    t = 0
    inference_cache = {}

    # Prepare prompt for current task
    prompt_dict = make_prompt(
        env=env,
        obs=obs,
        command=TASK_COMMAND["pick_place"][str(variation_id)],
        task_name="pick_place")

    while not done:

        tasks['reached'] = tasks['reached'] or np.linalg.norm(
            obs[obj_delta_key][:2]) < 0.03
        tasks['picked'] = tasks['picked'] or (
            tasks['reached'] and obs[obj_key][2] - start_z > 0.05)
        if baseline and len(states) >= 5:
            states, images = [], []
        states.append(np.concatenate(
            (obs['ee_aa'], obs['gripper_qpos'])).astype(np.float32)[None])

        if t == 0:

            prompt_token_type = torch.from_numpy(prompt_dict["prompt_token_type"]).to(
                device=gpu_id)[None]
            word_batch = prompt_dict["word_batch"].to(
                gpu_id)[None]
            image_batch = prompt_dict["image_batch"].to_torch_tensor(
                device=gpu_id)[None]

            inference_cache["obs_tokens"] = []
            inference_cache["obs_masks"] = []
            inference_cache["action_tokens"] = []

            # 1. Forward prompt assembly
            prompt_tokens, prompt_masks = model.forward_prompt_assembly(
                (prompt_token_type, word_batch, image_batch)
            )

        # Prepare obs
        obs = prepare_obs(env=env,
                          obs=obs,
                          views=['front'],
                          task_name="pick_place").to_torch_tensor(device=gpu_id)[None]

        # 2. Forward obs token
        obs_token_this_step, obs_mask_this_step = model.forward_obs_token(obs)

        # Dim: 1 B Num_obj Obj_Emb_size
        obs_token_this_step = rearrange(
            obs_token_this_step, 'B T O E -> T B O E')

        obs_mask_this_step = rearrange(obs_mask_this_step, 'B T O -> T B O')

        # prepare history
        inference_cache["obs_tokens"].append(
            obs_token_this_step[0])  # B O E
        inference_cache["obs_masks"].append(obs_mask_this_step[0])
        max_objs = max(x.shape[1] for x in inference_cache["obs_tokens"])
        obs_tokens_to_forward, obs_masks_to_forward = [], []
        obs_tokens_this_env, obs_masks_this_env = [], []
        for idx in range(len(inference_cache["obs_tokens"])):
            obs_this_env_this_step = inference_cache["obs_tokens"][idx]
            obs_mask_this_env_this_step = inference_cache["obs_masks"][idx]
            required_pad = max_objs - obs_this_env_this_step.shape[1]
            obs_tokens_this_env.append(
                obs_this_env_this_step
            )
            obs_masks_this_env.append(
                obs_mask_this_env_this_step
            )

        obs_tokens_to_forward = any_stack(obs_tokens_this_env, dim=0)
        obs_masks_to_forward = any_stack(obs_masks_this_env, dim=0)

        if t == 0:
            action_tokens_to_forward = None
        else:
            action_tokens_to_forward = any_stack(
                inference_cache["action_tokens"], dim=0)

        obs_token = obs_tokens_to_forward
        obs_mask = obs_masks_to_forward
        action_token = action_tokens_to_forward
        prompt_token = prompt_tokens
        prompt_token_mask = prompt_masks

        # Compute action distribution
        out = model.forward_single_step(obs_token,
                                        obs_mask,
                                        action_token,
                                        prompt_token,
                                        prompt_token_mask)

        # Compute the action component class
        predicted_actions = dict()
        for k, v in out["dist_dict"].items():
            predicted_actions[k] = torch.reshape(
                v.mode(), (1, 1, 1)) if k == 'gripper_action' else v.mode()

        actions_to_embed = predicted_actions
        action_tokens = model.forward_action_token(
            actions_to_embed)  # (1, B, E)
        action_tokens = action_tokens.squeeze(0)  # (B, E)
        inference_cache["action_tokens"].append(action_tokens)

        # Compute the predicted action
        action_dict = model._de_discretize_actions(predicted_actions)

        actions_to_embed = predicted_actions
        action_tokens = model.forward_action_token(
            actions_to_embed)  # (1, B, E)
        action_tokens = action_tokens.squeeze(0)  # (B, E)
        inference_cache["action_tokens"].append(action_tokens)

        # Perform predicted action
        action = []
        for component_key in action_dict.keys():
            action_component = action_dict[component_key].cpu().numpy()[0][0]
            for a in action_component:
                action.append(a)
        action = np.asarray(action)
        action = denormalize_action(norm_action=action,
                                    action_ranges=action_ranges)
        obs, reward, env_done, info = perform_primitive(
            env, action=action, primitive="reaching", trajectory=traj)

        # try:
        # cv2.imwrite(
        #     f"{n_steps}.png", obs['camera_front_image'][:, :, ::-1])
        # except:
        # print("Exception during step")

        # traj.append(obs, reward, done, info, action)

        # tasks['success'] = reward or tasks['success']
        # n_steps += 1
        # if env_done or reward or n_steps > max_T:
        #     done = True

    env.close()
    del env
    del states
    del images
    del model

    return traj, tasks


def pick_place_eval_demo_cond(model, target_obj_dec, env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False, action_ranges=[]):
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
            obs['camera_front_image'][:, :, ::-1])[None])
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


def pick_place_eval(model, target_obj_dec, env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False, action_ranges=[], model_name=None):

    if "vima" in model_name:
        return pick_place_eval_vima(model=model,
                                    env=env,
                                    gpu_id=gpu_id,
                                    variation_id=variation_id,
                                    max_T=max_T,
                                    baseline=baseline,
                                    action_ranges=action_ranges)
    else:
        return pick_place_eval_demo_cond(model=model,
                                         target_obj_dec=target_obj_dec,
                                         env=env,
                                         context=context,
                                         gpu_id=gpu_id,
                                         variation_id=variation_id,
                                         img_formatter=img_formatter,
                                         max_T=max_T,
                                         baseline=baseline,
                                         action_ranges=action_ranges)


def reaching_primitive(env: object, action: np.array, trajectory: object):
    desired_position_reached = False
    while not desired_position_reached:
        # Action Logic
        # 1. Align with the object
        current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]
        action_alignment = action
        # change z value with the current value
        action_alignment[2] = current_gripper_position[2]
        gripper_aligned = False
        while not gripper_aligned:
            env.step(action)


def perform_primitive(env: object, action: dict() = None, primitive: str = "reaching", trajectory: object = None):
    action_list = []
    for key in action.keys():
        action_component = action[key]
        action_list.append(action_component)

    if primitive == "reaching":
        return reaching_primitive(env, action, trajectory)
    elif primitive == "placing":
        pass
    else:
        pass
