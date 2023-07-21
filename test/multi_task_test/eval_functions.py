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
from multi_task_il.utils import denormalize_action, denormalize_action_vima
from multi_task_test import make_prompt, prepare_obs
from einops import rearrange, repeat
from multi_task_il.models.vima.utils import *
import robosuite.utils.transform_utils as T
from multi_task_il.models.cond_target_obj_detector.utils import project_bboxes
from torchvision.ops import box_iou

# Frezzes at this line if torchvision is imported
# cv2.imshow("debug", np.zeros((128, 128, 3), dtype=np.uint8))
# cv2.waitKey(1)
# cv2.destroyAllWindows()

# open command json file
import json
import multi_task_robosuite_env as mtre
commad_path = os.path.join(os.path.dirname(
    mtre.__file__), "../collect_data/command.json")
with open(commad_path) as f:
    TASK_COMMAND = json.load(f)

ENV_OBJECTS = {
    'pick_place': {
        'obj_names': ['greenbox', 'yellowbox', 'bluebox', 'redbox'],
        'ranges': [[-0.255, -0.195], [-0.105, -0.045], [0.045, 0.105], [0.195, 0.255]],
    },
    'nut_assembly': {
        'obj_names': ['round-nut', 'round-nut-2', 'round-nut-3', "peg1", "peg2", "peg3"],
        'splitted_obj_names': ['grey nut', 'brown nut', 'blue nut'],
        'ranges': [[-0.31, -0.10], [-0.10, 0.10], [0.10, 0.31]]
    }
}

DEBUG = False


def get_action(model, target_obj_dec, states, images, context, gpu_id, n_steps, max_T=80, baseline=None, action_ranges=[], target_obj_embedding=None):
    s_t = torch.from_numpy(np.concatenate(states, 0).astype(np.float32))[None]
    if isinstance(images[-1], np.ndarray):
        i_t = torch.from_numpy(np.concatenate(
            images, 0).astype(np.float32))[None]
    else:
        i_t = images[0][None]
    s_t, i_t = s_t.float().cuda(gpu_id), i_t.float().cuda(gpu_id)

    predicted_prob = None

    if baseline == 'daml':
        learner = model.clone()
        # Perform adaptation
        learner.adapt(
            learner(None, context[0], learned_loss=True)['learned_loss'], allow_nograd=True, allow_unused=True)
        out = model(states=s_t[0], images=i_t[0], ret_dist=True)
        action = out['action_dist'].sample()[-1].cpu().detach().numpy()
    else:
        with torch.no_grad():
            out = model(states=s_t, images=i_t, context=context, eval=True,
                        target_obj_embedding=target_obj_embedding, compute_activation_map=True)  # to avoid computing ATC loss
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
    return action, predicted_prob, target_obj_embedding, out.get('activation_map', None)


def startup_env(model, env, context, gpu_id, variation_id, baseline=None, bb_flag=False):
    done, states, images = False, [], []
    if baseline is None:
        states = deque(states, maxlen=1)
        images = deque(images, maxlen=1)  # NOTE: always use only one frame
        if bb_flag:
            bb = deque([], maxlen=1)
            gt_classes = deque([], maxlen=1)
    context = context.cuda(gpu_id).float()

    while True:
        try:
            obs = env.reset()
            # make a "null step" to stabilize all objects
            current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]
            current_gripper_orientation = T.quat2axisangle(T.mat2quat(np.reshape(
                env.sim.data.site_xmat[env.robots[0].eef_site_id], (3, 3))))
            current_gripper_pose = np.concatenate(
                (current_gripper_position, current_gripper_orientation, np.array([-1])), axis=-1)
            obs, reward, env_done, info = env.step(current_gripper_pose)
            break
        except:
            pass
    traj = Trajectory()
    traj.append(obs)
    tasks = {'success': False, 'reached': False,
             'picked': False, 'variation_id': variation_id}
    if bb_flag:
        return done, states, images, context, obs, traj, tasks, bb, gt_classes
    else:
        return done, states, images, context, obs, traj, tasks


def nut_assembly_eval(model, target_obj_dec, env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False, action_ranges=[], model_name=None):

    if "vima" in model_name:
        return nut_assembly_eval_vima(model=model,
                                      env=env,
                                      gpu_id=gpu_id,
                                      variation_id=variation_id,
                                      max_T=max_T,
                                      baseline=baseline,
                                      action_ranges=action_ranges)
    else:
        return nut_assembly_eval_demo_cond(model=model,
                                           target_obj_dec=target_obj_dec,
                                           env=env,
                                           context=context,
                                           gpu_id=gpu_id,
                                           variation_id=variation_id,
                                           img_formatter=img_formatter,
                                           max_T=max_T,
                                           baseline=baseline,
                                           action_ranges=action_ranges)


def nut_assembly_eval_vima(model, env, gpu_id, variation_id, target_obj_dec=None, img_formatter=None, max_T=85, baseline=False, action_ranges=[]):

    done, states, images = False, [], []
    if baseline is None:
        states = deque(states, maxlen=10)
        images = deque(images, maxlen=10)  # NOTE: always use only one frame

    while True:
        try:
            obs = env.reset()
            # make a "null step" to stabilize all objects
            current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]
            current_gripper_orientation = T.quat2axisangle(T.mat2quat(np.reshape(
                env.sim.data.site_xmat[env.robots[0].eef_site_id], (3, 3))))
            current_gripper_pose = np.concatenate(
                (current_gripper_position, current_gripper_orientation, np.array([1])), axis=-1)
            obs, reward, env_done, info = env.step(current_gripper_pose)
            break
        except:
            pass
    traj = Trajectory()
    traj.append(obs)
    tasks = {'success': False, 'reached': False,
             'picked': False, 'variation_id': variation_id}

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

    n_steps = 0

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

    inference_cache = {}
    avg_prediction = 0

    # Prepare prompt for current task
    prompt_dict = make_prompt(
        env=env,
        obs=obs,
        command=TASK_COMMAND["nut_assembly"][str(variation_id)],
        task_name="nut_assembly")

    while not done:

        tasks['reached'] = tasks['reached'] or np.linalg.norm(
            handle_loc - obs['eef_pos']) < 0.045
        tasks['picked'] = tasks['picked'] or (
            tasks['reached'] and obs[obj_key][2] - start_z > 0.05)
        if baseline and len(states) >= 5:
            states, images = [], []
        states.append(np.concatenate(
            (obs['ee_aa'], obs['gripper_qpos'])).astype(np.float32)[None])

        if n_steps == 0:

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
        obs_to_tokenize = prepare_obs(env=env,
                                      obs=obs,
                                      views=['front'],
                                      task_name="nut_assembly").to_torch_tensor(device=gpu_id)[None]

        # 2. Forward obs token
        obs_token_this_step, obs_mask_this_step = model.forward_obs_token(
            obs_to_tokenize)

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

        if n_steps == 0:
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
        action = denormalize_action_vima(norm_action=action,
                                         action_ranges=action_ranges)
        if n_steps == 0:
            obs, reward, env_done, info, success = perform_primitive(
                env, action=action, primitive="reaching", trajectory=traj)
        elif n_steps == 1:
            obs, reward, env_done, info, success = perform_primitive(
                env, action=action, primitive="assembly", trajectory=traj)

        if not success:
            tasks['success'] = False
            done = True
        else:
            tasks['success'] = reward or tasks['success']
            n_steps += 1
            if env_done or reward or n_steps > 1:
                done = True

        # TODO ADD target pred
        if target_obj_dec is not None:
            target_pred = None
            info['target_pred'] = None
            info['target_gt'] = agent_target_obj_position
            if np.argmax(target_pred) == agent_target_obj_position:
                avg_prediction += 1

    env.close()
    tasks['avg_pred'] = avg_prediction/len(traj)
    del env
    del states
    del images
    del model

    return traj, tasks


def nut_assembly_eval_demo_cond(model, target_obj_dec, env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False, action_ranges=[]):

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
    target_obj_emb = None
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
        action, target_pred, target_obj_emb, activation_map = get_action(
            model,
            target_obj_dec,
            states,
            images,
            context,
            gpu_id,
            n_steps,
            max_T,
            baseline,
            action_ranges,
            target_obj_emb)

        obs, reward, env_done, info = env.step(action)
        # obs['activation_map'] = activation_map
        # cv2.imwrite("prova_activation_map.png", activation_map.numpy())
        # traj.append(obs, reward, done, info, action)

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
        action = get_action(model,
                            states,
                            images,
                            context,
                            gpu_id,
                            n_steps,
                            max_T,
                            baseline)

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
    dist = 0.025
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
            handle_loc - obs['eef_pos']) < 0.0275
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
    dist = 0.026
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


def pick_place_eval_vima(model, env, gpu_id, variation_id, target_obj_dec=None, img_formatter=None, max_T=85, baseline=False, action_ranges=[]):
    print(f"Max-t {max_T}")

    done, states, images = False, [], []
    if baseline is None:
        states = deque(states, maxlen=10)
        images = deque(images, maxlen=10)  # NOTE: always use only one frame

    while True:
        try:
            obs = env.reset()
            # make a "null step" to stabilize all objects
            current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]
            current_gripper_orientation = T.quat2axisangle(T.mat2quat(np.reshape(
                env.sim.data.site_xmat[env.robots[0].eef_site_id], (3, 3))))
            current_gripper_pose = np.concatenate(
                (current_gripper_position, current_gripper_orientation, np.array([1])), axis=-1)
            obs, reward, env_done, info = env.step(current_gripper_pose)
            break
        except:
            pass
    traj = Trajectory()
    traj.append(obs)
    tasks = {'success': False, 'reached': False,
             'picked': False, 'variation_id': variation_id}
    cv2.imwrite("pre_reaching.png", np.array(
        obs['camera_front_image'][:, :, ::-1]))
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

    n_steps = 0

    object_name = env.objects[env.object_id].name
    obj_delta_key = object_name + '_to_robot0_eef_pos'
    obj_key = object_name + '_pos'

    start_z = obs[obj_key][2]

    inference_cache = {}
    avg_prediction = 0

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

        if n_steps == 0:

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
        obs_to_tokenize = prepare_obs(env=env,
                                      obs=obs,
                                      views=['front'],
                                      task_name="pick_place").to_torch_tensor(device=gpu_id)[None]

        # 2. Forward obs token
        obs_token_this_step, obs_mask_this_step = model.forward_obs_token(
            obs_to_tokenize)

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

        if n_steps == 0:
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
        action = denormalize_action_vima(norm_action=action,
                                         action_ranges=action_ranges)
        if n_steps == 0:
            obs, reward, env_done, info, success = perform_primitive(
                env, action=action, primitive="reaching", trajectory=traj)
            cv2.imwrite("post_reaching.png", np.array(
                obs['camera_front_image'][:, :, ::-1]))
        elif n_steps == 1:
            obs, reward, env_done, info, success = perform_primitive(
                env, action=action, primitive="placing", trajectory=traj)
            cv2.imwrite("post_placing.png", np.array(
                obs['camera_front_image'][:, :, ::-1]))

        if not success:
            tasks['success'] = False
            done = True
        else:
            tasks['success'] = reward or tasks['success']
            n_steps += 1
            if env_done or reward or n_steps > 1:
                done = True

        # TODO ADD target pred
        if target_obj_dec is not None:
            target_pred = None
            info['target_pred'] = None
            info['target_gt'] = agent_target_obj_position
            if np.argmax(target_pred) == agent_target_obj_position:
                avg_prediction += 1

    env.close()
    tasks['avg_pred'] = avg_prediction/len(traj)
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
    target_obj_emb = None

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
        # convert observation from BGR to RGB
        images.append(img_formatter(
            obs['camera_front_image'][:, :, ::-1])[None])

        action, target_pred, target_obj_emb, activation_map = get_action(
            model=model,
            target_obj_dec=target_obj_dec,
            states=states,
            images=images,
            context=context,
            gpu_id=gpu_id,
            n_steps=n_steps,
            max_T=max_T,
            baseline=baseline,
            action_ranges=action_ranges,
            target_obj_embedding=target_obj_emb
        )
        try:
            obs, reward, env_done, info = env.step(action)
            cv2.imwrite(
                f"step_test.png", obs['camera_front_image'][:, :, ::-1])
        except:
            print("Exception during step")
        if target_obj_dec is not None:
            info['target_pred'] = target_pred
            info['target_gt'] = agent_target_obj_position
            if np.argmax(target_pred) == agent_target_obj_position:
                avg_prediction += 1

        if activation_map is not None:
            obs['activation_map'] = activation_map
            cv2.imwrite("prova_activation_map.png", activation_map)

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


def object_detection_inference(model, env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False, task_name="pick_place", controller=None):

    done, states, images, context, obs, traj, tasks, bb, gt_classes = \
        startup_env(model,
                    env,
                    context,
                    gpu_id,
                    variation_id,
                    baseline=baseline,
                    bb_flag=True)
    n_steps = 0
    iou = 0
    while not done:

        # Get GT Bounding Box
        agent_target_obj_id = traj.get(0)['obs']['target-object']
        for id, obj_name in enumerate(ENV_OBJECTS['pick_place']['obj_names']):
            if id == agent_target_obj_id:
                top_left_x = traj.get(
                    n_steps)['obs']['obj_bb']["camera_front"][ENV_OBJECTS[task_name]['obj_names'][agent_target_obj_id]]['bottom_right_corner'][0]
                top_left_y = traj.get(
                    n_steps)['obs']['obj_bb']["camera_front"][ENV_OBJECTS[task_name]['obj_names'][agent_target_obj_id]]['bottom_right_corner'][1]
                # print(f"Top-Left X {top_left_x} - Top-Left Y {top_left_y}")
                bottom_right_x = traj.get(
                    n_steps)['obs']['obj_bb']["camera_front"][ENV_OBJECTS[task_name]['obj_names'][agent_target_obj_id]]['upper_left_corner'][0]
                bottom_right_y = traj.get(
                    n_steps)['obs']['obj_bb']["camera_front"][ENV_OBJECTS[task_name]['obj_names'][agent_target_obj_id]]['upper_left_corner'][1]
                bb_t = np.array(
                    [[top_left_x, top_left_y, bottom_right_x, bottom_right_y]])
                gt_t = np.array(1)

        if baseline and len(states) >= 5:
            states, images = [], []
        states.append(np.concatenate(
            (obs['ee_aa'], obs['gripper_qpos'])).astype(np.float32)[None])

        # convert observation from BGR to RGB
        formatted_img, bb_t = img_formatter(
            obs['camera_front_image'][:, :, ::-1], bb_t)

        model_input = dict()
        model_input['demo'] = context.to(device=gpu_id)
        model_input['images'] = formatted_img[None][None].to(device=gpu_id)
        model_input['gt_bb'] = torch.from_numpy(
            bb_t[None][None]).to(device=gpu_id)
        model_input['gt_classes'] = torch.from_numpy(
            gt_t[None][None][None]).to(device=gpu_id)
        # Perform bb detection
        prediction = model(model_input, inference=True)

        # Project bb over image
        if prediction['conf_scores_final'][0] != -1:
            predicted_bb = project_bboxes(bboxes=prediction['proposals'][0][None],
                                          width_scale_factor=model._agent_backone.width_scale_factor,
                                          height_scale_factor=model._agent_backone.height_scale_factor,
                                          mode='a2p')
            try:
                max_conf_score_indx = torch.argmax(
                    prediction['conf_scores_final'][0])
            except:
                print("Argmax error")
            predicted_bb = predicted_bb[max_conf_score_indx]
            if True:
                image = np.array(np.moveaxis(
                    formatted_img[:, :, :].cpu().numpy()*255, 0, -1), dtype=np.uint8)

                image = cv2.rectangle(np.ascontiguousarray(image),
                                      (int(predicted_bb[0]),
                                       int(predicted_bb[1])),
                                      (int(predicted_bb[2]),
                                       int(predicted_bb[3])),
                                      color=(0, 0, 255), thickness=1)
                image = cv2.rectangle(np.ascontiguousarray(image),
                                      (int(bb_t[0][0]),
                                       int(bb_t[0][1])),
                                      (int(bb_t[0][2]),
                                       int(bb_t[0][3])),
                                      color=(250, 0, 0), thickness=1)
                cv2.imwrite("predicted_bb.png", image)
            obs['predicted_bb'] = predicted_bb.cpu().numpy()
            obs['gt_bb'] = bb_t
            # compute IoU over time
            iou_t = box_iou(boxes1=torch.from_numpy(
                bb_t).to(device=gpu_id), boxes2=predicted_bb[None])
            obs['iou'] = iou_t[0][0].cpu().numpy()
            iou += iou_t[0][0].cpu().numpy()
            traj.append(obs)
        else:
            obs['predicted_bb'] = predicted_bb.cpu().numpy()
            obs['gt_bb'] = bb_t
            obs['iou'] = 0
            traj.append(obs)
        try:
            if controller is not None:
                # compute the action for the current state
                action, status = controller.act(obs)
                obs, reward, env_done, info = env.step(action)
                cv2.imwrite(
                    f"step_test.png", obs['camera_front_image'][:, :, ::-1])
        except:
            print("Exception during step")

        n_steps += 1
        if n_steps > max_T or env_done:
            done = True

    env.close()
    tasks['avg_iou'] = iou/(n_steps-1)
    del env
    del states
    del images
    del model

    return traj, tasks


def pick_place_eval(model, target_obj_dec, env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False, action_ranges=[], model_name=None, task_name="pick_place"):

    if "vima" in model_name:
        return pick_place_eval_vima(model=model,
                                    env=env,
                                    gpu_id=gpu_id,
                                    variation_id=variation_id,
                                    max_T=max_T,
                                    baseline=baseline,
                                    action_ranges=action_ranges)
    elif "cond_target_obj_detector" in model_name:
        # Instantiate Controller
        if task_name == "pick_place":
            from multi_task_robosuite_env.controllers.controllers.expert_pick_place import PickPlaceController
            controller = PickPlaceController(
                env=env.env,
                tries=[],
                ranges=[],
                object_set=2)
        return object_detection_inference(model=model,
                                          env=env,
                                          context=context,
                                          gpu_id=gpu_id,
                                          variation_id=variation_id,
                                          img_formatter=img_formatter,
                                          max_T=max_T,
                                          baseline=baseline,
                                          controller=controller
                                          )
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


def reaching_primitive(env: object, desired_action: np.array, trajectory: object):

    # Action Logic

    # 1. Align with the object
    # get the current gripper position
    current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]
    action_alignment = copy.copy(desired_action)
    # change z value with the current value
    action_alignment[2] = current_gripper_position[2]
    t = 0
    while np.linalg.norm(current_gripper_position[:2]-action_alignment[:2]) > 0.005 and t < 100:
        delta_position = np.round(
            action_alignment[:3] - current_gripper_position, 2)
        delta_position = np.clip(delta_position, -0.02, 0.02)
        action_to_perform = copy.copy(action_alignment)
        action_to_perform[:3] = current_gripper_position + delta_position
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
        delta_position = np.round(
            action_pick_object[:3] - current_gripper_position, 2)
        delta_position = np.clip(delta_position, -0.02, 0.02)
        action_to_perform = copy.copy(action_pick_object)
        action_to_perform[:3] = current_gripper_position + delta_position
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
    action_pick_object = np.concatenate(
        (action_pick_object, np.array([1])), axis=-1)
    obs, reward, env_done, info = env.step(action_pick_object)
    cv2.imwrite("debug.png", obs['camera_front_image'][:, :, ::-1])
    trajectory.append(obs, reward, False, info, desired_action)

    return obs, reward, env_done, info, True


def placing_primitive(env: object, desired_action: np.array, trajectory: object):
    # Action Logic

    # 1. Move up from current position
    current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]
    current_gripper_orientation = T.quat2axisangle(T.mat2quat(np.reshape(
        env.sim.data.site_xmat[env.robots[0].eef_site_id], (3, 3))))
    current_gripper_pose = np.concatenate(
        (current_gripper_position, current_gripper_orientation, np.array([1])), axis=-1)

    start_gripper_pose = current_gripper_pose
    t = 0
    while abs(current_gripper_pose[2] - start_gripper_pose[2]) < 0.10 and t < 100:
        delta_position = np.array([0, 0, 0.02])
        action_to_perform = copy.copy(current_gripper_pose)
        action_to_perform[:3] = action_to_perform[:3] + delta_position
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

    # 4. Placing object
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
