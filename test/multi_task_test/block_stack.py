from collections import deque
import torch
import numpy as np
from multi_task_il.datasets import Trajectory
import cv2
from multi_task_il.utils import denormalize_action_vima
from einops import rearrange
from multi_task_il.models.vima.utils import *
import robosuite.utils.transform_utils as T
from multi_task_test.primitive import *
from multi_task_test.utils import *
from multi_task_il.models.cond_target_obj_detector.utils import project_bboxes


def block_stack_eval_vima(model, env, gpu_id, variation_id, target_obj_dec=None, img_formatter=None, max_T=85, baseline=False, action_ranges=[]):
    print(f"Max-t {max_T}")
    return NotImplementedError


def block_stack_eval_demo_cond(model, env, context, gpu_id, variation_id, img_formatter, max_T=85, concat_bb=False, baseline=False, action_ranges=[], gt_env=None, controller=None, task_name=None, config=None, predict_gt_bb=False, sub_action=False, gt_action=4, real=True, gt_file=None):

    start_up_env_return = \
        startup_env(model=model,
                    env=env,
                    gt_env=gt_env,
                    context=context,
                    gpu_id=gpu_id,
                    variation_id=variation_id,
                    baseline=baseline,
                    bb_flag=concat_bb
                    )
    if concat_bb:
        done, states, images, context, obs, traj, tasks, bb, gt_classes, gt_obs, current_gripper_pose = start_up_env_return
    else:
        done, states, images, context, obs, traj, tasks, gt_obs, current_gripper_pose = start_up_env_return
        bb = None
        gt_classes = None

    n_steps = 0
    target_obj_loc = env.sim.data.body_xpos[env.cubeA_body_id]
    obj_key = 'cubeA_pos'
    start_z = obs[obj_key][2]

    # compute the average prediction over the whole trajectory
    avg_prediction = 0
    target_obj_emb = None

    print(f"Max-t {max_T}")
    tasks["reached_wrong"] = 0.0
    tasks["picked_wrong"] = 0.0
    tasks['place_correct_wrong_obj'] = 0.0
    tasks['fall_correct_obj'] = 0.0
    tasks['fall_wrong_obj'] = 0.0
    tasks['fall_correct_obj'] = 0.0
    reward_cnt = 0
    reward_flag = False

    while not done:
        tasks['reached'] = tasks['reached'] or np.linalg.norm(
            target_obj_loc - obs['eef_pos']) < 0.045
        tasks['picked'] = tasks['picked'] or (
            tasks['reached'] and obs[obj_key][2] - start_z > 0.05)
        if baseline and len(states) >= 5:
            states, images = [], []

        for i, obj in enumerate(env.cubes[1:]):
            obj_name = obj.name
            other_obj_loc = env.sim.data.body_xpos[getattr(
                env, f"{obj_name}_body_id")]

            tasks['reached_wrong'] = tasks['reached_wrong'] or np.linalg.norm(
                other_obj_loc - obs['eef_pos']) < 0.045
            tasks['picked_wrong'] = tasks['picked_wrong'] or (
                tasks['reached_wrong'] and (other_obj_loc[2] - start_z) > 0.05)

        states.append(np.concatenate(
            (obs['joint_pos'], obs['gripper_qpos'])).astype(np.float32)[None])

        obs, reward, info, action, env_done, time_action = task_run_action(
            traj=traj,
            obs=obs,
            task_name=task_name,
            env=env,
            real=real,
            gpu_id=gpu_id,
            config=config,
            images=images,
            img_formatter=img_formatter,
            model=model,
            predict_gt_bb=predict_gt_bb,
            bb=bb,
            gt_classes=gt_classes,
            concat_bb=concat_bb,
            states=states,
            context=context,
            n_steps=n_steps,
            max_T=max_T,
            baseline=baseline,
            action_ranges=action_ranges,
            sub_action=sub_action,
            gt_action=gt_action,
            controller=controller,
            target_obj_emb=target_obj_emb)

        traj.append(obs, reward, done, info, action)

        if bool(reward):
            reward_cnt += 1
            reward_flag = True

        if reward_flag and reward != 1.0:
            tasks['success'] = 0.0
            tasks['fall_correct_obj'] = 1.0

        if reward_cnt > 3:
            tasks['success'] = 1.0

        if tasks['picked_wrong']:
            picked_obj_loc = env.sim.data.body_xpos[env.cubeC_body_id]
            placed_obj_loc = env.sim.data.body_xpos[env.cubeB_body_id]
            if (abs(picked_obj_loc[0]-placed_obj_loc[0]) < 0.03 and abs(picked_obj_loc[1]-placed_obj_loc[1]) < 0.03 and abs(picked_obj_loc[2]-placed_obj_loc[2]) > 0.02) or tasks['place_correct_wrong_obj']:
                tasks['place_correct_wrong_obj'] = 1.0
            elif (abs(picked_obj_loc[0]-placed_obj_loc[0]) < 0.10 and abs(picked_obj_loc[1]-placed_obj_loc[1]) < 0.10 and abs(picked_obj_loc[2]-placed_obj_loc[2]) < 0.03) or tasks['fall_wrong_obj']:
                tasks['fall_wrong_obj'] = 1.0
                tasks['place_correct_wrong_obj'] = 0.0

        # check if the object has been placed in a different bin
        if not tasks['success']:
            pass

        n_steps += 1
        if env_done or tasks['success'] or n_steps > max_T:
            done = True
    env.close()
    if getattr(model, 'first_phase', None) is not None:
        model.first_phase = True
    tasks['avg_pred'] = avg_prediction/len(traj)
    del env
    del states
    del images
    del model

    return traj, tasks


def block_stack_eval(model, env, gt_env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False, action_ranges=[], model_name=None, task_name="pick_place", config=None, gt_file=None, gt_bb=False, sub_action=False, gt_action=4, real=True, expert_traj=None):

    if "vima" in model_name:
        return block_stack_eval_vima(model=model,
                                     env=env,
                                     gpu_id=gpu_id,
                                     variation_id=variation_id,
                                     max_T=max_T,
                                     baseline=baseline,
                                     action_ranges=action_ranges)
    elif "cond_target_obj_detector" in model_name:
        controller = None
        policy = False
        if gt_file is None:
            # Instantiate Controller
            if task_name == "stack_block" and "CondPolicy" not in model_name:
                from multi_task_robosuite_env.controllers.controllers.expert_block_stacking import BlockStackingController
                controller = BlockStackingController(
                    env=env.env,
                    tries=[],
                    ranges=[],
                    object_set=1)
                policy = False
            else:
                controller = None
                policy = True
                # map between task and number of tasks
                n_tasks = []
                tasks = dict()
                start = 0
                for i, task in enumerate(config.tasks):
                    n_tasks.append(task['n_tasks'])
                    tasks[task['name']] = (start, task['n_tasks'])
                    start += task['n_tasks']

                config.policy.n_tasks = n_tasks
                config.dataset_cfg.tasks = tasks
                config.dataset_cfg.n_tasks = int(np.sum(n_tasks))

                from multi_task_robosuite_env.controllers.controllers.expert_block_stacking import BlockStackingController
                controller = BlockStackingController(
                    env=env.env,
                    tries=[],
                    ranges=[],
                    object_set=1)

        return object_detection_inference(model=model,
                                          env=env,
                                          context=context,
                                          gpu_id=gpu_id,
                                          variation_id=variation_id,
                                          img_formatter=img_formatter,
                                          max_T=max_T,
                                          baseline=baseline,
                                          controller=controller,
                                          action_ranges=action_ranges,
                                          policy=policy,
                                          perform_augs=config.dataset_cfg.get(
                                              'perform_augs', True),
                                          config=config,
                                          gt_traj=gt_file,
                                          task_name=task_name,
                                          real=real
                                          )
    else:
        # Instantiate Controller
        if gt_env is not None:
            from multi_task_robosuite_env.controllers.controllers.expert_block_stacking import BlockStackingController
            controller = BlockStackingController(
                env=env.env,
                tries=[],
                ranges=[],
                object_set=1)

        return block_stack_eval_demo_cond(model=model,
                                          env=env,
                                          gt_env=gt_env,
                                          controller=controller,
                                          context=context,
                                          gpu_id=gpu_id,
                                          variation_id=variation_id,
                                          img_formatter=img_formatter,
                                          max_T=max_T,
                                          baseline=baseline,
                                          action_ranges=action_ranges,
                                          concat_bb=config.policy.get(
                                              "concat_bb", False),
                                          task_name=task_name,
                                          config=config,
                                          predict_gt_bb=gt_bb,
                                          sub_action=sub_action,
                                          gt_action=gt_action,
                                          real=real
                                          )


# def block_stack_eval(model, env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False):
#     done, states, images, context, obs, traj, tasks = \
#         startup_env(model, env, context, gpu_id,
#                     variation_id, baseline=baseline)
#     n_steps = 0
#     obj_loc = env.sim.data.body_xpos[env.cubeA_body_id]
#     obj_key = 'cubeA_pos'
#     start_z = obs[obj_key][2]
#     while not done:
#         tasks['reached'] = tasks['reached'] or np.linalg.norm(
#             obj_loc - obs['eef_pos']) < 0.045
#         tasks['picked'] = tasks['picked'] or (
#             tasks['reached'] and obs[obj_key][2] - start_z > 0.05)
#         if baseline and len(states) >= 5:
#             states, images = [], []
#         states.append(np.concatenate(
#             (obs['ee_aa'], obs['gripper_qpos'])).astype(np.float32)[None])
#         images.append(img_formatter(obs['camera_front_image'])[None])

#         action = get_action(model, states, images, context,
#                             gpu_id, n_steps, max_T, baseline)

#         obs, reward, env_done, info = env.step(action)
#         traj.append(obs, reward, done, info, action)

#         tasks['success'] = reward or tasks['success']
#         n_steps += 1
#         if env_done or reward or n_steps > max_T:
#             done = True
#     env.close()
#     del env
#     del states
#     del images
#     del model

#     return traj, tasks
