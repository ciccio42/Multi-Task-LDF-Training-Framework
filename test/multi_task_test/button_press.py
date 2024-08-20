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


def press_button_eval_vima(model, env, gpu_id, variation_id, target_obj_dec=None, img_formatter=None, max_T=85, baseline=False, action_ranges=[]):
    print(f"Max-t {max_T}")
    return NotImplementedError


def press_button_eval_demo_cond(model, env, context, gpu_id, variation_id, img_formatter, max_T=85, concat_bb=False, baseline=False, action_ranges=[], gt_env=None, controller=None, task_name=None, config=None, predict_gt_bb=False, sub_action=False, gt_action=4, real=True, gt_file=None, expert_traj=None, place=False):

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

    button_loc = np.array(env.sim.data.site_xpos[env.target_button_id])
    dist = 0.015

    # compute the average prediction over the whole trajectory
    avg_prediction = 0
    target_obj_emb = None

    print(f"Max-t {max_T}")
    tasks["reached_wrong"] = 0.0
    tasks["picked_wrong"] = 0.0
    tasks["pressed_wrong"] = 0.0

    while not done:
        button_loc = np.array(env.sim.data.site_xpos[env.target_button_id])
        if abs(obs['eef_pos'][0] - button_loc[0]) < 0.02 and abs(obs['eef_pos'][1] - button_loc[1]) < 0.02 and abs(obs['eef_pos'][2] - button_loc[2]) < 0.01 or tasks['reached']:
            tasks['reached'] = 1.0
            tasks['picked'] = tasks['picked'] or \
                (tasks['reached'])

        for obj_id, obj_name, in enumerate(env.env.names):
            if obj_name != traj.get(0)['obs']['target-object']:
                other_button_loc = np.array(env.sim.data.site_xpos[env.env.sim.model.site_name2id(
                    env.env.names[obj_id])])
                if abs(obs['eef_pos'][0] - other_button_loc[0]) < 0.02 and abs(obs['eef_pos'][1] - other_button_loc[1]) < 0.02 and abs(obs['eef_pos'][2] - other_button_loc[2]) < 0.01 or tasks['reached_wrong']:
                    tasks['reached_wrong'] = 1.0
                    tasks['picked_wrong'] = tasks['picked_wrong'] or \
                        (tasks['reached_wrong'])

                qpos = env.sim.data.get_joint_qpos(
                    env.objects[obj_id//3].joints[obj_id % 3])
                if qpos >= 0.04:
                    print("Pressed Wrong")
                    tasks["pressed_wrong"] = 1.0

            #        qpos = self.sim.data.get_joint_qpos(
            #     self.objects[self.task_id // 3].joints[self.task_id % 3])
            # if qpos >= 0.04:
            #     return True
            # else:
            #     return False

        if baseline and len(states) >= 5:
            states, images, bb = [], [], []

        # states.append(np.concatenate(
        #     (obs['ee_aa'], obs['gripper_qpos'])).astype(np.float32)[None])

        # states.append(np.concatenate(
        #     (obs['joint_pos'], obs['gripper_qpos'])).astype(np.float32)[None])
        if n_steps == 0:
            gripper_state = -1
        else:
            gripper_state = action[-1]
        states.append(np.concatenate(
            (obs['joint_pos'], [gripper_state])).astype(np.float32)[None])

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
            target_obj_emb=target_obj_emb,
            expert_traj=expert_traj,
            place=place)

        traj.append(obs, reward, done, info, action)

        tasks['success'] = reward or tasks['success']

        if tasks['success'] and not tasks['reached']:
            tasks['reached'] = 1.0
            tasks['picked'] = 1.0

        if tasks['pressed_wrong'] and not tasks['reached_wrong']:
            tasks['reached_wrong'] = 1.0
            tasks['picked_wrong'] = 1.0

        n_steps += 1
        if env_done or tasks['success'] or tasks['pressed_wrong'] or n_steps > max_T:
            done = True
    env.close()
    tasks['avg_pred'] = avg_prediction/len(traj)
    if getattr(model, 'first_phase', None) is not None:
        model.first_phase = True
    del env
    del states
    del images
    del model

    return traj, tasks


def press_button_eval(model, env, gt_env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False, action_ranges=[], model_name=None, task_name="nut_assembly", config=None, gt_file=None, gt_bb=False, sub_action=False, gt_action=4, real=True, place_bb_flag=False, **kwargs):

    if "vima" in model_name:
        return press_button_eval_vima(model=model,
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
            if task_name == "button" and "CondPolicy" not in model_name:
                from multi_task_robosuite_env.controllers.controllers.expert_button import ButtonPressController
                controller = ButtonPressController(
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

                from multi_task_robosuite_env.controllers.controllers.expert_button import ButtonPressController
                controller = ButtonPressController(
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
                                          real=real,
                                          expert_traj=kwargs.get(
                                              'expert_traj', None)
                                          )
    else:
        # Instantiate Controller
        if gt_env is not None:
            from multi_task_robosuite_env.controllers.controllers.expert_button import ButtonPressController
            controller = ButtonPressController(
                env=env.env,
                tries=[],
                ranges=[],
                object_set=1)

        return press_button_eval_demo_cond(model=model,
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
                                           real=real,
                                           gt_file=gt_file,
                                           expert_traj=kwargs.get(
                                               'expert_traj', None),
                                           place=place_bb_flag
                                           )
