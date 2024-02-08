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


def press_button_eval_demo_cond(model, env, context, gpu_id, variation_id, img_formatter, max_T=85, concat_bb=False, baseline=False, action_ranges=[], gt_env=None, controller=None, task_name=None, config=None, predict_gt_bb=False,  sub_action=False, gt_action=4, real=True):

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
        if abs(obs['eef_pos'][0] - button_loc[0]) < 0.05 and abs(obs['eef_pos'][1] - button_loc[1]) < 0.05 and abs(obs['eef_pos'][2] - button_loc[2]) < 0.03 or tasks['reached']:
            tasks['reached'] = 1.0
            tasks['picked'] = tasks['picked'] or \
                (tasks['reached'])

        for obj_id, obj_name, in enumerate(env.env.names):
            if obj_name != traj.get(0)['obs']['target-object']:
                other_button_loc = np.array(env.sim.data.site_xpos[env.env.sim.model.site_name2id(
                    env.env.names[obj_id])])
                if abs(obs['eef_pos'][0] - other_button_loc[0]) < 0.05 and abs(obs['eef_pos'][1] - other_button_loc[1]) < 0.05 and abs(obs['eef_pos'][2] - other_button_loc[2]) < 0.03 or tasks['reached']:
                    tasks['reached'] = 1.0
                    tasks['picked'] = tasks['picked'] or \
                        (tasks['reached'])

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

        states.append(np.concatenate(
            (obs['ee_aa'], obs['gripper_qpos'])).astype(np.float32)[None])

        # Get GT BB
        # if concat_bb:
        bb_t, gt_t = get_gt_bb(
            env=env,
            traj=traj,
            obs=obs,
            task_name=task_name,
            real=real
        )
        previous_predicted_bb = []
        previous_predicted_bb.append(torch.tensor(
            [.0, .0, .0, .0]).to(
            device=gpu_id).float())

        # convert observation from BGR to RGB
        if config.augs.get("old_aug", True):
            images.append(img_formatter(
                obs['camera_front_image'][:, :, ::-1])[None])
        else:
            img_aug, bb_t_aug = img_formatter(
                obs['camera_front_image'][:, :, ::-1], bb_t)
            images.append(img_aug[None])
            if model._object_detector is not None or predict_gt_bb:
                bb.append(bb_t_aug[None][None])
                gt_classes.append(torch.from_numpy(
                    gt_t[None][None][None]).to(device=gpu_id))

        if concat_bb:
            action, target_pred, target_obj_emb, activation_map, prediction_internal_obj, predicted_bb = get_action(
                model=model,
                target_obj_dec=None,
                states=states,
                bb=bb,
                predict_gt_bb=predict_gt_bb,
                gt_classes=gt_classes[0],
                images=images,
                context=context,
                gpu_id=gpu_id,
                n_steps=n_steps,
                max_T=max_T,
                baseline=baseline,
                action_ranges=action_ranges,
                target_obj_embedding=target_obj_emb
            )
        else:
            action, target_pred, target_obj_emb, activation_map, prediction_internal_obj, predicted_bb = get_action(
                model=model,
                target_obj_dec=None,
                states=states,
                bb=None,
                predict_gt_bb=False,
                gt_classes=None,
                images=images,
                context=context,
                gpu_id=gpu_id,
                n_steps=n_steps,
                max_T=max_T,
                baseline=baseline,
                action_ranges=action_ranges,
                target_obj_embedding=target_obj_emb
            )

        if concat_bb and model._object_detector is not None and not predict_gt_bb:
            prediction = prediction_internal_obj

        try:

            if sub_action:
                if n_steps < gt_action:
                    action, _ = controller.act(obs)

            obs, reward, env_done, info = env.step(action)
            if concat_bb and not predict_gt_bb:
                # get predicted bb from prediction
                # 1. Get the index with target class
                target_indx_flags = prediction['classes_final'][0] == 1
                if torch.sum((target_indx_flags == True).int()) != 0:
                    # 2. Get the confidence scores for the target predictions and the the max
                    target_max_score_indx = torch.argmax(
                        prediction['conf_scores_final'][0][target_indx_flags])
                    max_score_target = prediction['conf_scores_final'][0][target_indx_flags][target_max_score_indx]
                    if max_score_target != -1:
                        scale_factor = model._object_detector.get_scale_factors()
                        predicted_bb = project_bboxes(bboxes=prediction['proposals'][0][None][None],
                                                      width_scale_factor=scale_factor[0],
                                                      height_scale_factor=scale_factor[1],
                                                      mode='a2p')[0][target_indx_flags][target_max_score_indx]
                        previous_predicted_bb[0] = torch.round(
                            predicted_bb).int()
                        # replace bb
                        bb.append(torch.round(
                            predicted_bb[None][None].to(device=gpu_id)).int())
                    else:
                        bb.append(torch.round(previous_predicted_bb[None][None].to(
                            device=gpu_id)).int())

                    obs['predicted_bb'] = torch.round(
                        predicted_bb).cpu().numpy()
                    obs['predicted_score'] = max_score_target.cpu().numpy()
                    obs['gt_bb'] = bb_t_aug
                    # adjust bb
                    adj_predicted_bb = adjust_bb(bb=predicted_bb,
                                                 crop_params=config.get('tasks_cfgs').get(task_name).get('crop'))
                    image = np.array(cv2.rectangle(
                        np.array(obs['camera_front_image'][:, :, ::-1]),
                        (int(adj_predicted_bb[0]),
                         int(adj_predicted_bb[1])),
                        (int(adj_predicted_bb[2]),
                         int(adj_predicted_bb[3])),
                        (255, 0, 0), 1))
                else:
                    obs['gt_bb'] = bb_t_aug
                    image = np.array(obs['camera_front_image'][:, :, ::-1])
            elif concat_bb and predict_gt_bb:
                obs['gt_bb'] = bb_t_aug[0]
                obs['predicted_bb'] = bb_t_aug[0]
                # adjust bb
                adj_predicted_bb = adjust_bb(bb=bb_t_aug[0],
                                             crop_params=config.get('tasks_cfgs').get(task_name).get('crop'))
                image = np.array(cv2.rectangle(
                    np.array(obs['camera_front_image'][:, :, ::-1]),
                    (int(adj_predicted_bb[0]),
                     int(adj_predicted_bb[1])),
                    (int(adj_predicted_bb[2]),
                     int(adj_predicted_bb[3])),
                    (0, 255, 0), 1))
            else:
                image = np.array(obs['camera_front_image'][:, :, ::-1])

            cv2.imwrite(
                f"step_test.png", image)
            # if controller is not None and gt_env is not None:
            #     gt_action, gt_status = controller.act(gt_obs)
            #     gt_obs, gt_reward, gt_env_done, gt_info = gt_env.step(
            #         gt_action)
            #     cv2.imwrite(
            #         f"gt_step_test.png", gt_obs['camera_front_image'][:, :, ::-1])
        except Exception as e:
            print(f"Exception during step {e}")

        traj.append(obs, reward, done, info, action)

        tasks['success'] = reward or tasks['success']

        if tasks['success'] and not tasks['reached']:
            print("Fermati")
        # check if the object has been placed in a different bin
        if not tasks['success']:
            pass

        n_steps += 1
        if env_done or tasks['success'] or tasks['pressed_wrong'] or n_steps > max_T:
            done = True
    env.close()
    tasks['avg_pred'] = avg_prediction/len(traj)
    del env
    del states
    del images
    del model

    return traj, tasks


def press_button_eval(model, env, gt_env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False, action_ranges=[], model_name=None, task_name="pick_place", config=None, gt_file=None, gt_bb=False, sub_action=False, gt_action=4, real=True):

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
                                          task_name=task_name,
                                          config=config,
                                          gt_traj=gt_file,
                                          real=real
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
                                           real=real
                                           )
