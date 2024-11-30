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
from sklearn.metrics import mean_squared_error
from robosuite.utils.transform_utils import quat2axisangle
import time

OBJECT_SET = 2


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


def pick_place_eval_demo_cond(model, env, context, gpu_id, variation_id, img_formatter, max_T=85, concat_bb=False, baseline=False, action_ranges=[], gt_env=None, controller=None, task_name=None, config=None, gt_traj=None, perform_augs=True, predict_gt_bb=False, sub_action=False, gt_action=4, real=True, place=False, convert_action=False):

    if gt_traj is None: # inizialize env and get first state
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

        object_name_target = env.objects[env.object_id].name.lower()
        obj_delta_key = object_name_target + '_to_robot0_eef_pos'
        obj_key = object_name_target + '_pos'

        start_z = obs[obj_key][2]

        # compute the average prediction over the whole trajectory
        avg_prediction = 0
        target_obj_emb = None
        consecutive_gt_action_cnt = 0

        print(f"Max-t {max_T}")
        tasks["reached_wrong"] = 0.0
        tasks["picked_wrong"] = 0.0
        tasks["place_wrong"] = 0.0
        tasks["place_wrong_correct_obj"] = 0.0
        tasks["place_wrong_wrong_obj"] = 0.0
        tasks["place_correct_bin_wrong_obj"] = 0.0
        elapsed_time = 0.0
        while not done:

            tasks['reached'] = check_reach(threshold=0.03,
                                           obj_distance=obs[obj_delta_key][:2],
                                           current_reach=tasks['reached']
                                           )

            tasks['picked'] = check_pick(threshold=0.05,
                                         obj_z=obs[obj_key][2],
                                         start_z=start_z,
                                         reached=tasks['reached'],
                                         picked=tasks['picked'])

            for obj_id, obj_name, in enumerate(env.env.obj_names):
                if obj_id != traj.get(0)['obs']['target-object'] and obj_name != "bin":
                    if check_reach(threshold=0.03,
                                   obj_distance=obs[obj_name.lower() +
                                                    '_to_robot0_eef_pos'],
                                   current_reach=tasks.get(
                                       "reached_wrong", 0.0)
                                   ):
                        tasks['reached_wrong'] = 1.0
                    if check_pick(threshold=0.05,
                                  obj_z=obs[obj_name.lower() + "_pos"][2],
                                  start_z=start_z,
                                  reached=tasks['reached_wrong'],
                                  picked=tasks.get(
                                      "picked_wrong", 0.0)):
                        tasks['picked_wrong'] = 1.0

            if baseline and len(states) >= 5:
                states, images, bb = [], [], []

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
                place=place,
                convert_action=convert_action
            )

            traj.append(obs, reward, done, info, action)
            elapsed_time += time_action

            tasks['success'] = reward or tasks['success']

            # check if the object has been placed in a different bin
            if not tasks['success']:
                for i, bin_name in enumerate(ENV_OBJECTS['pick_place']['bin_names']):
                    if i != obs['target-box-id']:
                        bin_pos = obs[f"{bin_name}_pos"]
                        if check_bin(threshold=0.03,
                                     bin_pos=bin_pos,
                                     obj_pos=obs[f"{object_name_target}_pos"],
                                     current_bin=tasks.get(
                                         "place_wrong_correct_obj", 0.0)
                                     ):
                            tasks["place_wrong_correct_obj"] = 1.0

                for obj_id, obj_name, in enumerate(env.env.obj_names):
                    if obj_id != traj.get(0)['obs']['target-object'] and obj_name != "bin":
                        for i, bin_name in enumerate(ENV_OBJECTS['pick_place']['bin_names']):
                            if i != obs['target-box-id']:
                                bin_pos = obs[f"{bin_name}_pos"]
                                if check_bin(threshold=0.03,
                                             bin_pos=bin_pos,
                                             obj_pos=obs[f"{obj_name}_pos"],
                                             current_bin=tasks.get(
                                                 "place_wrong_wrong_obj", 0.0)
                                             ):
                                    tasks["place_wrong_wrong_obj"] = 1.0
                            else:
                                bin_pos = obs[f"{bin_name}_pos"]
                                if check_bin(threshold=0.03,
                                             bin_pos=bin_pos,
                                             obj_pos=obs[f"{obj_name}_pos"],
                                             current_bin=tasks.get(
                                                 "place_correct_bin_wrong_obj", 0.0)
                                             ):
                                    tasks["place_correct_bin_wrong_obj"] = 1.0

            n_steps += 1
            if env_done or reward or n_steps > max_T:
                done = True
        print(tasks)
        env.close()
        mean_elapsed_time = elapsed_time/n_steps
        print(f"Mean elapsed time {mean_elapsed_time}")
        if getattr(model, 'first_phase', None) is not None:
            model.first_phase = True
        tasks['avg_pred'] = avg_prediction/len(traj)
        del env
        del states
        del images
        del model

        return traj, tasks
    else:
        target_obj_emb = None
        states = deque([], maxlen=1)
        images = deque([], maxlen=1)
        bb = deque([], maxlen=1)
        gt_classes = deque([], maxlen=1)
        fp = 0
        tp = 0
        fn = 0
        iou = 0
        info = {}
        error = []
        # take current observation
        for t in range(len(gt_traj)-1):
            if True:  # t == 1:
                agent_obs = gt_traj[t]['obs']['camera_front_image']
                bb_t, gt_t = get_gt_bb(traj=gt_traj,
                                       obs=gt_traj[t]['obs'],
                                       task_name=task_name,
                                       t=t)
                previous_predicted_bb = []
                previous_predicted_bb.append(torch.tensor(
                    [.0, .0, .0, .0]).to(
                    device=gpu_id).float())

                state_components = []
                for k in config.dataset_cfg.state_spec:
                    if isinstance(gt_traj[t]['obs'][k], int):
                        state_component = np.array(
                            [gt_traj[t]['obs'][k]], dtype=np.float32)
                    else:
                        state_component = np.array(
                            gt_traj[t]['obs'][k], dtype=np.float32)
                    state_components.extend(state_component)
                states.append([state_components])

                # convert observation from BGR to RGB
                if perform_augs:
                    formatted_img, bb_t = img_formatter(
                        agent_obs, bb_t, agent=True)
                    context = context.to(device=gpu_id)
                    bb_t_aug = bb_t.copy()

                    images.append(formatted_img[None])
                    if model._object_detector is not None or predict_gt_bb:
                        bb.append(bb_t_aug[None][None])
                        gt_classes.append(torch.from_numpy(
                            gt_t[None][None][None]).to(device=gpu_id))
                else:
                    cv2.imwrite("obs.png", agent_obs)
                    formatted_img = ToTensor()(agent_obs.copy()).to(device=gpu_id)
                    context = context.to(device=gpu_id)
                    bb_t_aug = bb_t.copy()
                    images.append(formatted_img[None])
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
                        n_steps=t,
                        max_T=max_T,
                        baseline=baseline,
                        action_ranges=action_ranges,
                        target_obj_embedding=target_obj_emb,
                        real=True
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
                        n_steps=t,
                        max_T=max_T,
                        baseline=baseline,
                        action_ranges=action_ranges,
                        target_obj_embedding=target_obj_emb
                    )

                # Compute MSE between predicted action and GT
                gt_action = np.zeros(7)
                gt_action[:3] = gt_traj[t+1]['action'][:3]
                gt_rot = gt_traj[t+1]['action'][3:7]
                gt_action[3:6] = quat2axisangle(gt_rot)
                gt_action[6] = gt_traj[t+1]['action'][7]

                pos_error_t = gt_action[:3] - action[:3]  # np.linalg.norm(
                # [gt_action[:3]] - np.array([action[:3]]), axis=1)
                # print(pos_error_t)
                error.append(pos_error_t)

                if concat_bb and model._object_detector is not None and not predict_gt_bb:
                    prediction = prediction_internal_obj

                # Project bb over image
                # 1. Get the index with target class
                target_indx_flags = prediction['classes_final'][0] == 1
                # 2. Get the confidence scores for the target predictions and the the max
                try:
                    target_max_score_indx = torch.argmax(
                        prediction['conf_scores_final'][0][target_indx_flags])  # prediction['conf_scores_final'][0][target_indx_flags]
                    max_score_target = prediction['conf_scores_final'][0]
                except:
                    print("No target bb found")
                    max_score_target = [-1]
                if max_score_target[0] != -1:
                    if perform_augs:
                        scale_factor = model._object_detector.get_scale_factors()
                        image = np.array(np.moveaxis(
                            formatted_img[:, :, :].cpu().numpy()*255, 0, -1), dtype=np.uint8)
                        predicted_bb = project_bboxes(bboxes=prediction['proposals'][0][None][None],
                                                      width_scale_factor=scale_factor[0],
                                                      height_scale_factor=scale_factor[1],
                                                      mode='a2p')[0][target_indx_flags][target_max_score_indx][None]
                    else:
                        image = np.array(np.moveaxis(
                            formatted_img[:, :, :].cpu().numpy()*255, 0, -1), dtype=np.uint8)
                        scale_factor = model._object_detector.get_scale_factors()
                        predicted_bb = project_bboxes(bboxes=prediction['proposals'][0][None][None],
                                                      width_scale_factor=scale_factor[0],
                                                      height_scale_factor=scale_factor[1],
                                                      mode='a2p')[0][target_indx_flags][target_max_score_indx][None]

                    if True:
                        for indx, bbox in enumerate(predicted_bb):
                            color = (255, 0, 0)
                            image = cv2.rectangle(np.ascontiguousarray(image),
                                                  (int(bbox[0]),
                                                   int(bbox[1])),
                                                  (int(bbox[2]),
                                                   int(bbox[3])),
                                                  color=color, thickness=1)
                            image = cv2.putText(image, "Score {:.2f}".format(max_score_target[indx]),
                                                (int(bbox[0]),
                                                int(bbox[1])),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.3,
                                                (0, 0, 255),
                                                1,
                                                cv2.LINE_AA)
                        image = cv2.rectangle(np.ascontiguousarray(image),
                                              (int(bb_t[0][0]),
                                               int(bb_t[0][1])),
                                              (int(bb_t[0][2]),
                                               int(bb_t[0][3])),
                                              color=(0, 255, 0), thickness=1)

                        gt_traj[t]['obs']['predicted_bb'] = predicted_bb
                        gt_traj[t]['obs']['gt_bb'] = bb_t
                        cv2.imwrite("predicted_bb.png", image)

                        # compute IoU over time
                        iou_t = box_iou(boxes1=torch.from_numpy(
                            bb_t).to(device=gpu_id), boxes2=predicted_bb)
                        gt_traj[t]['obs']['iou'] = iou_t[0][0].cpu().numpy()

                        # check if there are TP
                        if iou_t[0][0].cpu().numpy() < 0.5:
                            fp += 1
                            gt_traj[t]['obs']['fp'] = 1

                        else:
                            tp += 1
                            gt_traj[t]['obs']['tp'] = 1

                        iou += iou_t[0][0].cpu().numpy()
                        gt_traj[t]['obs']['iou'] = iou_t[0][0].cpu().numpy()
                        # traj.append(obs)

                else:
                    gt_traj[t]['obs']['predicted_bb'] = np.array([0, 0, 0, 0])[
                        None]
                    gt_traj[t]['obs']['gt_bb'] = bb_t
                    gt_traj[t]['obs']['iou'] = 0.0
                    gt_traj[t]['obs']['fn'] = 1
                    fn += 1

        info["avg_iou"] = iou/(len(gt_traj)-1)
        info["avg_tp"] = tp/(len(gt_traj)-1)
        info["avg_fp"] = fp/(len(gt_traj)-1)
        info["avg_fn"] = fn/(len(gt_traj)-1)
        info["error"] = np.mean(error, axis=0)
        return gt_traj, info


def pick_place_eval(model, env, gt_env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False, action_ranges=[], model_name=None, task_name="pick_place", config=None, gt_file=None, gt_bb=False, sub_action=False, gt_action=4, real=True, expert_traj=None, place_bb_flag=False, convert_action=False):

    print(f"Model name {model_name}")

    if "vima" in model_name:
        return pick_place_eval_vima(model=model,
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
            if task_name == "pick_place" and "CondPolicy" not in model_name:
                from multi_task_robosuite_env.controllers.controllers.expert_pick_place import PickPlaceController
                controller = PickPlaceController(
                    env=env.env,
                    tries=[],
                    ranges=[],
                    object_set=OBJECT_SET)
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

                from multi_task_robosuite_env.controllers.controllers.expert_pick_place import PickPlaceController
                controller = PickPlaceController(
                    env=env.env,
                    tries=[],
                    ranges=[],
                    object_set=OBJECT_SET)

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
                                          real=real,
                                          place_bb_flag=place_bb_flag
                                          )
    else:
        # Instantiate Controller
        if task_name == "pick_place":
            if env != None:
                from multi_task_robosuite_env.controllers.controllers.expert_pick_place import PickPlaceController
                controller = PickPlaceController(
                    env=env.env,
                    tries=[],
                    ranges=[],
                    object_set=OBJECT_SET)
            else:
                controller = None

        return pick_place_eval_demo_cond(model=model,
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
                                         gt_traj=gt_file,
                                         perform_augs=config.dataset_cfg.get(
                                             'perform_augs', True),
                                         predict_gt_bb=gt_bb,
                                         sub_action=sub_action,
                                         gt_action=gt_action,
                                         real=real,
                                         place=place_bb_flag,
                                         convert_action=convert_action)
