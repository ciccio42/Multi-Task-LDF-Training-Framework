from collections import deque
import torch
import numpy as np
from multi_task_il.datasets import Trajectory
import cv2
from multi_task_il.utils import denormalize_action_vima
from multi_task_test import make_prompt, prepare_obs, adjust_bb
from einops import rearrange
from multi_task_il.models.vima.utils import *
import robosuite.utils.transform_utils as T
from multi_task_test import ENV_OBJECTS, TASK_COMMAND, startup_env, get_action, object_detection_inference, check_pick, check_reach, get_gt_bb
from multi_task_test.primitive import *
from multi_task_il.models.cond_target_obj_detector.utils import project_bboxes


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


def pick_place_eval_demo_cond(model, object_detector, env, context, gpu_id, variation_id, img_formatter, max_T=85, concat_bb=False, baseline=False, action_ranges=[], gt_env=None, controller=None, task_name=None, config=None, prediction_with_gt_bb=False):

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

    object_name = env.objects[env.object_id].name
    obj_delta_key = object_name + '_to_robot0_eef_pos'
    obj_key = object_name + '_pos'

    start_z = obs[obj_key][2]

    # Compute the target obj-slot
    if object_detector != None:
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

        tasks['reached'] = check_reach(threshold=0.03,
                                       obj_distance=obs[obj_delta_key][:2],
                                       current_reach=tasks['reached']
                                       )

        tasks['picked'] = check_pick(threshold=0.05,
                                     obj_z=obs[obj_key][2],
                                     start_z=start_z,
                                     reached=tasks['reached'],
                                     picked=tasks['picked'])

        if baseline and len(states) >= 5:
            states, images, bb = [], [], []

        states.append(np.concatenate(
            (obs['ee_aa'], obs['gripper_qpos'])).astype(np.float32)[None])

        # Get GT BB
        if concat_bb:
            bb_t, gt_t = get_gt_bb(traj=traj,
                                   obs=obs,
                                   task_name=task_name)
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
            if model._object_detector is not None or prediction_with_gt_bb:
                bb.append(bb_t_aug[None][None])
                gt_classes.append(torch.from_numpy(
                    gt_t[None][None][None]).to(device=gpu_id))

        # Perform inference with object-detector
        if object_detector is not None and model._object_detector is None and not prediction_with_gt_bb:
            model_input = dict()
            model_input['demo'] = context.to(device=gpu_id)
            model_input['images'] = img_aug[None][None].to(device=gpu_id)
            model_input['gt_bb'] = torch.from_numpy(
                bb_t_aug[None][None]).float().to(device=gpu_id)
            model_input['gt_classes'] = torch.from_numpy(
                gt_t[None][None][None]).to(device=gpu_id)
            model_input['states'] = torch.from_numpy(
                np.array(states)).to(device=gpu_id)

            prediction_external_obj = object_detector(model_input,
                                                      inference=True)
            # 1. Get the index with target class
            target_indx_flags = prediction_external_obj['classes_final'][0] == 1
            if torch.sum((target_indx_flags == True).int()) != 0:
                # 2. Get the confidence scores for the target predictions and the the max
                target_max_score_indx = torch.argmax(
                    prediction_external_obj['conf_scores_final'][0][target_indx_flags])
                max_score_target = prediction_external_obj['conf_scores_final'][
                    0][target_indx_flags][target_max_score_indx]
                if max_score_target != -1:
                    scale_factor = object_detector.get_scale_factors()
                    predicted_bb = project_bboxes(bboxes=prediction_external_obj['proposals'][0][None][None],
                                                  width_scale_factor=scale_factor[0],
                                                  height_scale_factor=scale_factor[1],
                                                  mode='a2p')[0][target_indx_flags][target_max_score_indx]
                    previous_predicted_bb[0] = torch.round(predicted_bb).int()
                    # replace bb
                    bb.append(torch.round(
                        predicted_bb[None][None].to(device=gpu_id)).int())
                else:
                    bb.append(torch.round(previous_predicted_bb[0][None][None].to(
                        device=gpu_id)).int())
            else:
                bb.append(torch.round(previous_predicted_bb[0][None][None].to(
                    device=gpu_id)).int())

            gt_classes.append(torch.from_numpy(
                np.array([1])[None][None]).to(device=gpu_id))

        if concat_bb:
            action, target_pred, target_obj_emb, activation_map, prediction_internal_obj, predicted_bb = get_action(
                model=model,
                target_obj_dec=None,
                states=states,
                bb=bb,
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

        if concat_bb and model._object_detector is None and not prediction_with_gt_bb:
            prediction = prediction_external_obj
        elif concat_bb and model._object_detector is not None and not prediction_with_gt_bb:
            prediction = prediction_internal_obj

        try:
            obs, reward, env_done, info = env.step(action)
            if concat_bb and not prediction_with_gt_bb:
                # get predicted bb from prediction
                # 1. Get the index with target class
                target_indx_flags = prediction['classes_final'][0] == 1
                if torch.sum((target_indx_flags == True).int()) != 0:
                    # 2. Get the confidence scores for the target predictions and the the max
                    target_max_score_indx = torch.argmax(
                        prediction['conf_scores_final'][0][target_indx_flags])
                    max_score_target = prediction['conf_scores_final'][0][target_indx_flags][target_max_score_indx]
                    if max_score_target != -1:
                        scale_factor = object_detector.get_scale_factors()
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
                        (0, 0, 255), 1))
                else:
                    obs['gt_bb'] = bb_t_aug
                    image = np.array(obs['camera_front_image'][:, :, ::-1])
            elif concat_bb and prediction_with_gt_bb:
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
                    (0, 0, 255), 1))
            else:
                image = np.array(obs['camera_front_image'][:, :, ::-1])

            cv2.imwrite(
                f"step_test.png", image)
            if controller is not None and gt_env is not None:
                gt_action, gt_status = controller.act(gt_obs)
                gt_obs, gt_reward, gt_env_done, gt_info = gt_env.step(
                    gt_action)
                cv2.imwrite(
                    f"gt_step_test.png", gt_obs['camera_front_image'][:, :, ::-1])
        except Exception as e:
            print(f"Exception during step {e}")
        # if target_obj_dec is not None:
        #     info['target_pred'] = target_pred
        #     info['target_gt'] = agent_target_obj_position
        #     if np.argmax(target_pred) == agent_target_obj_position:
        #         avg_prediction += 1

        # if activation_map is not None:
        #     obs['activation_map'] = activation_map
        #     cv2.imwrite("prova_activation_map.png", activation_map)

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


def pick_place_eval(model, object_detector, env, gt_env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False, action_ranges=[], model_name=None, task_name="pick_place", config=None, gt_file=None, gt_bb=False):

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
                    object_set=2)
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
                    object_set=2)

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
                                          gt_traj=gt_file
                                          )
    else:
        # Instantiate Controller
        if task_name == "pick_place":
            from multi_task_robosuite_env.controllers.controllers.expert_pick_place import PickPlaceController
            controller = PickPlaceController(
                env=env.env,
                tries=[],
                ranges=[],
                object_set=2)

        return pick_place_eval_demo_cond(model=model,
                                         object_detector=object_detector,
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
                                         prediction_with_gt_bb=gt_bb
                                         )
