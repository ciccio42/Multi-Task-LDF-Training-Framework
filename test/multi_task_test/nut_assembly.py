from collections import deque
import torch
import numpy as np
from multi_task_il.datasets import Trajectory
from multi_task_il.utils import denormalize_action_vima
from einops import rearrange
from multi_task_il.models.vima.utils import *
import robosuite.utils.transform_utils as T
from multi_task_test.primitive import *
from multi_task_test.utils import *
from multi_task_il.models.cond_target_obj_detector.utils import project_bboxes


def _clip_delta(delta, max_step=0.015):
    norm_delta = np.linalg.norm(delta)
    if norm_delta < max_step:
        return delta
    return delta / norm_delta * max_step


def nut_assembly_eval(model, env, gt_env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False, action_ranges=[], model_name=None, task_name="nut_assembly", config=None, gt_file=None, gt_bb=False, sub_action=False, gt_action=4, real=True, place_bb_flag=True, **kwargs):

    if "vima" in model_name:
        return nut_assembly_eval_vima(model=model,
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
            if task_name == "nut_assembly" and "CondPolicy" not in model_name:
                from multi_task_robosuite_env.controllers.controllers.expert_nut_assembly import NutAssemblyController
                controller = NutAssemblyController(
                    env=env.env,
                    tries=0,
                    ranges=[])
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

                from multi_task_robosuite_env.controllers.controllers.expert_nut_assembly import NutAssemblyController
                controller = NutAssemblyController(
                    env=env.env,
                    tries=[],
                    ranges=[])

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
                                          place_bb_flag=place_bb_flag
                                          )
    else:
        # Instantiate Controller
        if task_name == "nut_assembly":
            from multi_task_robosuite_env.controllers.controllers.expert_nut_assembly import NutAssemblyController
            controller = NutAssemblyController(
                env=env.env,
                tries=0,
                ranges=[])
        return nut_assembly_eval_demo_cond(model=model,
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
                                           place=place_bb_flag
                                           )


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
        # states.append(np.concatenate(
        #     (obs['ee_aa'], obs['gripper_qpos'])).astype(np.float32)[None])
        if n_steps == 0:
            gripper_state = -1
        else:
            gripper_state = action[-1]
        states.append(np.concatenate(
            (obs['joint_pos'], [gripper_state])).astype(np.float32)[None])

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


def nut_assembly_eval_demo_cond(model, env, context, gpu_id, variation_id, img_formatter, max_T=85, concat_bb=False, baseline=False, action_ranges=[], gt_env=None, controller=None, task_name=None, config=None, predict_gt_bb=False, sub_action=False, gt_action=4, real=True, gt_file=None, place=False):

    start_up_env_return = \
        startup_env(model=model,
                    env=env,
                    gt_env=gt_env,
                    context=context,
                    gpu_id=gpu_id,
                    variation_id=variation_id,
                    baseline=baseline,
                    bb_flag=concat_bb,
                    gt_file=gt_file
                    )

    if concat_bb:
        done, states, images, context, obs, traj, tasks, bb, gt_classes, gt_obs, current_gripper_pose = start_up_env_return
    else:
        done, states, images, context, obs, traj, tasks, gt_obs, current_gripper_pose = start_up_env_return
        bb = None
        gt_classes = None

    prev_action = current_gripper_pose
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

    object_name = env.nuts[env.nut_id].name
    obj_delta_key = object_name + '_to_robot0_eef_pos'
    obj_key = object_name + '_pos'

    start_z = obs[obj_key][2]
    n_steps = 0

    # compute the average prediction over the whole trajectory
    avg_prediction = 0
    target_obj_emb = None
    tasks["reached_wrong"] = 0.0
    tasks["picked_wrong"] = 0.0
    tasks["place_wrong"] = 0.0
    tasks["place_wrong_correct_obj"] = 0.0
    tasks["place_wrong_wrong_obj"] = 0.0
    tasks["place_correct_wrong_obj"] = 0.0
    print(f"Max-t {max_T}")

    while not done:

        tasks['reached'] = tasks['reached'] or np.linalg.norm(
            handle_loc - obs['eef_pos']) < 0.045

        tasks['picked'] = tasks['picked'] or (
            tasks['reached'] and obs[obj_key][2] - start_z > 0.05)

        if not tasks['reached']:
            for obj_id, obj_name, in enumerate(ENV_OBJECTS["nut_assembly"]['obj_names'][:3]):
                handle_obj_loc = env.sim.data.site_xpos[env.sim.model.site_name2id(
                    f'{obj_name}_handle_site')]
                if obj_id != traj.get(0)['obs']['target-object'] and obj_name != "bin":
                    tasks['reached_wrong'] = tasks['reached_wrong'] or np.linalg.norm(
                        handle_obj_loc - obs['eef_pos']) < 0.045
                    tasks['picked_wrong'] = tasks['picked_wrong'] or (
                        tasks['reached_wrong'] and handle_obj_loc[2] - start_z > 0.05)

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
            place=place)

        traj.append(obs, reward, done, info, action)

        tasks['success'] = (reward or tasks['success']) and (
            tasks['reached'] and tasks['picked'])

        if not tasks['success']:
            for i, peg_name in enumerate(ENV_OBJECTS['nut_assembly']['peg_names']):
                if i != obs['target-peg']:
                    peg_pos = obs[f"peg{i+1}_pos"]
                    obj_pos = obs[f"{ENV_OBJECTS['nut_assembly']['obj_names'][obs['target-object']]}_pos"]
                    if check_peg(peg_pos=peg_pos,
                                 obj_pos=obj_pos,
                                 current_peg=tasks.get("place_wrong_correct_obj", 0.0)):
                        tasks["place_wrong_correct_obj"] = 1.0
                # if i != obs['target-box-id']:
                #     bin_pos = obs[f"{bin_name}_pos"]
                #     if check_bin(threshold=0.03,
                #                  bin_pos=bin_pos,
                #                  obj_pos=obs[f"{object_name_target}_pos"],
                #                  current_bin=tasks.get(
                #                      "place_wrong", 0.0)
                #                  ):
                #         tasks["place_wrong"] = 1.0

            for obj_id, obj_name, in enumerate(env.env.obj_names):
                if obj_id != traj.get(0)['obs']['target-object']:
                    for i, peg_name in enumerate(ENV_OBJECTS['nut_assembly']['peg_names']):
                        if i != obs['target-peg']:
                            peg_pos = obs[f"peg{i+1}_pos"]
                            obj_pos = obs[f"{ENV_OBJECTS['nut_assembly']['obj_names'][obj_id]}_pos"]
                            if check_peg(peg_pos=peg_pos,
                                         obj_pos=obj_pos,
                                         current_peg=tasks.get("place_wrong_wrong_obj", 0.0)):
                                tasks["place_wrong_wrong_obj"] = 1.0
                        else:
                            peg_pos = obs[f"peg{i+1}_pos"]
                            obj_pos = obs[f"{ENV_OBJECTS['nut_assembly']['obj_names'][obj_id]}_pos"]
                            if check_peg(peg_pos=peg_pos,
                                         obj_pos=obj_pos,
                                         current_peg=tasks.get("place_correct_wrong_obj", 0.0)):
                                tasks["place_correct_wrong_obj"] = 1.0

        n_steps += 1
        if env_done or reward or n_steps > max_T:
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
