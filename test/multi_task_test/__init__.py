# from multi_task_test.eval_functions import *
import os
import torch
import numpy as np
from torch.multiprocessing import set_start_method
import json
import cv2
import robosuite.utils.transform_utils as T
from collections import deque
from tokenizers import Tokenizer
from tokenizers import AddedToken
from einops import rearrange
from multi_task_il.datasets.savers import Trajectory
import json
import multi_task_robosuite_env as mtre
from multi_task_il.utils import normalize_action, denormalize_action
from multi_task_il.models.cond_target_obj_detector.utils import project_bboxes
from torchvision.ops import box_iou
import copy

DEBUG = False

commad_path = os.path.join(os.path.dirname(
    mtre.__file__), "../collect_data/command.json")
with open(commad_path) as f:
    TASK_COMMAND = json.load(f)


set_start_method('forkserver', force=True)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

_kwargs = {
    "single_word": True,
    "lstrip": False,
    "rstrip": False,
    "normalized": True,
}

PLACEHOLDER_TOKENS = [
    AddedToken("{pick_object}", **_kwargs),
]
# PLACEHOLDERS = [token.content for token in PLACEHOLDER_TOKENS]
# tokenizer = Tokenizer.from_pretrained("t5-base")
# tokenizer.add_tokens(PLACEHOLDER_TOKENS)

ENV_OBJECTS = {
    'pick_place': {
        'obj_names': ['greenbox', 'yellowbox', 'bluebox', 'redbox', 'bin'],
        'ranges': [[-0.255, -0.195], [-0.105, -0.045], [0.045, 0.105], [0.195, 0.255]],
        'splitted_obj_names': ['green box', 'yellow box', 'blue box', 'red box'],
        'bin_position': [0.18, 0.00, 0.75],
        'obj_dim': {'greenbox': [0.05, 0.055, 0.045],  # W, H, D
                    'yellowbox': [0.05, 0.055, 0.045],
                    'bluebox': [0.05, 0.055, 0.045],
                    'redbox': [0.05, 0.055, 0.045],
                    'bin': [0.6, 0.06, 0.15]},
    },
    'nut_assembly': {
        'obj_names': ['round-nut', 'round-nut-2', 'round-nut-3', "peg1", "peg2", "peg3"],
        'splitted_obj_names': ['grey nut', 'brown nut', 'blue nut'],
        'ranges': [[0.10, 0.31], [-0.10, 0.10], [-0.31, -0.10]]
    }
}


def make_prompt(env: object, obs: object, command: str, task_name: str):
    ret_dict = {'states': [],
                'actions': [],
                'prompt': None}

    # 1. Get target object id

    # add special token
    color = command.split(" ")[2]
    object = command.split(" ")[3]
    command = command.replace(
        f"{color} {object}", "{pick_object}")
    ret_dict['prompt'] = command
    print(f"Command {ret_dict['prompt']}")
    prompt_assets = _create_prompt_assets(
        obs=obs,
        task_name=task_name,
        views=["front"],
        modalities=['rgb'],
    )

    prompt_token_type, word_batch, image_batch = _prepare_prompt(
        obs=obs,
        task_name=task_name,
        prompt=command,
        prompt_assets=prompt_assets,
        views=["front"],
    )

    ret_dict['prompt_token_type'] = prompt_token_type
    ret_dict['word_batch'] = word_batch
    ret_dict['image_batch'] = image_batch

    return ret_dict


def _create_prompt_assets(obs, task_name, views, modalities):
    prompt_assets = dict()
    prompt_assets['pick_object'] = dict()

    if task_name == 'pick_place' or task_name == 'nut_assembly':
        prompt_assets['pick_object']['rgb'] = dict()
        prompt_assets['pick_object']['segm'] = dict({'obj_info': dict()})
        prompt_assets['pick_object']['placeholder_type'] = 'object'
        # For each modality fill the prompt asset
        for modality in modalities:
            # For each modality and for each view fill the prompt asset
            for view in views:
                if view not in prompt_assets['pick_object'][modality].keys():
                    prompt_assets['pick_object'][modality][view] = dict()
                target_obj_id = obs['target-object']
                target_obj_name = ENV_OBJECTS[task_name]['obj_names'][target_obj_id]
                # assign prompt assets
                prompt_assets['pick_object'][modality][view] = obs['camera_front_image'][:, :, ::-1]
                prompt_assets['pick_object']['segm']['obj_info']['obj_id'] = target_obj_id
                prompt_assets['pick_object']['segm']['obj_info']['obj_name'] = ENV_OBJECTS[task_name]['splitted_obj_names'][target_obj_id]
                prompt_assets['pick_object']['segm']['obj_info']['obj_color'] = ENV_OBJECTS[task_name]['splitted_obj_names'][target_obj_id].split(" ")[
                    0]

    return prompt_assets


def _prepare_prompt(obs, task_name, prompt, prompt_assets, views):
    views = sorted(views)
    encoding = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_ids, prompt_tokens = encoding.ids, encoding.tokens
    assert set(prompt_assets.keys()) == set(
        [token[1:-1] for token in prompt_tokens if token in PLACEHOLDERS]
    )
    filled_prompt = []
    for id, token in zip(prompt_ids, prompt_tokens):
        if token not in PLACEHOLDERS:
            assert "{" not in token and "}" not in token
            filled_prompt.append(id)
        else:
            assert token.startswith("{") and token.endswith("}")
            asset_name = token[1:-1]
            assert asset_name in prompt_assets, f"missing prompt asset {asset_name}"
            asset = prompt_assets[asset_name]
            obj_info = asset["segm"]["obj_info"]
            placeholder_type = asset["placeholder_type"]
            if placeholder_type == "object":
                objects = [obj_info["obj_id"]]
            elif placeholder_type == "scene":
                objects = [each_info["obj_id"] for each_info in obj_info]
            obj_repr = {
                "cropped_img": {view: [] for view in views},
                "bbox": {view: [] for view in views},
            }

            for view in views:
                modality_view = asset['rgb'][view]
                bboxes = []
                cropped_imgs = []
                # for each object, crop the image around the target object
                for i, obj_id in enumerate(objects):
                    # Create bounding box for the target object
                    if task_name == 'pick_place' or task_name == 'nut_assembly':
                        if i == 0:
                            # In pick-place the first object is the target object
                            target_obj_id = obs['target-object']
                            target_obj_name = ENV_OBJECTS[task_name]['obj_names'][target_obj_id]
                            target_obj_bb = obs['obj_bb']['camera_front'][target_obj_name]
                            upper_left_corner = target_obj_bb['upper_left_corner']
                            bottom_right_corner = target_obj_bb['bottom_right_corner']
                            object_center = target_obj_bb['center']
                            # get prompt observation
                            rgb_this_view = asset['rgb'][view]
                            prompt_img = cv2.rectangle(
                                np.array(rgb_this_view), upper_left_corner, bottom_right_corner, (255, 0, 0), 1)
                            # cv2.imwrite("rgb_this_view.png",
                            #             np.array(prompt_img))

                            # bounding box center, height and width
                            x_center, y_center = object_center[0], object_center[1]
                            h, w = upper_left_corner[1] - \
                                bottom_right_corner[1], upper_left_corner[0] - \
                                bottom_right_corner[0]
                            bboxes.append(
                                [int(x_center), int(y_center), int(h), int(w)])
                            # crop image
                            cropped_img = np.array(rgb_this_view[
                                bottom_right_corner[1]:upper_left_corner[1] + 1, bottom_right_corner[0]:upper_left_corner[0] + 1, :])
                            cv2.imwrite(f"prompt_cropped_img_{target_obj_name}.png",
                                        np.array(cropped_img))
                            # pad if dimensions are different
                            if cropped_img.shape[0] != cropped_img.shape[1]:
                                diff = abs(
                                    cropped_img.shape[0] - cropped_img.shape[1])
                                pad_before, pad_after = int(
                                    diff / 2), diff - int(diff / 2)
                                if cropped_img.shape[0] > cropped_img.shape[1]:
                                    pad_width = (
                                        (0, 0), (pad_before, pad_after), (0, 0))
                                else:
                                    pad_width = (
                                        (pad_before, pad_after), (0, 0), (0, 0))
                                cropped_img = np.pad(
                                    cropped_img,
                                    pad_width,
                                    mode="constant",
                                    constant_values=0,
                                )
                                assert cropped_img.shape[0] == cropped_img.shape[1], "INTERNAL"
                            cropped_img = np.asarray(cropped_img)
                            # cv2.imwrite("cropped_img.png", cropped_img)
                            cropped_img = cv2.resize(
                                cropped_img,
                                (32, 32),
                                interpolation=cv2.INTER_AREA,
                            )
                            cropped_img = rearrange(
                                cropped_img, "h w c -> c h w")
                            cropped_imgs.append(cropped_img)

                bboxes = np.asarray(bboxes)
                cropped_imgs = np.asarray(cropped_imgs)
                obj_repr["bbox"][view] = bboxes
                obj_repr["cropped_img"][view] = cropped_imgs
            filled_prompt.append(obj_repr)
    raw_prompt = [filled_prompt]
    max_n_objs_prompt = {view: 0 for view in views}
    for prompt in raw_prompt:
        for token in prompt:
            if isinstance(token, dict):
                for view in views:
                    max_n_objs_prompt[view] = max(
                        max_n_objs_prompt[view], len(
                            token["cropped_img"][view])
                    )
    raw_prompt_token_type, word_batch, image_batch = [], [], []
    for prompt in raw_prompt:
        token_type = []
        for token in prompt:
            if isinstance(token, int):
                token_type.append(0)
                word_batch.append(token)
            elif isinstance(token, dict):
                token_type.append(1)
                n_objs_prompt = {
                    view: len(token["cropped_img"][view]) for view in views
                }
                # add mask
                token["mask"] = {
                    view: np.ones((n_objs_prompt[view],), dtype=bool)
                    for view in views
                }
                n_objs_to_pad = {
                    view: max_n_objs_prompt[view] - n_objs_prompt[view]
                    for view in views
                }
                objs_pad = {
                    "bbox": {
                        view: np.zeros(
                            (n_objs_to_pad[view], 4), dtype=np.int64)
                        for view in views
                    },
                    "cropped_img": {
                        view: np.zeros(
                            (n_objs_to_pad[view], 3, 32, 32),
                            dtype=np.uint8,
                        )
                        for view in views
                    },
                    "mask": {
                        view: np.zeros((n_objs_to_pad[view]), dtype=bool)
                        for view in views
                    },
                }
                token = any_concat([token, objs_pad], dim=0)
                image_batch.append(token)
        raw_prompt_token_type.append(token_type)
    assert sum([len(prompt) for prompt in raw_prompt_token_type]) == len(
        word_batch) + len(image_batch)

    raw_prompt_token_type = np.array(raw_prompt_token_type[0])
    word_batch = any_stack(word_batch, dim=0)
    image_batch = any_to_datadict(stack_sequence_fields(image_batch))

    word_batch = any_to_torch_tensor(word_batch)
    image_batch = image_batch.to_torch_tensor()
    return raw_prompt_token_type, word_batch, image_batch


def prepare_obs(env, obs, views, task_name):
    obs_list = {
        "ee": None,
        "objects": {
            "cropped_img": {view: [] for view in views},
            "bbox": {view: [] for view in views},
            "mask": {view: [] for view in views},
        },
    }

    obs_list['ee'] = torch.from_numpy(
        np.array([0]))

    for view in views:
        # get observation at timestamp t
        obs_t = obs
        rgb_this_view = obs_t['camera_front_image'][:, :, ::-1]
        # cv2.imwrite("rgb_this_view.png", np.array(rgb_this_view))
        bboxes = []
        cropped_imgs = []
        n_pad = 0

        # cut the image around each object in the scene
        for obj_name in ENV_OBJECTS[task_name]['obj_names']:

            # get object bb
            obj_bb = obs_t['obj_bb']['camera_front'][obj_name]
            upper_left_corner = obj_bb['upper_left_corner']
            bottom_right_corner = obj_bb['bottom_right_corner']
            object_center = obj_bb['center']
            # bounding box center, height and width
            x_center, y_center = object_center[0], object_center[1]
            h, w = upper_left_corner[1] - \
                bottom_right_corner[1], upper_left_corner[0] - \
                bottom_right_corner[0]
            bboxes.append(
                [int(x_center), int(y_center), int(h), int(w)])
            # crop image
            cropped_img = np.array(rgb_this_view[
                bottom_right_corner[1]:upper_left_corner[1] + 1, bottom_right_corner[0]:upper_left_corner[0] + 1, :])
            cv2.imwrite(f"cropped_img_{obj_name}.png",
                        np.array(cropped_img))

            # pad if dimensions are different
            if cropped_img.shape[0] != cropped_img.shape[1]:
                diff = abs(
                    cropped_img.shape[0] - cropped_img.shape[1])
                pad_before, pad_after = int(
                    diff / 2), diff - int(diff / 2)
                if cropped_img.shape[0] > cropped_img.shape[1]:
                    pad_width = (
                        (0, 0), (pad_before, pad_after), (0, 0))
                else:
                    pad_width = (
                        (pad_before, pad_after), (0, 0), (0, 0))
                cropped_img = np.pad(
                    cropped_img,
                    pad_width,
                    mode="constant",
                    constant_values=0,
                )
                assert cropped_img.shape[0] == cropped_img.shape[1], "INTERNAL"
            cropped_img = np.asarray(cropped_img)
            cropped_img = cv2.resize(
                cropped_img,
                (32, 32),
                interpolation=cv2.INTER_AREA,
            )
            cropped_img = rearrange(cropped_img, "h w c -> c h w")
            cropped_imgs.append(cropped_img)
        bboxes = np.asarray(bboxes)
        cropped_imgs = np.asarray(cropped_imgs)
        mask = np.ones(len(bboxes), dtype=bool)

        obs_list["objects"]["bbox"][view].append(bboxes)
        obs_list["objects"]["cropped_img"][view].append(cropped_imgs)
        obs_list["objects"]["mask"][view].append(mask)

    for view in views:
        obs_list["objects"]["bbox"][view] = np.stack(
            obs_list["objects"]["bbox"][view], axis=0
        )
        obs_list["objects"]["cropped_img"][view] = np.stack(
            obs_list["objects"]["cropped_img"][view], axis=0
        )
        obs_list["objects"]["mask"][view] = np.stack(
            obs_list["objects"]["mask"][view], axis=0
        )

    obs = any_to_datadict(obs_list)
    obs = obs.to_torch_tensor()
    # obs = any_transpose_first_two_axes(obs)

    return obs


def get_action(model, target_obj_dec, bb, states, images, context, gpu_id, n_steps, max_T=80, baseline=None, action_ranges=[], target_obj_embedding=None):
    s_t = torch.from_numpy(np.concatenate(states, 0).astype(np.float32))[None]
    if isinstance(images[-1], np.ndarray):
        i_t = torch.from_numpy(np.concatenate(
            images, 0).astype(np.float32))[None]
    else:
        i_t = images[0][None]
    s_t, i_t = s_t.float().cuda(gpu_id), i_t.float().cuda(gpu_id)

    if bb is not None:
        bb = torch.from_numpy(bb[0]).float().cuda(gpu_id)

    predicted_prob = None

    if baseline == 'daml':
        learner = model.clone()
        # Perform adaptation
        learner.adapt(
            learner(None, context[0], learned_loss=True)['learned_loss'],
            allow_nograd=True,
            allow_unused=True)
        out = model(states=s_t[0], images=i_t[0], ret_dist=True)
        action = out['action_dist'].sample()[-1].cpu().detach().numpy()
    else:
        with torch.no_grad():
            out = model(states=s_t,
                        images=i_t,
                        context=context,
                        bb=bb,
                        eval=True,
                        target_obj_embedding=target_obj_embedding,
                        compute_activation_map=True)  # to avoid computing ATC loss
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


def startup_env(model, env, gt_env, context, gpu_id, variation_id, baseline=None, bb_flag=False):

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

    gt_obs = None
    if gt_env != None:
        while True:
            try:
                gt_obs = gt_env.reset()
                for obj_name in env.object_to_id.keys():
                    gt_obj = gt_env.objects[env.object_to_id[obj_name]]
                    # set object position based on trajectory file
                    obj_pos = obs[f"{obj_name}_pos"]
                    obj_quat = obs[f"{obj_name}_quat"]
                    gt_env.env.sim.data.set_joint_qpos(
                        gt_obj.joints[0], np.concatenate([obj_pos, obj_quat]))

                # make a "null step" to stabilize all objects
                current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]
                current_gripper_orientation = T.quat2axisangle(T.mat2quat(np.reshape(
                    env.sim.data.site_xmat[env.robots[0].eef_site_id], (3, 3))))
                current_gripper_pose = np.concatenate(
                    (current_gripper_position, current_gripper_orientation, np.array([-1])), axis=-1)
                gt_obs, gt_reward, gt_env_done, gt_info = env.step(
                    current_gripper_pose)
                break
            except:
                pass

    traj = Trajectory()
    traj.append(obs)
    tasks = {'success': False, 'reached': False,
             'picked': False, 'variation_id': variation_id}
    if bb_flag:
        return done, states, images, context, obs, traj, tasks, bb, gt_classes, gt_obs, current_gripper_pose
    else:
        return done, states, images, context, obs, traj, tasks, gt_obs, current_gripper_pose


def get_gt_bb(traj=None, obs=None, task_name=None, t=0):
    # Get GT Bounding Box
    agent_target_obj_id = traj.get(t)['obs']['target-object']
    for id, obj_name in enumerate(ENV_OBJECTS['pick_place']['obj_names']):
        if id == agent_target_obj_id:
            top_left_x = obs['obj_bb']["camera_front"][ENV_OBJECTS[task_name]
                                                       ['obj_names'][agent_target_obj_id]]['bottom_right_corner'][0]
            top_left_y = obs['obj_bb']["camera_front"][ENV_OBJECTS[task_name]
                                                       ['obj_names'][agent_target_obj_id]]['bottom_right_corner'][1]
            # print(f"Top-Left X {top_left_x} - Top-Left Y {top_left_y}")
            bottom_right_x = obs['obj_bb']["camera_front"][ENV_OBJECTS[task_name]
                                                           ['obj_names'][agent_target_obj_id]]['upper_left_corner'][0]
            bottom_right_y = obs['obj_bb']["camera_front"][ENV_OBJECTS[task_name]
                                                           ['obj_names'][agent_target_obj_id]]['upper_left_corner'][1]
            bb_t = np.array(
                [[top_left_x, top_left_y, bottom_right_x, bottom_right_y]])
            gt_t = np.array(1)

    return bb_t, gt_t


def object_detection_inference(model, env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False, task_name="pick_place", controller=None, action_ranges=[], policy=True, perform_augs=False, config=None, gt_traj=None):

    if gt_traj is None:
        done, states, images, context, obs, traj, tasks, bb, gt_classes, _, prev_action = \
            startup_env(model=model,
                        env=env,
                        gt_env=None,
                        context=context,
                        gpu_id=gpu_id,
                        variation_id=variation_id,
                        baseline=baseline,
                        bb_flag=True)
        n_steps = 0
        fp = 0
        tp = 0
        fn = 0
        iou = 0
        prev_action = normalize_action(
            action=prev_action,
            n_action_bin=256,
            action_ranges=action_ranges)

        object_name = env.objects[env.object_id].name
        obj_delta_key = object_name + '_to_robot0_eef_pos'
        obj_key = object_name + '_pos'

        start_z = obs[obj_key][2]
        bb_queue = []
        while not done:

            tasks['reached'] = check_reach(threshold=0.03,
                                           obj_distance=obs[obj_delta_key][:2],
                                           current_reach=tasks['reached'])

            tasks['picked'] = check_pick(threshold=0.05,
                                         obj_z=obs[obj_key][2],
                                         start_z=start_z,
                                         reached=tasks['reached'],
                                         picked=tasks['picked'])

            bb_t, gt_t = get_gt_bb(traj=traj,
                                   obs=obs,
                                   task_name=task_name)

            if baseline and len(states) >= 5:
                states, images = [], []
            states.append(prev_action.astype(np.float32)[None])

            # convert observation from BGR to RGB
            if perform_augs:
                formatted_img, bb_t = img_formatter(
                    obs['camera_front_image'][:, :, ::-1], bb_t)
            else:
                formatted_img = torch.from_numpy(
                    np.array(obs['camera_front_image'][:, :, ::-1]))

            model_input = dict()
            model_input['demo'] = context.to(device=gpu_id)
            model_input['images'] = formatted_img[None][None].to(device=gpu_id)
            model_input['gt_bb'] = torch.from_numpy(
                bb_t[None][None]).float().to(device=gpu_id)
            model_input['gt_classes'] = torch.from_numpy(
                gt_t[None][None][None]).to(device=gpu_id)
            model_input['states'] = torch.from_numpy(
                np.array(states)).to(device=gpu_id)

            if task_name in config.dataset_cfg.get("tasks").keys():
                task_one_hot = np.zeros((1, config.dataset_cfg.n_tasks))
                task_one_hot[0][config.dataset_cfg.tasks[task_name]
                                [0]+variation_id] = 1
                model_input['task_id'] = torch.from_numpy(
                    np.array(task_one_hot)).to(device=gpu_id)

            with torch.no_grad():
                # Perform  detection
                if policy:
                    prediction = model(model_input,
                                       inference=True,
                                       oracle=False,
                                       bb_queue=bb_queue)
                else:
                    prediction = model(model_input,
                                       inference=True)

            if controller is not None and not policy:
                prediction = prediction
            else:
                bc_distrib = prediction['bc_distrib']
                bb_queue = prediction['prediction_target_obj_detector']['proposals']
                prediction = prediction['prediction_target_obj_detector']

            # Project bb over image
            if prediction['conf_scores_final'][0] != -1:
                if perform_augs:
                    scale_factor = model.get_scale_factors()
                    image = np.array(np.moveaxis(
                        formatted_img[:, :, :].cpu().numpy()*255, 0, -1), dtype=np.uint8)
                    predicted_bb = project_bboxes(bboxes=prediction['proposals'][0][None][None],
                                                  width_scale_factor=scale_factor[0],
                                                  height_scale_factor=scale_factor[1],
                                                  mode='a2p')[0][0]
                else:
                    image = formatted_img.cpu().numpy()
                    predicted_bb = prediction['proposals'][0]

                if True:

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
                                          color=(0, 255, 0), thickness=1)
                    cv2.imwrite("predicted_bb.png", image)
                obs['predicted_bb'] = predicted_bb.cpu().numpy()
                obs['gt_bb'] = bb_t
                # compute IoU over time
                iou_t = box_iou(boxes1=torch.from_numpy(
                    bb_t).to(device=gpu_id), boxes2=predicted_bb[None])
                obs['iou'] = iou_t[0][0].cpu().numpy()

                if iou_t[0][0].cpu().numpy() < 0.5:
                    fp += 1
                    obs['fp'] = 1
                else:
                    tp += 1
                    obs['tp'] = 1

                iou += iou_t[0][0].cpu().numpy()
                traj.append(obs)
            else:
                scale_factor = model.get_scale_factors()
                predicted_bb = project_bboxes(bboxes=prediction['proposals'][0][None].float(),
                                              width_scale_factor=scale_factor[0],
                                              height_scale_factor=scale_factor[1],
                                              mode='a2p')[0]
                obs['predicted_bb'] = predicted_bb.cpu().numpy()
                obs['gt_bb'] = bb_t
                obs['iou'] = 0
                iou += 0
                fn += 1
                obs['fn'] = 1
                traj.append(obs)

            if controller is not None:
                # compute the action for the current state
                action_gt, status = controller.act(obs)
                action = action_gt
                # action_norm = bc_distrib.sample().cpu().numpy()[0]
                # prev_action = action_norm
                # action = denormalize_action(action_norm,
                #                             action_ranges)
            else:
                action = bc_distrib.sample().cpu().numpy()[0]
                prev_action = action
                action = denormalize_action(action,
                                            action_ranges)

                # action = clip_action(action)
            obs, reward, env_done, info = env.step(action)
            cv2.imwrite(
                f"step_test.png", obs['camera_front_image'][:, :, ::-1])

            n_steps += 1
            if n_steps >= 1 or env_done or reward:
                done = True

        env.close()
        tasks['avg_iou'] = iou/(n_steps)
        tasks['avg_tp'] = tp/(n_steps)
        tasks['avg_fp'] = fp/(n_steps)
        tasks['avg_fn'] = fn/(n_steps)
        del env
        del states
        del images
        del model

        return traj, tasks
    else:
        states = deque([], maxlen=1)
        images = deque([], maxlen=1)
        bb = deque([], maxlen=1)
        gt_classes = deque([], maxlen=1)
        fp = 0
        tp = 0
        fn = 0
        iou = 0
        info = {}
        # take current observation
        for t in range(len(gt_traj)):
            if t == 1:
                agent_obs = gt_traj[t]['obs']['camera_front_image']
                bb_t, gt_t = get_gt_bb(traj=gt_traj,
                                       obs=gt_traj[t]['obs'],
                                       task_name=task_name,
                                       t=t)
                formatted_img, bb_t = img_formatter(
                    agent_obs[:, :, ::-1], bb_t)

                model_input = dict()
                model_input['demo'] = context.to(device=gpu_id)
                model_input['images'] = formatted_img[None][None].to(
                    device=gpu_id)
                model_input['gt_bb'] = torch.from_numpy(
                    bb_t[None][None]).float().to(device=gpu_id)
                model_input['gt_classes'] = torch.from_numpy(
                    gt_t[None][None][None]).to(device=gpu_id)

                prediction = model(model_input,
                                   inference=True)

                # Project bb over image
                if prediction['conf_scores_final'][0] != -1:
                    if perform_augs:
                        scale_factor = model.get_scale_factors()
                        image = np.array(np.moveaxis(
                            formatted_img[:, :, :].cpu().numpy()*255, 0, -1), dtype=np.uint8)
                        predicted_bb = project_bboxes(bboxes=prediction['proposals'][0][None][None],
                                                      width_scale_factor=scale_factor[0],
                                                      height_scale_factor=scale_factor[1],
                                                      mode='a2p')[0][0]
                    else:
                        image = formatted_img.cpu().numpy()
                        predicted_bb = prediction['proposals'][0]

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
                                          color=(0, 255, 0), thickness=1)
                    cv2.imwrite("predicted_bb.png", image)

                    # compute IoU over time
                    iou_t = box_iou(boxes1=torch.from_numpy(
                        bb_t).to(device=gpu_id), boxes2=predicted_bb[None])
                    iou += iou_t[0][0].cpu().numpy()

                    if iou_t[0][0].cpu().numpy() < 0.5:
                        fp += 1
                    else:
                        tp += 1

                else:
                    fn += 1

        info["avg_iou"] = iou
        info["avg_tp"] = tp  # /(len(gt_traj)-1)
        info["avg_fp"] = fp  # /(len(gt_traj)-1)
        info["avg_fn"] = fn  # /(len(gt_traj)-1)
        return None, info


def select_random_frames(frames, n_select, sample_sides=True, random_frames=True):
    selected_frames = []
    def clip(x): return int(max(0, min(x, len(frames) - 1)))
    per_bracket = max(len(frames) / n_select, 1)

    if random_frames:
        for i in range(n_select):
            n = clip(np.random.randint(
                int(i * per_bracket), int((i + 1) * per_bracket)))
            if sample_sides and i == n_select - 1:
                n = len(frames) - 1
            elif sample_sides and i == 0:
                n = 1
            selected_frames.append(n)
    else:
        for i in range(n_select):
            # get first frame
            if i == 0:
                n = 1
            # get the last frame
            elif i == n_select - 1:
                n = len(frames) - 1
            elif i == 1:
                obj_in_hand = 0
                # get the first frame with obj_in_hand and the gripper is closed
                for t in range(1, len(frames)):
                    state = frames.get(t)['info']['status']
                    trj_t = frames.get(t)
                    gripper_act = trj_t['action'][-1]
                    if state == 'obj_in_hand' and gripper_act == 1:
                        obj_in_hand = t
                        n = t
                        break
            elif i == 2:
                # get the middle moving frame
                start_moving = 0
                end_moving = 0
                for t in range(obj_in_hand, len(frames)):
                    state = frames.get(t)['info']['status']
                    if state == 'moving' and start_moving == 0:
                        start_moving = t
                    elif state != 'moving' and start_moving != 0 and end_moving == 0:
                        end_moving = t
                        break
                n = start_moving + int((end_moving-start_moving)/2)
            selected_frames.append(n)

    if isinstance(frames, (list, tuple)):
        return [frames[i] for i in selected_frames]
    elif isinstance(frames, Trajectory):
        return [frames[i]['obs']['camera_front_image'] for i in selected_frames]
        # return [frames[i]['obs']['image-state'] for i in selected_frames]
    return frames[selected_frames]


def check_pick(threshold: float, obj_z: float, start_z: float, reached: bool, picked: bool):
    return picked or (reached and obj_z - start_z > threshold)


def check_reach(threshold: float, obj_distance: np.array, current_reach: bool):
    return current_reach or np.linalg.norm(
        obj_distance) < threshold


def clip_action(action, prev_action):
    prev_pos = prev_action[:3]
    current_action = copy.deepcopy(action)
    delta = prev_pos - current_action[:3]

    for i, delta_component in enumerate(delta):
        if abs(delta_component) > 0.02:
            if prev_pos[i] > current_action[i]:
                current_action[i] = prev_action[i] - 0.02
            else:
                current_action[i] = prev_action[i] + 0.02

    return current_action
