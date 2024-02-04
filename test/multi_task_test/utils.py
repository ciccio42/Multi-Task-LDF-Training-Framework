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
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import resized_crop
from torchvision.transforms import ToTensor
from robosuite import load_controller_config
from multi_task_test import ENV_OBJECTS, TASK_MAP

DEBUG = False
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


def adjust_bb(bb, crop_params=[20, 25, 80, 75]):

    x1_old, y1_old, x2_old, y2_old = bb
    x1_old = int(x1_old)
    y1_old = int(y1_old)
    x2_old = int(x2_old)
    y2_old = int(y2_old)

    top, left = crop_params[0], crop_params[2]
    img_height, img_width = 200, 360
    box_h, box_w = img_height - top - \
        crop_params[1], img_width - left - crop_params[3]

    # Modify bb based on computed resized-crop
    # 1. Take into account crop and resize
    x_scale = 180/box_w
    y_scale = 100/box_h
    x1 = int((x1_old/x_scale)+left)
    x2 = int((x2_old/x_scale)+left)
    y1 = int((y1_old/y_scale)+top)
    y2 = int((y2_old/y_scale)+top)
    return [x1, y1, x2, y2]


def get_action(model, target_obj_dec, bb, predict_gt_bb, gt_classes, states, images, context, gpu_id, n_steps, max_T=80, baseline=None, action_ranges=[], target_obj_embedding=None, t=-1):
    s_t = torch.from_numpy(np.concatenate(states, 0).astype(np.float32))[None]
    if isinstance(images[-1], np.ndarray):
        i_t = torch.from_numpy(np.concatenate(
            images, 0).astype(np.float32))[None]
    else:
        i_t = images[0][None]
    s_t, i_t = s_t.float().cuda(gpu_id), i_t.float().cuda(gpu_id)

    if bb is not None:
        if isinstance(bb[0], np.ndarray):
            bb = torch.from_numpy(bb[0]).float().cuda(gpu_id)
        else:
            bb = bb[0][None]
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
                        gt_classes=gt_classes,
                        predict_gt_bb=predict_gt_bb,
                        eval=True,
                        target_obj_embedding=target_obj_embedding,
                        compute_activation_map=True,
                        t=t)  # to avoid computing ATC loss

            target_obj_embedding = out.get('target_obj_embedding', None)

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
    return action, predicted_prob, target_obj_embedding, out.get('activation_map', None), out.get('target_obj_prediction', None), out.get('predicted_bb', None)


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
                if gt_env.env_name == 'nut_assembly':
                    print(f"GT_ENV for {gt_env.env_name} not implemented")
                elif gt_env.env_name == 'press_button':
                    print(f"GT_ENV for {gt_env.env_name} not implemented")
                elif gt_env.env_name == "pick_place":
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
                elif gt_env.env_name == "block_stack":
                    gt_obs = gt_env.reset()
                    for i, gt_obj in enumerate(env.cubes):
                        obj_name = gt_obj.name
                        obj_pos = np.array(
                            env.sim.data.body_xpos[env.sim.model.body_name2id(gt_obj.root_body)])
                        obj_quat = T.convert_quat(
                            env.sim.data.body_xquat[env.sim.model.body_name2id(
                                gt_obj.root_body)], to="xyzw"
                        )
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


def get_gt_bb(env=None, traj=None, obs=None, task_name=None, t=0, real=True):
    # Get GT Bounding Box
    if task_name != 'stack_block':
        agent_target_obj_id = traj.get(t)['obs']['target-object']
    else:
        agent_target_obj_id = "cubeA"

    if env is not None:

        if env.env.env_name == "press_button":
            obj_name_list = env.env.names
        elif env.env.env_name == "block_stack":
            obj_name_list = env.env.cube_names.keys()
        else:
            obj_name_list = env.env.obj_names
    else:
        obj_name_list = ENV_OBJECTS[task_name]["obj_names"]
    for id, obj_name in enumerate(obj_name_list):
        if id == agent_target_obj_id or obj_name == agent_target_obj_id:
            try:
                if real:
                    top_left_x = obs['obj_bb']["camera_front"][obj_name]['upper_left_corner'][0]
                    top_left_y = obs['obj_bb']["camera_front"][obj_name]['upper_left_corner'][1]

                    bottom_right_x = obs['obj_bb']["camera_front"][obj_name]['bottom_right_corner'][0]
                    bottom_right_y = obs['obj_bb']["camera_front"][obj_name]['bottom_right_corner'][1]
                else:
                    top_left_x = obs['obj_bb']["camera_front"][obj_name]['bottom_right_corner'][0]
                    top_left_y = obs['obj_bb']["camera_front"][obj_name]['bottom_right_corner'][1]
                    # print(f"Top-Left X {top_left_x} - Top-Left Y {top_left_y}")
                    bottom_right_x = obs['obj_bb']["camera_front"][obj_name]['upper_left_corner'][0]
                    bottom_right_y = obs['obj_bb']["camera_front"][obj_name]['upper_left_corner'][1]
            except:
                top_left_x = obs['obj_bb'][obj_name]['upper_left_corner'][0]
                top_left_y = obs['obj_bb'][obj_name]['upper_left_corner'][1]
                # print(f"Top-Left X {top_left_x} - Top-Left Y {top_left_y}")
                bottom_right_x = obs['obj_bb'][obj_name]['bottom_right_corner'][0]
                bottom_right_y = obs['obj_bb'][obj_name]['bottom_right_corner'][1]
            bb_t = np.array(
                [[top_left_x, top_left_y, bottom_right_x, bottom_right_y]])
            gt_t = np.array(1)

    return bb_t, gt_t


def compute_activation_map(model, agent_obs, prediction):

    model.zero_grad()
    one_hot_output = torch.FloatTensor(1, 2).zero_()
    one_hot_output[0][1] = 1
    target_indx_flags = prediction['classes_final'][0] == 1
    target_max_score_indx = torch.argmax(
        prediction['conf_scores_final'][0][target_indx_flags])
    output = prediction['cls_scores'][target_max_score_indx][None]
    output.requires_grad = True
    output.backward(gradient=one_hot_output.cuda())

    # Get the gradients and the features
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = model.get_activations(agent_obs).detach()

    # Weight the channels by corresponding gradients
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    # Average the channels of the activations
    activation_map = torch.mean(
        activations, dim=1).squeeze().cpu().numpy()

    return activation_map


def plot_activation_map(activation_map, agent_obs, save_path="activation_map.png"):
    """
    Compute and plot the activation map overlaid on the original image.

    Parameters:
    - model: The pre-trained model.
    - agent_obs: The input tensor to the model. Expected shape is [B, C, H, W].
    - target_class: The target class for which the activation map should be computed.
    - save_path: Path to save the overlaid image.

    Returns:
    - None. The function saves the overlaid image to the specified path.
    """

    # Get the first image from the batch
    input_image = np.array((agent_obs[0, :, :, :].cpu(
    ).numpy() * 255).transpose((1, 2, 0)), dtype=np.uint8)

    # Resize the activation map to match the input image size
    heatmap_resized = cv2.resize(
        activation_map, (input_image.shape[1], input_image.shape[0]))

    # Convert the heatmap values between 0 and 255 for visualization
    heatmap_np = np.uint8(255 * heatmap_resized)

    # Convert the heatmap into a colormap
    heatmap_colormap = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)

    # Overlay the colormap on the original image
    overlaid_image = cv2.addWeighted(
        input_image, 0.5, heatmap_colormap, 0.5, 0)

    # Display the image using matplotlib
    plt.imshow(cv2.cvtColor(overlaid_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Activation Map Overlaid on Image")
    plt.savefig(save_path)
    plt.show()


def object_detection_inference(model, env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False, task_name="pick_place", controller=None, action_ranges=[], policy=True, perform_augs=False, config=None, gt_traj=None, activation_map=True, real=True):

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

        print(f"Max T {max_T}")
        status = None
        if task_name == "pick_place":
            object_name = env.objects[env.object_id].name
            obj_delta_key = object_name + '_to_robot0_eef_pos'

        elif task_name == "nut_assembly":
            object_name = env.nuts[env.nut_id].name
            obj_key = object_name + '_pos'
            if env.nut_id == 0:
                handle_loc = env.sim.data.site_xpos[env.sim.model.site_name2id(
                    'round-nut_handle_site')]
            elif env.nut_id == 1:
                handle_loc = env.sim.data.site_xpos[env.sim.model.site_name2id(
                    'round-nut-2_handle_site')]
            else:
                handle_loc = env.sim.data.site_xpos[env.sim.model.site_name2id(
                    'round-nut-3_handle_site')]
        elif task_name == 'button':
            button_loc = np.array(env.sim.data.site_xpos[env.target_button_id])
            dist = 0.090
        elif task_name == 'stack_block':
            object_name = 'cubeA'
            target_obj_loc = env.sim.data.body_xpos[env.cubeA_body_id]

        if task_name != 'button':
            obj_key = object_name + '_pos'
            start_z = obs[obj_key][2]
        bb_queue = []
        while not done:
            if task_name == 'pick_place':
                tasks['reached'] = check_reach(threshold=0.03,
                                               obj_distance=obs[obj_delta_key][:2],
                                               current_reach=tasks['reached'])

                tasks['picked'] = check_pick(threshold=0.05,
                                             obj_z=obs[obj_key][2],
                                             start_z=start_z,
                                             reached=tasks['reached'],
                                             picked=tasks['picked'])

            elif task_name == "nut_assembly":
                tasks['reached'] = tasks['reached'] or np.linalg.norm(
                    handle_loc - obs['eef_pos']) < 0.045
                tasks['picked'] = tasks['picked'] or (
                    tasks['reached'] and obs[obj_key][2] - start_z > 0.05)
            elif task_name == 'button':
                button_loc = np.array(
                    env.sim.data.site_xpos[env.target_button_id])
                tasks['reached'] = tasks['reached'] or \
                    np.linalg.norm(obs['eef_pos'] - button_loc) < dist
                tasks['picked'] = tasks['picked'] or \
                    (tasks['reached'])
            elif task_name == 'stack_block':
                tasks['reached'] = tasks['reached'] or np.linalg.norm(
                    target_obj_loc - obs['eef_pos']) < 0.045
                tasks['picked'] = tasks['picked'] or (
                    tasks['reached'] and obs[obj_key][2] - start_z > 0.05)

            bb_t, gt_t = get_gt_bb(traj=traj,
                                   obs=obs,
                                   task_name=task_name,
                                   real=real
                                   )

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

            if activation_map:
                pass
                # last_conv_layer = model._backbone.
                # model.register_forward_hook(last_conv_layer)
                # model.register_backward_hook(last_conv_layer)
                # actvation_map = compute_activation_map(model=model,
                #                                        agent_obs=model_input,
                #                                        prediction=prediction)

            # Project bb over image
            # 1. Get the index with target class
            target_indx_flags = prediction['classes_final'][0] == 1
            if torch.sum((target_indx_flags == True).int()) != 0:
                # 2. Get the confidence scores for the target predictions and the the max
                target_max_score_indx = torch.argmax(
                    prediction['conf_scores_final'][0][target_indx_flags])
                max_score_target = prediction['conf_scores_final'][0][target_indx_flags][target_max_score_indx][None]
                if perform_augs:
                    scale_factor = model.get_scale_factors()
                    image = np.array(np.moveaxis(
                        formatted_img[:, :, :].cpu().numpy()*255, 0, -1), dtype=np.uint8)
                    predicted_bb = project_bboxes(bboxes=prediction['proposals'][0][None][None],
                                                  width_scale_factor=scale_factor[0],
                                                  height_scale_factor=scale_factor[1],
                                                  mode='a2p')[0][target_indx_flags][target_max_score_indx][None]
                else:
                    image = formatted_img.cpu().numpy()
                    predicted_bb = prediction['proposals'][0]

                if True:
                    for indx, bb in enumerate(predicted_bb):
                        image = cv2.rectangle(np.ascontiguousarray(image),
                                              (int(bb[0]),
                                               int(bb[1])),
                                              (int(bb[2]),
                                               int(bb[3])),
                                              color=(0, 0, 255), thickness=1)
                        image = cv2.putText(image, "Score {:.2f}".format(max_score_target[indx]),
                                            (int(bb[0]),
                                             int(bb[1])),
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

                    cv2.imwrite("predicted_bb.png", image)
                obs['predicted_bb'] = predicted_bb.cpu().numpy()
                obs['predicted_score'] = max_score_target.cpu().numpy()
                obs['gt_bb'] = bb_t
                # compute IoU over time
                iou_t = box_iou(boxes1=torch.from_numpy(
                    bb_t).to(device=gpu_id), boxes2=predicted_bb)
                obs['iou'] = iou_t[0][0].cpu().numpy()

                # check if there are TP
                if iou_t[0][0].cpu().numpy() < 0.5:
                    fp += 1
                    obs['fp'] = 1
                else:
                    tp += 1
                    obs['tp'] = 1

                iou += iou_t[0][0].cpu().numpy()
                traj.append(obs)
            else:
                # scale_factor = model.get_scale_factors()
                # predicted_bb = project_bboxes(bboxes=prediction['proposals'][0][None].float(),
                #                               width_scale_factor=scale_factor[0],
                #                               height_scale_factor=scale_factor[1],
                #                               mode='a2p')[0]
                # obs['predicted_bb'] = predicted_bb.cpu().numpy()
                obs['gt_bb'] = bb_t
                obs['iou'] = 0
                iou += 0
                fn += 1
                obs['fn'] = 1
                traj.append(obs)

            if controller is not None:
                # compute the action for the current state
                if task_name == "nut_assembly":
                    action_gt, status = controller.act(obs, status)
                else:
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
            tasks['success'] = reward or tasks['success']
            if n_steps >= max_T or env_done or reward:
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
            if True:  # t == 1:
                agent_obs = gt_traj[t]['obs']['camera_front_image']
                bb_t, gt_t = get_gt_bb(traj=gt_traj,
                                       obs=gt_traj[t]['obs'],
                                       task_name=task_name,
                                       t=t)

                if perform_augs:
                    formatted_img, bb_t = img_formatter(
                        agent_obs[:, :, ::-1], bb_t)
                else:
                    cv2.imwrite("agent_obs.png", agent_obs)
                    formatted_img = ToTensor()(agent_obs.copy())

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
                        scale_factor = model.get_scale_factors()
                        image = np.array(np.moveaxis(
                            formatted_img[:, :, :].cpu().numpy()*255, 0, -1), dtype=np.uint8)
                        predicted_bb = project_bboxes(bboxes=prediction['proposals'][0][None][None],
                                                      width_scale_factor=scale_factor[0],
                                                      height_scale_factor=scale_factor[1],
                                                      mode='a2p')[0][:10]
                    else:
                        image = np.array(np.moveaxis(
                            formatted_img[:, :, :].cpu().numpy()*255, 0, -1), dtype=np.uint8)
                        scale_factor = model.get_scale_factors()
                        predicted_bb = project_bboxes(bboxes=prediction['proposals'][0][None][None],
                                                      width_scale_factor=scale_factor[0],
                                                      height_scale_factor=scale_factor[1],
                                                      mode='a2p')[0][target_indx_flags][target_max_score_indx][None]

                    if True:
                        for indx, bb in enumerate(predicted_bb):
                            color = (255, 0, 0)
                            image = cv2.rectangle(np.ascontiguousarray(image),
                                                  (int(bb[0]),
                                                   int(bb[1])),
                                                  (int(bb[2]),
                                                   int(bb[3])),
                                                  color=color, thickness=1)
                            image = cv2.putText(image, "Score {:.2f}".format(max_score_target[indx]),
                                                (int(bb[0]),
                                                int(bb[1])),
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
        return gt_traj, info


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


def check_bin(threshold: float, bin_pos: np.array, obj_pos: np.array, current_bin: bool):
    bin_x_low = bin_pos[0]
    bin_y_low = bin_pos[1]
    bin_x_low -= 0.16 / 2
    bin_y_low -= 0.16 / 2

    bin_x_high = bin_x_low + 0.16
    bin_y_high = bin_y_low + 0.16
    # print(bin_pos, obj_pos)
    res = False
    if (
            bin_x_low < obj_pos[0] < bin_x_high
            and bin_y_low < obj_pos[1] < bin_y_high
            and bin_pos[2] < obj_pos[2] < bin_pos[2] + 0.1
    ):
        res = True
    return (current_bin or res)


def check_peg(peg_pos: np.array, obj_pos: np.array, current_peg: bool):

    # print(bin_pos, obj_pos)
    res = False
    if (
            abs(obj_pos[0] - peg_pos[0]) < 0.03
            and abs(obj_pos[1] - peg_pos[1]) < 0.03
            and obj_pos[2] < 0.860 + 0.05
    ):
        res = True
    return res or current_peg


def clip_action(action, prev_action):
    prev_pos = prev_action[:3]
    current_action = copy.deepcopy(action)
    delta = prev_pos - current_action[:3]

    for i, delta_component in enumerate(delta):
        if abs(delta_component) > 0.005:
            if prev_pos[i] > current_action[i]:
                current_action[i] = prev_action[i] - 0.005
            else:
                current_action[i] = prev_action[i] + 0.005

    return current_action


def build_tvf_formatter(config, env_name='stack'):
    """Use this for torchvision.transforms in multi-task dataset, 
    note eval_fn always feeds in traj['obs']['images'], i.e. shape (h,w,3)
    """
    dataset_cfg = config.train_cfg.dataset
    height, width = dataset_cfg.get(
        'height', 100), dataset_cfg.get('width', 180)
    task_spec = config.tasks_cfgs.get(env_name, dict())

    crop_params = task_spec.get('crop', [0, 0, 0, 0])
    # print(crop_params)
    top, left = crop_params[0], crop_params[2]

    def resize_crop(img):
        if len(img.shape) == 4:
            img = img[0]
        img_h, img_w = img.shape[0], img.shape[1]
        assert img_h != 3 and img_w != 3, img.shape
        box_h, box_w = img_h - top - \
            crop_params[1], img_w - left - crop_params[3]
        # cv2.imwrite("obs.png", np.array(img))
        obs = ToTensor()(img.copy())
        obs = resized_crop(obs, top=top, left=left, height=box_h, width=box_w,
                           size=(height, width))
        cv2.imwrite("resized_test.png",
                    np.moveaxis(obs.numpy(), 0, -1)*255)

        # weak_scale = config.augs.get('weak_crop_scale', (0.8, 1.0))
        # weak_ratio = [1.0, 1.0]
        # randcrop = RandomResizedCrop(
        #     size=(height, width), scale=weak_scale, ratio=weak_ratio)
        # cv2.imwrite("obs_cropped.png", np.moveaxis(obs.numpy(), 0, -1)*255)
        # # obs = Normalize(mean=[0.485, 0.456, 0.406],
        # #                 std=[0.229, 0.224, 0.225])(obs)
        # obs = randcrop(obs)
        cv2.imwrite("random_resized_crop_test.png",
                    np.moveaxis(obs.numpy(), 0, -1)*255)
        return obs
    return resize_crop


def build_tvf_formatter_obj_detector(config, env_name):
    """Use this for torchvision.transforms in multi-task dataset, 
    note eval_fn always feeds in traj['obs']['images'], i.e. shape (h,w,3)
    """

    def resize_crop(img, bb=None):
        img_height, img_width = img.shape[:2]
        """applies to every timestep's RGB obs['camera_front_image']"""
        task_spec = config.tasks_cfgs.get(env_name, dict())
        crop_params = task_spec.get('crop', [0, 0, 0, 0])
        top, left = crop_params[0], crop_params[2]
        img_height, img_width = img.shape[0], img.shape[1]
        box_h, box_w = img_height - top - \
            crop_params[1], img_width - left - crop_params[3]

        img = transforms.ToTensor()(img.copy())
        # ---- Resized crop ----#
        img = resized_crop(img, top=top, left=left, height=box_h,
                           width=box_w, size=(config.dataset_cfg.height, config.dataset_cfg.width))
        # transforms_pipe = transforms.Compose([
        #     transforms.ColorJitter(
        #         brightness=list(config.augs.get(
        #             "brightness", [0.875, 1.125])),
        #         contrast=list(config.augs.get(
        #             "contrast", [0.5, 1.5])),
        #         saturation=list(config.augs.get(
        #             "contrast", [0.5, 1.5])),
        #         hue=list(config.augs.get("hue", [-0.05, 0.05]))
        #     ),
        # ])
        # img = transforms_pipe(img)

        cv2.imwrite("resized_target_obj.png", np.moveaxis(
            img.numpy()*255, 0, -1))

        if bb is not None:
            from multi_task_il.datasets.utils import adjust_bb
            bb = adjust_bb(dataset_loader=config.dataset_cfg,
                           bb=bb,
                           obs=img,
                           img_height=img_height,
                           img_width=img_width,
                           top=top,
                           left=left,
                           box_w=box_w,
                           box_h=box_h)

            # image = cv2.rectangle(np.ascontiguousarray(np.array(np.moveaxis(
            #     img.numpy()*255, 0, -1), dtype=np.uint8)),
            #     (bb[0][0],
            #      bb[0][1]),
            #     (bb[0][2],
            #      bb[0][3]),
            #     color=(0, 0, 255),
            #     thickness=1)
            # cv2.imwrite("bb_cropped.png", image)
            return img, bb

        return img

    return resize_crop


def build_env(ctr=0, env_name='nut', heights=100, widths=200, size=False, shape=False, color=False, gpu_id=0, variation=None, controller_path=None):

    # create_seed = random.Random(None)
    # create_seed = create_seed.getrandbits(32)
    if controller_path == None:
        controller = load_controller_config(default_controller='IK_POSE')
    else:
        # load custom controller
        controller = load_controller_config(
            custom_fpath=controller_path)
    # assert gpu_id != -1
    build_task = TASK_MAP.get(env_name, None)
    assert build_task, 'Got unsupported task '+env_name
    div = int(build_task['num_variations'])
    env_fn = build_task['env_fn']
    agent_name, teacher_name = build_task['agent-teacher']

    if variation == None:
        variation = ctr % div
    else:
        variation = variation

    if 'Stack' in teacher_name:

        agent_env = env_fn(agent_name,
                           size=size,
                           shape=shape,
                           color=color,
                           controller_type=controller,
                           task=variation,
                           ret_env=True,
                           seed=create_seed,
                           gpu_id=gpu_id,
                           object_set=TASK_MAP[env_name]['object_set'])
    else:

        agent_env = env_fn(agent_name,
                           controller_type=controller,
                           task=variation,
                           ret_env=True,
                           seed=create_seed,
                           gpu_id=gpu_id,
                           object_set=TASK_MAP[env_name]['object_set'])

    return agent_env, variation


def build_env_context(img_formatter, T_context=4, ctr=0, env_name='nut', heights=100, widths=200, size=False, shape=False, color=False, gpu_id=0, variation=None, random_frames=True, controller_path=None, ret_gt_env=False, seed=42):

    print(f"Seed: {seed}")
    if controller_path == None:
        controller = load_controller_config(default_controller='IK_POSE')
    else:
        # load custom controller
        controller = load_controller_config(
            custom_fpath=controller_path)
    # assert gpu_id != -1
    build_task = TASK_MAP.get(env_name, None)
    assert build_task, 'Got unsupported task '+env_name
    div = int(build_task['num_variations'])
    env_fn = build_task['env_fn']
    agent_name, teacher_name = build_task['agent-teacher']

    if variation == None:
        variation = ctr % div
    else:
        variation = variation

    # if 'Stack' in teacher_name:
    #     teacher_expert_rollout = env_fn(teacher_name,
    #                                     controller_type=controller,
    #                                     task=variation,
    #                                     size=size,
    #                                     shape=shape,
    #                                     color=color,
    #                                     seed=seed,
    #                                     gpu_id=gpu_id,
    #                                     object_set=TASK_MAP[env_name]['object_set'])
    #     agent_env = env_fn(agent_name,
    #                        size=size,
    #                        shape=shape,
    #                        color=color,
    #                        controller_type=controller,
    #                        task=variation,
    #                        ret_env=True,
    #                        seed=seed,
    #                        gpu_id=gpu_id,
    #                        object_set=TASK_MAP[env_name]['object_set'])
    # else:
    teacher_expert_rollout = env_fn(teacher_name,
                                    controller_type=controller,
                                    task=variation,
                                    seed=seed,
                                    gpu_id=gpu_id,
                                    object_set=TASK_MAP[env_name]['object_set'])

    agent_env = env_fn(agent_name,
                       controller_type=controller,
                       task=variation,
                       ret_env=True,
                       seed=seed,
                       gpu_id=gpu_id,
                       object_set=TASK_MAP[env_name]['object_set'])

    if ret_gt_env:
        gt_env = env_fn(agent_name,
                        controller_type=controller,
                        task=variation,
                        ret_env=True,
                        seed=seed,
                        gpu_id=gpu_id,
                        object_set=TASK_MAP[env_name]['object_set'])

    assert isinstance(teacher_expert_rollout, Trajectory)
    context = select_random_frames(
        teacher_expert_rollout, T_context, sample_sides=True, random_frames=random_frames)
    # convert BGR context image to RGB and scale to 0-1
    # for i, img in enumerate(context):
    #     cv2.imwrite(f"context_{i}.png", np.array(img[:, :, ::-1]))
    context = [img_formatter(i[:, :, ::-1])[None] for i in context]
    # assert len(context ) == 6
    if isinstance(context[0], np.ndarray):
        context = torch.from_numpy(np.concatenate(context, 0))[None]
    else:
        context = torch.cat(context, dim=0)[None]

    if ret_gt_env:
        return agent_env, context, variation, teacher_expert_rollout, gt_env
    else:
        return agent_env, context, variation, teacher_expert_rollout


def build_context(img_formatter, T_context=4, ctr=0, env_name='nut', heights=100, widths=200, size=False, shape=False, color=False, gpu_id=0, variation=None, random_frames=True, controller_path=None, ret_gt_env=False, seed=42):

    print(f"Seed: {seed}")
    if controller_path == None:
        controller = load_controller_config(default_controller='IK_POSE')
    else:
        # load custom controller
        controller = load_controller_config(
            custom_fpath=controller_path)
    # assert gpu_id != -1
    build_task = TASK_MAP.get(env_name, None)
    assert build_task, 'Got unsupported task '+env_name
    div = int(build_task['num_variations'])
    env_fn = build_task['env_fn']
    agent_name, teacher_name = build_task['agent-teacher']

    if variation == None:
        variation = ctr % div
    else:
        variation = variation
    TASK_MAP[env_name]['object_set'] = 2
    if 'Stack' in teacher_name:
        teacher_expert_rollout = env_fn(teacher_name,
                                        controller_type=controller,
                                        task=variation,
                                        size=size,
                                        shape=shape,
                                        color=color,
                                        seed=seed,
                                        gpu_id=gpu_id,
                                        object_set=TASK_MAP[env_name]['object_set'])
    else:
        teacher_expert_rollout = env_fn(teacher_name,
                                        controller_type=controller,
                                        task=variation,
                                        seed=seed,
                                        gpu_id=gpu_id,
                                        object_set=TASK_MAP[env_name]['object_set'])

    assert isinstance(teacher_expert_rollout, Trajectory)
    context = select_random_frames(
        teacher_expert_rollout, T_context, sample_sides=True, random_frames=random_frames)
    # convert BGR context image to RGB and scale to 0-1
    for i, img in enumerate(context):
        cv2.imwrite(f"context_{i}.png", np.array(img[:, :, ::-1]))
    context = [img_formatter(i[:, :, ::-1])[None] for i in context]
    # assert len(context ) == 6
    if isinstance(context[0], np.ndarray):
        context = torch.from_numpy(np.concatenate(context, 0))[None]
    else:
        context = torch.cat(context, dim=0)[None]

    return context, variation, teacher_expert_rollout


def get_eval_fn(env_name):
    if "pick_place" in env_name:
        from multi_task_test.pick_place import pick_place_eval
        return pick_place_eval
    elif "nut_assembly" in env_name:
        from multi_task_test.nut_assembly import nut_assembly_eval
        return nut_assembly_eval
    elif "button" in env_name:
        from multi_task_test.button_press import press_button_eval
        return press_button_eval
    elif "stack" in env_name:
        from multi_task_test.block_stack import block_stack_eval
        return block_stack_eval
    else:
        assert NotImplementedError
