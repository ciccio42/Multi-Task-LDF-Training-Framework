import hydra
# from multi_task_test.eval_functions import *

import random
import copy
import os
from collections import defaultdict
import torch
import numpy as np
import pickle as pkl
import imageio
import functools
from torch.multiprocessing import Pool, set_start_method
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import cv2
import random
from tokenizers import Tokenizer
from tokenizers import AddedToken
import vima.nn as vnn
from vima.utils import *
from einops import rearrange, repeat

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
PLACEHOLDERS = [token.content for token in PLACEHOLDER_TOKENS]
tokenizer = Tokenizer.from_pretrained("t5-base")
tokenizer.add_tokens(PLACEHOLDER_TOKENS)

ENV_OBJECTS = {
    'pick_place': {
        'obj_names': ['greenbox', 'yellowbox', 'bluebox', 'redbox', 'bin'],
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
