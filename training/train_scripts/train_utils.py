import learn2learn as l2l
import os
import json
import yaml
import copy
import torch
import hydra
import random
import argparse
import datetime
import pickle as pkl
import numpy as np
import torch.nn as nn
from os.path import join
import torch.nn.functional as F
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from einops import rearrange, reduce, repeat, parse_shape
from multi_task_il.models.discrete_logistic import DiscreteMixLogistic
from collections import defaultdict, OrderedDict
from hydra.utils import instantiate
# need for val. loader
from multi_task_il.datasets.multi_task_datasets import DIYBatchSampler, collate_by_task
from torch.nn import CrossEntropyLoss, BCELoss
from torchmetrics.classification import Accuracy
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
# for visualization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1))
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1))


def loss_to_scalar(loss):
    x = loss.item()
    return float("{:.5f}".format(x))


def check_train_val_overlap(train_dataset, val_dataset):
    same_agent_file_cnt = 0
    same_demo_file_cnt = 0
    for task in train_dataset.agent_files.keys():
        for id in train_dataset.agent_files[task].keys():
            for agent_trj in train_dataset.agent_files[task][id]:
                if agent_trj in val_dataset.agent_files[task][id]:
                    same_agent_file_cnt += 1

    for task in train_dataset.demo_files.keys():
        for id in train_dataset.demo_files[task].keys():
            for demo_trj in train_dataset.demo_files[task][id]:
                if demo_trj in val_dataset.demo_files[task][id]:
                    same_demo_file_cnt += 1
    print(f"Overlapping counter {same_agent_file_cnt} - {same_demo_file_cnt}")


def make_data_loaders(config, dataset_cfg):
    """ Use .yaml cfg to create both train and val dataloaders """
    assert '_target_' in dataset_cfg.keys(), "Let's use hydra-config from now on. "
    print("Initializing {} with hydra config. \n".format(dataset_cfg._target_))

    dataset_cfg.mode = 'train'
    dataset = instantiate(dataset_cfg)
    train_step = int(config.get('epochs') *
                     int(len(dataset)/config.get('bsize')))
    samplerClass = DIYBatchSampler
    train_sampler = samplerClass(
        task_to_idx=dataset.task_to_idx,
        subtask_to_idx=dataset.subtask_to_idx,
        tasks_spec=dataset_cfg.tasks_spec,
        object_distribution_to_indx=dataset.object_distribution_to_indx,
        sampler_spec=config.samplers,
        n_step=train_step)
    train_loader = DataLoader(
        dataset,
        batch_sampler=train_sampler,
        num_workers=config.get('loader_workers', cpu_count()),
        worker_init_fn=lambda w: np.random.seed(
            np.random.randint(2 ** 29) + w),
        collate_fn=collate_by_task
    )

    dataset_cfg.mode = 'val'
    val_dataset = instantiate(dataset_cfg)
    # allow validation batch to have a different size
    config.samplers.batch_size = config.train_cfg.val_size
    val_step = int(config.get('epochs') *
                   int(len(val_dataset)/config.get('vsize')))
    val_sampler = samplerClass(
        task_to_idx=val_dataset.task_to_idx,
        subtask_to_idx=val_dataset.subtask_to_idx,
        tasks_spec=dataset_cfg.tasks_spec,
        object_distribution_to_indx=None,
        sampler_spec=config.samplers,
        n_step=val_step
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=config.get('loader_workers', cpu_count()),
        worker_init_fn=lambda w: np.random.seed(
            np.random.randint(2 ** 29) + w),
        collate_fn=collate_by_task
    )

    # check_train_val_overlap(train_dataset=dataset, val_dataset=val_dataset)
    return train_loader, val_loader


def collect_stats(step, task_losses, raw_stats, prefix='train'):
    """ create/append to stats dict of a one-layer dict structure:
        {'task_name/loss_key': [..], 'loss_key/task_name':[...]}"""
    task_names = sorted(task_losses.keys())
    for task, stats in task_losses.items():
        # expects: {'button': {"loss_sum": 1, "l_bc": 1}}
        for k, v in stats.items():
            for log_key in [f"{prefix}/{task}/{k}", f"{prefix}/{k}/{task}"]:
                if log_key not in raw_stats.keys():
                    raw_stats[log_key] = []
                raw_stats[log_key].append(loss_to_scalar(v))
        if "step" in raw_stats.keys():
            raw_stats["step"].append(int(step))
        else:
            raw_stats["step"] = [int(step)]
    tr_print = ""
    for i, task in enumerate(task_names):
        tr_print += "[{0:<9}] l_tot: {1:.1f} l_bc: {2:.1f} l_inv: {3: 1f} l_rep: {4: 1f} l_pnt: {5:.1f} l_aux: {6:.1f} ".format(
            task,
            raw_stats[f"{prefix}/{task}/loss_sum"][-1],
            raw_stats[f"{prefix}/{task}/l_bc"][-1],
            raw_stats.get(f"{prefix}/{task}/l_inv", [0])[-1],
            raw_stats.get(f"{prefix}/{task}/rep_loss", [0])[-1],
            raw_stats.get(f"{prefix}/{task}/point_loss", [0])[-1],
            raw_stats.get(f"{prefix}/{task}/l_aux", [0])[-1],
        )
        if i % 3 == 2:  # use two lines to print
            tr_print += "\n"

    return tr_print


def generate_figure(images, context, fname='burner.png'):
    _B, T_im, _, _H, _W = images.shape
    T_con = context.shape[1]
    print("Images value range: ", images.min(), images.max(), context.max())
    print("Generating figures from images shape {}, context shape {} \n".format(
        images.shape, context.shape))
    npairs = 7
    skip = 8
    ncols = 4
    fig, axs = plt.subplots(nrows=npairs * 2, ncols=ncols, figsize=(
        ncols*3.5, npairs*2*2.8), subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(left=0.03, right=0.97, hspace=0.3, wspace=0.05)
    for img_index in range(npairs):
        show_img = images[img_index*skip].cpu().numpy() * STD + MEAN
        show_con = context[img_index*skip].cpu().numpy() * STD + MEAN
        for count in range(ncols):
            axs[img_index*2, count].imshow(show_img[count].transpose(1, 2, 0))
            if count < T_con:
                axs[img_index*2+1,
                    count].imshow(show_con[count].transpose(1, 2, 0))

    plt.tight_layout()
    print("Saving figure to: ", fname)
    plt.savefig(fname)


def calculate_maml_loss(config, device, meta_model, model_inputs):
    states, actions = model_inputs['states'], model_inputs['actions']
    images, context = model_inputs['images'], model_inputs['demo']
    aux = model_inputs['aux_pose']

    meta_model = meta_model.to(device)
    inner_iters = config.daml.get('inner_iters', 1)
    l2error = torch.nn.MSELoss()

    # error = 0
    bc_loss, aux_loss = [], []

    for task in range(states.shape[0]):
        learner = meta_model.module.clone()
        for _ in range(inner_iters):
            learner.adapt(
                learner(None, context[task], learned_loss=True)['learned_loss'], allow_nograd=True, allow_unused=True)
        out = learner(states[task], images[task], ret_dist=False)
        l_aux = l2error(out['aux'], aux[task][None])[None]
        mu, sigma_inv, alpha = out['action_dist']
        action_distribution = DiscreteMixLogistic(
            mu[:-1], sigma_inv[:-1], alpha[:-1])
        l_bc = -torch.mean(action_distribution.log_prob(actions[task]))[None]
        bc_loss.append(l_bc)
        aux_loss.append(l_aux)

    return torch.cat(bc_loss, dim=0), torch.cat(aux_loss, dim=0)


def calculate_obj_pos_loss(config, train_cfg, device, model, task_inputs, loss, accuracy):
    model_inputs = defaultdict(list)
    task_to_idx = dict()
    target_obj_pos_one_hot = OrderedDict()
    start = 0
    for idx, (task_name, inputs) in enumerate(task_inputs.items()):

        for key in ['images', 'images_cp']:
            model_inputs[key].append(inputs['traj'][key].to(device))
        for key in ['demo', 'demo_cp']:
            model_inputs[key].append(inputs['demo_data'][key].to(device))

        task_inputs[task_name]['traj']['target_position_one_hot'].requires_grad = True
        obj_position = task_inputs[task_name]['traj']['target_position_one_hot'].to(
            device)
        obj_position.requires_grad = True
        target_obj_pos_one_hot[task_name] = obj_position
        task_bsize = inputs['traj']['images'].shape[0]
        task_to_idx[task_name] = [start + i for i in range(task_bsize)]
        start += task_bsize

    for key in model_inputs.keys():
        model_inputs[key] = torch.cat(model_inputs[key], dim=0)

    all_losses = OrderedDict()
    out = model(
        images=model_inputs['images'], images_cp=model_inputs['images_cp'],
        context=model_inputs['demo'],  context_cp=model_inputs['demo_cp'])

    ##### ---- #####
    # ToDo generalize to multiple-task
    ##### ---- #####
    for task_name in target_obj_pos_one_hot.keys():
        all_losses[task_name] = dict()
        # for each task compute the cross-entropy loss
        # B - T - Number Classes
        gt = target_obj_pos_one_hot[task_name].permute(0, 2, 1)
        gt.requires_grad = True
        prediction = out['target_obj_pred'].permute(0, 2, 1)
        all_losses[task_name]['ce_loss'] = loss(prediction, gt)

    all_accuracy = OrderedDict()
    for task_name in target_obj_pos_one_hot.keys():
        all_accuracy[task_name] = dict()
        gt = torch.argmax(
            target_obj_pos_one_hot[task_name].permute(0, 2, 1), dim=1)
        prediction = torch.argmax(
            out['target_obj_pred'].permute(0, 2, 1), dim=1)
        all_accuracy[task_name]['accuracy'] = accuracy(prediction, gt)

    return all_losses, all_accuracy


def calculate_task_loss_vima(config, train_cfg, device, model, task_inputs, mode='train'):
    model_inputs = defaultdict()
    task_to_idx = dict()
    task_losses = OrderedDict()
    start = 0
    for idx, (task_name, inputs) in enumerate(task_inputs.items()):
        traj = inputs['sample']
        input_keys = ['states',
                      'actions',
                      'prompt',
                      'prompt_token_type',
                      'word_batch',
                      'image_batch',
                      'obs']

        for key in input_keys:
            if key != 'prompt' and key != 'prompt_token_type':
                if key == 'image_batch' or key == 'obs':
                    model_inputs[key] = traj[key].to_torch_tensor(
                        device=device)
                else:
                    model_inputs[key] = traj[key].to(device)
            else:
                model_inputs[key] = traj[key]

        task_bsize = traj['actions'].shape[0]
        task_to_idx[task_name] = [start + i for i in range(task_bsize)]
        task_losses[task_name] = OrderedDict()
        start += task_bsize

    # for key in model_inputs.keys():
    #     model_inputs[key] = torch.cat(model_inputs[key], dim=0)
    all_losses = dict()

    out = model(
        input=model_inputs,
        mode=mode
    )

    # Compute Categorical-Cross-Entropy for each command component
    loss = CrossEntropyLoss(reduction="mean")
    position_accuracy_x = Accuracy(
        task="multiclass", num_classes=256).to(device=0)
    for key in out.keys():
        if "position_x_logits" == key:
            prediction_x = rearrange(
                out['position_x_logits'][:, :, :], 'T B C -> (T B) C').to(torch.float32)
            x_true = rearrange(rearrange(F.one_hot(
                model_inputs['actions'][:, :, 0].to(torch.int64), out['position_x_logits'][:, :, :].shape[-1]), 'B T C -> T B C'), 'T B C -> (T B) C').to(torch.float32)
            if mode == 'train':
                x_true.requires_grad = True
            loss_x = loss(prediction_x, x_true.to(torch.float32))

            # gt_action_class_x = torch.argmax(x_true, dim=1)
            # predicted_action_class_x = torch.argmax(prediction_x, dim=1)
            # accuracy_x = position_accuracy_x(
            #     predicted_action_class_x, gt_action_class_x)

        elif "position_y_logits" == key:
            y_true = rearrange(rearrange(F.one_hot(
                model_inputs['actions'][:, :, 1].to(torch.int64), out['position_y_logits'][:, :, :].shape[-1]), 'B T C -> T B C'), 'T B C -> (T B) C').to(torch.float32)
            if mode == 'train':
                y_true.requires_grad = True
            loss_y = loss(rearrange(
                out['position_y_logits'][:, :, :], 'T B C -> (T B) C').to(torch.float32), y_true.to(torch.float32))

        elif "position_z_logits" == key:
            z_true = rearrange(rearrange(F.one_hot(
                model_inputs['actions'][:, :, 2].to(torch.int64), out['position_z_logits'][:, :, :].shape[-1]), 'B T C -> T B C'), 'T B C -> (T B) C').to(torch.float32)
            if mode == 'train':
                z_true.requires_grad = True
            loss_z = loss(rearrange(
                out['position_z_logits'][:, :, :], 'T B C -> (T B) C').to(torch.float32), z_true.to(torch.float32))

        elif "rotation_r_logits" == key:
            r_true = rearrange(rearrange(F.one_hot(
                model_inputs['actions'][:, :, 3].to(torch.int64), out['rotation_r_logits'][:, :, :].shape[-1]), 'B T C -> T B C'), 'T B C -> (T B) C').to(torch.float32)
            if mode == 'train':
                r_true.requires_grad = True
            loss_r = loss(rearrange(
                out['rotation_r_logits'][:, :, :], 'T B C -> (T B) C').to(torch.float32), r_true.to(torch.float32))

        elif "rotation_p_logits" == key:
            p_true = rearrange(rearrange(F.one_hot(
                model_inputs['actions'][:, :, 4].to(torch.int64), out['rotation_p_logits'][:, :, :].shape[-1]), 'B T C -> T B C'), 'T B C -> (T B) C').to(torch.float32)
            if mode == 'train':
                p_true.requires_grad = True
            loss_p = loss(rearrange(
                out['rotation_p_logits'][:, :, :], 'T B C -> (T B) C').to(torch.float32), p_true.to(torch.float32))

        elif "rotation_y_logits" == key:
            yaw_true = rearrange(rearrange(F.one_hot(
                model_inputs['actions'][:, :, 5].to(torch.int64), out['rotation_y_logits'][:, :, :].shape[-1]), 'B T C -> T B C'), 'T B C -> (T B) C').to(torch.float32)
            if mode == 'train':
                yaw_true.requires_grad = True
            loss_yaw = loss(rearrange(
                out['rotation_y_logits'][:, :, :], 'T B C -> (T B) C').to(torch.float32), yaw_true.to(torch.float32))

        elif "gripper_logits" == key:
            gripper_true = rearrange(rearrange(F.one_hot(
                model_inputs['actions'][:, :, 6].to(torch.int64), out['gripper_logits'][:, :, :].shape[-1]), 'B T C -> T B C'), 'T B C -> (T B) C').to(torch.float32)
            if mode == 'train':
                gripper_true.requires_grad = True
            loss_gripper = loss(rearrange(
                out['gripper_logits'][:, :, :], 'T B C -> (T B) C').to(torch.float32), gripper_true.to(torch.float32))

    all_losses['l_bc'] = loss_x + loss_y + \
        loss_z + loss_r + loss_p + loss_yaw + loss_gripper

    all_losses["loss_sum"] = all_losses["l_bc"]
    # flatten here to avoid headache
    for (task_name, idxs) in task_to_idx.items():
        for (loss_name, loss_val) in all_losses.items():
            task_losses[task_name][loss_name] = torch.mean(loss_val)

    return task_losses


def calculate_task_loss(config, train_cfg, device, model, task_inputs):
    """Assumes inputs are collated by task names already, organize things properly before feeding into the model s.t.
        for each batch input, the model does only one forward pass."""

    model_inputs = defaultdict(list)
    task_to_idx = dict()
    task_losses = OrderedDict()
    start = 0
    for idx, (task_name, inputs) in enumerate(task_inputs.items()):
        traj = inputs['traj']
        input_keys = ['states', 'actions', 'images', 'images_cp']
        if config.use_daml:
            input_keys.append('aux_pose')
        for key in input_keys:
            model_inputs[key].append(traj[key].to(device))

        model_inputs['points'].append(traj['points'].to(device).long())
        for key in ['demo', 'demo_cp']:
            model_inputs[key].append(inputs['demo_data'][key].to(device))

        task_bsize = traj['actions'].shape[0]
        task_to_idx[task_name] = [start + i for i in range(task_bsize)]
        task_losses[task_name] = OrderedDict()
        start += task_bsize

    for key in model_inputs.keys():
        model_inputs[key] = torch.cat(model_inputs[key], dim=0)
    all_losses = dict()

    if config.use_daml:
        bc_loss, aux_loss = calculate_maml_loss(model, model_inputs)
        all_losses["l_bc"] = bc_loss
        all_losses["l_aux"] = aux_loss
        all_losses["loss_sum"] = bc_loss + aux_loss
    else:
        if config.policy._target_ == 'multi_task_il.models.mt_rep.VideoImitation':
            model = model.to(device)
            out = model(
                images=model_inputs['images'], images_cp=model_inputs['images_cp'],
                context=model_inputs['demo'],  context_cp=model_inputs['demo_cp'],
                states=model_inputs['states'], ret_dist=False,
                actions=model_inputs['actions'])
        else:  # other baselines
            out = model(
                images=model_inputs['images'],
                context=model_inputs['demo'],
                states=model_inputs['states'],
                ret_dist=False)

        # forward & backward action pred
        actions = model_inputs['actions']
        # mu_bc.shape: B, 7, 8, 4]) but actions.shape: B, 6, 8
        mu_bc, scale_bc, logit_bc = out['bc_distrib']
        action_distribution = DiscreteMixLogistic(
            mu_bc[:, :-1], scale_bc[:, :-1], logit_bc[:, :-1])
        act_prob = rearrange(- action_distribution.log_prob(actions),
                             'B n_mix act_dim -> B (n_mix act_dim)')

        all_losses["l_bc"] = train_cfg.bc_loss_mult * \
            torch.mean(act_prob, dim=-1)
        # compute inverse model density
        inv_distribution = DiscreteMixLogistic(*out['inverse_distrib'])
        inv_prob = rearrange(- inv_distribution.log_prob(actions),
                             'B n_mix act_dim -> B (n_mix act_dim)')
        all_losses["l_inv"] = train_cfg.inv_loss_mult * \
            torch.mean(inv_prob, dim=-1)

        if 'point_ll' in out:
            pnts = model_inputs['points']
            point_ll = - train_cfg.pnt_loss_mult * out['point_ll'][range(pnts.shape[0]),
                                                                   pnts[:, -1, 0], pnts[:, -1, 1]]
            # l_point = train_cfg.pnt_loss_mult * \
            #     torch.mean(-out['point_ll'][range(pnts.shape[0]),
            #                                 pnts[:, -1, 0], pnts[:, -1, 1]], dim=-1)

            all_losses["point_loss"] = point_ll

        # NOTE: the model should output calculated rep-learning loss
        if not model._load_target_obj_detector or not model._freeze_target_obj_detector:
            rep_loss = torch.zeros_like(all_losses["l_bc"])
            for k, v in out.items():
                if k in train_cfg.rep_loss_muls.keys():
                    v = torch.mean(v, dim=-1)  # just return size (B,) here
                    v = v * train_cfg.rep_loss_muls.get(k, 0)
                    all_losses[k] = v
                    rep_loss = rep_loss + v
            all_losses["rep_loss"] = rep_loss
        else:
            rep_loss = 0

        all_losses["loss_sum"] = all_losses["l_bc"] + \
            all_losses["l_inv"] + rep_loss
        all_losses["loss_sum"] = all_losses["loss_sum"] + \
            all_losses["point_loss"] if 'point_ll' in out else all_losses["loss_sum"]

    # flatten here to avoid headache
    for (task_name, idxs) in task_to_idx.items():
        for (loss_name, loss_val) in all_losses.items():
            if len(loss_val.shape) > 0:
                task_losses[task_name][loss_name] = torch.mean(loss_val[idxs])
    return task_losses
