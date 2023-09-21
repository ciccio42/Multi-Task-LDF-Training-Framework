import os
import json
import yaml
import copy
import torch
import hydra
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
from multi_task_il.datasets.utils import DIYBatchSampler, collate_by_task
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from omegaconf import OmegaConf
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup
from multi_task_il.utils.lr_scheduler import build_scheduler
from multi_task_il.utils.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
import wandb
from torchsummary import summary
from tqdm import tqdm
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import learn2learn as l2l
from torchvision.ops import box_iou
from multi_task_il.models.cond_target_obj_detector.utils import project_bboxes


torch.autograd.set_detect_anomaly(True)
# for visualization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 3, 1, 1))
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 3, 1, 1))
DEBUG = False


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
        collate_fn=collate_by_task,
        pin_memory=True,
        prefetch_factor=5,
        persistent_workers=True
    )

    val_loader = None
    if dataset_cfg.split[1] > 0.0:
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
            collate_fn=collate_by_task,
            pin_memory=False,
            prefetch_factor=5,
            persistent_workers=True
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
        tr_print += "[{0:<9}] l_tot: {1:.4f} l_bc: {2:.4f} l_inv: {3: 4f} l_rep: {4: 4f} l_pnt: {5:.4f} l_aux: {6:.4f} avg_prec {7:.4f}".format(
            task,
            raw_stats.get(f"{prefix}/{task}/loss_sum", [0])[-1],
            raw_stats.get(f"{prefix}/{task}/l_bc", [0])[-1],
            raw_stats.get(f"{prefix}/{task}/l_inv", [0])[-1],
            raw_stats.get(f"{prefix}/{task}/rep_loss", [0])[-1],
            raw_stats.get(f"{prefix}/{task}/point_loss", [0])[-1],
            raw_stats.get(f"{prefix}/{task}/l_aux", [0])[-1],
            raw_stats.get(f"{prefix}/{task}/avg_prec", [0])[-1],
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
        learner = meta_model.clone()
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


def loss_func_bb(config, train_cfg, device, model, inputs, w_conf=1, w_reg=5):

    def calc_cls_loss(conf_scores_pos, conf_scores_neg, batch_size):
        target_pos = torch.ones_like(conf_scores_pos)
        target_neg = torch.zeros_like(conf_scores_neg)

        target = torch.cat((target_pos, target_neg))
        inputs = torch.cat((conf_scores_pos, conf_scores_neg))

        loss = F.binary_cross_entropy_with_logits(
            inputs, target, reduction='sum') * 1. / batch_size

        return loss

    def calc_bbox_reg_loss(gt_offsets, reg_offsets_pos, batch_size):
        assert gt_offsets.size() == reg_offsets_pos.size()
        loss = F.smooth_l1_loss(reg_offsets_pos, gt_offsets,
                                reduction='sum') * 1. / batch_size
        return loss

    def compute_avg_prec(gt_bb=None, predicted_bb=None, thr=0.7):
        from multi_task_il.models.cond_target_obj_detector.utils import get_iou_mat
        # compute IoU over time
        gt_bb = rearrange(gt_bb, 'B T N BB -> (B T N) BB')
        predicted_bb = rearrange(predicted_bb, 'B T N BB -> (B T N) BB')
        iou_t = torch.diagonal(
            box_iou(boxes1=gt_bb, boxes2=predicted_bb))  # (B T N) BB
        tp = (torch.where(iou_t > thr, 1.0, 0.0) == 1.0).sum(dim=0)
        return tp/gt_bb.shape[0]

    model_inputs = defaultdict(list)
    task_to_idx = dict()
    task_losses = OrderedDict()
    start = 0
    for idx, (task_name, inputs) in enumerate(inputs.items()):
        traj = inputs['traj']

        for key in traj.keys():
            model_inputs[key].append(traj[key].to(device))

        for key in inputs['demo_data'].keys():
            model_inputs[key].append(inputs['demo_data'][key].to(device))

        task_bsize = traj['images'].shape[0]
        task_to_idx[task_name] = [start + i for i in range(task_bsize)]
        task_losses[task_name] = OrderedDict()
        start += task_bsize

    for key in model_inputs.keys():
        model_inputs[key] = torch.cat(model_inputs[key], dim=0)

    all_losses = dict()

    model = model.to(device)
    predictions_dict = model(model_inputs, inference=False)
    # compute detection loss
    cls_loss = calc_cls_loss(predictions_dict['conf_scores_pos'],
                             predictions_dict['conf_scores_neg'],
                             traj['images'].shape[0]*traj['images'].shape[1])
    bb_reg_loss = calc_bbox_reg_loss(predictions_dict['GT_offsets'],
                                     predictions_dict['offsets_pos'],
                                     traj['images'].shape[0]*traj['images'].shape[1])

    # all_losses["cls_loss"] = cls_loss
    # all_losses["bb_reg_loss"] = bb_reg_loss
    # all_losses["loss_sum"] = w_conf*cls_loss + w_reg*bb_reg_loss

    predictions_dict = model(model_inputs, inference=False)
    # compute detection loss
    cls_loss = calc_cls_loss(predictions_dict['conf_scores_pos'],
                             predictions_dict['conf_scores_neg'],
                             traj['images'].shape[0]*traj['images'].shape[1])
    bb_reg_loss = calc_bbox_reg_loss(predictions_dict['GT_offsets'],
                                     predictions_dict['offsets_pos'],
                                     traj['images'].shape[0]*traj['images'].shape[1])

    all_losses["cls_loss"] = cls_loss
    all_losses["bb_reg_loss"] = bb_reg_loss
    all_losses["loss_sum"] = w_conf*cls_loss + w_reg*bb_reg_loss

    # compute average precision
    proposals = predictions_dict['proposals'][:, None, None, :]
    # take the bounding box with the highest confidence-score and compute the IoU with
    scale_factor = model.get_scale_factors()
    proposals = project_bboxes(bboxes=proposals,
                               width_scale_factor=scale_factor[0],
                               height_scale_factor=scale_factor[1],
                               mode='a2p')[:, None, :, :]
    all_losses["avg_prec"] = compute_avg_prec(gt_bb=model_inputs['gt_bb'],
                                              predicted_bb=proposals)

    if DEBUG:
        import cv2
        for indx in range(inputs['traj']['images'].shape[0]):
            image = np.array(np.moveaxis(
                inputs['traj']['images'][indx, 0, :, :, :].cpu().numpy()*255, 0, -1), dtype=np.uint8)
            proposal = proposals[indx].cpu()
            bb_gt = inputs['traj']['gt_bb'][indx][0][0].cpu()
            image = cv2.rectangle(np.ascontiguousarray(image),
                                  (int(proposal[0, 0, 0]), int(
                                      proposal[0, 0, 1])),
                                  (int(proposal[0, 0, 2]), int(
                                      proposal[0, 0, 3])),
                                  color=(0, 0, 255), thickness=1)
            image = cv2.rectangle(np.ascontiguousarray(image),
                                  (int(bb_gt[0]), int(
                                      bb_gt[1])),
                                  (int(bb_gt[2]), int(
                                      bb_gt[3])),
                                  color=(0, 255, 0), thickness=1)
            cv2.imwrite("prova_predictions_eval.png", image)

    # flatten here to avoid headache
    for (task_name, idxs) in task_to_idx.items():
        for (loss_name, loss_val) in all_losses.items():
            if len(loss_val.shape) > 0:
                task_losses[task_name][loss_name] = torch.mean(loss_val[idxs])
            else:
                task_losses[task_name][loss_name] = loss_val
    return task_losses


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
        # obj_position.requires_grad = True
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
        # gt.requires_grad = True
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


def loss_function_vima(config, train_cfg, device, model, task_inputs, mode='train'):
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
    for key in out.keys():
        if "position_x_logits" == key:
            prediction_x = rearrange(
                out['position_x_logits'][:, :, :], 'T B C -> (T B) C').to(torch.float32)
            x_true = rearrange(rearrange(F.one_hot(
                model_inputs['actions'][:, :, 0].to(torch.int64), out['position_x_logits'][:, :, :].shape[-1]), 'B T C -> T B C'), 'T B C -> (T B) C').to(torch.float32)
            if mode == 'train':
                x_true.requires_grad = True
            loss_x = loss(prediction_x, x_true.to(torch.float32))

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

    all_losses['l_bc'] = loss_x + loss_y + \
        loss_z + loss_r + loss_p + loss_yaw  # + loss_gripper

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
        input_keys = traj.keys()

        if config.get('use_daml', False):
            input_keys.append('aux_pose')
        for key in input_keys:
            model_inputs[key].append(traj[key].to(device))

        # if 'points' in traj.keys():
        #     model_inputs['points'].append(traj['points'].to(device).long())

        for key in inputs['demo_data'].keys():
            model_inputs[key].append(inputs['demo_data'][key].to(device))

        task_bsize = traj['actions'].shape[0]
        task_to_idx[task_name] = [start + i for i in range(task_bsize)]
        task_losses[task_name] = OrderedDict()
        start += task_bsize

    for key in model_inputs.keys():
        model_inputs[key] = torch.cat(model_inputs[key], dim=0)
    all_losses = dict()

    if config.get('use_daml', False):
        bc_loss, aux_loss = calculate_maml_loss(
            config=config,
            device=device,
            meta_model=model,
            model_inputs=model_inputs)
        all_losses["l_bc"] = bc_loss
        all_losses["l_aux"] = aux_loss
        all_losses["loss_sum"] = bc_loss + aux_loss
    else:
        if config.policy._target_ == 'multi_task_il.models.mt_rep.VideoImitation':
            out = model(
                images=model_inputs['images'],
                images_cp=model_inputs['images_cp'],
                context=model_inputs['demo'],
                context_cp=model_inputs['demo_cp'],
                states=model_inputs['states'],
                bb=model_inputs['gt_bb'],
                ret_dist=False,
                actions=model_inputs['actions'])
        elif "CondPolicy" in config.policy._target_:
            out = model(
                inputs=model_inputs,
                inference=False,
                oracle=False)
        else:  # other baselines
            out = model(
                images=model_inputs['images'],
                context=model_inputs['demo'],
                states=model_inputs['states'],
                ret_dist=False)

        # forward & backward action pred
        actions = model_inputs['actions']
        if "CondPolicy" not in config.policy._target_:
            # mu_bc.shape: B, 7, 8, 4]) but actions.shape: B, 6, 8
            mu_bc, scale_bc, logit_bc = out['bc_distrib']
            action_distribution = DiscreteMixLogistic(
                mu_bc[:, :-1], scale_bc[:, :-1], logit_bc[:, :-1])
            act_prob = rearrange(- action_distribution.log_prob(actions),
                                 'B n_mix act_dim -> B (n_mix act_dim)')

        else:
            actions = rearrange(actions, 'B T act_dim -> (B T) act_dim')
            act_prob = - out['bc_distrib'].log_prob(actions)
            if len(act_prob.shape) == 1:
                act_prob = rearrange(
                    act_prob, '(B T) -> B T',
                    B=model_inputs['actions'].shape[0],
                    T=model_inputs['actions'].shape[1])
            else:
                act_prob = torch.sum(act_prob, dim=-1)
                act_prob = rearrange(
                    act_prob, '(B T) -> B T',
                    B=model_inputs['actions'].shape[0],
                    T=model_inputs['actions'].shape[1])

        all_losses["l_bc"] = train_cfg.bc_loss_mult * \
            torch.mean(act_prob, dim=-1)

        if 'inverse_distrib' in out.keys():
            # compute inverse model density
            inv_distribution = DiscreteMixLogistic(*out['inverse_distrib'])
            inv_prob = rearrange(- inv_distribution.log_prob(actions),
                                 'B n_mix act_dim -> B (n_mix act_dim)')
            all_losses["l_inv"] = train_cfg.inv_loss_mult * \
                torch.mean(inv_prob, dim=-1)

        if 'point_ll' in out:
            pnts = model_inputs['points']
            l_point = train_cfg.pnt_loss_mult * out['point_ll'][range(pnts.shape[0]),
                                                                pnts[:, -1, 0].long(), pnts[:, -1, 1].long()]

            all_losses["point_loss"] = l_point

        # NOTE: the model should output calculated rep-learning loss
        if hasattr(model, "_load_target_obj_detector") and hasattr(model, "_freeze_target_obj_detector"):
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
                all_losses["rep_loss"] = 0
        else:
            pass

        loss_sum = 0
        for loss_key in ['l_bc', 'l_inv', 'rep_loss']:
            loss_sum += all_losses[loss_key] if loss_key in all_losses.keys() else 0.0
        all_losses["loss_sum"] = loss_sum

        all_losses["loss_sum"] = all_losses["loss_sum"] + \
            all_losses["point_loss"] if 'point_ll' in out else all_losses["loss_sum"]

    # flatten here to avoid headache
    for (task_name, idxs) in task_to_idx.items():
        for (loss_name, loss_val) in all_losses.items():
            if len(loss_val.shape) > 0:
                task_losses[task_name][loss_name] = torch.mean(loss_val[idxs])
    return task_losses


class Trainer:

    def __init__(self, allow_val_grad=False, hydra_cfg=None):
        assert hydra_cfg is not None, "Need to start with hydra-enabled yaml file!"
        self.config = hydra_cfg
        self.train_cfg = hydra_cfg.train_cfg
        # initialize device
        def_device = hydra_cfg.device if hydra_cfg.device != -1 else 0
        self._device = torch.device("cuda:{}".format(def_device))
        self._device_list = None
        self._allow_val_grad = allow_val_grad
        # set of file saving

        if not os.path.exists(self.config.save_path):
            os.makedirs(self.config.save_path)

        assert self.config.exp_name != -1, 'Specify an experiment name for log data!'
        self._best_validation_loss = float('inf')
        self._best_validation_weights = None

        append = "-Batch{}".format(int(self.config.bsize))
        if 'mosaic' in hydra_cfg.policy:
            append = "-Batch{}-{}gpu-Attn{}ly{}-Act{}ly{}mix{}".format(
                int(self.config.bsize), int(torch.cuda.device_count()),
                int(self.config.policy.attn_cfg.n_attn_layers), int(
                    self.config.policy.attn_cfg.attn_ff),
                int(self.config.policy.action_cfg.n_layers), int(
                    self.config.policy.action_cfg.out_dim),
                int(self.config.policy.action_cfg.n_mixtures))

            if self.config.policy.concat_demo_head:
                append += "-headCat"
            elif self.config.policy.concat_demo_act:
                append += "-actCat"
            else:
                append += "-noCat"
            if 'mosaic' in hydra_cfg.policy:
                append += "-simclr{}x{}".format(int(self.config.policy.simclr_config.compressor_dim), int(
                    self.config.policy.simclr_config.hidden_dim))

        self.config.exp_name += append

        save_dir = join(self.config.get('save_path', './'),
                        str(self.config.exp_name))
        save_dir = os.path.expanduser(save_dir)
        self._save_fname = join(save_dir, 'model_save')
        self.save_dir = save_dir
        print(f"Saving dir {self.save_dir}")
        self._step = None
        if self.config.wandb_log:
            config_keys = ['train_cfg', 'tasks',
                           'samplers', 'dataset_cfg', 'policy']
            # for k in config_keys:
            #     print(k, self.config.get(k))
            #     print(k, dict(self.config.get(k)))
            #     print('-'*20)
            wandb_config = {k: self.config.get(k) for k in config_keys}
            run = wandb.init(project=self.config.project_name,
                             name=self.config.exp_name, config=wandb_config, sync_tensorboard=False)

        # create early stopping object
        self._early_stopping = EarlyStopping(patience=self.train_cfg.early_stopping.patience,
                                             verbose=True,
                                             delta=self.train_cfg.early_stopping.delta,
                                             path=self.save_dir
                                             )

    def train(self, model, weights_fn=None, save_fn=None, optim_weights=None, optimizer_state_dict=None, loss_function=None):

        self._train_loader, self._val_loader = make_data_loaders(
            self.config, self.train_cfg.dataset)
        # wrap model in DataParallel if needed and transfer to correct device
        print('\n-------------------\nTraining stage\nFound {} GPU devices \n'.format(self.device_count))
        model = model.to(self._device)
        if self.device_count > 1 and not isinstance(model, nn.DataParallel):
            print("Training stage \n Device list: {}".format(self.device_list))
            model = nn.DataParallel(model, device_ids=self.device_list)

        # save model
        # save the model's state dictionary to a file
        if self.config.wandb_log:
            wandb.watch(model)

        # initialize optimizer and lr scheduler
        optim_weights = optim_weights if optim_weights is not None else model.parameters()
        optimizer, scheduler = self._build_optimizer_and_scheduler(
            self.config.train_cfg.optimizer, optim_weights, optimizer_state_dict, self.train_cfg)

        if self.config.cosine_annealing:
            scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                      first_cycle_steps=10000,
                                                      cycle_mult=1.0,
                                                      max_lr=0.0005,
                                                      min_lr=0.00001,
                                                      warmup_steps=2500,
                                                      gamma=1.0)
        # initialize constants:
        # compute epochs
        if self.config.resume:
            epochs = self.config.epochs - \
                int(self.config.resume_step/len(self._train_loader))
            print(f"\n---- Remaining epochs {epochs} ----\n")
            self._step = int(self.config.resume_step)
            print(f"\n----Starting step {self._step} ----\n")
        else:
            epochs = self.train_cfg.get('epochs', 1)
            self._step = 0

        vlm_alpha = self.train_cfg.get('vlm_alpha', 0.6)
        log_freq = self.train_cfg.get('log_freq', 1000)
        val_freq = self.train_cfg.get('val_freq', 1000)
        print_freq = self.train_cfg.get('print_freq', 10000)
        save_freq = self.train_cfg.get('save_freq', 10000)

        try:
            print("\n----Loss multipliers: \n BC: {} inv: {} Point: {}\n----".format(
                self.train_cfg.bc_loss_mult, self.train_cfg.inv_loss_mult, self.train_cfg.pnt_loss_mult))

            print(
                {name: mul for name, mul in self.train_cfg.rep_loss_muls.items() if mul != 0})
            if self.train_cfg.bc_loss_mult == 0 and self.train_cfg.inv_loss_mult == 0:
                assert sum([v for k, v in self.train_cfg.rep_loss_muls.items()]
                           ) != 0, self.train_cfg.rep_loss_muls
        except:
            pass

        self.tasks = self.config.tasks
        num_tasks = len(self.tasks)
        sum_mul = sum([task.get('loss_mul', 1) for task in self.tasks])
        task_loss_muls = {task.name:
                          float("{:3f}".format(task.get('loss_mul', 1) / sum_mul)) for task in self.tasks}
        print(" Weighting each task loss separately:", task_loss_muls)
        self.generated_png = False
        raw_stats = dict()
        if self._val_loader != None:
            val_iter = iter(self._val_loader)
            print(f"Training for {epochs} epochs train dataloader has length {len(self._train_loader)}, \ which sums to {epochs * len(self._train_loader)} total train steps, \ validation loader has length {len(self._val_loader)}")
        else:
            print(
                f"Training for {epochs} epochs train dataloader has length {len(self._train_loader)}")

        model = model.train()
        model = model.to(self._device)
        # summary(model)

        for e in range(epochs):
            frac = e / epochs
            with tqdm(self._train_loader, unit="batch") as tepoch:
                for inputs in tepoch:
                    tolog = {}
                    # Save stats
                    if save_freq != 0 and self._step % save_freq == 0:  # stats
                        self.save_checkpoint(
                            model, optimizer, weights_fn, save_fn)
                        if save_fn is not None:
                            save_fn(self._save_fname, self._step)
                        else:
                            save_module = model
                            if weights_fn is not None:
                                save_module = weights_fn()
                            elif isinstance(model, nn.DataParallel):
                                save_module = model.module
                            torch.save(save_module.state_dict(),
                                       self._save_fname + '-{}.pt'.format(self._step))
                        if self.config.get('save_optim', False):
                            torch.save(optimizer.state_dict(
                            ), self._save_fname + '-optim-{}.pt'.format(self._step))

                        stats_save_name = join(
                            self.save_dir, 'stats', '{}.json'.format('train_val_stats'))
                        json.dump({k: str(v) for k, v in raw_stats.items()},
                                  open(stats_save_name, 'w'))

                    torch.cuda.empty_cache()

                    optimizer.zero_grad()
                    # self.batch_distribution(inputs)

                    # calculate loss here:
                    task_losses = loss_function(
                        self.config, self.train_cfg, self._device, model, inputs)
                    task_names = sorted(task_losses.keys())
                    weighted_task_loss = sum(
                        [l["loss_sum"] * task_loss_muls.get(name) for name, l in task_losses.items()])
                    weighted_task_loss.backward()
                    optimizer.step()

                    ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
                    # calculate train iter stats
                    if self._step % log_freq == 0:
                        train_print = collect_stats(
                            self._step, task_losses, raw_stats, prefix='train')
                        if self.config.wandb_log:
                            tolog['Train Step'] = self._step
                            for task_name, losses in task_losses.items():
                                for loss_name, loss_val in losses.items():
                                    tolog[f'train/{loss_name}/{task_name}'] = loss_val
                                    tolog[f'train/{task_name}/{loss_name}'] = loss_val

                        if self._step % print_freq == 0:
                            print(
                                'Training epoch {1}/{2}, step {0}: \t '.format(self._step, e, epochs))
                            print(train_print)

                    #### ---- Validation step ----####
                    if self._step % val_freq == 0 and not self.config.get("use_daml", False):
                        rollout = self.config.get("rollout", False)
                        model = model.eval()
                        if not rollout:
                            # exhaust all data in val loader and take avg loss
                            all_val_losses = {task: defaultdict(
                                list) for task in task_names}
                            # val_iter = iter(self._val_loader)
                            for i, val_inputs in tqdm(enumerate(self._val_loader), total=len(self._val_loader), leave=False):
                                use_daml = self.config.get("use_daml", False)
                                if use_daml:  # allow grad!
                                    val_task_losses = loss_function(
                                        self.config, self.train_cfg,  self._device, model, val_inputs)
                                else:
                                    with torch.no_grad():
                                        val_task_losses = loss_function(
                                            self.config, self.train_cfg, self._device, model, val_inputs)

                                for task, losses in val_task_losses.items():
                                    for k, v in losses.items():
                                        all_val_losses[task][k].append(v)

                            # take average across all batches in the val loader
                            avg_losses = dict()
                            for task, losses in all_val_losses.items():
                                avg_losses[task] = {
                                    k: torch.mean(torch.stack(v)) for k, v in losses.items()}

                            if self.config.wandb_log:
                                tolog['Validation Step'] = self._step
                                for task_name, losses in avg_losses.items():
                                    for loss_name, loss_val in losses.items():
                                        tolog[f'val/{loss_name}/{task_name}'] = loss_val
                                        tolog[f'val/{task_name}/{loss_name}'] = loss_val

                            val_print = collect_stats(
                                self._step, avg_losses, raw_stats, prefix='val')
                            if (self._step % len(self._train_loader) == 0):
                                print('Validation step {}:'.format(self._step))
                                print(val_print)

                            # compute the sum of validation losses
                            weighted_task_loss_val = sum(
                                [l["loss_sum"] * task_loss_muls.get(name) for name, l in avg_losses.items()])
                            if self.config.train_cfg.lr_schedule != 'None':
                                # perform lr-scheduling step
                                scheduler.step(val_loss=weighted_task_loss_val)
                                if self.config.wandb_log:
                                    # log learning-rate
                                    tolog['Validation Step'] = self._step
                                    tolog['learning_rate'] = scheduler._schedule.optimizer.param_groups[0]['lr']

                            # check for early stopping
                            if self.train_cfg.early_stopping.patience != -1:
                                self._early_stopping(
                                    weighted_task_loss_val, model, self._step, optimizer)
                        elif rollout and self._step != 0:
                            from multi_task_test.test_any_task import _proc
                            import functools
                            from torch.multiprocessing import Pool
                            target_obj_dec = None
                            controller_path = "/home/frosa_loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json"
                            model_name = self.config.policy._target_
                            for task in self.tasks:
                                import random
                                task_name = task['name']
                                results_dir = os.path.join(
                                    self.save_dir, 'results_{}_{}/'.format(task_name, i))
                                os.makedirs(results_dir, exist_ok=True)
                                best_fp = np.inf
                                best_avg_success = 0
                                f = functools.partial(_proc,
                                                      model,
                                                      target_obj_dec,
                                                      self.config,
                                                      results_dir,
                                                      200,
                                                      360,
                                                      False,
                                                      False,
                                                      False,
                                                      task_name,
                                                      None,
                                                      None,
                                                      70,
                                                      controller_path,
                                                      model_name,
                                                      self.device,
                                                      False)
                                random.seed(42)
                                np.random.seed(42)
                                n_run = task['n_tasks']*2
                                seeds = [(random.getrandbits(32), i)
                                         for i in range(n_run)]
                                with Pool(10) as p:
                                    task_success_flags = p.starmap(f, seeds)
                                if "CondTargetObjectDetector" in self.config.policy._target_:
                                    all_mean_iou = [t['avg_iou']
                                                    for t in task_success_flags]
                                    all_fp = [t['num_false_positive']
                                              for t in task_success_flags]
                                    tolog['avg_iou'] = np.mean(all_mean_iou)
                                    tolog['fp'] = np.mean(all_fp)
                                    if tolog['fp'] <= best_fp:
                                        print(
                                            f"Saving best model, from {best_fp} to {tolog['fp']}")
                                        best_fp = tolog['fp']
                                        self.save_checkpoint(
                                            model, optimizer, weights_fn, save_fn)
                                else:
                                    all_succ_flags = [t['success']
                                                      for t in task_success_flags]
                                    all_reached_flags = [t['reached']
                                                         for t in task_success_flags]
                                    all_picked_flags = [t['picked']
                                                        for t in task_success_flags]
                                    all_avg_pred = [t['avg_pred']
                                                    for t in task_success_flags]
                                    tolog['avg_success'] = np.mean(
                                        all_succ_flags)
                                    tolog['avg_reached'] = np.mean(
                                        all_reached_flags)
                                    tolog['avg_picked'] = np.mean(
                                        all_picked_flags)
                                    tolog['avg_prediction'] = np.mean(
                                        all_avg_pred)

                                    if best_avg_success <= tolog['avg_success']:
                                        print(
                                            f"Save model best_avg_success from {best_avg_success} to {tolog['avg_success']}")
                                        best_avg_success = tolog['avg_success']
                                        self.save_checkpoint(
                                            model, optimizer, weights_fn, save_fn)

                                if self.config.wandb_log:
                                    wandb.log(tolog)

                        model = model.train()
                        if self._early_stopping.early_stop:
                            break

                    if scheduler != 'None' and self.config.cosine_annealing:
                        if self.config.wandb_log:
                            # log learning-rate
                            tolog['Train Step'] = self._step
                            tolog['learning_rate'] = scheduler.optimizer.param_groups[0]['lr']

                    if self.config.wandb_log:
                        wandb.log(tolog)
                    self._step += 1
                    try:
                        if not model._load_target_obj_detector or not model._freeze_target_obj_detector:
                            # update target params
                            mod = model.module if isinstance(
                                model, nn.DataParallel) else model
                            if self.train_cfg.target_update_freq > -1:
                                mod.momentum_update(frac)
                                if self._step % self.train_cfg.target_update_freq == 0:
                                    mod.soft_param_update()
                    except:
                        pass
                    if self._early_stopping.early_stop:
                        print("----Stop training for early-stopping----")
                        break

        # when all epochs are done, save model one last time
        self.save_checkpoint(model, optimizer, weights_fn, save_fn)

    def save_checkpoint(self, model, optimizer, weights_fn=None, save_fn=None):
        if save_fn is not None:
            save_fn(self._save_fname, self._step)
        else:
            save_module = model
            if weights_fn is not None:
                save_module = weights_fn()
            elif isinstance(model, nn.DataParallel):
                save_module = model.module
            torch.save(save_module.state_dict(),
                       self._save_fname + '-{}.pt'.format(self._step))
        if self.config.get('save_optim', False):
            torch.save(optimizer.state_dict(), self._save_fname +
                       '-optim-{}.pt'.format(self._step))
        print(f'Model checkpoint saved at step {self._step}')
        return

    @property
    def device_count(self):
        if self._device_list is None:
            return torch.cuda.device_count()
        return len(self._device_list)

    @property
    def device_list(self):
        if self._device_list is None:
            return [i for i in range(torch.cuda.device_count())]
        return copy.deepcopy(self._device_list)

    @property
    def device(self):
        return copy.deepcopy(self._device)

    def _build_optimizer_and_scheduler(self, optimizer, optim_weights, optimizer_state_dict, cfg):
        assert self.device_list is not None, str(self.device_list)
        if optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                optim_weights,
                cfg.lr,
                weight_decay=cfg.get('weight_decay', 0))
        elif optimizer == 'RMSProp':
            optimizer = torch.optim.RMSprop(
                optim_weights,
                cfg.lr,
                weight_decay=cfg.get('weight_decay', 0))
        elif optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(
                optim_weights,
                cfg.lr,
                weight_decay=cfg.weight_decay)
        if optimizer_state_dict:
            optimizer.load_state_dict(optimizer_state_dict)

        print(
            f"Creating {optimizer}, with lr {optimizer.param_groups[0]['lr']}")

        lr_schedule = dict()
        if cfg.lr_schedule == 'None':
            lr_schedule['type'] = None
        else:
            lr_schedule['type'] = cfg.lr_schedule
        print(f"Lr-scheduler {cfg.lr_schedule}")
        return optimizer, build_scheduler(optimizer, lr_schedule)

    def _loss_to_scalar(self, loss):
        """For more readable logging"""
        x = loss.item()
        return float("{:.3f}".format(x))

    @property
    def step(self):
        if self._step is None:
            raise Exception("Optimization has not begun!")
        return self._step

    @property
    def is_img_log_step(self):
        return self._step % self._img_log_freq == 0


class Workspace(object):
    """ Initializes the policy model and prepare for Trainer.train() """

    def __init__(self, cfg):
        self.trainer = Trainer(allow_val_grad=False, hydra_cfg=cfg)
        print("Finished initializing trainer")
        config = self.trainer.config

        # map between task and number of tasks
        n_tasks = []
        tasks = dict()
        start = 0
        for i, task in enumerate(cfg.tasks):
            n_tasks.append(task['n_tasks'])
            tasks[task['name']] = (start, task['n_tasks'])
            start += task['n_tasks']

        resume = config.get('resume', False)

        # config.policy.n_tasks = n_tasks
        # config.dataset_cfg.tasks = tasks
        # config.dataset_cfg.n_tasks = int(np.sum(n_tasks))
        self.action_model = hydra.utils.instantiate(config.policy)
        try:
            config.use_daml = 'DAMLNetwork' in cfg.policy._target_
            if config.use_daml:
                print("Switching to l2l.algorithms.MAML")
                self.action_model = l2l.algorithms.MAML(
                    self.action_model,
                    lr=config['policy']['maml_lr'],
                    first_order=config['policy']['first_order'],
                    allow_unused=True)
        except:
            print("use_daml not in configuration file")

        print("Model initialized to: {}".format(config.policy._target_))
        if resume:
            self._rpath = join(cfg.save_path, cfg.resume_path,
                               f"model_save-{cfg.resume_step}.pt")
            assert os.path.exists(self._rpath), "Can't seem to find {} anywhere".format(
                config.resume_path)
            print('load model from ...%s' % self._rpath)
            self.action_model.load_state_dict(torch.load(
                self._rpath, map_location=torch.device('cpu')))
            # create path for loading state dict
            optimizer_state_dict = join(
                cfg.save_path, cfg.resume_path, f"model_save-optim-{cfg.resume_step}.pt")
            self.optimizer_state_dict = torch.load(
                optimizer_state_dict, map_location=torch.device('cpu'))
        else:
            self.optimizer_state_dict = None

        self.config = config
        self.train_cfg = config.train_cfg

        # move log path to here!
        print('\n----Done initializing Workspace, saving config.yaml to directory: {}----\n'.format(
            self.trainer.save_dir))

        try:
            os.makedirs(self.trainer.save_dir, exist_ok=(
                'burn' in self.trainer.save_dir))
            os.makedirs(join(self.trainer.save_dir, 'stats'), exist_ok=True)
        except:
            pass

        save_config = copy.deepcopy(self.trainer.config)
        OmegaConf.save(config=save_config, f=join(
            self.trainer.save_dir, 'config.yaml'))

    def run(self):
        loss_function = None
        if "VideoImitation" in self.config.policy._target_ or "InverseImitation" in self.config.policy._target_ or "DAMLNetwork" in self.config.policy._target_ or "CondPolicy" in self.config.policy._target_:
            loss_function = calculate_task_loss
        elif "vima" in self.config.policy._target_:
            loss_function = loss_function_vima
        elif "cond_target_obj_detector" in self.config.policy._target_:
            loss_function = loss_func_bb

        self.trainer.train(model=self.action_model,
                           optimizer_state_dict=self.optimizer_state_dict,
                           loss_function=loss_function)

        print("Done training")
