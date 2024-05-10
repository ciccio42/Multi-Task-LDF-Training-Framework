"""
Evaluate each task for the same number of --eval_each_task times. 
"""
import warnings
import cv2
import random
import os
from os.path import join
from collections import defaultdict
import torch
from multi_task_il.datasets import Trajectory
import numpy as np
import pickle as pkl
import functools
from torch.multiprocessing import Pool, set_start_method
import json
import wandb
from collections import OrderedDict
import hydra
from omegaconf import OmegaConf
import learn2learn as l2l
import re
from hydra.utils import instantiate
from multi_task_test import TASK_MAP
from multi_task_test.utils import *
import pickle


set_start_method('forkserver', force=True)
LOG_PATH = None
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def extract_last_number(path):
    # Use regular expression to find the last number in the path
    check_point_number = path.split(
        '/')[-1].split('_')[-1].split('-')[-1].split('.')[0]
    return int(check_point_number)


def object_detection_inference(model, config, ctr, heights=100, widths=200, size=0, shape=0, color=0, max_T=150, env_name='place', gpu_id=-1, baseline=None, variation=None, controller_path=None, seed=None, action_ranges=[], model_name=None, traj_file=None, context_file=None, gt_bb=False, build_context_flag=False):

    if gpu_id == -1:
        gpu_id = int(ctr % torch.cuda.device_count())
    print(f"Model GPU id {gpu_id}")
    model = model.cuda(gpu_id)

    T_context = config.train_cfg.dataset.get('T_context', None)
    random_frames = config.dataset_cfg.get('select_random_frames', False)
    if not T_context:
        assert 'multi' in config.train_cfg.dataset._target_, config.train_cfg.dataset._target_
        T_context = config.train_cfg.dataset.demo_T

    # Build pre-processing module
    img_formatter = build_tvf_formatter_obj_detector(config, env_name)
    traj_data_trj = None

    # Build environments
    print(f"Create expert demonstration")
    if build_context_flag:
        context, variation_id, expert_traj = build_context(img_formatter,
                                                           T_context=T_context,
                                                           ctr=ctr,
                                                           env_name=env_name,
                                                           heights=heights,
                                                           widths=widths,
                                                           size=size,
                                                           shape=shape,
                                                           color=color,
                                                           gpu_id=gpu_id,
                                                           variation=variation, random_frames=random_frames,
                                                           controller_path=controller_path,
                                                           seed=seed)
    else:
        env = None
        variation_id = None
        expert_traj = None
        # open context pk file
        print(
            f"Considering task {variation} - context {context_file.split('/')[-1]} - agent {traj_file.split('/')[-1]}")
        import pickle
        with open(context_file, "rb") as f:
            context_data = pickle.load(f)

        # create context data
        context_data_trj = context_data['traj']
        assert isinstance(context_data_trj, Trajectory)
        context = select_random_frames(
            context_data_trj, T_context, sample_sides=True, random_frames=random_frames)
        # convert BGR context image to RGB and scale to 0-1
        for i, img in enumerate(context):
            cv2.imwrite(f"context_{i}.png", np.array(img[:, :, ::-1]))
        context = [img_formatter(i[:, :, ::-1])[None] for i in context]
        # assert len(context ) == 6
        if isinstance(context[0], np.ndarray):
            context = torch.from_numpy(np.concatenate(context, 0))[None]
        else:
            context = torch.cat(context, dim=0)[None]

    print(
        f"Considering task {variation} - agent {traj_file.split('/')[-1]}")
    with open(traj_file, "rb") as f:
        traj_data = pickle.load(f)

    traj_data_trj = traj_data['traj']
    # create context data

    build_task = TASK_MAP.get(env_name, None)
    assert build_task, 'Got unsupported task '+env_name
    eval_fn = get_eval_fn(env_name=env_name)
    # config.dataset_cfg['perform_augs'] = True if "real" not in config.dataset_cfg['agent_name'] else False
    traj, info = eval_fn(model=model,
                         env=None,
                         gt_env=None,
                         context=context,
                         gpu_id=gpu_id,
                         variation_id=variation_id,
                         img_formatter=img_formatter,
                         baseline=baseline,
                         max_T=max_T,
                         action_ranges=action_ranges,
                         model_name=model_name,
                         task_name=env_name,
                         config=config,
                         gt_file=traj_data_trj,
                         gt_bb=gt_bb)

    if "cond_target_obj_detector" in model_name:
        print("Evaluated traj #{}, task #{}, TP {}, FP {}, FN {}".format(
            ctr, variation_id, info['avg_tp'], info['avg_fp'], info['avg_fn']))
    else:
        print("Evaluated traj #{}, task#{}, reached? {} picked? {} success? {} ".format(
            ctr, variation_id, info['reached'], info['picked'], info['success']))
        (f"Avg prediction {info['avg_pred']}")
    return traj, context, info


def rollout_imitation(model, config, ctr,
                      heights=100, widths=200, size=0, shape=0, color=0, max_T=150, env_name='place', gpu_id=-1, baseline=None, variation=None, controller_path=None, seed=None, action_ranges=[], model_name=None, traj_file=None, context_file=None, gt_bb=False, build_context_flag=False):
    if gpu_id == -1:
        gpu_id = int(ctr % torch.cuda.device_count())
    print(f"Model GPU id {gpu_id}")
    try:
        model = model.cuda(gpu_id)
    except:
        print("Error")

    if "CondPolicy" not in model_name and config.augs.get("old_aug", True):
        img_formatter = build_tvf_formatter(config, env_name)
    else:
        img_formatter = build_tvf_formatter_obj_detector(config=config,
                                                         env_name=env_name)

    T_context = config.train_cfg.dataset.get('demo_T', None)
    random_frames = config.dataset_cfg.get('select_random_frames', False)

    # Build environments
    print(f"Create expert demonstration")
    if build_context_flag:
        context, variation_id, expert_traj = build_context(img_formatter,
                                                           T_context=T_context,
                                                           ctr=ctr,
                                                           env_name=env_name,
                                                           heights=heights,
                                                           widths=widths,
                                                           size=size,
                                                           shape=shape,
                                                           color=color,
                                                           gpu_id=gpu_id,
                                                           variation=variation, random_frames=random_frames,
                                                           controller_path=controller_path,
                                                           seed=seed)
    else:
        env = None
        variation_id = None
        expert_traj = None
        # open context pk file
        print(
            f"Considering task {variation} - context {context_file.split('/')[-1]} - agent {traj_file.split('/')[-1]}")
        import pickle
        with open(context_file, "rb") as f:
            context_data = pickle.load(f)

        # create context data
        context_data_trj = context_data['traj']
        assert isinstance(context_data_trj, Trajectory)
        context = select_random_frames(
            context_data_trj, T_context, sample_sides=True, random_frames=random_frames)
        # convert BGR context image to RGB and scale to 0-1
        for i, img in enumerate(context):
            cv2.imwrite(f"context_{i}.png", np.array(img[:, :, ::-1]))
        context = [img_formatter(i[:, :, ::-1])[None] for i in context]
        # assert len(context ) == 6
        if isinstance(context[0], np.ndarray):
            context = torch.from_numpy(np.concatenate(context, 0))[None]
        else:
            context = torch.cat(context, dim=0)[None]

    print(
        f"Considering task {variation} - agent {traj_file.split('/')[-1]}")
    with open(traj_file, "rb") as f:
        traj_data = pickle.load(f)

    traj_data_trj = traj_data['traj']

    build_task = TASK_MAP.get(env_name, None)
    assert build_task, 'Got unsupported task '+env_name
    eval_fn = get_eval_fn(env_name=env_name)
    # if "real" not in config.dataset_cfg['agent_name'] else False
    config.dataset_cfg['perform_augs'] = True
    traj, info = eval_fn(model=model,
                         env=None,
                         gt_env=None,
                         context=context,
                         gpu_id=gpu_id,
                         variation_id=variation_id,
                         img_formatter=img_formatter,
                         baseline=baseline,
                         max_T=max_T,
                         action_ranges=action_ranges,
                         model_name=model_name,
                         task_name=env_name,
                         config=config,
                         gt_file=traj_data_trj,
                         gt_bb=gt_bb)

    for k in info.keys():
        print(f"Key: {k}, Value: {info[k]}\n")
    # if "cond_target_obj_detector" not in model_name:
    #     print("Evaluated traj #{}, task#{}, reached? {} picked? {} success? {} ".format(
    #         ctr, variation_id, info['reached'], info['picked'], info['success']))
    #     print(f"Avg prediction {info['avg_pred']}")
    # else:
    #     print()
    return traj, context, info


def _proc(model, config, results_dir, heights, widths, size, shape, color, env_name, baseline, variation, max_T, controller_path, model_name, gpu_id, save, gt_bb, seed, n, traj_file, context_file):
    json_name = results_dir + '/traj{}.json'.format(n)
    pkl_name = results_dir + '/traj{}.pkl'.format(n)
    if os.path.exists(json_name) and os.path.exists(pkl_name):
        f = open(json_name)
        task_success_flags = json.load(f)
        print(f"Using previous results")
        # print("Using previous results at {}. Loaded eval traj #{}, task#{}, reached? {} picked? {} success? {} ".format(
        #     json_name, n, task_success_flags['variation_id'], task_success_flags['reached'], task_success_flags['picked'], task_success_flags['success']))
    else:
        if variation is not None:
            variation_id = variation[n % len(variation)]
        else:
            variation_id = variation
        if ("cond_target_obj_detector" not in model_name) or ("CondPolicy" in model_name):
            pass
            return_rollout = rollout_imitation(model,
                                               config,
                                               n,
                                               heights,
                                               widths,
                                               size,
                                               shape,
                                               color,
                                               max_T=max_T,
                                               env_name=env_name,
                                               baseline=baseline,
                                               variation=variation_id,
                                               controller_path=controller_path,
                                               seed=seed,
                                               action_ranges=np.array(
                                                   config.dataset_cfg.get('normalization_ranges', [])),
                                               model_name=model_name,
                                               gpu_id=gpu_id,
                                               traj_file=traj_file,
                                               context_file=context_file,
                                               build_context_flag=False)
        else:
            # Perform object detection inference
            return_rollout = object_detection_inference(model=model,
                                                        config=config,
                                                        ctr=n,
                                                        heights=heights,
                                                        widths=widths,
                                                        size=size,
                                                        shape=shape,
                                                        color=color,
                                                        max_T=max_T,
                                                        env_name=env_name,
                                                        baseline=baseline,
                                                        variation=variation_id,
                                                        controller_path=controller_path,
                                                        seed=seed,
                                                        action_ranges=np.array(
                                                            config.dataset_cfg.get('normalization_ranges', [])),
                                                        model_name=model_name,
                                                        gpu_id=gpu_id,
                                                        traj_file=traj_file,
                                                        context_file=context_file,
                                                        build_context_flag=False)

        if "vima" not in model_name:
            rollout, context, task_success_flags = return_rollout
            if save:
                pkl.dump(rollout, open(
                    results_dir+'/traj{}.pkl'.format(n), 'wb'))
                # pkl.dump(expert_traj, open(
                #     results_dir+'/demo{}.pkl'.format(n), 'wb'))
                pkl.dump(context, open(
                    results_dir+'/context{}.pkl'.format(n), 'wb'))
                res_dict = dict()
                for k, v in task_success_flags.items():
                    if not isinstance(v, np.ndarray):
                        if v == True or v == False:
                            res_dict[k] = int(v)
                        else:
                            res_dict[k] = v
                    else:
                        res_dict[k] = v
                json.dump(res_dict, open(
                    results_dir+'/traj{}.json'.format(n), 'w'))
        else:
            rollout, task_success_flags = return_rollout
            if save:
                pkl.dump(rollout, open(
                    results_dir+'/traj{}.pkl'.format(n), 'wb'))
                res_dict = dict()
                for k, v in task_success_flags.items():
                    if v == True or v == False:
                        res_dict[k] = int(v)
                    else:
                        res_dict[k] = v
                json.dump(res_dict, open(
                    results_dir+'/traj{}.json'.format(n), 'w'))
    del model
    return task_success_flags


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--wandb_log', action='store_true')
    parser.add_argument('--project_name', '-pn', default="mosaic", type=str)
    parser.add_argument('--config', default='')
    parser.add_argument('--N', default=-1, type=int)
    parser.add_argument('--use_h', default=-1, type=int)
    parser.add_argument('--use_w', default=-1, type=int)
    parser.add_argument('--num_workers', default=3, type=int)
    # for block stacking only!
    parser.add_argument('--size', action='store_true')
    parser.add_argument('--shape', action='store_true')
    parser.add_argument('--color', action='store_true')
    parser.add_argument('--env', '-e', default='door', type=str)
    parser.add_argument('--eval_each_task',  default=30, type=int)
    parser.add_argument('--eval_subsets',  default=0, type=int)
    parser.add_argument('--saved_step', '-s', default=1000, type=int)
    parser.add_argument('--baseline', '-bline', default=None, type=str,
                        help='baseline uses more frames at each test-time step')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--results_name', default=None, type=str)
    parser.add_argument('--variation', default=None, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--controller_path', default=None, type=str)
    parser.add_argument('--gpu_id', default=-1, type=int)
    parser.add_argument('--save_path', default=None, type=str)
    parser.add_argument('--test_gt', action='store_true')
    parser.add_argument('--save_files', action='store_true')
    parser.add_argument('--gt_bb', action='store_true')

    args = parser.parse_args()

    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    random.seed(42)
    np.random.seed(42)
    # torch.manual_seed(42)
    # os.environ["PYTHONHASHSEED"] = "42"
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.set_num_threads(1)

    try_path = args.model
    # if 'log' not in args.model and 'mosaic' not in args.model:
    #     print("Appending dir to given exp_name: ", args.model)
    #     try_path = join(LOG_PATH, args.model)
    #     assert os.path.exists(try_path), f"Cannot find {try_path} anywhere"
    try_path_list = []
    if 'model_save' not in args.model:
        print(args.saved_step)
        if args.saved_step != -1:
            print("Appending saved step {}".format(args.saved_step))
            try_path = join(
                try_path, 'model_save-{}.pt'.format(args.saved_step))
            try_path_list.append(try_path)
            assert os.path.exists(
                try_path), "Cannot find anywhere: " + str(try_path)
        else:
            import glob
            print(f"Finding checkpoints in {try_path}")
            check_point_list = glob.glob(
                os.path.join(try_path, "model_save-*.pt"))
            exclude_pattern = r'model_save-optim.pt'
            check_point_list = [
                path for path in check_point_list if not re.search(exclude_pattern, path)]
            check_point_list = sorted(
                check_point_list, key=extract_last_number)
            # take the last check point
            try_paths = check_point_list
            epoch_numbers = len(try_paths)
            try_path_list = try_paths[-1:]

    for try_path in try_path_list:

        model_path = os.path.expanduser(try_path)
        print(f"Testing model {model_path}")
        assert args.env in TASK_MAP.keys(), "Got unsupported environment {}".format(args.env)

        if args.variation and args.save_path is None:
            results_dir = os.path.join(
                os.path.dirname(model_path), 'results_{}_{}/'.format(args.env, args.variation))
        elif not args.variation and args.save_path is None:
            results_dir = os.path.join(os.path.dirname(
                model_path), 'results_{}/'.format(args.env))
        elif args.save_path is not None:
            results_dir = os.path.join(args.save_path)

        print(f"Result dir {results_dir}")

        assert args.env != '', 'Must specify correct task to evaluate'
        os.makedirs(results_dir, exist_ok=True)
        model_saved_step = model_path.split("/")[-1].split("-")
        model_saved_step.remove("model_save")
        model_saved_step = model_saved_step[0][:-3]
        print("loading model from saved training step %s" % model_saved_step)
        results_dir = os.path.join(results_dir, 'step-'+model_saved_step)
        os.makedirs(results_dir, exist_ok=True)
        print("Made new path for results at: %s" % results_dir)
        config_path = os.path.expanduser(args.config) if args.config else os.path.join(
            os.path.dirname(model_path), 'config.yaml')

        config = OmegaConf.load(config_path)
        print('Multi-task dataset, tasks used: ', config.tasks)

        model = hydra.utils.instantiate(config.policy)

        if args.wandb_log:
            model_name = model_path.split("/")[-2]
            run = wandb.init(project=args.project_name,
                             job_type='test',
                             group=model_name.split("-1gpu")[0],
                             reinit=True)
            run.name = model_name + f'-Test_{args.env}-Step_{model_saved_step}'
            wandb.config.update(args)

        # assert torch.cuda.device_count() <= 5, "Let's restrict visible GPUs to not hurt other processes. E.g. export CUDA_VISIBLE_DEVICES=0,1"
        build_task = TASK_MAP.get(args.env, None)
        assert build_task, 'Got unsupported task '+args.env

        if args.N == -1:
            if not args.variation:
                if len(config["tasks_cfgs"][args.env].get('skip_ids', [])) == 0:
                    args.N = int(args.eval_each_task *
                                 config["tasks_cfgs"][args.env]["n_tasks"])
                    if args.eval_subsets:
                        print("evaluating only first {} subtasks".format(
                            args.eval_subsets))
                        args.N = int(args.eval_each_task * args.eval_subsets)
                        args.variation = [i for i in range(args.eval_subsets)]
                else:
                    args.N = int(args.eval_each_task *
                                 len(config["tasks_cfgs"][args.env].get('skip_ids', [])))
                    args.variation = config["tasks_cfgs"][args.env].get(
                        'skip_ids', [])

            else:
                args.N = int(args.eval_each_task*len(args.variation))

        assert args.N, "Need pre-define how many trajs to test for each env"
        print('Found {} GPU devices, using {} parallel workers for evaluating {} total trajectories\n'.format(
            torch.cuda.device_count(), args.num_workers, args.N))

        T_context = config.dataset_cfg.demo_T  # a different set of config scheme
        # heights, widths = config.train_cfg.dataset.get('height', 100), config.train_cfg.dataset.get('width', 200)
        heights, widths = build_task.get('render_hw', (100, 180))
        if args.use_h != -1 and args.use_w != -1:
            print(
                f"Reset to non-default render sizes {args.use_h}-by-{args.use_w}")
            heights, widths = args.use_h, args.use_w

        print("Renderer is using size {} \n".format((heights, widths)))

        model._2_point = None
        model._target_embed = None

        loaded = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(loaded)

        model = model.eval()  # .cuda()
        n_success = 0
        size = args.size
        shape = args.shape
        color = args.color
        variation = args.variation
        seed = args.seed
        max_T = 95

        parallel = args.num_workers > 1

        model_name = config.policy._target_

        dataset = None
        from hydra.utils import instantiate
        # config.tasks[0] = {
        #     "name": "pick_place",
        #     "n_tasks": 4,
        #     "crop": [0,
        #              30,
        #              50,
        #              0],
        #     "n_per_task": 2,
        #     "task_ids":
        #     [0,
        #      1,
        #      2,
        #      3],
        #     "loss_mul": 1,
        #     "task_per_batch": 16,
        #     "traj_per_subtask": 36,
        #     "demo_per_subtask": 100}

        config.dataset_cfg.mode = "val"
        config.dataset_cfg.agent_name = "real_ur5e"
        config.dataset_cfg.change_command_epoch = False
        config.dataset_cfg.root_dir = "/raid/home/frosa_Loc/opt_dataset"
        dataset = instantiate(config.get('dataset_cfg', None))
        dataset._mix_demo_agent = False
        # get list of pkl files
        build_context_flag = False
        if build_context_flag:
            pkl_file_dict = dataset.agent_files
            pkl_file_list = []
            for task_name in pkl_file_dict.keys():
                for task_id in pkl_file_dict[task_name].keys():
                    for pkl_file in pkl_file_dict[task_name][task_id]:
                        pkl_file_list.append(pkl_file)
        else:
            file_pairs = dataset.all_file_pairs
            pkl_file_list = []
            for pkl_file in file_pairs.values():
                pkl_file_list.append((pkl_file[3], pkl_file[2]))
            # for pkl_file in file_pairs.values():
            #     if 'traj010.pkl' in pkl_file[3]:
            #         pkl_file_list.append((pkl_file[3], '/raid/home/frosa_Loc/opt_dataset/pick_place/panda_pick_place/task_00/traj000.pkl'
            #                               ))
            args.N = len(pkl_file_list)

        print(f"---- Testing model {model_name} ----")
        f = functools.partial(_proc,
                              model,
                              config,
                              results_dir,
                              heights,
                              widths,
                              size,
                              shape,
                              color,
                              args.env,
                              args.baseline,
                              variation,
                              max_T,
                              args.controller_path,
                              model_name,
                              args.gpu_id,
                              args.save_files,
                              args.gt_bb)

        seeds = []
        for i in range(args.N):
            if build_context_flag:
                seeds.append((random.getrandbits(32), i,
                              pkl_file_list[i % len(pkl_file_list)], -1))
            else:
                seeds.append((random.getrandbits(32), i,
                              pkl_file_list[i % len(pkl_file_list)][0],
                              pkl_file_list[i % len(pkl_file_list)][1]))

        if parallel:
            with Pool(args.num_workers) as p:
                task_success_flags = p.starmap(f, seeds)
        else:
            task_success_flags = [f(seeds[i][0], seeds[i][1], seeds[i][2], seeds[i][3])
                                  for i, _ in enumerate(seeds)]
        if args.wandb_log:

            for i, t in enumerate(task_success_flags):
                log = dict()
                log['episode'] = i
                for k in t.keys():
                    if "error" not in k:
                        log[k] = float(
                            t[k]) if k != "variation_id" else int(t[k])
                    else:
                        log[f"{k}_x"] = float(
                            t[k][0]) if k != "variation_id" else int(t[k])
                        log[f"{k}_y"] = float(
                            t[k][1]) if k != "variation_id" else int(t[k])
                        log[f"{k}_z"] = float(
                            t[k][2]) if k != "variation_id" else int(t[k])
                wandb.log(log)

            to_log = dict()
            flags = dict()
            for t in task_success_flags:
                for k in t.keys():
                    if flags.get(k, None) is None:
                        flags[k] = [t[k]]
                    else:
                        flags[k].append(t[k])
            for k in flags.keys():
                avg_flags = np.mean(flags[k])
                to_log[f'avg_{k}'] = avg_flags

            json.dump({k: v for k, v in to_log.items()}, open(
                results_dir+'/test_across_{}trajs.json'.format(args.N), 'w'))
            wandb.log(to_log)
