"""
Evaluate each task for the same number of --eval_each_task times.
"""
from colorama import Style
from colorama import Fore
from colorama import init as colorama_init
import re
from omegaconf import OmegaConf
import hydra
from collections import OrderedDict
import wandb
import json
from torch.multiprocessing import Pool
import functools
import pickle as pkl
import numpy as np
import random
import os
from os.path import join
from collections import defaultdict
import torch
from multi_task_test_gnn.utils import seed_everything, build_tvf_formatter, build_env_context, get_eval_fn
from multi_task_test_gnn import TASK_MAP
colorama_init()


def extract_last_number(path):
    # Use regular expression to find the last number in the path
    check_point_number = path.split(
        '/')[-1].split('_')[-1].split('-')[-1].split('.')[0]
    return int(check_point_number)


def rollout_imitation(model, config, ctr,
                      heights=100, widths=200, max_T=150, env_name='place', gpu_id=-1, baseline=None, variation=None, controller_path=None, seed=None, model_name=None, real=True):
    print(f"{Fore.YELLOW} Running rollout for trajectory {ctr}{Style.RESET_ALL}")

    if gpu_id == -1:
        gpu_id = int(ctr % torch.cuda.device_count())
    print(f"Model GPU id {gpu_id}")
    try:
        model = model.cuda(gpu_id)
    except:
        print("Error")

    img_formatter = build_tvf_formatter(config=config,
                                        env_name=env_name)

    T_context = config.train_cfg.dataset.demo_T
    random_frames = config.dataset_cfg.get('select_random_frames', False)

    env, context, variation_id, expert_traj, gt_env = build_env_context(img_formatter,
                                                                        T_context=T_context,
                                                                        ctr=ctr,
                                                                        env_name=env_name,
                                                                        heights=heights,
                                                                        widths=widths,
                                                                        gpu_id=gpu_id,
                                                                        variation=variation, random_frames=random_frames,
                                                                        controller_path=controller_path,
                                                                        ret_gt_env=True,
                                                                        seed=seed)

    build_task = TASK_MAP.get(env_name, None)
    assert build_task, 'Got unsupported task '+env_name
    eval_fn = get_eval_fn(env_name=env_name)
    traj, info = eval_fn(model,
                         env,
                         gt_env,
                         context,
                         gpu_id,
                         variation_id,
                         img_formatter,
                         baseline=baseline,
                         max_T=max_T,
                         model_name=model_name,
                         config=config,
                         task_name=env_name,
                         real=real)
    print("Evaluated traj #{}, task#{}, reached? {} picked? {} success? {} ".format(
        ctr, variation_id, info['reached'], info['picked'], info['success']))
    # print(f"Avg prediction {info['avg_pred']}")
    return traj, info, expert_traj, context


def _proc(model, config, results_dir, heights, widths, env_name, baseline, variation, max_T, controller_path, model_name, gpu_id, save, real, seed, n, gt_file):
    print(f"{Fore.BLUE} Running proc {Style.RESET_ALL}")
    json_name = results_dir + '/traj{}.json'.format(n)
    pkl_name = results_dir + '/traj{}.pkl'.format(n)
    if os.path.exists(json_name) and os.path.exists(pkl_name):
        f = open(json_name)
        task_success_flags = json.load(f)
        print("Using previous results at {}. Loaded eval traj #{}, task#{}, reached? {} picked? {} success? {} ".format(
            json_name, n, task_success_flags['variation_id'], task_success_flags['reached'], task_success_flags['picked'], task_success_flags['success']))
    else:
        if variation is not None:
            variation_id = variation[n % len(variation)]
        else:
            variation_id = variation
            return_rollout = rollout_imitation(model,
                                               config,
                                               n,
                                               heights,
                                               widths,
                                               max_T=max_T,
                                               env_name=env_name,
                                               baseline=baseline,
                                               variation=variation_id,
                                               controller_path=controller_path,
                                               seed=seed,
                                               model_name=model_name,
                                               gpu_id=gpu_id,
                                               real=real)
            rollout, task_success_flags, expert_traj, context = return_rollout
            if save:
                pkl.dump(rollout, open(
                    results_dir+'/traj{}.pkl'.format(n), 'wb'))
                pkl.dump(expert_traj, open(
                    results_dir+'/demo{}.pkl'.format(n), 'wb'))
                pkl.dump(context, open(
                    results_dir+'/context{}.pkl'.format(n), 'wb'))
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
    parser.add_argument('--save_files', action='store_true')

    args = parser.parse_args()
    seed_everything(seed=42)

    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    try_path = args.model
    real = True if "Real" in try_path else False
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
            try_path_list = try_paths[-5:]

    for try_path in try_path_list:

        model_path = os.path.expanduser(try_path)
        print(f"{Fore.GREEN}Testing model {model_path}{Style.RESET_ALL}")
        assert args.env in TASK_MAP.keys(), "Got unsupported environment {}".format(args.env)

        if args.variation and args.save_path is None:
            results_dir = os.path.join(
                os.path.dirname(model_path), 'results_{}_{}/'.format(args.env, args.variation))
        elif not args.variation and args.save_path is None:
            results_dir = os.path.join(os.path.dirname(
                model_path), 'results_{}/'.format(args.env))
        elif args.save_path is not None:
            results_dir = os.path.join(args.save_path)

        print(f"{Fore.GREEN}Result dir {results_dir}{Style.RESET_ALL}")

        assert args.env != '', 'Must specify correct task to evaluate'
        os.makedirs(results_dir, exist_ok=True)
        model_saved_step = model_path.split("/")[-1].split("-")
        model_saved_step.remove("model_save")
        model_saved_step = model_saved_step[0][:-3]
        print("loading model from saved training step %s" %
              model_saved_step)
        results_dir = os.path.join(results_dir, 'step-'+model_saved_step)
        os.makedirs(results_dir, exist_ok=True)
        print("Made new path for results at: %s" % results_dir)
        config_path = os.path.expanduser(args.config) if args.config else os.path.join(
            os.path.dirname(model_path), 'config.yaml')

        config = OmegaConf.load(config_path)
        print('Multi-task dataset, tasks used: ', config.tasks)

        config.gnn_policy_cfg.lcgnet_conf.gpu_id = args.gpu_id
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
                                 build_task.get('num_variations', 0))
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
                    print(
                        f"Testing model on variation {config['tasks_cfgs'][args.env].get('skip_ids', [])}")

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

        loaded = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(loaded)

        model = model.eval()  # .cuda()
        n_success = 0
        variation = args.variation
        seed = args.seed
        max_T = 95

        parallel = args.num_workers > 1

        model_name = config.policy._target_

        print(f"{Fore.GREEN}---- Testing model {model_name} ----{Style.RESET_ALL}")
        f = functools.partial(_proc,
                              model,
                              config,
                              results_dir,
                              heights,
                              widths,
                              args.env,
                              args.baseline,
                              variation,
                              max_T,
                              args.controller_path,
                              model_name,
                              args.gpu_id,
                              args.save_files,
                              real)

        seeds = [(random.getrandbits(32), i, None) for i in range(args.N)]

        if parallel:
            with Pool(args.num_workers) as p:
                task_success_flags = p.starmap(f, seeds)
        else:
            task_success_flags = [f(seeds[i][0], seeds[i][1], seeds[i][2])
                                  for i, n in enumerate(range(args.N))]

        if "cond_target_obj_detector" not in model_name:
            final_results = dict()
            for k in task_success_flags[0].keys():
                n_success = sum([t[k] for t in task_success_flags])
                print('Task {}, rate {}'.format(k, n_success / float(args.N)))
                final_results[k] = n_success / float(args.N)
            variation_ids = defaultdict(list)
            for t in task_success_flags:
                _id = t['variation_id']
                variation_ids[_id].append(t['success'])
            for _id in variation_ids.keys():
                num_eval = len(variation_ids[_id])
                rate = sum(variation_ids[_id]) / num_eval
                final_results['task#'+str(_id)] = rate
                print('Success rate on task#'+str(_id), rate)

            final_results['N'] = int(args.N)
            final_results['model_saved'] = model_saved_step
            json.dump({k: v for k, v in final_results.items()}, open(
                results_dir+'/test_across_{}trajs.json'.format(args.N), 'w'))
        else:
            all_avg_iou = np.mean([t['avg_iou'] for t in task_success_flags])
            all_avg_tp = np.mean([t['avg_tp'] for t in task_success_flags])
            all_avg_fp = np.mean([t['avg_fp'] for t in task_success_flags])
            all_avg_fn = np.mean([t['avg_fn'] for t in task_success_flags])
            print(f"TP {all_avg_tp} - FP {all_avg_fp} - FN {all_avg_fn}")
            final_results = dict()
            final_results['N'] = int(args.N)
            final_results['model_saved'] = model_saved_step
            final_results['avg_iou'] = all_avg_iou
            final_results['avg_tp'] = all_avg_tp
            final_results['avg_fp'] = all_avg_fp
            final_results['avg_fn'] = all_avg_fn

            json.dump({k: v for k, v in final_results.items()}, open(
                results_dir+'/test_across_{}trajs.json'.format(args.N), 'w'))

        if args.wandb_log:

            for i, t in enumerate(task_success_flags):
                log = dict()
                log['episode'] = i
                for k in t.keys():
                    log[k] = float(t[k]) if k != "variation_id" else int(t[k])
                wandb.log(log)

            if "cond_target_obj_detector" not in model_name:
                to_log = dict()
                flags = dict()
                for t in task_success_flags:
                    for k in t.keys():
                        if flags.get(k, None) is None:
                            flags[k] = [int(t[k])]
                        else:
                            flags[k].append(int(t[k]))
                for k in flags.keys():
                    avg_flags = np.mean(flags[k])
                    to_log[f'avg_{k}'] = avg_flags
                wandb.log(to_log)
            else:
                wandb.log({
                    'all_avg_iou': all_avg_iou})
                wandb.log({
                    'all_avg_tp': all_avg_tp})
                wandb.log({
                    'all_avg_fp': all_avg_fp})
                wandb.log({
                    'all_avg_fn': all_avg_fn})
