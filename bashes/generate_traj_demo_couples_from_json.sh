#!/bin/bash


python -u ../training/multi_task_il/datasets/command_encoder/generate_traj_demo_couples_from_json.py \
        --trajectory_json_file_path='val_pkl_paths.json'

python -u ../training/multi_task_il/datasets/command_encoder/generate_traj_demo_couples_from_json.py \
        --trajectory_json_file_path='train_pkl_paths.json'

python -u ../training/multi_task_il/datasets/command_encoder/generate_traj_demo_couples_from_json.py \
        --trajectory_json_file_path='all_pkl_paths.json'