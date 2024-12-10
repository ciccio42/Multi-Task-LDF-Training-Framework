#!/bin/bash


DATASET_FOLDER='/user/frosa/multi_task_lfd/datasets'

python -u ../training/multi_task_il/datasets/command_encoder/generate_train_val_paths_finetuning.py \
        --dataset_folder=${DATASET_FOLDER} \
        --write_train_pkl_path \
        --write_val_pkl_path \
        --split='0.9,0.1'
