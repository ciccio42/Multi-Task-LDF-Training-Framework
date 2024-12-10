#!/bin/bash


DATASET_FOLDER='/user/frosa/multi_task_lfd/datasets'

python -u ../training/multi_task_il/datasets/command_encoder/generate_train_val_paths_finetuning.py \
        --dataset_folder=${DATASET_FOLDER}
