#!/bin/bash
SAVE_GIF_NAME='prova_from_bash'
DATASET='real_ur5e'

if [ "$DATASET" == 'real_ur5e' ]; then 
    python -u ../training/multi_task_il/datasets/command_encoder/visualize_action_single_dataset.py \
    --dataset_index 5 \
    --save_gif_name $SAVE_GIF_NAME
fi