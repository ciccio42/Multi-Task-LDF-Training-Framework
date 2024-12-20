#!/bin/bash

export MUJOCO_PY_MUJOCO_PATH=/home/frosa_Loc/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

WEIGHTS_PATH='/user/frosa/multi_task_lfd/checkpoint_save_folder/test_2_pretraining_5_datasets_no_droid_4_augmentations_10_epoch-Batch32/model_save-2530.pt'
CUDA_DEVICE=1
DEBUG=True
BLACK_LIST=droid_converted,droid_converted_old


###TODO: test per ogni checkpoint -> fare una sorta di collage

python -u ../training/multi_task_il/datasets/command_encoder/test_cond_module_finetuning.py \
    --weights_path=${WEIGHTS_PATH} \
    --cuda_device=${CUDA_DEVICE} \
    --debug=${DEBUG} \
    --black_list=${BLACK_LIST}
