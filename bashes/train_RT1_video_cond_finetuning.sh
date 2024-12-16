#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0,1,2,3
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1   # Request 1 GPU
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16

# export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

export MUJOCO_PY_MUJOCO_PATH=/home/frosa_Loc/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

export PYTHONPATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/models/rt1/repo
export PYTHONPATH=${PYTHONPATH}:/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/datasets/command_encoder
echo "pythonpath: " $PYTHONPATH

export HYDRA_FULL_ERROR=1
echo $1
# TASK_NAME="$1"
TASK_NAME="pick_place"

SAVE_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder
# SAVE_PATH=/raid/home/frosa_Loc/multi_task_lfd/checkpoint_save_folder
POLICY='${rt1_video_cond}'
TARGET='multi_task_il.models.mt_rep.VideoImitation'
# COND_MODULE_PATH='/user/frosa/multi_task_lfd/checkpoint_save_folder/1Task-pick_place-cond_module_no_lr_1e-4-Batch32/model_save-96.pt'
COND_MODULE_PATH='/user/frosa/multi_task_lfd/checkpoint_save_folder/1Task-pick_place-cond_module_lr_1e-4_good_split-Batch32/model_save-225.pt'

SAVE_FREQ=-1
LOG_FREQ=10
VAL_FREQ=-1
# DEVICE=0    # cuda gpu selection
DEVICE=3   # cuda gpu selection
DEBUG=true
WANDB_LOG=false
ROLLOUT=false
EPOCH=90
LOADER_WORKERS=16
CONFIG_PATH=../experiments
CONFIG_NAME=config_RT1_finetuning.yaml
CONCAT_IMG_EMB=true 
CONCAT_DEMO_EMB=true

LOAD_TARGET_OBJ_DETECTOR=false
CONCAT_BB=false

CHANGE_COMMAND_EPOCH=true
SPLIT_PICK_PLACE=false

LOAD_CONTRASTIVE=true
LOAD_INV=true

CONCAT_STATE=true

RESUME=false
BSIZE=32 #32 #128 #64 #32
# Policy 1: At each slot is assigned a RandomSampler
SET_SAME_N=2

NORMALIZE_ACTION=true
EARLY_STOPPING_PATIECE=-1
OPTIMIZER='AdamW'
# LR=0.0005
LR=0.0005
WEIGHT_DECAY=0.0
SCHEDULER=None

DROP_DIM=4      # 2    # 3
HEIGHT=100
WIDTH=180

TASK_str="pick_place" #[pick_place,nut_assembly,stack_block,button]
EXP_NAME="pick_place_panda_dem_ur5e_agent_cond_module_freezed" #1Task-pick_place-Panda_dem_sim_agent_ur5e_sim_cond_module_h100_w180_good_condmodule #1Task-pick_place-Panda_dem_sim_agent_ur5e_sim_cond_module_h100_w180_good_condmodule #1Task-${TASK_str}-Panda_dem_sim_agent_ur5e_sim_cond_module_h100_w180_condmodule_lr1e-4_step24   #1Task-${TASK_str}-MOSAIC-Rollout
PROJECT_NAME=${EXP_NAME}


# srun --output=training_${EXP_NAME}.txt --job-name=training_${EXP_NAME}
python -u ../training/train_scripts/train_any.py \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME} \
    policy=${POLICY} \
    device=${DEVICE} \
    set_same_n=${SET_SAME_N} \
    task_names=${TASK_str} \
    exp_name=${EXP_NAME} \
    save_freq=${SAVE_FREQ} \
    log_freq=${LOG_FREQ} \
    val_freq=${VAL_FREQ} \
    bsize=${BSIZE} \
    vsize=${BSIZE} \
    epochs=${EPOCH} \
    rollout=${ROLLOUT} \
    debug=${DEBUG} \
    wandb_log=${WANDB_LOG} \
    resume=${RESUME} \
    loader_workers=${LOADER_WORKERS} \
    save_path=${SAVE_PATH} \
    optimizer=${OPTIMIZER} \
    cond_module_path=${COND_MODULE_PATH}
    # width=${WIDTH} \
    # height=${HEIGHT}
