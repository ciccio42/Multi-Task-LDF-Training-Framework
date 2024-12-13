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

# export PYTHONPATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/models/rt1/repo
# echo "pythonpath: " $PYTHONPATH

export HYDRA_FULL_ERROR=1
echo $1
# TASK_NAME="$1"
TASK_NAME="pick_place"

# EXPERT_DATA=/raid/home/frosa_Loc/opt_dataset/ 
SAVE_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder
# SAVE_PATH=/raid/home/frosa_Loc/multi_task_lfd/checkpoint_save_folder
POLICY='${cond_module}'

SAVE_FREQ=-1
LOG_FREQ=10
VAL_FREQ=-1
DEVICE=3 # cuda gpu selection
DEBUG=false
WANDB_LOG=true
ROLLOUT=false
EPOCH=20
LOADER_WORKERS=16
CONFIG_PATH=../experiments
CONFIG_NAME=config_cond_module_finetuning.yaml

RESUME=false

BSIZE=32 #32 #128 #64 #32
# Policy 1: At each slot is assigned a RandomSampler
# SET_SAME_N=2

OPTIMIZER='AdamW'
LR=0.0005 # not used

HEIGHT=100 # not used
WIDTH=180 # not used

TASK_str="pick_place" #[pick_place,nut_assembly,stack_block,button]
    # EXP_NAME=1Task-${TASK_str}-cond_module_no_lr_1e-4   #1Task-${TASK_str}-Panda_dem_sim_agent_ur5e_sim_2      #1Task-${TASK_str}-MOSAIC-Rollout
EXP_NAME=prima_vera_prova_cond_module_lr_5e-4_finetuning_batch_size-96   #1Task-${TASK_str}-Panda_dem_sim_agent_ur5e_sim_2      #1Task-${TASK_str}-MOSAIC-Rollout

# srun --output=training_${EXP_NAME}.txt --job-name=training_${EXP_NAME}
python -u ../training/train_scripts/train_any.py \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME} \
    policy=${POLICY} \
    device=${DEVICE} \
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
    
    
