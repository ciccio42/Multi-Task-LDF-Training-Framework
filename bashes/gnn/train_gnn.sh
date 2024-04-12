#!/bin/sh
# export MUJOCO_PY_MUJOCO_PATH=/user/frosa/.mujoco/mujoco210
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user/frosa/.mujoco/mujoco210/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user/frosa/miniconda3/envs/multi_task_lfd/lib
export MUJOCO_PY_MUJOCO_PATH=/home/frosa_Loc/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CUDA_VISIBLE_DEVICES=2
export HYDRA_FULL_ERROR=1

EXPERT_DATA=/raid/home/frosa_Loc/opt_dataset/geometric_graphs
DEMO_DATA=/raid/home/frosa_Loc/opt_dataset
SAVE_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder
POLICY='${gnn_policy}'

SAVE_FREQ=-1
LOG_FREQ=100
VAL_FREQ=-1
DEVICE=0
DEBUG=true
WAND_LOG=false

EXP_NAME=Prova_Train_GNN
PROJECT_NAME=${EXP_NAME}
TASK_str=pick_place 


ROLLOUT=false

RESUME_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/1Task-BUTTON-MOSAIC-ALL-OBJ-One-Variation-Left-Batch25/
RESUME_STEP=6402
RESUME=false

EPOCH=90
BSIZE=27 #32 #128 #64 #32
# Policy 1: At each slot is assigned a RandomSampler
BALANCING_POLICY=0
SET_SAME_N=3
CONFIG_PATH=../../experiments
CONFIG_NAME=config_gnn.yaml
LOADER_WORKERS=1


EARLY_STOPPING_PATIECE=-1
OPTIMIZER='AdamW'
LR=0.0005
WEIGHT_DECAY=0.0
SCHEDULER=None

HEIGHT=100
WIDTH=180

python ../../training/train_scripts/gnn/train.py \
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
    dataset_cfg.height=${HEIGHT} \
    dataset_cfg.width=${WIDTH} \
    samplers.balancing_policy=${BALANCING_POLICY} \
    project_name=${PROJECT_NAME} \
    EXPERT_DATA=${EXPERT_DATA} \
    DEMO_DATA=${DEMO_DATA} \
    save_path=${SAVE_PATH} \
    resume_path=${RESUME_PATH} \
    resume_step=${RESUME_STEP} \
    optimizer=${OPTIMIZER} \
    train_cfg.lr=${LR} \
    train_cfg.weight_decay=${WEIGHT_DECAY} \
    train_cfg.lr_schedule=${SCHEDULER} \
    debug=${DEBUG} \
    wandb_log=${WAND_LOG} \
    resume=${RESUME} \
    loader_workers=${LOADER_WORKERS}
