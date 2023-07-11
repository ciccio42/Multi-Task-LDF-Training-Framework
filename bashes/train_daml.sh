#!/bin/sh
export MUJOCO_PY_MUJOCO_PATH="/home/frosa_loc/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1

EXPERT_DATA=/home/frosa_loc/Multi-Task-LFD-Framework/ur_multitask_dataset
SAVE_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline
POLICY='${daml}'

SAVE_FREQ=8100
LOG_FREQ=100
VAL_FREQ=4050
PRINT_FREQ=1

EXP_NAME=1Task-Nut-Assembly-Target-Slot-MAML-224_224
TASK_str=nut_assembly
PROJECT_NAME="ur_${TASK_str}_daml"

EPOCH=40
BSIZE=18 #128 #64 #32
COMPUTE_OBJ_DISTRIBUTION=false
# Policy 1: At each slot is assigned a RandomSampler
BALANCING_POLICY=0
SET_SAME_N=2
CONFIG_PATH=../experiments
CONFIG_NAME=config.yaml
LOADER_WORKERS=16
NORMALIZE_ACTION=true

LOAD_CONTRASTIVE=false
CONTRASTIVE_PRE=1.0
CONTRASTIVE_POS=1.0
MUL_INTM=0

ACTION_DIM=7
N_MIXTURES=3       # Nut-Assembly 3 # Pick-place 6

EARLY_STOPPING_PATIECE=-1
OPTIMIZER='AdamW'
LR=0.0005
WEIGHT_DECAY=0.0
SCHEDULER=ReduceLROnPlateau


RESUME_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline/1Task-Pick-Place-Target-Slot-Mosaic-200-360-Batch32
RESUME_STEP=10000
RESUME=false

COSINE_ANNEALING=false
USE_DAML=true

python ../training/train_scripts/train_any.py \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME} \
    policy=${POLICY} \
    set_same_n=${SET_SAME_N} \
    task_names=${TASK_str} \
    exp_name=${EXP_NAME} \
    save_freq=${SAVE_FREQ} \
    log_freq=${LOG_FREQ} \
    val_freq=${VAL_FREQ} \
    print_freq=${PRINT_FREQ} \
    bsize=${BSIZE} \
    vsize=${BSIZE} \
    epochs=${EPOCH} \
    dataset_cfg.normalize_action=${NORMALIZE_ACTION} \
    dataset_cfg.compute_obj_distribution=${COMPUTE_OBJ_DISTRIBUTION} \
    samplers.balancing_policy=${BALANCING_POLICY} \
    daml.adim=${ACTION_DIM} \
    daml.n_mix=${N_MIXTURES} \
    project_name=${PROJECT_NAME} \
    EXPERT_DATA=${EXPERT_DATA} \
    save_path=${SAVE_PATH} \
    resume_path=${RESUME_PATH} \
    resume_step=${RESUME_STEP} \
    optimizer=${OPTIMIZER} \
    train_cfg.lr=${LR} \
    train_cfg.weight_decay=${WEIGHT_DECAY} \
    train_cfg.lr_schedule=${SCHEDULER} \
    simclr.mul_pre=${CONTRASTIVE_PRE} \
    simclr.mul_pos=${CONTRASTIVE_POS} \
    simclr.mul_intm=${MUL_INTM} \
    debug=false \
    wandb_log=true \
    resume=${RESUME} \
    loader_workers=${LOADER_WORKERS} \
    cosine_annealing=${COSINE_ANNEALING}
