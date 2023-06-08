#!/bin/sh
export MUJOCO_PY_MUJOCO_PATH="/home/frosa_loc/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1

EXPERT_DATA=/home/frosa_loc/Multi-Task-LFD-Framework/ur_multitask_dataset
SAVE_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline
POLICY='${tosil}'

SAVE_FREQ=1000
LOG_FREQ=1000
VAL_FREQ=1000

EXP_NAME=1Task-Pick-Place-Tosil-cropped-no-normalized
TASK_str=pick_place
EPOCH=40
BSIZE=32 #128 #64 #32
COMPUTE_OBJ_DISTRIBUTION=false
# Policy 1: At each slot is assigned a RandomSampler
BALANCING_POLICY=0
SET_SAME_N=2
CONFIG_PATH=../experiments/
PROJECT_NAME="ur_tosil_baseline_cropped-no-normalized"
CONFIG_NAME=config.yaml
LOADER_WORKERS=16
NORMALIZE_ACTION=true

LOAD_TARGET_OBJ_DETECTOR=false
TARGET_OBJ_DETECTOR_STEP=17204
TARGET_OBJ_DETECTOR_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/baseline-1/1Task-Pick-Place-Target-Obj-Random-Frames-Batch128-1gpu-Attn2ly128-Act2ly256mix4-actCat
FREEZE_TARGET_OBJ_DETECTOR=false
CONCAT_STATE=false

ACTION_DIM=7
N_MIXTURES=6

EARLY_STOPPING_PATIECE=-1
OPTIMIZER='Adam'
LR=0.0005
WEIGHT_DECAY=0
SCHEDULER=None

RESUME_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline/1Task-Pick-Place-Mosaic-cropped-no-normalized-Batch32
RESUME_STEP=84100
RESUME=false

python ../training/train_scripts/train_any.py \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME} \
    policy=${POLICY} \
    set_same_n=${SET_SAME_N} \
    task_names=${TASK_str} \
    exp_name=${EXP_NAME} \
    bsize=${BSIZE} \
    vsize=${BSIZE} \
    epochs=${EPOCH} \
    save_freq=${SAVE_FREQ} \
    log_freq=${LOG_FREQ} \
    val_freq=${VAL_FREQ} \
    dataset_cfg.normalize_action=${NORMALIZE_ACTION} \
    dataset_cfg.compute_obj_distribution=${COMPUTE_OBJ_DISTRIBUTION} \
    samplers.balancing_policy=${BALANCING_POLICY} \
    tosil.load_target_obj_detector=${LOAD_TARGET_OBJ_DETECTOR} \
    tosil.target_obj_detector_step=${TARGET_OBJ_DETECTOR_STEP} \
    tosil.target_obj_detector_path=${TARGET_OBJ_DETECTOR_PATH} \
    tosil.freeze_target_obj_detector=${FREEZE_TARGET_OBJ_DETECTOR} \
    tosil.adim=${ACTION_DIM} \
    tosil.concat_state=${CONCAT_STATE} \
    early_stopping_cfg.patience=${EARLY_STOPPING_PATIECE} \
    project_name=${PROJECT_NAME} \
    EXPERT_DATA=${EXPERT_DATA} \
    save_path=${SAVE_PATH} \
    resume_path=${RESUME_PATH} \
    resume_step=${RESUME_STEP} \
    optimizer=${OPTIMIZER} \
    train_cfg.lr=${LR} \
    train_cfg.weight_decay=${WEIGHT_DECAY} \
    train_cfg.lr_schedule=${SCHEDULER} \
    debug=false \
    wandb_log=true \
    resume=false \
    loader_workers=${LOADER_WORKERS}
