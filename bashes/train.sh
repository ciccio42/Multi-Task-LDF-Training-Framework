#!/bin/sh
export MUJOCO_PY_MUJOCO_PATH="/home/frosa_loc/.mujoco/mujoco210"
export CUDA_VISIBLE_DEVICES=2
export HYDRA_FULL_ERROR=1

EXPERT_DATA=/home/frosa_loc/multitask_dataset_ur/multitask_dataset_language_command
SAVE_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline
POLICY='${mosaic}'

EXP_NAME=1Task-Pick-Place-Mosaic-No-Obj-Detector-Ur5e
TASK_str=pick_place
EPOCH=250
BSIZE=128 #64 #32
CONFIG_PATH=../experiments/
PROJECT_NAME="ur_mosaic_baseline_no_obj_detector"
CONFIG_NAME=config.yaml
LOADER_WORKERS=20

LOAD_TARGET_OBJ_DETECTOR=false
TARGET_OBJ_DETECTOR_STEP=17204
TARGET_OBJ_DETECTOR_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/baseline-1/1Task-Pick-Place-Target-Obj-Random-Frames-Batch128-1gpu-Attn2ly128-Act2ly256mix4-actCat
FREEZE_TARGET_OBJ_DETECTOR=false
CONCAT_STATE=false

ACTION_DIM=7
EARLY_STOPPING_PATIECE=20
OPTIMIZER='AdamW'
LR=0.0001
WEIGHT_DECAY=0.05
SCHEDULER=None

RESUME_PATH=None
RESUME_STEP=-1

python ../training/train_scripts/train_any.py \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME} \
    policy=${POLICY} \
    task_names=${TASK_str} \
    exp_name=${EXP_NAME} \
    bsize=${BSIZE} \
    vsize=${BSIZE} \
    epochs=${EPOCH} \
    mosaic.load_target_obj_detector=${LOAD_TARGET_OBJ_DETECTOR} \
    mosaic.target_obj_detector_step=${TARGET_OBJ_DETECTOR_STEP} \
    mosaic.target_obj_detector_path=${TARGET_OBJ_DETECTOR_PATH} \
    mosaic.freeze_target_obj_detector=${FREEZE_TARGET_OBJ_DETECTOR} \
    actions.adim=${ACTION_DIM} \
    mosaic.concat_state=${CONCAT_STATE} \
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
