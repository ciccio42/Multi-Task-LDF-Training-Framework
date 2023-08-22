#!/bin/sh
export MUJOCO_PY_MUJOCO_PATH="/home/frosa_loc/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
# export WANDB_CACHE_DIR=/mnt/sdc1/frosa/wandb
# export TMPDIR=/mnt/sdc1/frosa/tmp

EXPERT_DATA=/home/frosa_loc/Multi-Task-LFD-Framework/ur_multitask_dataset
SAVE_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline/TARGET_OBJ_DETECTOR_SLOT
POLICY='${cond_target_obj_detector}'
DATASET_TARGET=multi_task_il.datasets.multi_task_cond_target_obj_dataset.CondTargetObjDetectorDataset

SAVE_FREQ=0
LOG_FREQ=100
VAL_FREQ=2025
PRINT_FREQ=100

EXP_NAME=1Task-Pick-Place-Cond-Target-Obj-Detector-Only-First-Frame
TASK_str=pick_place
EPOCH=30 # start from 16
BSIZE=64 #32
COMPUTE_OBJ_DISTRIBUTION=false
CONFIG_PATH=../experiments/
PROJECT_NAME="pick_place_cond_target_obj_detector_only_first_frame"
CONFIG_NAME=config_cond_target_obj_detector.yaml
LOADER_WORKERS=32
BALANCING_POLICY=0
SET_SAME_N=4
OBS_T=7

EARLY_STOPPING_PATIECE=10
OPTIMIZER='AdamW'
LR=0.0001
WEIGHT_DECAY=5
SCHEDULER='ReduceLROnPlateau'
FIRST_FRAMES=false
ONLY_FIRST_FRAMES=true
ROLLOUT=false

RESUME_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline/TARGET_OBJ_DETECTOR_SLOT/1Task-Pick-Place-Cond-Target-Obj-Detector-Mixed-Batch64
RESUME_STEP=4050
RESUME=false

python ../training/train_scripts/train_any.py \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME} \
    policy=${POLICY} \
    task_names=${TASK_str} \
    set_same_n=${SET_SAME_N} \
    rollout=${ROLLOUT} \
    exp_name=${EXP_NAME} \
    save_freq=${SAVE_FREQ} \
    log_freq=${LOG_FREQ} \
    val_freq=${VAL_FREQ} \
    print_freq=${PRINT_FREQ} \
    dataset_target=${DATASET_TARGET} \
    bsize=${BSIZE} \
    vsize=${BSIZE} \
    epochs=${EPOCH} \
    dataset_cfg.obs_T=${OBS_T} \
    dataset_cfg.select_random_frames=true \
    dataset_cfg.compute_obj_distribution=${COMPUTE_OBJ_DISTRIBUTION} \
    dataset_cfg.first_frames=${FIRST_FRAMES} \
    dataset_cfg.only_first_frame=${ONLY_FIRST_FRAMES} \
    samplers.balancing_policy=${BALANCING_POLICY} \
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
    resume=${RESUME} \
    loader_workers=${LOADER_WORKERS}
