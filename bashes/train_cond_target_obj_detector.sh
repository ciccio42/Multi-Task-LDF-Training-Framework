#!/bin/sh
export MUJOCO_PY_MUJOCO_PATH="/user/frosa/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user/frosa/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
export WANDB_CACHE_DIR=/mnt/sdc1/frosa/wandb
export TMPDIR=/mnt/sdc1/frosa/tmp

EXPERT_DATA=/mnt/sdc1/frosa/ur_baseline_dataset/
SAVE_PATH=/user/frosa/multi_task_lfd/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/TARGET_OBJ_DETECTOR_SLOT
POLICY='${cond_target_obj_detector}'
DATASET_TARGET=multi_task_il.datasets.multi_task_cond_target_obj_dataset.CondTargetObjDetectorDataset

SAVE_FREQ=0
LOG_FREQ=100
VAL_FREQ=4050
PRINT_FREQ=100

EXP_NAME=1Task-Pick-Place-Cond-Target-Obj-Detector
TASK_str=pick_place
EPOCH=20
BSIZE=32 #64 #32
COMPUTE_OBJ_DISTRIBUTION=false
CONFIG_PATH=../experiments/
PROJECT_NAME="pick_place_cond_target_obj_detector"
CONFIG_NAME=config_cond_target_obj_detector.yaml
LOADER_WORKERS=16
BALANCING_POLICY=0
SET_SAME_N=2
OBS_T=7

EARLY_STOPPING_PATIECE=10
OPTIMIZER='AdamW'
LR=0.0001
WEIGHT_DECAY=5
SCHEDULER='ReduceLROnPlateau'

RESUME_PATH=/user/frosa/multi_task_lfd/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/TARGET_OBJ_DETECTOR_SLOT/1Task-Pick-Place-Cond-Target-Obj-Detector-Batch32
RESUME_STEP=32400

python ../training/train_scripts/train_any.py \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME} \
    policy=${POLICY} \
    task_names=${TASK_str} \
    set_same_n=${SET_SAME_N} \
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
    resume=true \
    loader_workers=${LOADER_WORKERS}
