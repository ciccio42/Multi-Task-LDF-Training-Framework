#!/bin/sh
export MUJOCO_PY_MUJOCO_PATH="/home/frosa_loc/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
# export WANDB_CACHE_DIR=/user/frosa
# export TMPDIR=/user/frosa/tmp

ROLLOUT=false
EXPERT_DATA=/home/frosa_loc/Multi-Task-LFD-Framework/ur_multitask_dataset
SAVE_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline/TARGET_OBJ_DETECTOR_SLOT
POLICY='${cond_policy}'
DATASET_TARGET=multi_task_il.datasets.multi_task_cond_target_obj_dataset.CondTargetObjDetectorDataset

SAVE_FREQ=2025
LOG_FREQ=100
VAL_FREQ=2025
PRINT_FREQ=100

EXP_NAME=1Task-Pick-Place-Cond-Target-Obj-Detector-Policy-GT-BB-Low-Variance
PROJECT_NAME="pick_place_cond_target_obj_detector_policy_gt_bb"

TASK_str=pick_place
EPOCH=40
BSIZE=64 #32
COMPUTE_OBJ_DISTRIBUTION=false
LOAD_ACTION=true
LOAD_STATE=true
BC_MUL=1.0

CONFIG_PATH=../experiments/
CONFIG_NAME=config_cond_target_obj_detector.yaml
LOADER_WORKERS=16
BALANCING_POLICY=0
SET_SAME_N=4
OBS_T=7
AUG_TWICE=false

EARLY_STOPPING_PATIECE=-1
OPTIMIZER='AdamW'
LR=0.000001
WEIGHT_DECAY=0
SCHEDULER=None

N_MIXTURES=6

RESUME_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline/TARGET_OBJ_DETECTOR_SLOT/1Task-Pick-Place-Cond-Target-Obj-Detector-Policy-Batch32
RESUME_STEP=238950
RESUME=false

COND_TARGET_OBJ_DETECTOR_PRE_TRAINED=true
COND_TARGET_OBJ_DETECTOR_WEIGHTS="/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline/TARGET_OBJ_DETECTOR_SLOT/1Task-Pick-Place-Cond-Target-Obj-Detector-First-Frame-Batch64"
COND_TARGET_OBJ_DETECTOR_STEP=32400
SPATIAL_SOFTMAX=false
GT_BB=true


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
    rollout=${ROLLOUT} \
    bsize=${BSIZE} \
    vsize=${BSIZE} \
    epochs=${EPOCH} \
    bc_mul=${BC_MUL} \
    actions.n_mixtures=${N_MIXTURES} \
    cond_policy.cond_target_obj_detector_pretrained=${COND_TARGET_OBJ_DETECTOR_PRE_TRAINED} \
    cond_policy.cond_target_obj_detector_weights=${COND_TARGET_OBJ_DETECTOR_WEIGHTS} \
    cond_policy.cond_target_obj_detector_step=${COND_TARGET_OBJ_DETECTOR_STEP} \
    cond_policy.spatial_softmax=${SPATIAL_SOFTMAX} \
    cond_policy.gt_bb=${GT_BB} \
    dataset_cfg.obs_T=${OBS_T} \
    dataset_cfg.select_random_frames=true \
    dataset_cfg.compute_obj_distribution=${COMPUTE_OBJ_DISTRIBUTION} \
    dataset_cfg.load_action=${LOAD_ACTION} \
    dataset_cfg.load_state=${LOAD_STATE} \
    dataset_cfg.aug_twice=${AUG_TWICE} \
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
    wandb_log=false \
    resume=${RESUME} \
    loader_workers=${LOADER_WORKERS}
