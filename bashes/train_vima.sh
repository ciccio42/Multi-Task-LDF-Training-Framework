#!/bin/sh
export MUJOCO_PY_MUJOCO_PATH="/home/frosa_loc/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1

EXPERT_DATA=/home/frosa_loc/Multi-Task-LFD-Framework/ur_multitask_dataset
SAVE_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline
POLICY='${vima}'

EXP_NAME=1Task-Pick-Place-VIMA
TASK_str=pick_place
EPOCH=30
BSIZE=1 #32 #64 #32
COMPUTE_OBJ_DISTRIBUTION=false
CONFIG_PATH=../experiments/
PROJECT_NAME="ur_vima"
CONFIG_NAME=config_vima.yaml
LOADER_WORKERS=1
SET_SAME_N=1
NORMALIZE_ACTION=true
NORMALIZE_ACTION=true
OBS_T=20

LOAD_TARGET_OBJ_DETECTOR=false
TARGET_OBJ_DETECTOR_STEP=17204
TARGET_OBJ_DETECTOR_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/baseline-1/1Task-Pick-Place-Target-Obj-Random-Frames-Batch128-1gpu-Attn2ly128-Act2ly256mix4-actCat
FREEZE_TARGET_OBJ_DETECTOR=false
CONCAT_STATE=false
ACTION_DIM=7
EARLY_STOPPING_PATIECE=30
OPTIMIZER='Adam'
LR=0.0005
WEIGHT_DECAY=0
SCHEDULER=None

RESUME_PATH=None
RESUME_STEP=-1

DATASET_TARGET=multi_task_il.datasets.multi_task_datasets_prompt.MultiTaskPairedDataset

python ../training/train_scripts/train_vima.py \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME} \
    policy=${POLICY} \
    task_names=${TASK_str} \
    set_same_n=${SET_SAME_N} \
    exp_name=${EXP_NAME} \
    dataset_target=${DATASET_TARGET} \
    bsize=${BSIZE} \
    vsize=${BSIZE} \
    epochs=${EPOCH} \
    dataset_cfg.obs_T=${OBS_T} \
    dataset_cfg.normalize_action=${NORMALIZE_ACTION} \
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
    debug=true \
    wandb_log=false \
    resume=false \
    loader_workers=${LOADER_WORKERS}
