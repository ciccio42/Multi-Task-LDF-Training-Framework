#!/bin/sh
export MUJOCO_PY_MUJOCO_PATH="/home/frosa_loc/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

EXPERT_DATA=/home/frosa_loc/Multi-Task-LFD-Framework/ur_multitask_dataset
SAVE_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline
POLICY='${target_obj_detector}'

SAVE_FREQ=1000000
LOG_FREQ=100
VAL_FREQ=4050

EXP_NAME=1Task-Pick-Place-Mosaic-200-360-Target-Obj-Detector
TASK_str=pick_place
EPOCH=20
BSIZE=32 #128 #64 #32
COMPUTE_OBJ_DISTRIBUTION=true
# Policy 1: At each slot is assigned a RandomSampler
BALANCING_POLICY=0
SET_SAME_N=2
CONFIG_PATH=../experiments
PROJECT_NAME="ur_pick_place_200_360_target_obj_detector"
CONFIG_NAME=config_target_obj.yaml
LOADER_WORKERS=16

ATTN_FF=128 # 256
CONCAT_DEMO_HEAD=true
CONCAT_DEMO_ACT=false
PRETRAINED=false

EARLY_STOPPING_PATIECE=5
OPTIMIZER='AdamW'
LR=0.0005
WEIGHT_DECAY=0
SCHEDULER=None

DROP_DIM=3      # 2    # 3
OUT_FEATURE=256 # 512 # 256
DIM_H=14        # 13        # 7 (100 DROP_DIM 3)        #8         # 4         # 7
DIM_W=14        # 23        # 12 (180 DROP_DIM 3)        #8         # 6         # 12

RESUME_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline/1Task-Pick-Place-Mosaic-cropped-no-normalized-drop-2-Batch32
RESUME_STEP=95000
RESUME=false

python ../training/train_scripts/train_target_obj_pos.py \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME} \
    policy=${POLICY} \
    set_same_n=${SET_SAME_N} \
    task_names=${TASK_str} \
    exp_name=${EXP_NAME} \
    save_freq=${SAVE_FREQ} \
    log_freq=${LOG_FREQ} \
    val_freq=${VAL_FREQ} \
    bsize=${BSIZE} \
    vsize=${BSIZE} \
    epochs=${EPOCH} \
    dataset_cfg.compute_obj_distribution=${COMPUTE_OBJ_DISTRIBUTION} \
    samplers.balancing_policy=${BALANCING_POLICY} \
    target_obj_detector.dim_H=${DIM_H} \
    target_obj_detector.dim_W=${DIM_W} \
    attn.img_cfg.pretrained=${PRETRAINED} \
    attn.attn_ff=${ATTN_FF} \
    attn.img_cfg.drop_dim=${DROP_DIM} \
    attn.img_cfg.out_feature=${OUT_FEATURE} \
    target_obj_detector.concat_state=${CONCAT_STATE} \
    target_obj_detector.concat_demo_head=${CONCAT_DEMO_HEAD} \
    target_obj_detector.concat_demo_act=${CONCAT_DEMO_ACT} \
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
