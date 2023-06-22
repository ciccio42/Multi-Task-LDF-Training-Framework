#!/bin/sh
export MUJOCO_PY_MUJOCO_PATH="/home/frosa_loc/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

EXPERT_DATA=/home/frosa_loc/Multi-Task-LFD-Framework/ur_multitask_dataset
SAVE_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline
POLICY='${mosaic}'

SAVE_FREQ=1000
LOG_FREQ=1000
VAL_FREQ=1000

EXP_NAME=1Task-Pick-Place-Target-Slot-Mosaic-200-360
TASK_str=pick_place
EPOCH=40
BSIZE=32 #128 #64 #32
COMPUTE_OBJ_DISTRIBUTION=false
# Policy 1: At each slot is assigned a RandomSampler
BALANCING_POLICY=0
SET_SAME_N=2
CONFIG_PATH=../experiments
PROJECT_NAME="ur_pick_place_target_slot_200_360"
CONFIG_NAME=config.yaml
LOADER_WORKERS=16
NORMALIZE_ACTION=true

LOAD_CONTRASTIVE=false
CONTRASTIVE_PRE=1.0
CONTRASTIVE_POS=1.0
MUL_INTM=0

LOAD_TARGET_OBJ_DETECTOR=true
TARGET_OBJ_DETECTOR_STEP=18000
TARGET_OBJ_DETECTOR_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline/TARGET_OBJ_DETECTOR_SLOT/1Task-Pick-Place-Mosaic-200-360-Target-Obj-Detector-Batch32-1gpu-Attn2ly128-Act2ly256mix4-headCat
FREEZE_TARGET_OBJ_DETECTOR=false
REMOVE_CLASS_LAYERS=true
CONCAT_TARGET_OBJ_EMBEDDING=true
CONCAT_STATE=false

ACTION_DIM=7
N_MIXTURES=2       # Nut-Assembly 2 # Pick-place 6
OUT_DIM=128        # 64                  # 128
ATTN_FF=256        # 128                 # 256
COMPRESSOR_DIM=256 # 128          # 256
HIDDEN_DIM=512     # 256              # 512
CONCAT_DEMO_HEAD=false
CONCAT_DEMO_ACT=true
PRETRAINED=false

EARLY_STOPPING_PATIECE=-1
OPTIMIZER='AdamW'
LR=0.0001
WEIGHT_DECAY=0.0
SCHEDULER=None

DROP_DIM=3      # 2    # 3
OUT_FEATURE=256 # 512 # 256
DIM_H=13        # 7 (100 DROP_DIM 3)        #8         # 4         # 7
DIM_W=23        # 12 (180 DROP_DIM 3)        #8         # 6         # 12

RESUME_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline/1Task-Pick-Place-Mosaic-cropped-no-normalized-drop-2-Batch32
RESUME_STEP=95000
RESUME=false

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
    bsize=${BSIZE} \
    vsize=${BSIZE} \
    epochs=${EPOCH} \
    dataset_cfg.normalize_action=${NORMALIZE_ACTION} \
    dataset_cfg.compute_obj_distribution=${COMPUTE_OBJ_DISTRIBUTION} \
    samplers.balancing_policy=${BALANCING_POLICY} \
    mosaic.load_target_obj_detector=${LOAD_TARGET_OBJ_DETECTOR} \
    mosaic.target_obj_detector_step=${TARGET_OBJ_DETECTOR_STEP} \
    mosaic.target_obj_detector_path=${TARGET_OBJ_DETECTOR_PATH} \
    mosaic.freeze_target_obj_detector=${FREEZE_TARGET_OBJ_DETECTOR} \
    mosaic.remove_class_layers=${REMOVE_CLASS_LAYERS} \
    mosaic.dim_H=${DIM_H} \
    mosaic.dim_W=${DIM_W} \
    mosaic.load_contrastive=${LOAD_CONTRASTIVE} \
    mosaic.concat_target_obj_embedding=${CONCAT_TARGET_OBJ_EMBEDDING} \
    attn.img_cfg.pretrained=${PRETRAINED} \
    actions.adim=${ACTION_DIM} \
    actions.n_mixtures=${N_MIXTURES} \
    actions.out_dim=${OUT_DIM} \
    attn.attn_ff=${ATTN_FF} \
    attn.img_cfg.drop_dim=${DROP_DIM} \
    attn.img_cfg.out_feature=${OUT_FEATURE} \
    simclr.compressor_dim=${COMPRESSOR_DIM} \
    simclr.hidden_dim=${HIDDEN_DIM} \
    mosaic.concat_state=${CONCAT_STATE} \
    mosaic.concat_demo_head=${CONCAT_DEMO_HEAD} \
    mosaic.concat_demo_act=${CONCAT_DEMO_ACT} \
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
    simclr.mul_pre=${CONTRASTIVE_PRE} \
    simclr.mul_pos=${CONTRASTIVE_POS} \
    simclr.mul_intm=${MUL_INTM} \
    debug=false \
    wandb_log=true \
    resume=${RESUME} \
    loader_workers=${LOADER_WORKERS}
