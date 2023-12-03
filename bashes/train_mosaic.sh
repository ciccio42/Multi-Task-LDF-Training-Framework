#!/bin/sh
export MUJOCO_PY_MUJOCO_PATH=/user/frosa/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user/frosa/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user/frosa/miniconda3/envs/multi_task_lfd/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

EXPERT_DATA=/user/frosa/multi_task_lfd/ur_multitask_dataset/opt_dataset
SAVE_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder
POLICY='${mosaic}'

SAVE_FREQ=-1
LOG_FREQ=100
VAL_FREQ=-1

EXP_NAME=2Task-Nut-Assembly-Pick-Place-100-180-Prova
PROJECT_NAME=${EXP_NAME}
TASK_str=[pick_place,nut_assembly]
ROLLOUT=false
EPOCH=90
BSIZE=27 #32 #128 #64 #32
COMPUTE_OBJ_DISTRIBUTION=false
# Policy 1: At each slot is assigned a RandomSampler
BALANCING_POLICY=0
SET_SAME_N=2
CONFIG_PATH=../experiments
CONFIG_NAME=config.yaml
LOADER_WORKERS=8
NORMALIZE_ACTION=true

LOAD_CONTRASTIVE=true
CONTRASTIVE_PRE=1.0
CONTRASTIVE_POS=1.0
MUL_INTM=0
BC_MUL=1.0
INV_MUL=1.0

LOAD_TARGET_OBJ_DETECTOR=false
TARGET_OBJ_DETECTOR_STEP=40455
TARGET_OBJ_DETECTOR_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/1Task-Nut-Assembly-Cond-Target-Obj-Detector-separate-demo-agent-Batch54
FREEZE_TARGET_OBJ_DETECTOR=false
REMOVE_CLASS_LAYERS=false
CONCAT_TARGET_OBJ_EMBEDDING=false
CONCAT_STATE=false

ACTION_DIM=7
N_MIXTURES=7 #3 Pick-place
OUT_DIM=64 #128 Pick-place
ATTN_FF=128 #256 Pick-place
COMPRESSOR_DIM=128 #256 Pick-place
HIDDEN_DIM=128 #512 Pick-place
CONCAT_DEMO_HEAD=false
CONCAT_DEMO_ACT=true
PRETRAINED=false
CONCAT_BB=false
NULL_BB=false

EARLY_STOPPING_PATIECE=-1
OPTIMIZER='AdamW'
LR=0.0005
WEIGHT_DECAY=0.0
SCHEDULER=None

DROP_DIM=3      # 2    # 3
OUT_FEATURE=256 # 512 # 256
DIM_H=7 #14        # 7 (100 DROP_DIM 3)        #8         # 4         # 7
DIM_W=12 #14        # 12 (180 DROP_DIM 3)        #8         # 6         # 12
HEIGHT=100
WIDTH=180

RESUME_PATH="/user/frosa/multi_task_lfd/checkpoint_save_folder/1Task-Nut-Assembly-Mosaic-200-360-Batch27"
RESUME_STEP=108000
RESUME=false

COSINE_ANNEALING=false

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
    rollout=${ROLLOUT} \
    dataset_cfg.normalize_action=${NORMALIZE_ACTION} \
    dataset_cfg.compute_obj_distribution=${COMPUTE_OBJ_DISTRIBUTION} \
    dataset_cfg.height=${HEIGHT} \
    dataset_cfg.width=${WIDTH} \
    samplers.balancing_policy=${BALANCING_POLICY} \
    mosaic.load_target_obj_detector=${LOAD_TARGET_OBJ_DETECTOR} \
    mosaic.target_obj_detector_step=${TARGET_OBJ_DETECTOR_STEP} \
    mosaic.target_obj_detector_path=${TARGET_OBJ_DETECTOR_PATH} \
    mosaic.freeze_target_obj_detector=${FREEZE_TARGET_OBJ_DETECTOR} \
    mosaic.remove_class_layers=${REMOVE_CLASS_LAYERS} \
    mosaic.dim_H=${DIM_H} \
    mosaic.dim_W=${DIM_W} \
    mosaic.concat_bb=${CONCAT_BB} \
    mosaic.load_contrastive=${LOAD_CONTRASTIVE} \
    mosaic.concat_target_obj_embedding=${CONCAT_TARGET_OBJ_EMBEDDING} \
    augs.null_bb=${NULL_BB} \
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
    bc_mul=${BC_MUL} \
    inv_mul=${INV_MUL} \
    cosine_annealing=${COSINE_ANNEALING} \
    debug=true \
    wandb_log=false \
    resume=${RESUME} \
    loader_workers=${LOADER_WORKERS}
