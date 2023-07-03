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

EXP_NAME=1Task-Pick-Place-Target-Slot-MAML-224_224
TASK_str=pick_place
EPOCH=40
BSIZE=16 #128 #64 #32
COMPUTE_OBJ_DISTRIBUTION=false
# Policy 1: At each slot is assigned a RandomSampler
BALANCING_POLICY=0
SET_SAME_N=1
CONFIG_PATH=../experiments
PROJECT_NAME="ur_pick_place_daml"
CONFIG_NAME=config.yaml
LOADER_WORKERS=4
NORMALIZE_ACTION=true

LOAD_CONTRASTIVE=false
CONTRASTIVE_PRE=1.0
CONTRASTIVE_POS=1.0
MUL_INTM=0

LOAD_TARGET_OBJ_DETECTOR=true
TARGET_OBJ_DETECTOR_STEP=13000
TARGET_OBJ_DETECTOR_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline/1Task-Pick-Place-Mosaic-200-360-Target-Obj-Detector-Batch32-1gpu-Attn2ly128-Act2ly256mix4-headCat
FREEZE_TARGET_OBJ_DETECTOR=false
REMOVE_CLASS_LAYERS=true
CONCAT_TARGET_OBJ_EMBEDDING=true
CONCAT_STATE=false

ACTION_DIM=7
N_MIXTURES=5       # Nut-Assembly 3 # Pick-place 6
OUT_DIM=128        # 64                  # 128
ATTN_FF=256        # 128                 # 256
COMPRESSOR_DIM=256 # 128          # 256
HIDDEN_DIM=512     # 256              # 512
CONCAT_DEMO_HEAD=false
CONCAT_DEMO_ACT=true
PRETRAINED=false

EARLY_STOPPING_PATIECE=-1
OPTIMIZER='AdamW'
LR=0.0005
WEIGHT_DECAY=0.0
SCHEDULER=ReduceLROnPlateau

DROP_DIM=3      # 2    # 3
OUT_FEATURE=256 # 512 # 256
DIM_H=14        # 7 (100 DROP_DIM 3)        #8         # 4         # 7
DIM_W=14        # 12 (180 DROP_DIM 3)        #8         # 6         # 12

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
    debug=true \
    wandb_log=false \
    resume=${RESUME} \
    loader_workers=${LOADER_WORKERS} \
    cosine_annealing=${COSINE_ANNEALING}