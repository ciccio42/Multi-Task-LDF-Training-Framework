#!/bin/sh
export MUJOCO_PY_MUJOCO_PATH="/home/frosa_Loc/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1


EXPERT_DATA=/raid/home/frosa_Loc/ur_multitask_dataset
SAVE_PATH=/raid/home/frosa_Loc/checkpoint_save_folder
POLICY='${cond_target_obj_detector}'
DATASET_TARGET=multi_task_il.datasets.multi_task_cond_target_obj_dataset.CondTargetObjDetectorDataset

SAVE_FREQ=1620
LOG_FREQ=100
VAL_FREQ=1620
PRINT_FREQ=100

EXP_NAME=1Task-Pick-Place-Cond-Target-Obj-Detector-only-first-frame
PROJECT_NAME="pick_place_cond_target_obj_detector_only_first_frame"
TASK_str=pick_place
EPOCH=30 # start from 16
BSIZE=80 #16 #32
COMPUTE_OBJ_DISTRIBUTION=false
CONFIG_PATH=../experiments/
CONFIG_NAME=config_cond_target_obj_detector.yaml
LOADER_WORKERS=16
BALANCING_POLICY=0
SET_SAME_N=5
OBS_T=0

EARLY_STOPPING_PATIECE=-1
OPTIMIZER='AdamW'
LR=0.000001
WEIGHT_DECAY=5
SCHEDULER='ReduceLROnPlateau'
FIRST_FRAMES=false
ONLY_FIRST_FRAMES=true
ROLLOUT=false
PERFORM_AUGS=true
NON_SEQUENTIAL=false

RESUME_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline/TARGET_OBJ_DETECTOR_SLOT/1Task-Pick-Place-Cond-Target-Obj-Detector-3D-Resnet-Batch16
RESUME_STEP=2025
RESUME=false

DROP_DIM=3      # 2    # 3
OUT_FEATURE=256 # 512 # 256
DIM_H=7 #14        # 7 (100 DROP_DIM 3)        #8         # 4         # 7
DIM_W=12 #14        # 12 (180 DROP_DIM 3)        #8         # 6         # 12
HEIGHT=100
WIDTH=180

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
    dataset_cfg.non_sequential=${NON_SEQUENTIAL} \
    dataset_cfg.compute_obj_distribution=${COMPUTE_OBJ_DISTRIBUTION} \
    dataset_cfg.first_frames=${FIRST_FRAMES} \
    dataset_cfg.only_first_frame=${ONLY_FIRST_FRAMES} \
    dataset_cfg.height=${HEIGHT} \
    dataset_cfg.width=${WIDTH} \
    dataset_cfg.perform_augs=${PERFORM_AUGS} \
    samplers.balancing_policy=${BALANCING_POLICY} \
    early_stopping_cfg.patience=${EARLY_STOPPING_PATIECE} \
    cond_target_obj_detector_cfg.height=${HEIGHT} \
    cond_target_obj_detector_cfg.width=${WIDTH} \
    cond_target_obj_detector_cfg.dim_H=${DIM_H} \
    cond_target_obj_detector_cfg.dim_W=${DIM_W} \
    cond_target_obj_detector_cfg.n_channels=${OUT_FEATURE} \
    cond_target_obj_detector_cfg.conv_drop_dim=${DROP_DIM} \
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
