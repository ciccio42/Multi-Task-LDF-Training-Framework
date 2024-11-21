#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0,1,2,3
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1   # Request 1 GPU
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16

# export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

export MUJOCO_PY_MUJOCO_PATH=/home/frosa_Loc/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

export HYDRA_FULL_ERROR=1
echo $1
# TASK_NAME="$1"
TASK_NAME="pick_place"

EXPERT_DATA=/raid/home/frosa_Loc/opt_dataset/
# SAVE_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder
SAVE_PATH=/raid/home/frosa_Loc/multi_task_lfd/checkpoint_save_folder
POLICY='${mosaic}'
TARGET='multi_task_il.models.mt_rep.VideoImitation'

SAVE_FREQ=-1
LOG_FREQ=10
VAL_FREQ=-1
DEVICE=0    # cuda gpu selection
DEBUG=true
WANDB_LOG=false # weight and bias
ROLLOUT=false
EPOCH=90
LOADER_WORKERS=16
CONFIG_PATH=../experiments
CONFIG_NAME=config.yaml
CONCAT_IMG_EMB=true
CONCAT_DEMO_EMB=true

LOAD_TARGET_OBJ_DETECTOR=false
CONCAT_BB=false

CHANGE_COMMAND_EPOCH=true
SPLIT_PICK_PLACE=false

LOAD_CONTRASTIVE=true
LOAD_INV=true

CONCAT_STATE=true

if [ "$TASK_NAME" == 'nut_assembly' ]; then
    echo "NUT-ASSEMBLY"
    #SBATCH --job-name=nut_assembly
    ### Nut-Assembly ###
    RESUME_PATH=1Task-nut_assembly-Double-Policy-Contrastive-false-Inverse-false-trial-2-Batch27
    RESUME_STEP=18640
    RESUME=false

    TARGET_OBJ_DETECTOR_STEP=53091 #68526 #129762 #198900 #65250
    TARGET_OBJ_DETECTOR_PATH=${SAVE_PATH}/1Task-Nut-Assemly-KP-Batch63

    BSIZE=27 #32 #128 #64 #32
    COMPUTE_OBJ_DISTRIBUTION=false
    # Policy 1: At each slot is assigned a RandomSampler
    BALANCING_POLICY=0
    SET_SAME_N=3
    NORMALIZE_ACTION=true

    CONTRASTIVE_PRE=1.0
    CONTRASTIVE_POS=1.0
    MUL_INTM=0
    BC_MUL=1.0
    INV_MUL=1.0

    FREEZE_TARGET_OBJ_DETECTOR=false
    REMOVE_CLASS_LAYERS=false
    CONCAT_TARGET_OBJ_EMBEDDING=false

    ACTION_DIM=7
    N_MIXTURES=3       #14 MT #7 2Task, Nut, button, stack #3 Pick-place #2 Nut-Assembly
    OUT_DIM=128        #64 MT #64 2Task, Nut, button, stack #128 Pick-place
    ATTN_FF=256        #256 MT #128 2Task, Nut, button, stack #256 Pick-place
    COMPRESSOR_DIM=256 #256 MT #128 2Task, Nut, button, stack #256 Pick-place
    HIDDEN_DIM=512     #256 MT #128 2Task, Nut, button, stack #512 Pick-place
    CONCAT_DEMO_HEAD=false
    CONCAT_DEMO_ACT=true
    PRETRAINED=false
    NULL_BB=false

    EARLY_STOPPING_PATIECE=-1
    OPTIMIZER='AdamW'
    LR=0.0002
    WEIGHT_DECAY=0.0
    SCHEDULER=None

    DROP_DIM=4      # 2    # 3
    OUT_FEATURE=128 # 512 # 256
    DIM_H=13        #14        # 7 (100 DROP_DIM 3)        #8         # 4         # 7
    DIM_W=23        #14        # 12 (180 DROP_DIM 3)        #8         # 6         # 12
    HEIGHT=100
    WIDTH=180

    COSINE_ANNEALING=false

    TASK_str="nut_assembly" #[pick_place,nut_assembly,stack_block,button]
    EXP_NAME=1Task-${TASK_str}-MOSAIC-rollout
    PROJECT_NAME=${EXP_NAME}
elif [ "$TASK_NAME" == 'button' ] || [ "$TASK_NAME" == 'press_button_close_after_reaching' ]; then
    echo "BUTTON"
    RESUME_PATH=1Task-press_button_close_after_reaching-Double-Policy-Contrastive-false-Inverse-false-Batch18
    RESUME_STEP=3624
    RESUME=false

    TARGET_OBJ_DETECTOR_STEP=44625 #68526 #129762 #198900 #65250
    TARGET_OBJ_DETECTOR_PATH=${SAVE_PATH}/Task-button-KP-no-scaled-Batch36

    BSIZE=27 #32 #128 #64 #32
    COMPUTE_OBJ_DISTRIBUTION=false
    # Policy 1: At each slot is assigned a RandomSampler
    BALANCING_POLICY=0
    SET_SAME_N=3
    NORMALIZE_ACTION=true

    CONTRASTIVE_PRE=1.0
    CONTRASTIVE_POS=1.0
    MUL_INTM=0
    BC_MUL=1.0
    INV_MUL=1.0

    FREEZE_TARGET_OBJ_DETECTOR=false
    REMOVE_CLASS_LAYERS=false
    CONCAT_TARGET_OBJ_EMBEDDING=false

    ACTION_DIM=7
    N_MIXTURES=3       #14 MT #7 2Task, Nut, button, stack #3 Pick-place #2 Nut-Assembly
    OUT_DIM=128        #64 MT #64 2Task, Nut, button, stack #128 Pick-place
    ATTN_FF=256        #256 MT #128 2Task, Nut, button, stack #256 Pick-place
    COMPRESSOR_DIM=256 #256 MT #128 2Task, Nut, button, stack #256 Pick-place
    HIDDEN_DIM=512     #256 MT #128 2Task, Nut, button, stack #512 Pick-place
    CONCAT_DEMO_HEAD=false
    CONCAT_DEMO_ACT=true
    PRETRAINED=false
    NULL_BB=false

    EARLY_STOPPING_PATIECE=-1
    OPTIMIZER='AdamW'
    LR=0.0005
    WEIGHT_DECAY=0.0
    SCHEDULER=None

    DROP_DIM=4      # 2    # 3
    OUT_FEATURE=128 # 512 # 256
    DIM_H=13        #14        # 7 (100 DROP_DIM 3)        #8         # 4         # 7
    DIM_W=23        #14        # 12 (180 DROP_DIM 3)        #8         # 6         # 12
    HEIGHT=100
    WIDTH=180

    COSINE_ANNEALING=false

    TASK_str=${TASK_NAME} #[pick_place,nut_assembly,stack_block,button]
    EXP_NAME=1Task-press_button-MOSAIC-State_pos_gripper
    PROJECT_NAME=${EXP_NAME}
elif [ "$TASK_NAME" == 'stack_block' ]; then
    echo "STACK_BLOCK"
    RESUME_PATH=1Task-press_button_close_after_reaching-Double-Policy-Contrastive-false-Inverse-false-Batch18
    RESUME_STEP=3624
    RESUME=false

    TARGET_OBJ_DETECTOR_STEP=37665 #68526 #129762 #198900 #65250
    TARGET_OBJ_DETECTOR_PATH=${SAVE_PATH}/1Task-stack_block-CTOD-KP-Batch36

    BSIZE=27 #32 #128 #64 #32
    COMPUTE_OBJ_DISTRIBUTION=false
    # Policy 1: At each slot is assigned a RandomSampler
    BALANCING_POLICY=0
    SET_SAME_N=3
    NORMALIZE_ACTION=true

    CONTRASTIVE_PRE=1.0
    CONTRASTIVE_POS=1.0
    MUL_INTM=0
    BC_MUL=1.0
    INV_MUL=1.0

    FREEZE_TARGET_OBJ_DETECTOR=false
    REMOVE_CLASS_LAYERS=false
    CONCAT_TARGET_OBJ_EMBEDDING=false

    ACTION_DIM=7
    N_MIXTURES=3       #14 MT #7 2Task, Nut, button, stack #3 Pick-place #2 Nut-Assembly
    OUT_DIM=128        #64 MT #64 2Task, Nut, button, stack #128 Pick-place
    ATTN_FF=256        #256 MT #128 2Task, Nut, button, stack #256 Pick-place
    COMPRESSOR_DIM=256 #256 MT #128 2Task, Nut, button, stack #256 Pick-place
    HIDDEN_DIM=512     #256 MT #128 2Task, Nut, button, stack #512 Pick-place
    CONCAT_DEMO_HEAD=false
    CONCAT_DEMO_ACT=true
    PRETRAINED=false
    NULL_BB=false

    EARLY_STOPPING_PATIECE=-1
    OPTIMIZER='AdamW'
    LR=0.0005
    WEIGHT_DECAY=0.0
    SCHEDULER=None

    DROP_DIM=4      # 2    # 3
    OUT_FEATURE=128 # 512 # 256
    DIM_H=13        #14        # 7 (100 DROP_DIM 3)        #8         # 4         # 7
    DIM_W=23        #14        # 12 (180 DROP_DIM 3)        #8         # 6         # 12
    HEIGHT=100
    WIDTH=180

    COSINE_ANNEALING=false

    TASK_str=${TASK_NAME} #[pick_place,nut_assembly,stack_block,button]
    EXP_NAME=1Task-${TASK_str}-MOSAIC-State_pos_gripper
    PROJECT_NAME=${EXP_NAME}

elif [ "$TASK_NAME" == 'pick_place' ]; then
    echo "Pick-Place"
    ### Pick-Place ###
    RESUME_PATH=""
    RESUME_STEP=""
    RESUME=false

    TARGET_OBJ_DETECTOR_STEP=37476 #68526 #129762 #198900 #65250
    TARGET_OBJ_DETECTOR_PATH=${SAVE_PATH}/1Task-Pick-Place-KP-Batch112

    BSIZE=32 #32 #128 #64 #32
    COMPUTE_OBJ_DISTRIBUTION=false
    # Policy 1: At each slot is assigned a RandomSampler
    BALANCING_POLICY=0
    SET_SAME_N=2

    NORMALIZE_ACTION=true

    CONTRASTIVE_PRE=1.0
    CONTRASTIVE_POS=1.0
    MUL_INTM=0
    BC_MUL=1.0
    INV_MUL=1.0

    FREEZE_TARGET_OBJ_DETECTOR=false
    REMOVE_CLASS_LAYERS=false
    CONCAT_TARGET_OBJ_EMBEDDING=false

    ACTION_DIM=7
    N_MIXTURES=3       #14 MT #7 2Task, Nut, button, stack #3 Pick-place #2 Nut-Assembly
    OUT_DIM=128        #64 MT #64 2Task, Nut, button, stack #128 Pick-place
    ATTN_FF=256        #256 MT #128 2Task, Nut, button, stack #256 Pick-place
    COMPRESSOR_DIM=256 #256 MT #128 2Task, Nut, button, stack #256 Pick-place
    HIDDEN_DIM=512     #256 MT #128 2Task, Nut, button, stack #512 Pick-place
    CONCAT_DEMO_HEAD=false
    CONCAT_DEMO_ACT=true
    PRETRAINED=false
    NULL_BB=false

    EARLY_STOPPING_PATIECE=-1
    OPTIMIZER='AdamW'
    LR=0.0005
    WEIGHT_DECAY=0.0
    SCHEDULER=None

    DROP_DIM=4      # 2    # 3
    OUT_FEATURE=128 # 512 # 256
    DIM_H=13        #14        # 7 (100 DROP_DIM 3)        #8         # 4         # 7
    DIM_W=23        #14        # 12 (180 DROP_DIM 3)        #8         # 6         # 12
    HEIGHT=100
    WIDTH=180

    COSINE_ANNEALING=false

    TASK_str="pick_place" #[pick_place,nut_assembly,stack_block,button]
    EXP_NAME=Provola      #1Task-${TASK_str}-MOSAIC-Rollout
    PROJECT_NAME=${EXP_NAME}
elif [ "$TASK_NAME" == 'multi' ]; then
    echo "Multi Task"
    ### Pick-Place ###
    RESUME_PATH=""
    RESUME_STEP=""
    RESUME=false

    TARGET_OBJ_DETECTOR_STEP=91800 #68526 #129762 #198900 #65250
    TARGET_OBJ_DETECTOR_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/4Task-CTOD-KP-Batch74/

    BSIZE=32 #32 #128 #64 #32
    COMPUTE_OBJ_DISTRIBUTION=false
    # Policy 1: At each slot is assigned a RandomSampler
    BALANCING_POLICY=0
    SET_SAME_N=2

    NORMALIZE_ACTION=true

    CONTRASTIVE_PRE=1.0
    CONTRASTIVE_POS=1.0
    MUL_INTM=0
    BC_MUL=1.0
    INV_MUL=1.0

    FREEZE_TARGET_OBJ_DETECTOR=false
    REMOVE_CLASS_LAYERS=false
    CONCAT_TARGET_OBJ_EMBEDDING=false

    ACTION_DIM=7
    N_MIXTURES=14      #14 MT #7 2Task, Nut, button, stack #3 Pick-place #2 Nut-Assembly
    OUT_DIM=64         #64 MT #64 2Task, Nut, button, stack #128 Pick-place
    ATTN_FF=256        #256 MT #128 2Task, Nut, button, stack #256 Pick-place
    COMPRESSOR_DIM=256 #256 MT #128 2Task, Nut, button, stack #256 Pick-place
    HIDDEN_DIM=256     #256 MT #128 2Task, Nut, button, stack #512 Pick-place
    CONCAT_DEMO_HEAD=false
    CONCAT_DEMO_ACT=true
    PRETRAINED=false
    NULL_BB=false

    EARLY_STOPPING_PATIECE=-1
    OPTIMIZER='AdamW'
    LR=0.0005
    WEIGHT_DECAY=0.0
    SCHEDULER=None

    DROP_DIM=4      # 2    # 3
    OUT_FEATURE=128 # 512 # 256
    DIM_H=13        #14        # 7 (100 DROP_DIM 3)        #8         # 4         # 7
    DIM_W=23        #14        # 12 (180 DROP_DIM 3)        #8         # 6         # 12
    HEIGHT=100
    WIDTH=180

    COSINE_ANNEALING=false

    TASK_str=[pick_place,nut_assembly,stack_block,press_button_close_after_reaching]
    EXP_NAME=Multi-Task-MOSAIC-State_true
    PROJECT_NAME=${EXP_NAME}
fi

# srun --output=training_${EXP_NAME}.txt --job-name=training_${EXP_NAME}
python -u ../training/train_scripts/train_any.py \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME} \
    policy=${POLICY} \
    device=${DEVICE} \
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
    dataset_cfg.change_command_epoch=${CHANGE_COMMAND_EPOCH} \
    dataset_cfg.height=${HEIGHT} \
    dataset_cfg.width=${WIDTH} \
    dataset_cfg.split_pick_place=${SPLIT_PICK_PLACE} \
    samplers.balancing_policy=${BALANCING_POLICY} \
    mosaic._target_=${TARGET} \
    mosaic.load_target_obj_detector=${LOAD_TARGET_OBJ_DETECTOR} \
    mosaic.target_obj_detector_step=${TARGET_OBJ_DETECTOR_STEP} \
    mosaic.target_obj_detector_path=${TARGET_OBJ_DETECTOR_PATH} \
    mosaic.freeze_target_obj_detector=${FREEZE_TARGET_OBJ_DETECTOR} \
    mosaic.remove_class_layers=${REMOVE_CLASS_LAYERS} \
    mosaic.dim_H=${DIM_H} \
    mosaic.dim_W=${DIM_W} \
    mosaic.concat_bb=${CONCAT_BB} \
    mosaic.load_contrastive=${LOAD_CONTRASTIVE} \
    mosaic.load_inv=${LOAD_INV} \
    mosaic.concat_target_obj_embedding=${CONCAT_TARGET_OBJ_EMBEDDING} \
    augs.null_bb=${NULL_BB} \
    attn.img_cfg.pretrained=${PRETRAINED} \
    actions.adim=${ACTION_DIM} \
    actions.n_mixtures=${N_MIXTURES} \
    actions.out_dim=${OUT_DIM} \
    actions.concat_img_emb=${CONCAT_IMG_EMB} \
    actions.concat_demo_emb=${CONCAT_DEMO_EMB} \
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
    debug=${DEBUG} \
    wandb_log=${WANDB_LOG} \
    resume=${RESUME} \
    loader_workers=${LOADER_WORKERS}
