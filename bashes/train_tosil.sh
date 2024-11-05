#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0,1,2,3
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1   # Request 1 GPU
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16

export MUJOCO_PY_MUJOCO_PATH="/home/rsofnc000/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

export HYDRA_FULL_ERROR=1
echo $1
TASK_NAME="$1"

EXPERT_DATA=/home/rsofnc000/dataset/opt_dataset/
SAVE_PATH=/home/rsofnc000/checkpoint_save_folder
POLICY='${tosil}'
TARGET='multi_task_il.models.mt_rep.VideoImitation'

SAVE_FREQ=-1
LOG_FREQ=10
VAL_FREQ=-1
DEVICE=0
DEBUG=false
WANDB_LOG=true
ROLLOUT=false
EPOCH=90
LOADER_WORKERS=16
CONFIG_PATH=../experiments
CONFIG_NAME=config.yaml

SPLIT_PICK_PLACE=false

BC_MUL=1.0
INV_MUL=1.0
PNT_MUL=1.0
LOAD_EEF_POINTS=true

EARLY_STOPPING_PATIECE=-1
OPTIMIZER='AdamW'
LR=0.0005
WEIGHT_DECAY=0.0
SCHEDULER=None

if [ "$TASK_NAME" == 'nut_assembly' ]; then
    echo "NUT-ASSEMBLY"
    ### Nut-Assembly ###
    RESUME_PATH=1Task-nut_assembly-Double-Policy-Contrastive-false-Inverse-false-trial-2-Batch27
    RESUME_STEP=18640
    RESUME=false

    BSIZE=27 #32 #128 #64 #32
    COMPUTE_OBJ_DISTRIBUTION=false
    # Policy 1: At each slot is assigned a RandomSampler
    BALANCING_POLICY=0
    SET_SAME_N=3
    NORMALIZE_ACTION=true
    CHANGE_COMMAND_EPOCH=true

    ACTION_DIM=7
    N_MIXTURES=7 #14 MT #7 2Task, Nut, button, stack #3 Pick-place #2 Nut-Assembly
    OUT_DIM=64   #64 MT #64 2Task, Nut, button, stack #128 Pick-place
    PRETRAINED=false
    NULL_BB=false

    EARLY_STOPPING_PATIECE=-1
    OPTIMIZER='AdamW'
    LR=0.0005
    WEIGHT_DECAY=0.0
    SCHEDULER=None

    HEIGHT=100
    WIDTH=180

    COSINE_ANNEALING=false

    TASK_str="nut_assembly" #[pick_place,nut_assembly,stack_block,button]
    EXP_NAME=1Task-TOSIL-${TASK_str}
    PROJECT_NAME=${EXP_NAME}
elif [ "$TASK_NAME" == 'button' ] || [ "$TASK_NAME" == 'press_button_close_after_reaching' ]; then
    echo "BUTTON"
    BSIZE=27 #32 #128 #64 #32
    COMPUTE_OBJ_DISTRIBUTION=false
    # Policy 1: At each slot is assigned a RandomSampler
    BALANCING_POLICY=0
    SET_SAME_N=3
    NORMALIZE_ACTION=true
    CHANGE_COMMAND_EPOCH=true

    ACTION_DIM=7
    N_MIXTURES=7 #14 MT #7 2Task, Nut, button, stack #3 Pick-place #2 Nut-Assembly
    OUT_DIM=64   #64 MT #64 2Task, Nut, button, stack #128 Pick-place
    PRETRAINED=false
    NULL_BB=false

    EARLY_STOPPING_PATIECE=-1
    OPTIMIZER='AdamW'
    LR=0.0005
    WEIGHT_DECAY=0.0
    SCHEDULER=None

    HEIGHT=100
    WIDTH=180

    COSINE_ANNEALING=false

    TASK_str=${TASK_NAME} #[pick_place,nut_assembly,stack_block,button]
    EXP_NAME=1Task-TOSIL-${TASK_str}
    PROJECT_NAME=${EXP_NAME}
elif [ "$TASK_NAME" == 'stack_block' ]; then
    echo "STACK_BLOCK"
    BSIZE=27 #32 #128 #64 #32
    COMPUTE_OBJ_DISTRIBUTION=false
    # Policy 1: At each slot is assigned a RandomSampler
    BALANCING_POLICY=0
    SET_SAME_N=2
    NORMALIZE_ACTION=true
    CHANGE_COMMAND_EPOCH=true

    ACTION_DIM=7
    N_MIXTURES=7 #14 MT #7 2Task, Nut, button, stack #3 Pick-place #2 Nut-Assembly
    OUT_DIM=64   #64 MT #64 2Task, Nut, button, stack #128 Pick-place
    PRETRAINED=false
    NULL_BB=false

    EARLY_STOPPING_PATIECE=-1
    OPTIMIZER='AdamW'
    LR=0.0005
    WEIGHT_DECAY=0.0
    SCHEDULER=None

    HEIGHT=100
    WIDTH=180

    COSINE_ANNEALING=false

    TASK_str=stack_block #[pick_place,nut_assembly,stack_block,button]
    EXP_NAME=1Task-TOSIL-${TASK_str}
    PROJECT_NAME=${EXP_NAME}

elif [ "$TASK_NAME" == 'pick_place' ]; then
    echo "Pick-Place"
    BSIZE=27 #32 #128 #64 #32
    COMPUTE_OBJ_DISTRIBUTION=false
    # Policy 1: At each slot is assigned a RandomSampler
    BALANCING_POLICY=0
    SET_SAME_N=2
    NORMALIZE_ACTION=true
    CHANGE_COMMAND_EPOCH=true

    ACTION_DIM=7
    N_MIXTURES=3 #14 MT #7 2Task, Nut, button, stack #3 Pick-place #2 Nut-Assembly
    OUT_DIM=128  #64 MT #64 2Task, Nut, button, stack #128 Pick-place
    PRETRAINED=false
    NULL_BB=false

    EARLY_STOPPING_PATIECE=-1
    OPTIMIZER='AdamW'
    LR=0.0005
    WEIGHT_DECAY=0.0
    SCHEDULER=None

    HEIGHT=100
    WIDTH=180

    COSINE_ANNEALING=false

    TASK_str=pick_place #[pick_place,nut_assembly,stack_block,button]
    EXP_NAME=1Task-TOSIL-${TASK_str}
    PROJECT_NAME=${EXP_NAME}
elif [ "$TASK_NAME" == 'multi' ]; then
    echo "Multi-Task"
    RESUME_PATH=4Task-TOSIL-multi-task-Batch74
    RESUME_STEP=138229
    RESUME=true

    BSIZE=27 #32 #128 #64 #32
    COMPUTE_OBJ_DISTRIBUTION=false
    # Policy 1: At each slot is assigned a RandomSampler
    BALANCING_POLICY=0
    SET_SAME_N=2
    NORMALIZE_ACTION=true
    CHANGE_COMMAND_EPOCH=true

    ACTION_DIM=7
    N_MIXTURES=7 #14 MT #7 2Task, Nut, button, stack #3 Pick-place #2 Nut-Assembly
    OUT_DIM=64   #64 MT #64 2Task, Nut, button, stack #128 Pick-place
    PRETRAINED=false
    NULL_BB=false

    EARLY_STOPPING_PATIECE=-1
    OPTIMIZER='AdamW'
    LR=0.0005
    WEIGHT_DECAY=0.0
    SCHEDULER=None

    HEIGHT=100
    WIDTH=180

    COSINE_ANNEALING=false

    TASK_str=[pick_place,nut_assembly,stack_block,press_button_close_after_reaching]
    EXP_NAME=4Task-TOSIL-multi-task
    PROJECT_NAME=${EXP_NAME}
fi

python ../training/train_scripts/train_any.py \
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
    dataset_cfg.load_eef_point=${LOAD_EEF_POINTS} \
    samplers.balancing_policy=${BALANCING_POLICY} \
    dataset_cfg.split_pick_place=${SPLIT_PICK_PLACE} \
    tosil.adim=${ACTION_DIM} \
    tosil.n_mixtures=${N_MIXTURES} \
    augs.null_bb=${NULL_BB} \
    actions.adim=${ACTION_DIM} \
    actions.n_mixtures=${N_MIXTURES} \
    actions.out_dim=${OUT_DIM} \
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
    bc_mul=${BC_MUL} \
    inv_mul=${INV_MUL} \
    pnt_mul=${PNT_MUL} \
    debug=${DEBUG} \
    wandb_log=${WANDB_LOG} \
    resume=${RESUME} \
    loader_workers=${LOADER_WORKERS}
