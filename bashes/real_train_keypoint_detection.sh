#!/bin/bash

#SBATCH --exclude=tnode[01-17]
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --export=ALL

export MUJOCO_PY_MUJOCO_PATH=/home/rsofnc000/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rsofnc000/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export HYDRA_FULL_ERROR=1

EXPERT_DATA=/home/rsofnc000/dataset/opt_dataset
SAVE_PATH=/home/rsofnc000/checkpoint_save_folder
POLICY='${cond_target_obj_detector}'
DATASET_TARGET=multi_task_il.datasets.multi_task_keypoint_dataset.MultiTaskPairedKeypointDetectionDataset
TASKS_CONFIG=7_tasks_real
AGENT_NAME=real_new_ur5e
echo $1
TASK_NAME="$1"

SAVE_FREQ=-1
LOG_FREQ=10
VAL_FREQ=-1
PRINT_FREQ=$LOG_FREQ
DEVICE=-1
DEBUG=false
WANDB_LOG=true

EPOCH=90 # start from 16
BSIZE=32 #16 #32

COMPUTE_OBJ_DISTRIBUTION=false
CONFIG_PATH=../experiments/
CONFIG_NAME=config_cond_target_obj_detector_real.yaml
LOADER_WORKERS=1
BALANCING_POLICY=0
OBS_T=7

EARLY_STOPPING_PATIECE=20
OPTIMIZER='AdamW'
LR=0.00001
WEIGHT_DECAY=5
SCHEDULER='ReduceLROnPlateau'
FIRST_FRAMES=false
ONLY_FIRST_FRAMES=false
ROLLOUT=false
PERFORM_AUGS=true
NON_SEQUENTIAL=true

DROP_DIM=4      # 2    # 3
OUT_FEATURE=128 # 512 # 256
DIM_H=13        #14        # 7 (100 DROP_DIM 3)        #8         # 4         # 7
DIM_W=23        #14        # 12 (180 DROP_DIM 3)        #8         # 6         # 12
HEIGHT=100
WIDTH=180
N_CLASSES=4
DAGGER=false

if [ "$TASK_NAME" == 'nut_assembly' ]; then
    echo "NUT-ASSEMBLY"
    TASK_str="nut_assembly"
    EXP_NAME=1Task-${TASK_str}-CTOD-KP_NO_0_4_8
    PROJECT_NAME=${EXP_NAME}
    SET_SAME_N=7
    RESUME_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/${EXP_NAME}-Batch74/
    RESUME_STEP=72675
    RESUME=false
elif [ "$TASK_NAME" == 'button' ] || [ "$TASK_NAME" == 'press_button_close_after_reaching' ]; then
    echo "BUTTON"
    TASK_str="press_button_close_after_reaching"
    EXP_NAME=1Task-press_button-CTOD-KP
    PROJECT_NAME=${EXP_NAME}

    RESUME_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/${EXP_NAME}-Batch74/
    RESUME_STEP=72675
    RESUME=false
elif [ "$TASK_NAME" == 'stack_block' ]; then
    echo "STACK_BLOCK"
    TASK_str="stack_block"
    EXP_NAME=1Task-${TASK_str}-CTOD-KP_NO_0_3_5
    PROJECT_NAME=${EXP_NAME}
    SET_SAME_N=7
    RESUME_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/${EXP_NAME}-Batch74/
    RESUME_STEP=72675
    RESUME=false
elif [ "$TASK_NAME" == 'pick_place' ]; then
    echo "Pick-Place"
    TASK_str="pick_place"
    EXP_NAME=Real-1Task-${TASK_str}-KP-Finetune
    PROJECT_NAME=${EXP_NAME}
    SET_SAME_N=2
    RESUME_PATH=/home/rsofnc000/checkpoint_save_folder/Real-1Task-pick_place-KP-Finetune-Batch32 #/home/rsofnc000/checkpoint_save_folder/1Task-Pick-Place-KP-Batch112
    RESUME_STEP=6
    RESUME=true
    FINETUNE=false
elif [ "$TASK_NAME" == 'multi' ]; then
    echo "Multi Task"
    TASK_str=["pick_place","nut_assembly","stack_block","press_button_close_after_reaching"]
    EXP_NAME=4Task-CTOD-KP #1Task-${TASK_str}-CTOD-KP
    PROJECT_NAME=${EXP_NAME}

    RESUME_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/${EXP_NAME}-Batch74/
    RESUME_STEP=72675
    RESUME=false
fi

while true; do
    highest_epoch=0

    if [ "$RESUME" = true ]; then
        FOLDER_PATH=${RESUME_PATH}
        while IFS= read -r file; do
            if [[ $file =~ model_save-([0-9]+)\.pt ]]; then
                epoch_number=${BASH_REMATCH[1]}
                if ((epoch_number > highest_epoch)); then
                    highest_epoch=$epoch_number
                fi
            fi
        done < <(find "$FOLDER_PATH" -type f -name 'model_save-*.pt')

        RESUME_STEP=$highest_epoch
        echo "Highest epoch number found: $RESUME_STEP"
    fi

    if ((highest_epoch != EPOCH && highest_epoch != EPOCH - 1)); then
        echo "Running srun command..."
        srun --output=training_${EXP_NAME}.txt --job-name=training_${EXP_NAME} python -u ../training/train_scripts/train_any.py \
            --config-path ${CONFIG_PATH} \
            --config-name ${CONFIG_NAME} \
            policy=${POLICY} \
            device=${DEVICE} \
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
            finetune=${FINETUNE} \
            dataset_cfg.agent_name=${AGENT_NAME} \
            dataset_cfg.obs_T=${OBS_T} \
            dataset_cfg.non_sequential=${NON_SEQUENTIAL} \
            dataset_cfg.compute_obj_distribution=${COMPUTE_OBJ_DISTRIBUTION} \
            dataset_cfg.first_frames=${FIRST_FRAMES} \
            dataset_cfg.only_first_frame=${ONLY_FIRST_FRAMES} \
            dataset_cfg.height=${HEIGHT} \
            dataset_cfg.width=${WIDTH} \
            dataset_cfg.perform_augs=${PERFORM_AUGS} \
            dataset_cfg.mix_sim_real=false \
            dataset_cfg.dagger=${DAGGER} \
            samplers.balancing_policy=${BALANCING_POLICY} \
            early_stopping_cfg.patience=${EARLY_STOPPING_PATIECE} \
            cond_target_obj_detector_cfg.height=${HEIGHT} \
            cond_target_obj_detector_cfg.width=${WIDTH} \
            cond_target_obj_detector_cfg.dim_H=${DIM_H} \
            cond_target_obj_detector_cfg.dim_W=${DIM_W} \
            cond_target_obj_detector_cfg.n_channels=${OUT_FEATURE} \
            cond_target_obj_detector_cfg.conv_drop_dim=${DROP_DIM} \
            cond_target_obj_detector_cfg.n_classes=${N_CLASSES} \
            project_name=${PROJECT_NAME} \
            EXPERT_DATA=${EXPERT_DATA} \
            save_path=${SAVE_PATH} \
            resume_path=${RESUME_PATH} \
            resume_step=${RESUME_STEP} \
            optimizer=${OPTIMIZER} \
            train_cfg.lr=${LR} \
            train_cfg.weight_decay=${WEIGHT_DECAY} \
            train_cfg.lr_schedule=${SCHEDULER} \
            debug=${DEBUG} \
            wandb_log=${WANDB_LOG} \
            resume=${RESUME} \
            loader_workers=${LOADER_WORKERS}
    else
        echo "The highest epoch number ($highest_epoch) is equal to EPOCH ($EPOCH) or EPOCH-1 ($((EPOCH - 1))). Exiting loop."
        break
    fi

    RESUME=true
    FOLDER_PATH=${SAVE_PATH}/${EXP_NAME}-Batch${BSIZE}
    sleep 1 # Optional: Add a sleep to avoid tight loop
done
