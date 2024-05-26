#!/bin/sh
# export MUJOCO_PY_MUJOCO_PATH=/user/frosa/.mujoco/mujoco210
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user/frosa/.mujoco/mujoco210/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user/frosa/miniconda3/envs/multi_task_lfd/lib
export MUJOCO_PY_MUJOCO_PATH="/home/frosa_Loc/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HYDRA_FULL_ERROR=1

### --- Single-Task --- ###
# BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
# PROJECT_NAME=1Task-stack_block-CTOD-KP
# BATCH=36
# NUM_WORKERS=7
# GPU_ID=2
# MODEL_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/${PROJECT_NAME}-Batch${BATCH}
# CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

# for MODEL in ${MODEL_PATH}; do
#     for S in -1; do
#         for TASK in stack_block; do
#             SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
#             python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log #--save_path ${SAVE_PATH} --wandb_log #--save_files #--wandb_log
#         done
#     done
# done

### --- Multi-Task --- ###
BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
PROJECT_NAME=4Task-CTOD-KP
BATCH=74
NUM_WORKERS=7
GPU_ID=3
MODEL_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/${PROJECT_NAME}-Batch${BATCH}
CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

for MODEL in ${MODEL_PATH}; do
    for S in 53550 61200; do
        for TASK in nut_assembly pick_place button stack_block; do
            SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
            python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log #--save_path ${SAVE_PATH} --save_files --wandb_log #--save_path ${SAVE_PATH} --wandb_log #--save_files #--wandb_log
        done
    done
done
