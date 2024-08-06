#!/bin/sh
# export MUJOCO_PY_MUJOCO_PATH=/user/frosa/.mujoco/mujoco210
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user/frosa/.mujoco/mujoco210/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user/frosa/miniconda3/envs/multi_task_lfd/lib
# export MUJOCO_PY_MUJOCO_PATH="/home/frosa_Loc/.mujoco/mujoco210"
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export HYDRA_FULL_ERROR=1

BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
PROJECT_NAME=Real-Pick-Place-MOSAIC-CTOD-Only-Front-Pos-State-Info-Reduced-Space-Extended-Next-Action
#1Task-Pick-Place-Cond-Target-Obj-Detector-separate-demo-agent #
BATCH=5
NUM_WORKERS=1
GPU_ID=1
MODEL_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/${PROJECT_NAME}-Batch${BATCH}/
CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

for MODEL in ${MODEL_PATH}; do
    for S in 45180; do #81000 89100; do
        for TASK in pick_place; do
            for COUNT in 1; do
                SAVE_PATH=${MODEL}/results_${TASK}/val/run_${COUNT}
                python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_real_world_dataset.py $MODEL --env $TASK --saved_step $S --eval_each_task 1 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --eval_subsets 1 --wandb_log
            done
        done
    done
done

#### ---- ####
# BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
# PROJECT_NAME=Real-Pick-Place-MOSAIC-CTOD-Only-Front-No-State-Info-Reduced-Space-Extended
# #1Task-Pick-Place-Cond-Target-Obj-Detector-separate-demo-agent #
# BATCH=5
# NUM_WORKERS=10
# GPU_ID=1
# MODEL_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/${PROJECT_NAME}-Batch${BATCH}/
# CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

# for MODEL in ${MODEL_PATH}; do
#     for S in 207180; do #81000 89100; do
#         for TASK in pick_place; do
#             for COUNT in 1; do
#                 SAVE_PATH=${MODEL}/results_${TASK}/val/run_${COUNT}
#                 python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_real_world_dataset.py $MODEL --env $TASK --saved_step $S --eval_each_task 1 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --eval_subsets 1 --wandb_log  #--save_path ${SAVE_PATH} --save_files --wandb_log
#             done
#         done
#     done
# done

# #### ---- ####
# BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
# PROJECT_NAME=Real-Pick-Place-MOSAIC-CTOD-Only-Front-Pos-State-Info-Reduced-Space-Extended-Next-Act
# #1Task-Pick-Place-Cond-Target-Obj-Detector-separate-demo-agent #
# BATCH=5
# NUM_WORKERS=10
# GPU_ID=1
# MODEL_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/${PROJECT_NAME}-Batch${BATCH}/
# CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

# for MODEL in ${MODEL_PATH}; do
#     for S in 207180; do #81000 89100; do
#         for TASK in pick_place; do
#             for COUNT in 1; do
#                 SAVE_PATH=${MODEL}/results_${TASK}/val/run_${COUNT}
#                 python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_real_world_dataset.py $MODEL --env $TASK --saved_step $S --eval_each_task 1 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --eval_subsets 1 --wandb_log  #--save_path ${SAVE_PATH} --save_files --wandb_log
#             done
#         done
#     done
# done

# #### ---- ####
# BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
# PROJECT_NAME=Real-Pick-Place-MOSAIC-CTOD-Only-Front-No-State-Info-Reduced-Space-Extended-Next-Act
# #1Task-Pick-Place-Cond-Target-Obj-Detector-separate-demo-agent #
# BATCH=5
# NUM_WORKERS=10
# GPU_ID=1
# MODEL_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/${PROJECT_NAME}-Batch${BATCH}/
# CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

# for MODEL in ${MODEL_PATH}; do
#     for S in 207180; do #81000 89100; do
#         for TASK in pick_place; do
#             for COUNT in 1; do
#                 SAVE_PATH=${MODEL}/results_${TASK}/val/run_${COUNT}
#                 python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_real_world_dataset.py $MODEL --env $TASK --saved_step $S --eval_each_task 1 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --eval_subsets 1 --wandb_log  #--save_path ${SAVE_PATH} --save_files --wandb_log
#             done
#         done
#     done
# done
