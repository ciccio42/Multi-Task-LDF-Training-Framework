#!/bin/sh
# export MUJOCO_PY_MUJOCO_PATH=/user/frosa/.mujoco/mujoco210
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user/frosa/.mujoco/mujoco210/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user/frosa/miniconda3/envs/multi_task_lfd/lib
export MUJOCO_PY_MUJOCO_PATH="/home/frosa_Loc/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HYDRA_FULL_ERROR=1

BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
PROJECT_NAME=Real-1Task-Pick-Place-Cond-Target-Obj-Detector
NUM_WORKERS=1
MODEL_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/Real-1Task-Pick-Place-Cond-Target-Obj-Detector-Batch8/
CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

for MODEL in ${MODEL_PATH}; do
    for S in 36000; do #81000 89100; do   
        for TASK in pick_place; do
            for COUNT in 1; do
                SAVE_PATH=${MODEL}/results_${TASK}/run_${COUNT}
                python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_real_world_dataset.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id 0 --eval_subsets 4 --debug #--wandb_log --save_path ${SAVE_PATH} --save_files 
            done
        done
    done
done

