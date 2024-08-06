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

echo $1
TASK_NAME="$1"
NUM_WORKERS=10
GPU_ID=0

BASE_PATH=/home/rsofnc000/Multi-Task-LFD-Framework
CKP_FOLDER=/home/rsofnc000/checkpoint_save_folder
if [ "$TASK_NAME" == 'pick_place' ]; then
    PROJECT_NAME=1Task-pick_place-MOSAIC_State_pos_gripper
    BATCH=32 #32
    MODEL_PATH=${CKP_FOLDER}/${PROJECT_NAME}-Batch${BATCH}
    CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json
    for MODEL in ${MODEL_PATH}; do
        for S in 288630; do #81000 89100; do
            for TASK in pick_place; do
                for COUNT in 1 2 3; do
                    if [ $COUNT -eq 1 ]; then
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        srun --output=${PROJECT_NAME}.txt --job-name=${PROJECT_NAME} python -u $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --save_path ${SAVE_PATH} --save_files
                    else
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        srun --output=${PROJECT_NAME}.txt --job-name=${PROJECT_NAME} python -u $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log
                    fi
                done
            done
        done
    done
elif [ "$TASK_NAME" == 'nut_assembly' ]; then
    PROJECT_NAME=1Task-nut_assembly-MOSAIC-State_pos_gripper
    BATCH=27 #32
    MODEL_PATH=${CKP_FOLDER}/${PROJECT_NAME}-Batch${BATCH}
    CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json
    for MODEL in ${MODEL_PATH}; do
        for S in 164032; do #81000 89100; do
            for TASK in nut_assembly; do
                for COUNT in 1 2 3; do
                    if [ $COUNT -eq 1 ]; then
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        srun --output=${PROJECT_NAME}.txt --job-name=${PROJECT_NAME} python -u $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --save_path ${SAVE_PATH} --save_files
                    else
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        srun --output=${PROJECT_NAME}.txt --job-name=${PROJECT_NAME} python -u $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log
                    fi
                done
            done
        done
    done
elif [ "$TASK_NAME" == 'button' ]; then
    PROJECT_NAME=1Task-press_button-MOSAIC-State_true
    BATCH=18 #32
    MODEL_PATH=${CKP_FOLDER}/${PROJECT_NAME}-Batch${BATCH}
    CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json
    for MODEL in ${MODEL_PATH}; do
        for S in 106304; do #81000 89100; do
            for TASK in button; do
                for COUNT in 1 2 3; do
                    if [ $COUNT -eq 1 ]; then
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        srun --output=${PROJECT_NAME}.txt --job-name=${PROJECT_NAME} python -u $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --save_path ${SAVE_PATH} --save_files
                    else
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        srun --output=${PROJECT_NAME}.txt --job-name=${PROJECT_NAME} python -u $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log
                    fi
                done
            done
        done
    done
elif [ "$TASK_NAME" == 'stack_block' ]; then
    PROJECT_NAME=1Task-stack_block-MOSAIC-State_true
    BATCH=18 #32
    MODEL_PATH=${CKP_FOLDER}/${PROJECT_NAME}-Batch${BATCH}
    CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json
    for MODEL in ${MODEL_PATH}; do
        for S in 157036; do
            for TASK in stack_block; do
                for COUNT in 1 2 3; do
                    if [ $COUNT -eq 1 ]; then
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        srun --output=${PROJECT_NAME}.txt --job-name=${PROJECT_NAME} python -u $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --save_path ${SAVE_PATH} --save_files
                    else
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        srun --output=${PROJECT_NAME}.txt --job-name=${PROJECT_NAME} python -u $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log
                    fi
                done
            done
        done
    done
elif [ "$TASK_NAME" == 'multi' ]; then
    echo "Multi Task"
    PROJECT_NAME=Multi-Task-MOSAIC-Generalization
    BATCH=52 #32
    MODEL_PATH=${CKP_FOLDER}/${PROJECT_NAME}-Batch${BATCH}
    CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

    for MODEL in ${MODEL_PATH}; do
        for S in 239105; do
            for TASK in stack_block pick_place nut_assembly; do
                for COUNT in 1 2 3; do
                    if [ $COUNT -eq 1 ]; then
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        srun --output=${PROJECT_NAME}.txt --job-name=${PROJECT_NAME} python -u $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --save_path ${SAVE_PATH} --save_files
                    else
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        srun --output=${PROJECT_NAME}.txt --job-name=${PROJECT_NAME} python -u $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log
                    fi
                done
            done
        done
    done

fi
