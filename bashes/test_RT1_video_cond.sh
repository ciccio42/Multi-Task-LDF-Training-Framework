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

export PYTHONPATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/models/rt1/repo
echo "pythonpath: " $PYTHONPATH

export HYDRA_FULL_ERROR=1

echo $1
TASK_NAME='pick_place' #"$1"
NUM_WORKERS=1 #10
GPU_ID=1 #0

BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
CKP_FOLDER=/raid/home/frosa_Loc/multi_task_lfd/checkpoint_save_folder
if [ "$TASK_NAME" == 'pick_place' ]; then
    PROJECT_NAME=1Task-pick_place-Panda_dem_sim_agent_ur5e_sim_2
    BATCH=32 #32
    MODEL_PATH=${CKP_FOLDER}/${PROJECT_NAME}-Batch${BATCH}
    CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json
    for MODEL in ${MODEL_PATH}; do
        for S in 25656; do #81000 89100 16035; do
            for TASK in pick_place; do
                for COUNT in 1 2 3; do
                    if [ $COUNT -eq 1 ]; then
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        # no wandb_log
                        # no --debug
                        python -u $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py \
                        $MODEL \
                        --env $TASK \
                        --saved_step $S \
                        --eval_each_task 10 \
                        --num_workers ${NUM_WORKERS} \
                        --project_name ${PROJECT_NAME} \
                        --controller_path ${CONTROLLER_PATH} \
                        --gpu_id ${GPU_ID} \
                        --save_path ${SAVE_PATH} \
                        --save_files
                        # with wandb_log
                        # python -u $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --debug --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --save_path ${SAVE_PATH} --save_files
                        # srun --output=${PROJECT_NAME}.txt --job-name=${PROJECT_NAME} python -u $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --save_path ${SAVE_PATH} --save_files
                    else
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        srun --output=${PROJECT_NAME}.txt --job-name=${PROJECT_NAME} python -u $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log
                    fi
                done
            done
        done
    done
fi
