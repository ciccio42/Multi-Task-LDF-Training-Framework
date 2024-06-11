export CUDA_VISIBLE_DEVICES=0,1,2,3
export MUJOCO_PY_MUJOCO_PATH="/home/frosa_Loc/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
NUM_WORKERS=7
GPU_ID=1
CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

### --- Multi-Task ---####

BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
PROJECT_NAME=4Task-TOSIL-multi-task
BATCH=74
MODEL_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/${PROJECT_NAME}-Batch${BATCH}/
STEP=-1

# for MODEL in ${MODEL_PATH}; do
#     for S in 253890; do #81000 89100; do
#         for TASK in pick_place nut_assembly stack_block button; do
#             echo Task ${TASK}
#             for COUNT in 1 2 3; do
#                 if [ $COUNT -eq 1 ]; then
#                     SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
#                     python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --save_path ${SAVE_PATH} --save_files
#                 else
#                     SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
#                     python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log #--sub_action --gt_action 2
#                 fi
#             done
#         done
#     done
# done

## --- Single-Task GT-ACTION ---####
for MODEL in ${MODEL_PATH}; do
    for S in 253890; do #81000 89100; do
        for TASK in pick_place nut_assembly stack_block button; do
            echo Task ${TASK}
            for COUNT in 1 2 3; do
                if [ $COUNT -eq 1 ]; then
                    SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}_gt_2
                    python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --sub_action --gt_action 2
                else
                    SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}_gt_2
                    python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --sub_action --gt_action 2
                fi
            done

            for COUNT in 1 2 3; do
                if [ $COUNT -eq 1 ]; then
                    SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}_gt_10
                    python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --sub_action --gt_action 10
                else
                    SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}_gt_10
                    python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --sub_action --gt_action 10
                fi
            done

            for COUNT in 1 2 3; do
                if [ $COUNT -eq 1 ]; then
                    SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}_gt_18
                    python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --sub_action --gt_action 18
                else
                    SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}_gt_18
                    python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --sub_action --gt_action 18
                fi
            done

        done
    done
done
