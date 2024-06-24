#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MUJOCO_PY_MUJOCO_PATH="/home/frosa_Loc/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

echo $1
TASK_NAME="$1"
NUM_WORKERS=7
GPU_ID=1

if [ "$TASK_NAME" == 'pick_place' ]; then
    BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
    PROJECT_NAME=1Task-pick_place-Double-Policy-Contrastive-false-Inverse-false-CONCAT_IMG_EMB-false-CONCAT_DEMO_EMB-true
    BATCH=32 #32
    MODEL_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/${PROJECT_NAME}-Batch${BATCH}
    CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json
    for MODEL in ${MODEL_PATH}; do
        for S in 259767; do #81000 89100; do
            for TASK in pick_place; do
                for COUNT in 1 2 3; do
                    if [ $COUNT -eq 1 ]; then
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --save_path ${SAVE_PATH} --save_files
                    else
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log
                    fi
                done
            done
        done
    done
elif [ "$TASK_NAME" == 'nut_assembly' ]; then
    BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
    PROJECT_NAME=1Task-nut_assembly-Double-Policy-Contrastive-false-Inverse-false-CONCAT_IMG_EMB-false-CONCAT_DEMO_EMB-true
    BATCH=27 #32
    MODEL_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/from_cluster/checkpoint_save_folder/${PROJECT_NAME}-Batch${BATCH}
    CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json
    for MODEL in ${MODEL_PATH}; do
        for S in 165896; do #81000 89100; do
            for TASK in nut_assembly; do
                for COUNT in 1 2 3; do
                    if [ $COUNT -eq 1 ]; then
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --save_path ${SAVE_PATH} --save_files
                    else
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log
                    fi
                done
            done
        done
    done
elif [ "$TASK_NAME" == 'button' ]; then
    BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
    PROJECT_NAME=1Task-press_button_close_after_reaching-Double-Policy-Contrastive-false-Inverse-false-CONCAT_IMG_EMB-false-CONCAT_DEMO_EMB-true
    BATCH=18 #32
    MODEL_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/from_cluster/checkpoint_save_folder/${PROJECT_NAME}-Batch${BATCH}
    CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json
    for MODEL in ${MODEL_PATH}; do
        for S in 101472; do #81000 89100; do
            for TASK in button; do
                for COUNT in 1 2 3; do
                    if [ $COUNT -eq 1 ]; then
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --save_path ${SAVE_PATH} --save_files
                    else
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log
                    fi
                done
            done
        done
    done
elif [ "$TASK_NAME" == 'stack_block' ]; then
    echo "STACK_BLOCK"
    BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
    PROJECT_NAME=1Task-stack_block-Double-Policy-Contrastive-false-Inverse-false-CONCAT_IMG_EMB-false-CONCAT_DEMO_EMB-true
    BATCH=18 #32
    MODEL_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/from_cluster/checkpoint_save_folder/${PROJECT_NAME}-Batch${BATCH}
    CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json
    for MODEL in ${MODEL_PATH}; do
        for S in 157036; do
            for TASK in stack_block; do
                for COUNT in 1 2 3; do
                    if [ $COUNT -eq 1 ]; then
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --save_path ${SAVE_PATH} --save_files
                    else
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log
                    fi
                done
            done
        done
    done
elif [ "$TASK_NAME" == 'multi' ]; then
    echo "Multi Task"
    BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
    PROJECT_NAME=1Task-Multi-Task-Double-Policy-Contrastive-false-Inverse-false
    BATCH=74 #32
    NUM_WORKERS=7
    GPU_ID=3
    MODEL_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/${PROJECT_NAME}-Batch${BATCH}
    CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

    for MODEL in ${MODEL_PATH}; do
        for S in 234143 251069; do
            for TASK in button stack_block pick_place nut_assembly; do
                for COUNT in 1 2 3; do
                    if [ $COUNT -eq 1 ]; then
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --wandb_log --save_path ${SAVE_PATH} --save_files
                    else
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                        python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log
                    fi
                done
            done
        done
    done

    for MODEL in ${MODEL_PATH}; do
        for S in 234143 251069; do
            for TASK in button stack_block pick_place nut_assembly; do
                for COUNT in 1 2 3; do
                    if [ $COUNT -eq 1 ]; then
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}_gt_bb
                        python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --wandb_log --save_path ${SAVE_PATH} --save_files --gt_bb
                    else
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}_gt_bb
                        python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --gt_bb
                    fi
                done
            done
        done
    done
fi
