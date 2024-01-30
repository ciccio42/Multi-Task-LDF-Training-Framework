export CUDA_VISIBLE_DEVICES=0,1,2,3
export MUJOCO_PY_MUJOCO_PATH="/home/frosa_Loc/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
# PROJECT_NAME=#2Task-Nut-Assembly-Pick-Place-100-180 #1Task-Pick-Place-100-180-All-Obj-One-Task-Left
# NUM_WORKERS=3
# GPU_ID=0
# MODEL_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/2Task-Nut-Assembly-Pick-Place-100-180-Batch50/
# CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

# for MODEL in ${MODEL_PATH}; do
#     for S in -1; do #81000 89100; do
#         for TASK in pick_place; do
#             echo Task ${TASK};
#             for COUNT in 1 2 3; do    
#                 if [ $COUNT -eq 1 ]; then
#                     SAVE_PATH=${MODEL_PATH}/results_${TASK}/mixed_obj/run_${COUNT}
#                     python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} #--eval_subsets 12 --wandb_log --save_path ${SAVE_PATH} --save_files 
#                 else
#                     SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
#                     python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} #--eval_subsets 12 --wandb_log
#                 fi
#             done
#         done
#     done
# done


BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
PROJECT_NAME=1Task-Pick-Place-Mosaic-cropped-no-normalized #1Task-MOSAIC-STACK-BLOCk
#2Task-Nut-Assembly-Pick-Place-100-180
NUM_WORKERS=7
GPU_ID=1
MODEL_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/${PROJECT_NAME}-Batch32/
CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

# for MODEL in ${MODEL_PATH}; do
#     for S in 120000; do #81000 89100; do
#         for TASK in pick_place; do
#             echo Task ${TASK}
#             for COUNT in 1 2 3; do    
#                 if [ $COUNT -eq 1 ]; then
#                     SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
#                     python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --debug #--wandb_log --save_path ${SAVE_PATH} --save_files
#                 else
#                     SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
#                     python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log
#                 fi
#             done
#         done
#     done
# done

for MODEL in ${MODEL_PATH}; do
    for S in 120000; do #81000 89100; do
        for TASK in pick_place; do
            echo Task ${TASK}
            for GT_ACTION in 2 10 18; do
                for COUNT in 1 2 3; do    
                    if [ $COUNT -eq 1 ]; then
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/gt_action_${GT_ACTION}/run_${COUNT}
                        python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --sub_action --gt_action ${GT_ACTION} --wandb_log --save_path ${SAVE_PATH} --save_files
                    else
                        SAVE_PATH=${MODEL_PATH}/results_${TASK}/gt_action_${GT_ACTION}/run_${COUNT}
                        python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --wandb_log --sub_action --gt_action ${GT_ACTION}
                    fi
                done
            done
        done
    done
done