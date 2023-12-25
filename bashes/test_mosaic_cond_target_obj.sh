export CUDA_VISIBLE_DEVICES=0,1,2,3
export MUJOCO_PY_MUJOCO_PATH="/home/frosa_Loc/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
PROJECT_NAME=2Task-Pick-Place-Nut-Assembly-Mosaic-100-180-Target-Obj-Detector-BB
NUM_WORKERS=7
MODEL_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/2Task-Pick-Place-Nut-Assembly-Mosaic-100-180-Target-Obj-Detector-BB-Batch50/
CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

# for MODEL in ${MODEL_PATH}; do
#     for S in 253814 266046; do #81000 89100; do
#         for TASK in nut_assembly pick_place; do
#             for COUNT in 1 2 3; do
#                 SAVE_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/2Task-Pick-Place-Nut-Assembly-Mosaic-100-180-Target-Obj-Detector-BB-Batch50/results_${TASK}/run_${COUNT}
#                 python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id 0 --wandb_log --save_path ${SAVE_PATH} --save_files
#             done
#         done
#     done
# done

for MODEL in ${MODEL_PATH}; do
    for S in 250756 269104 253814 266046; do #81000 89100; do
        for TASK in nut_assembly pick_place; do
            for COUNT in 1 2 3; do    
                if [ $COUNT -eq 1 ]; then
                    SAVE_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/2Task-Pick-Place-Nut-Assembly-Mosaic-100-180-Target-Obj-Detector-BB-Batch50/results_${TASK}_gt_bb/run_${COUNT}
                    python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id 3 --gt_bb --save_path ${SAVE_PATH} --save_files  --wandb_log
                else
                    SAVE_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/2Task-Pick-Place-Nut-Assembly-Mosaic-100-180-Target-Obj-Detector-BB-Batch50/results_${TASK}_gt_bb/run_${COUNT}
                    python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id 3 --gt_bb --wandb_log
                fi
            done
        done
    done
done
