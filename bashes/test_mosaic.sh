export CUDA_VISIBLE_DEVICES=1
export MUJOCO_PY_MUJOCO_PATH="/home/frosa_Loc/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
PROJECT_NAME=1Task-Nut-Assembly-100-180-PREDICTED-BB-2
NUM_WORKERS=10
MODEL_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/1Task-Pick-Place-100-180-BB-inference-Batch32/1Task-Pick-Place-100-180-BB-inference-Batch32
CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json
SAVE_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/1Task-Pick-Place-100-180-BB-inference-Batch32/1Task-Pick-Place-100-180-BB-inference-Batch32/result_pick_place

for MODEL in ${MODEL_PATH}; do
    for S in  85050; do #81000 89100; do
        for TASK in pick_place; do

            python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id 0  --wandb_log --save_path ${SAVE_PATH} --save_files 
        done
    done
done

for MODEL in ${MODEL_PATH}; do
    for S in  85050 85050; do #81000 89100; do
        for TASK in pick_place; do

            python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id 0  --wandb_log #--save_path ${SAVE_PATH} --save_files 
        done
    done
done

# BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
# PROJECT_NAME=1Task-Nut-Assembly-100-180-PREDICTED-BB-2
# NUM_WORKERS=8
# MODEL_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/1Task-Nut-Assembly-100-180-PREDICTED-BB-2-Batch27
# CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json
# SAVE_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/1Task-Nut-Assembly-100-180-PREDICTED-BB-2-Batch27/result_nut_assembly_gt_bb

# for MODEL in ${MODEL_PATH}; do
#     for S in  94500; do #81000 89100; do
#         for TASK in nut_assembly; do

#             python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id 0 --gt_bb --wandb_log --save_path ${SAVE_PATH} --save_files 
#         done
#     done
# done

# for MODEL in ${MODEL_PATH}; do
#     for S in  94500 94500; do #81000 89100; do
#         for TASK in nut_assembly; do

#             python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id 0 --gt_bb --wandb_log #--save_path ${SAVE_PATH} --save_files 
#         done
#     done
# done
