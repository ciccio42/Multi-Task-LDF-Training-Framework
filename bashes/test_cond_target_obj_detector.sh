export CUDA_VISIBLE_DEVICES=3
export MUJOCO_PY_MUJOCO_PATH="/home/frosa_Loc/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
# PROJECT_NAME=1Task-Nut-Assembly-Cond-Target-Obj-Detector-separate-demo-agent
# NUM_WORKERS=3
# MODEL_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/1Task-Nut-Assembly-Cond-Target-Obj-Detector-separate-demo-agent-Batch54
# CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

# for MODEL in ${MODEL_PATH}; do
#     for S in 60030 63945; do
#         for TASK in nut_assembly; do

#             python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH}  --gpu_id 0 --wandb_log --save_files --wandb_log

#         done
#     done
# done

# BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
# PROJECT_NAME=1Task-Pick-Place-Cond-Target-Obj-Detector-separate-demo-agent
# NUM_WORKERS=5
# MODEL_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/1Task-Pick-Place-Cond-Target-Obj-Detector-separate-demo-agent-Batch80/
# CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

# for MODEL in ${MODEL_PATH}; do
#     for S in 45198; do
#         for TASK in pick_place; do

#             python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH}  --gpu_id 0 --wandb_log --save_files --wandb_log

#         done
#     done
# done

BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
PROJECT_NAME=2Task-Pick-Place-Nut-Assembly-Cond-Target-Obj-Detector
NUM_WORKERS=5
MODEL_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/2Task-Pick-Place-Nut-Assembly-Cond-Target-Obj-Detector-Batch50/
CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

for MODEL in ${MODEL_PATH}; do
    for S in 40095; do
        for TASK in nut_assembly; do

            python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH}  --gpu_id 0 --wandb_log --save_files 

        done
    done
done
