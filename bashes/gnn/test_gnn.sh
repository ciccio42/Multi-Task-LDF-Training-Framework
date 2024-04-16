export CUDA_VISIBLE_DEVICES=0,1,2,3
export MUJOCO_PY_MUJOCO_PATH="/home/frosa_Loc/.mujoco/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

BASE_PATH=/raid/home/frosa_Loc/Multi-Task-LFD-Framework
PROJECT_NAME=Train_GNN_Pick_Place-lr-0.0005
BATCH=48
NUM_WORKERS=1
GPU_ID=3
MODEL_PATH=/user/frosa/multi_task_lfd/checkpoint_save_folder/${PROJECT_NAME}-Batch${BATCH}_first_try/
CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

for MODEL in ${MODEL_PATH}; do
    for S in 2700; do #81000 89100; do
        for TASK in pick_place; do
            echo Task ${TASK}
            for COUNT in 1; do    
                if [ $COUNT -eq 1 ]; then
                    SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                    python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test_gnn/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} --debug #--wandb_log --save_path ${SAVE_PATH} --save_files
                else
                    SAVE_PATH=${MODEL_PATH}/results_${TASK}/run_${COUNT}
                    python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test_gnn/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --gpu_id ${GPU_ID} #--wandb_log
                fi
            done
        done
    done
done



