#export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user/frosa/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=/user/frosa/miniconda3/envs/multi_task_lfd/lib:$LD_LIBRARY_PATH 

BASE_PATH=/user/frosa/multi_task_lfd/Multi-Task-LFD-Framework
PROJECT_NAME=ur_pick_place_daml
NUM_WORKERS=1
MODEL_PATH=$BASE_PATH/mosaic-baseline-sav-folder/DAML/1Task-Pick-Place-Target-Slot-MAML-224_224-Batch32
CONTROLLER_PATH=$BASE_PATH/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

for MODEL in ${MODEL_PATH}; do
    for S in 81000; do
        for TASK in pick_place; do

            python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/multi_task_test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 1 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --baseline "daml" --debug
        done
    done
done
