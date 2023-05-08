export CUDA_VISIBLE_DEVICES=3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

BASE_PATH=/home/frosa_loc/Multi-Task-LFD-Framework
PROJECT_NAME=ur_mosaic_baseline
NUM_WORKERS=1
MODEL_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/mosaic-baseline-sav-folder/ur-baseline/1Task-Pick-Place-Mosaic-No-Obj-Detector-Ur5e-Batch128
CONTROLLER_PATH=/home/frosa_loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/tasks/multi_task_robosuite_env/controllers/config/osc_pose.json

for MODEL in ${MODEL_PATH}; do
    for S in 39468; do
        for TASK in pick_place; do

            python $BASE_PATH/repo/Multi-Task-LFD-Training-Framework/test/test_any_task.py $MODEL --env $TASK --saved_step $S --eval_each_task 10 --num_workers ${NUM_WORKERS} --project_name ${PROJECT_NAME} --controller_path ${CONTROLLER_PATH} --debug
        done
    done
done
