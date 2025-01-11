#!/bin/bash

export MUJOCO_PY_MUJOCO_PATH=/home/frosa_Loc/.mujoco/mujoco210/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/frosa_Loc/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia


DEBUG=false

if [ "$DEBUG" = true ]; then
    python3 ../training/multi_task_il/datasets/command_encoder/find_min_max_actions.py --debug
elif [ "$DEBUG" = false ]; then
    python3 ../training/multi_task_il/datasets/command_encoder/find_min_max_actions.py
fi