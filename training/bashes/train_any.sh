#! bin/bash
export CUDA_VISIBLE_DEVICES=0
DEVICE=0
EXP_NAME=1Task-NUT-ASSEMBLY1
# name of the task to use for training
TASK_str=nut_assembly
EPOCH=20
BSIZE=9

# Mosaic paper parametes
N_MIXTURES=4
ACTIONS_OUT_DIM=256
ACTIONS_HIDDEN_DIM=128

python ../train_scripts/train_any.py policy='${mosaic}' single_task=${TASK_str} exp_name=${EXP_NAME} bsize=${BSIZE} actions.n_mixtures=${N_MIXTURES} actions.out_dim=${ACTIONS_OUT_DIM} actions.hidden_dim=${ACTIONS_HIDDEN_DIM} attn.attn_ff=128  simclr.mul_intm=0 simclr.compressor_dim=128 simclr.hidden_dim=256 epochs=${EPOCH} device=${DEVICE}