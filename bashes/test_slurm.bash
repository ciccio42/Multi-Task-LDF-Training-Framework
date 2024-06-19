#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1   # Request 1 GPU
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=test_cuda
#SBATCH --output=test_cuda_output.txt
srun python test_slurm.py
