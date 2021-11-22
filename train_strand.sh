#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH --job-name=all_sims
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=70:00:00
#SBATCH --account=sharmas
#SBATCH --partition=pGPU32
#SBATCH --exclusive
#SBATCH --output=/gpfs/work/sharmas/mc-snow-data/output_logs/slurm-%j.out


srun /gpfs/home/sharmas/miniconda3/envs/warm-rain/bin/python3 train_save.py $1
