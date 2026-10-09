#!/bin/bash

#SBATCH --job-name=gemma 4 test run
#SBATCH -p gpu
#SBATCH --account=yzhao010_1531
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH -o /home1/chunyint/cs566/
#SBATCH -e /home1/chunyint/cs566/

module purge
module load julia/1.11.2

source
python model_runner.py
