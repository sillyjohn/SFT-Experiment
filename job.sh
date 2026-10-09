#!/bin/bash

#SBATCH --job-name=gemma4_test
#SBATCH -p gpu
#SBATCH --account=yzhao010_1531
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH -o /home1/chunyint/cs566/gemma_%j.out
#SBATCH -e /home1/chunyint/cs566/gemma_%j.err

module purge
module load conda

eval "$(conda shell.bash hook)"
conda activate venv_1

cd /home1/chunyint/cs566

python model_runner.py