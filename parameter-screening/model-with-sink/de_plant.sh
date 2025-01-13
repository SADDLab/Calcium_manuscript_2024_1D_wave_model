#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=dumulis
#SBATCH --mem=32G
#SBATCH --job-name=plant_de_job
#SBATCH --time=2-00:00:00

module load anaconda/2024.02-py311
conda activate nk_env_test
python main_v6.py
