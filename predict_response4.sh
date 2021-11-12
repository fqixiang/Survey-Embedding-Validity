#!/bin/bash

# RUNTIME SETTINGS

# GENERAL
#SBATCH --job-name=prediction4.job
#SBATCH --time=100:00:00
#SBATCH --mem=60G

# CPU
#SBATCH --cpus-per-task=12

# OUTPUT
#SBATCH --error=sbatch_files/errors/error4.txt
#SBATCH --output=sbatch_files/outputs/output4.txt

/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data None --model rf --other_feature random --dim_size 1024 --save_model False