#!/bin/bash

# RUNTIME SETTINGS

# GENERAL
#SBATCH --job-name=prediction3.job
#SBATCH --time=100:00:00
#SBATCH --mem=50G

# CPU
#SBATCH --cpus-per-task=12

# OUTPUT
#SBATCH --error=sbatch_files/errors/error3.txt
#SBATCH --output=sbatch_files/outputs/output3.txt

/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_bert_large_uncased --model rf --output_name results3
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_all_distilroberta_v1 --model rf --output_name results3
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_all_mpnet_base_v2 --model rf --output_name results3
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_USE --model rf --output_name results3