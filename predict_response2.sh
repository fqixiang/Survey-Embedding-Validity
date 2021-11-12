#!/bin/bash

# RUNTIME SETTINGS

# GENERAL
#SBATCH --job-name=prediction2.job
#SBATCH --time=100:00:00
#SBATCH --mem=50G

# CPU
#SBATCH --cpus-per-task=12

# OUTPUT
#SBATCH --error=sbatch_files/errors/error2.txt
#SBATCH --output=sbatch_files/outputs/output2.txt

/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_paraphrase_mpnet_base_v2 --model rf --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_stsb_mpnet_base_v2 --model rf --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_stsb_roberta_base_v2 --model rf --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_stsb_roberta_large --model rf --save_model False