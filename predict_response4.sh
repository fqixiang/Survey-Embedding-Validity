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

/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data None --other_feature count --model lasso --output_name results4
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data None --other_feature tf_idf --model lasso --output_name results4
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data None --model lasso --other_feature random --dim_size 300 --output_name results4
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data None --model lasso --other_feature random --dim_size 768 --output_name results4
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data None --model lasso --other_feature random --dim_size 1024 --output_name results4
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_fasttext --model lasso --output_name results4
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_glove --model lasso --output_name results4
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_bert_base_uncased --model lasso --output_name results4
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_bert_large_uncased --model lasso --output_name results4
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_all_distilroberta_v1 --model lasso --output_name results4
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_all_mpnet_base_v2 --model lasso --output_name results4
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_USE --model lasso --output_name results4
