#!/bin/bash

# RUNTIME SETTINGS

# GENERAL
#SBATCH --job-name=prediction.job
#SBATCH --time=100:00:00
#SBATCH --mem=50G

# CPU
#SBATCH --cpus-per-task=12

# OUTPUT
#SBATCH --error=sbatch_files/errors/error.txt
#SBATCH --output=sbatch_files/outputs/output.txt

/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data None --other_feature count --model lasso --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data None --other_feature count --model ridge --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data None --other_feature count --model rf --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data None --other_feature tf_idf --model lasso --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data None --other_feature tf_idf --model ridge --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data None --other_feature tf_idf --model rf --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data None --model lasso --other_feature random --dim_size 300 --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data None --model ridge --other_feature random --dim_size 300 --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data None --model rf --other_feature random --dim_size 300 --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data None --model lasso --other_feature random --dim_size 768 --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data None --model ridge --other_feature random --dim_size 768 --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data None --model rf --other_feature random --dim_size 768 --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data None --model lasso --other_feature random --dim_size 1024 --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data None --model ridge --other_feature random --dim_size 1024 --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data None --model rf --other_feature random --dim_size 1024 --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_fasttext --model lasso --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_bert_base_uncased --model lasso --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_bert_large_uncased --model lasso --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_paraphrase_mpnet_base_v2 --model lasso --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_stsb_mpnet_base_v2 --model lasso --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_stsb_roberta_base_v2 --model lasso --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_stsb_roberta_large --model lasso --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_fasttext --model ridge --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_bert_base_uncased --model ridge --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_bert_large_uncased --model ridge --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_paraphrase_mpnet_base_v2 --model ridge --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_stsb_mpnet_base_v2 --model ridge --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_stsb_roberta_base_v2 --model ridge --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_stsb_roberta_large --model ridge --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_fasttext --model rf --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_bert_base_uncased --model rf --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_bert_large_uncased --model rf --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_paraphrase_mpnet_base_v2 --model rf --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_stsb_mpnet_base_v2 --model rf --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_stsb_roberta_base_v2 --model rf --save_model False
/hpc/local/CentOS7/uu_cs_nlpsoc/conda/envs/qf_survey_embeddings_venv/bin/python predict_response.py --embeddings_data ESS_stsb_roberta_large --model rf --save_model False