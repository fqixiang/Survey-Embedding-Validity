# Project: Sentence Embeddings for Survey Questions

### Preparation:
- Install S-BERT and download the pre-trained models

### Study 1:
**1.1 Create synthetic data set of survey questions**
```shell
python create_synthetic_questions.py
```

**1.2 Generate sentence embeddings**
```shell
python create_embeddings.py --datafile Synthetic_Questions_Controlled_Variants.xlsx --model count --savename synthetic
python create_embeddings.py --datafile Synthetic_Questions_Controlled_Variants.xlsx --model tf_idf --savename synthetic

python create_embeddings.py --datafile Synthetic_Questions_Controlled_Variants.xlsx --model random --size 300 --seed 42 --savename synthetic
python create_embeddings.py --datafile Synthetic_Questions_Controlled_Variants.xlsx --model random --size 768 --seed 42 --savename synthetic
python create_embeddings.py --datafile Synthetic_Questions_Controlled_Variants.xlsx --model random --size 1024 --seed 42 --savename synthetic

python create_embeddings.py --datafile Synthetic_Questions_Controlled_Variants.xlsx --model fasttext --savename synthetic
python create_embeddings.py --datafile Synthetic_Questions_Controlled_Variants.xlsx --model paraphrase-mpnet-base-v2 --savename synthetic
python create_embeddings.py --datafile Synthetic_Questions_Controlled_Variants.xlsx --model stsb-mpnet-base-v2 --savename synthetic
python create_embeddings.py --datafile Synthetic_Questions_Controlled_Variants.xlsx --model stsb-roberta-base-v2 --savename synthetic
python create_embeddings.py --datafile Synthetic_Questions_Controlled_Variants.xlsx --model stsb-roberta-large --savename synthetic
python create_BERT_embeddings.py --datafile Synthetic_Questions_Controlled_Variants.xlsx --model bert-base-uncased --savename synthetic
python create_BERT_embeddings.py --datafile Synthetic_Questions_Controlled_Variants.xlsx --model bert-large-uncased --savename synthetic
```
**1.3 Analysis of cosine similarities**

See ```./analysis/```.

### Study 2:
**2.1 Generate sentence embeddings for the survey questions**
```shell
python create_embeddings.py --datafile ESS09_Ordinal_20210623.xlsx --model fasttext --savename ESS
python create_embeddings.py --datafile ESS09_Ordinal_20210623.xlsx --model paraphrase-mpnet-base-v2 --savename ESS
python create_embeddings.py --datafile ESS09_Ordinal_20210623.xlsx --model stsb-mpnet-base-v2 --savename ESS
python create_embeddings.py --datafile ESS09_Ordinal_20210623.xlsx --model stsb-roberta-base-v2 --savename ESS
python create_embeddings.py --datafile ESS09_Ordinal_20210623.xlsx --model stsb-roberta-large --savename ESS
python create_BERT_embeddings.py --datafile ESS09_Ordinal_20210623.xlsx --model bert-base-uncased --savename ESS
python create_BERT_embeddings.py --datafile ESS09_Ordinal_20210623.xlsx --model bert-large-uncased --savename ESS
```

**2.2 Prepare ESS survey response data (data cleaning etc.)**
```shell
python prepare_survey_data.py
```

**2.3 Predict responses to survey questions using sentence embeddings**
```shell
python predict_response.py --embeddings_data None --other_feature count --model lasso --save_model False
python predict_response.py --embeddings_data None --other_feature count --model ridge --save_model False
python predict_response.py --embeddings_data None --other_feature count --model rf --save_model False

python predict_response.py --embeddings_data None --other_feature tf_idf --model lasso --save_model False
python predict_response.py --embeddings_data None --other_feature tf_idf --model ridge --save_model False
python predict_response.py --embeddings_data None --other_feature tf_idf --model rf --save_model False

python predict_response.py --embeddings_data None --model lasso --other_feature random --dim_size 300 --save_model False
python predict_response.py --embeddings_data None --model ridge --other_feature random --dim_size 300 --save_model False
python predict_response.py --embeddings_data None --model rf --other_feature random --dim_size 300 --save_model False

python predict_response.py --embeddings_data None --model lasso --other_feature random --dim_size 768 --save_model False
python predict_response.py --embeddings_data None --model ridge --other_feature random --dim_size 768 --save_model False
python predict_response.py --embeddings_data None --model rf --other_feature random --dim_size 768 --save_model False

python predict_response.py --embeddings_data None --model lasso --other_feature random --dim_size 1024 --save_model False
python predict_response.py --embeddings_data None --model ridge --other_feature random --dim_size 1024 --save_model False
python predict_response.py --embeddings_data None --model rf --other_feature random --dim_size 1024 --save_model False

python predict_response.py --embeddings_data ESS_fasttext --model lasso --save_model False
python predict_response.py --embeddings_data ESS_bert_base_uncased --model lasso --save_model False
python predict_response.py --embeddings_data ESS_bert_large_uncased --model lasso --save_model False
python predict_response.py --embeddings_data ESS_paraphrase_mpnet_base_v2 --model lasso --save_model False
python predict_response.py --embeddings_data ESS_stsb_mpnet_base_v2 --model lasso --save_model False
python predict_response.py --embeddings_data ESS_stsb_roberta_base_v2 --model lasso --save_model False
python predict_response.py --embeddings_data ESS_stsb_roberta_large --model lasso --save_model False

python predict_response.py --embeddings_data ESS_fasttext --model ridge --save_model False
python predict_response.py --embeddings_data ESS_bert_base_uncased --model ridge --save_model False
python predict_response.py --embeddings_data ESS_bert_large_uncased --model ridge --save_model False
python predict_response.py --embeddings_data ESS_paraphrase_mpnet_base_v2 --model ridge --save_model False
python predict_response.py --embeddings_data ESS_stsb_mpnet_base_v2 --model ridge --save_model False
python predict_response.py --embeddings_data ESS_stsb_roberta_base_v2 --model ridge --save_model False
python predict_response.py --embeddings_data ESS_stsb_roberta_large --model ridge --save_model False

python predict_response.py --embeddings_data ESS_fasttext --model rf --save_model False
python predict_response.py --embeddings_data ESS_bert_base_uncased --model rf --save_model False
python predict_response.py --embeddings_data ESS_bert_large_uncased --model rf --save_model False
python predict_response.py --embeddings_data ESS_paraphrase_mpnet_base_v2 --model rf --save_model False
python predict_response.py --embeddings_data ESS_stsb_mpnet_base_v2 --model rf --save_model False
python predict_response.py --embeddings_data ESS_stsb_roberta_base_v2 --model rf --save_model False
python predict_response.py --embeddings_data ESS_stsb_roberta_large --model rf --save_model False
```

**Content Validity Analysis**
```shell
python probing.py --features_data Synthetic_Questions_Controlled.xlsx --questions_data Synthetic_Questions_Controlled_Variants.xlsx --embeddings_data random --embeddings_size 300
python probing.py --features_data Synthetic_Questions_Controlled.xlsx --questions_data Synthetic_Questions_Controlled_Variants.xlsx --embeddings_data random --embeddings_size 768
python probing.py --features_data Synthetic_Questions_Controlled.xlsx --questions_data Synthetic_Questions_Controlled_Variants.xlsx --embeddings_data random --embeddings_size 1024
python probing.py --features_data Synthetic_Questions_Controlled.xlsx --questions_data Synthetic_Questions_Controlled_Variants.xlsx --embeddings_data synthetic_fasttext.pkl
python probing.py --features_data Synthetic_Questions_Controlled.xlsx --questions_data Synthetic_Questions_Controlled_Variants.xlsx --embeddings_data synthetic_bert_base_uncased.pkl
python probing.py --features_data Synthetic_Questions_Controlled.xlsx --questions_data Synthetic_Questions_Controlled_Variants.xlsx --embeddings_data synthetic_bert_large_uncased.pkl
python probing.py --features_data Synthetic_Questions_Controlled.xlsx --questions_data Synthetic_Questions_Controlled_Variants.xlsx --embeddings_data synthetic_stsb_mpnet_base_v2.pkl
python probing.py --features_data Synthetic_Questions_Controlled.xlsx --questions_data Synthetic_Questions_Controlled_Variants.xlsx --embeddings_data synthetic_paraphrase_mpnet_base_v2.pkl
python probing.py --features_data Synthetic_Questions_Controlled.xlsx --questions_data Synthetic_Questions_Controlled_Variants.xlsx --embeddings_data synthetic_stsb_roberta_base_v2.pkl
python probing.py --features_data Synthetic_Questions_Controlled.xlsx --questions_data Synthetic_Questions_Controlled_Variants.xlsx --embeddings_data synthetic_stsb_roberta_large.pkl
```

**Next Steps**
- Similarity analysis
  - Add baselines
    - Jaccard
    - TF-IDF
    - Static word embeddings
    - Pre-trained BERT
  - Other sentence embeddings techniques
  - Keep pre-training BERT models?

**Just some extra stuff**
```shell
python convert_pickle_to_csv.py --datafile synthetic_count
python convert_pickle_to_csv.py --datafile synthetic_tf_idf
python convert_pickle_to_csv.py --datafile synthetic_fasttext
python convert_pickle_to_csv.py --datafile synthetic_random300
python convert_pickle_to_csv.py --datafile synthetic_random768
python convert_pickle_to_csv.py --datafile synthetic_random1024
python convert_pickle_to_csv.py --datafile synthetic_bert_base_uncased
python convert_pickle_to_csv.py --datafile synthetic_bert_large_uncased
python convert_pickle_to_csv.py --datafile synthetic_paraphrase_mpnet_base_v2
python convert_pickle_to_csv.py --datafile synthetic_stsb_mpnet_base_v2
python convert_pickle_to_csv.py --datafile synthetic_stsb_roberta_base_v2
python convert_pickle_to_csv.py --datafile synthetic_stsb_roberta_large
```