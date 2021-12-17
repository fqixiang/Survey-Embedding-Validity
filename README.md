# Project: Sentence Embeddings for Survey Questions

### Preparation:
- Install the environment 
- Create synthetic data set of survey questions: `python create_synthetic_questions.py`
- Download the ESS data via ...

### 1. Analysis of Content Validity:
**1.1 Generate sentence embeddings**
```shell
python create_embeddings.py --datafile Synthetic_Questions_Controlled_Variants.xlsx --model count --savename synthetic
python create_embeddings.py --datafile Synthetic_Questions_Controlled_Variants.xlsx --model tf_idf --savename synthetic

python create_embeddings.py --datafile Synthetic_Questions_Controlled_Variants.xlsx --model random --size 300 --seed 42 --savename synthetic
python create_embeddings.py --datafile Synthetic_Questions_Controlled_Variants.xlsx --model random --size 768 --seed 42 --savename synthetic
python create_embeddings.py --datafile Synthetic_Questions_Controlled_Variants.xlsx --model random --size 1024 --seed 42 --savename synthetic

python create_embeddings.py --datafile Synthetic_Questions_Controlled_Variants.xlsx --model fasttext --savename synthetic
python create_embeddings.py --datafile Synthetic_Questions_Controlled_Variants.xlsx --model glove --savename synthetic

python create_embeddings.py --datafile Synthetic_Questions_Controlled_Variants.xlsx --model USE --savename synthetic

python create_BERT_embeddings.py --datafile Synthetic_Questions_Controlled_Variants.xlsx --model bert-base-uncased --savename synthetic
python create_BERT_embeddings.py --datafile Synthetic_Questions_Controlled_Variants.xlsx --model bert-large-uncased --savename synthetic

python create_embeddings.py --datafile Synthetic_Questions_Controlled_Variants.xlsx --model all-mpnet-base-v2 --savename synthetic
python create_embeddings.py --datafile Synthetic_Questions_Controlled_Variants.xlsx --model all-distilroberta-v1 --savename synthetic
```
**1.2 Probing analysis**
```shell
python probing.py --features_data Synthetic_Questions_Controlled.xlsx --questions_data Synthetic_Questions_Controlled_Variants.xlsx --embeddings_data synthetic_count.pkl
python probing.py --features_data Synthetic_Questions_Controlled.xlsx --questions_data Synthetic_Questions_Controlled_Variants.xlsx --embeddings_data synthetic_tf_idf.pkl
python probing.py --features_data Synthetic_Questions_Controlled.xlsx --questions_data Synthetic_Questions_Controlled_Variants.xlsx --embeddings_data random --embeddings_size 300
python probing.py --features_data Synthetic_Questions_Controlled.xlsx --questions_data Synthetic_Questions_Controlled_Variants.xlsx --embeddings_data random --embeddings_size 768
python probing.py --features_data Synthetic_Questions_Controlled.xlsx --questions_data Synthetic_Questions_Controlled_Variants.xlsx --embeddings_data random --embeddings_size 1024
python probing.py --features_data Synthetic_Questions_Controlled.xlsx --questions_data Synthetic_Questions_Controlled_Variants.xlsx --embeddings_data synthetic_fasttext.pkl
python probing.py --features_data Synthetic_Questions_Controlled.xlsx --questions_data Synthetic_Questions_Controlled_Variants.xlsx --embeddings_data synthetic_glove.pkl
python probing.py --features_data Synthetic_Questions_Controlled.xlsx --questions_data Synthetic_Questions_Controlled_Variants.xlsx --embeddings_data synthetic_bert_base_uncased.pkl
python probing.py --features_data Synthetic_Questions_Controlled.xlsx --questions_data Synthetic_Questions_Controlled_Variants.xlsx --embeddings_data synthetic_bert_large_uncased.pkl
python probing.py --features_data Synthetic_Questions_Controlled.xlsx --questions_data Synthetic_Questions_Controlled_Variants.xlsx --embeddings_data synthetic_all_distilroberta_v1.pkl
python probing.py --features_data Synthetic_Questions_Controlled.xlsx --questions_data Synthetic_Questions_Controlled_Variants.xlsx --embeddings_data synthetic_all_mpnet_base_v2.pkl
python probing.py --features_data Synthetic_Questions_Controlled.xlsx --questions_data Synthetic_Questions_Controlled_Variants.xlsx --embeddings_data synthetic_USE.pkl
```

### 2. Analysis of Convergent & Discriminant Validity
**2.1 Convert the pickle files to csv**
```shell
python convert_pickle_to_csv.py --datafile synthetic_count
python convert_pickle_to_csv.py --datafile synthetic_tf_idf
python convert_pickle_to_csv.py --datafile synthetic_fasttext
python convert_pickle_to_csv.py --datafile synthetic_glove
python convert_pickle_to_csv.py --datafile synthetic_random300
python convert_pickle_to_csv.py --datafile synthetic_random768
python convert_pickle_to_csv.py --datafile synthetic_random1024
python convert_pickle_to_csv.py --datafile synthetic_bert_base_uncased
python convert_pickle_to_csv.py --datafile synthetic_bert_large_uncased
python convert_pickle_to_csv.py --datafile synthetic_all_distilroberta_v1
python convert_pickle_to_csv.py --datafile synthetic_all_mpnet_base_v2
python convert_pickle_to_csv.py --datafile synthetic_stsb_roberta_base_v2
python convert_pickle_to_csv.py --datafile synthetic_USE
```

**2.2 For the concrete analysis, see ```./analysis/```.**

### 3. Analysis of Criterion Validity
**3.1 Generate sentence embeddings for the survey questions**
```shell
python create_embeddings.py --datafile ESS09_Ordinal_20210623.xlsx --model fasttext --savename ESS
python create_embeddings.py --datafile ESS09_Ordinal_20210623.xlsx --model glove --savename ESS

python create_embeddings.py --datafile ESS09_Ordinal_20210623.xlsx --model USE --savename ESS

python create_BERT_embeddings.py --datafile ESS09_Ordinal_20210623.xlsx --model bert-base-uncased --savename ESS
python create_BERT_embeddings.py --datafile ESS09_Ordinal_20210623.xlsx --model bert-large-uncased --savename ESS

python create_embeddings.py --datafile ESS09_Ordinal_20210623.xlsx --model all-mpnet-base-v2 --savename ESS
python create_embeddings.py --datafile ESS09_Ordinal_20210623.xlsx --model all-distilroberta-v1 --savename ESS
```

**3.2 Prepare ESS survey response data (data cleaning etc.)**
```shell
python prepare_survey_data.py
```

**3.3 Predict responses to survey questions using sentence embeddings**
```shell
python predict_response.py --embeddings_data None --other_feature count --model lasso --output_name results
python predict_response.py --embeddings_data None --other_feature count --model rf --output_name results

python predict_response.py --embeddings_data None --other_feature tf_idf --model lasso --output_name results
python predict_response.py --embeddings_data None --other_feature tf_idf --model rf --output_name results

python predict_response.py --embeddings_data None --model lasso --other_feature random --dim_size 300 --output_name results
python predict_response.py --embeddings_data None --model rf --other_feature random --dim_size 300 --output_name results

python predict_response.py --embeddings_data None --model lasso --other_feature random --dim_size 768 --output_name results
python predict_response.py --embeddings_data None --model rf --other_feature random --dim_size 768 --output_name results

python predict_response.py --embeddings_data None --model lasso --other_feature random --dim_size 1024 --output_name results
python predict_response.py --embeddings_data None --model rf --other_feature random --dim_size 1024 --output_name results

python predict_response.py --embeddings_data ESS_fasttext --model lasso --output_name results
python predict_response.py --embeddings_data ESS_glove --model lasso --output_name results
python predict_response.py --embeddings_data ESS_bert_base_uncased --model lasso --output_name results
python predict_response.py --embeddings_data ESS_bert_large_uncased --model lasso --output_name results
python predict_response.py --embeddings_data ESS_all_distilroberta_v1 --model lasso --output_name results
python predict_response.py --embeddings_data ESS_all_mpnet_base_v2 --model lasso --output_name results
python predict_response.py --embeddings_data ESS_USE --model lasso --output_name results

python predict_response.py --embeddings_data ESS_fasttext --model rf --output_name results
python predict_response.py --embeddings_data ESS_glove --model rf --output_name results
python predict_response.py --embeddings_data ESS_bert_base_uncased --model rf --output_name results
python predict_response.py --embeddings_data ESS_bert_large_uncased --model rf --output_name results
python predict_response.py --embeddings_data ESS_all_distilroberta_v1 --model rf --output_name results
python predict_response.py --embeddings_data ESS_all_mpnet_base_v2 --model rf --output_name results
python predict_response.py --embeddings_data ESS_USE --model rf --output_name results
```

