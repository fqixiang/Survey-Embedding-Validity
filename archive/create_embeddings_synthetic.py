# from prepare_survey_data import ESS09_question_names, ESS09_question_texts
from create_embeddings import create_embeddings
import pandas as pd
import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile",
                        type=str,
                        default=None)
    parser.add_argument("--model",
                        type=str,
                        default=None)
    parser.add_argument("--savename",
                        type=str,
                        default=None)

    args = parser.parse_args()
    datafile = args.datafile
    model = args.model
    savename = args.savename

    questions_df = pd.read_excel(datafile)

    questions = questions_df.rfa.to_list()
    question_names = questions_df.row_id.to_list()

    save_path = 'embeddings_' + savename + now +

    embeddings_df = create_embeddings(sentences=questions,
                                      question_ids=question_names,
                                      model_name=model,
                                      save_name=)

# %% model: stsb-roberta-large
synthetic_embedding_df_stsb_roberta_large = create_embeddings(sentences=synthetic_questions,
                                                              question_ids=synthetic_question_names,
                                                              model_name='stsb-roberta-large',
                                                              save_name='synthetic_controlled_20210614_',
                                                              save=True
                                                              )

print(synthetic_embedding_df_stsb_roberta_large)

# %% model: paraphrase-mpnet-base-v2
synthetic_embedding_df_paraphrase_mpnet_base_v2 = create_embeddings(sentences=synthetic_questions,
                                                              question_ids=synthetic_question_names,
                                                              model_name='paraphrase-mpnet-base-v2',
                                                              save_name='synthetic_controlled_20210614_',
                                                              save=True
                                                              )
print(synthetic_embedding_df_paraphrase_mpnet_base_v2)

# %% model: stsb-mpnet-base-v2
synthetic_embedding_df_stsb_mpnet_base_v2 = create_embeddings(sentences=synthetic_questions,
                                                              question_ids=synthetic_question_names,
                                                              model_name='stsb-mpnet-base-v2',
                                                              save_name='synthetic_controlled_20210614_',
                                                              save=True
                                                              )
print(synthetic_embedding_df_stsb_mpnet_base_v2)

# %% model: stsb-roberta-base-v2
synthetic_embedding_df_stsb_roberta_base_v2 = create_embeddings(sentences=synthetic_questions,
                                                              question_ids=synthetic_question_names,
                                                              model_name='stsb-roberta-base-v2',
                                                              save_name='synthetic_controlled_20210614_',
                                                              save=True
                                                              )
print(synthetic_embedding_df_stsb_roberta_base_v2)


# %%
synthetic_embedding_df_stsb_roberta_large.to_csv('./data/synthetic_embedding_df_stsb_roberta_large_20210614.csv',
                                                   index=False)

# %%
synthetic_embedding_df_paraphrase_mpnet_base_v2.to_csv('./data/synthetic_embedding_df_paraphrase_mpnet_base_v2_20210614.csv',
                                                   index=False)
synthetic_embedding_df_stsb_mpnet_base_v2.to_csv('./data/synthetic_embedding_df_paraphrase_stsb_mpnet_base_v2_20210614.csv',
                                                   index=False)
synthetic_embedding_df_stsb_roberta_base_v2.to_csv('./data/synthetic_embedding_df_stsb_roberta_base_v2_20210614.csv',
                                                   index=False)

