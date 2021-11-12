# from prepare_survey_data import ESS09_question_names, ESS09_question_texts
from create_embeddings import create_embeddings
import pandas as pd
from sentence_transformers import util

# %% read ESS question meta data
ESS_questions_df = pd.read_excel('./data/ESS09_Ordinal_20210623.xlsx')
print(ESS_questions_df)

# %% UK questions and simplified version
questions_UK = ESS_questions_df.question_UK.to_list()
questions_UK_simple = ESS_questions_df.question_simple_UK.to_list()
questions_names = ESS_questions_df.name.to_list()


# %% model: paraphrase-mpnet-base-v2
UK_question_embedding_df_paraphrase_mpnet_base_v2 = create_embeddings(sentences=questions_UK,
                                                                      question_ids=questions_names,
                                                                      model_name='paraphrase-mpnet-base-v2',
                                                                      save_name='ESS_UK_20210624_',
                                                                      save=True
                                                                      )

UK_question_simple_embedding_df_paraphrase_mpnet_base_v2 = create_embeddings(sentences=questions_UK_simple,
                                                                      question_ids=questions_names,
                                                                      model_name='paraphrase-mpnet-base-v2',
                                                                      save_name='ESS_UK_simple_20210624_',
                                                                      save=True
                                                                      )

# %% model: stsb-mpnet-base-v2
UK_question_embedding_df_stsb_mpnet_base_v2 = create_embeddings(sentences=questions_UK,
                                                              question_ids=questions_names,
                                                              model_name='stsb-mpnet-base-v2',
                                                              save_name='ESS_UK_20210624_',
                                                              save=True
                                                              )

UK_question_simple_embedding_df_stsb_mpnet_base_v2 = create_embeddings(sentences=questions_UK_simple,
                                                              question_ids=questions_names,
                                                              model_name='stsb-mpnet-base-v2',
                                                              save_name='ESS_UK_simple_20210624_',
                                                              save=True
                                                              )

# %% model: stsb-roberta-base-v2
UK_question_embedding_df_stsb_roberta_base_v2 = create_embeddings(sentences=questions_UK,
                                                                question_ids=questions_names,
                                                                model_name='stsb-roberta-base-v2',
                                                                save_name='ESS_UK_20210624_',
                                                                save=True
                                                                )

UK_question_simple_embedding_df_stsb_roberta_base_v2 = create_embeddings(sentences=questions_UK_simple,
                                                                question_ids=questions_names,
                                                                model_name='stsb-roberta-base-v2',
                                                                save_name='ESS_UK_simple_20210624_',
                                                                save=True
                                                                )

# %% model: stsb-roberta-large
UK_question_embedding_df_stsb_roberta_large = create_embeddings(sentences=questions_UK,
                                                              question_ids=questions_names,
                                                              model_name='stsb-roberta-large',
                                                              save_name='ESS_UK_20210624_',
                                                              save=True
                                                              )

UK_question_simple_embedding_df_stsb_roberta_large = create_embeddings(sentences=questions_UK_simple,
                                                              question_ids=questions_names,
                                                              model_name='stsb-roberta-large',
                                                              save_name='ESS_UK_simple_20210624_',
                                                              )



# %% calculate cosine similarity
cosine_ls = list()

for row_id in range(len(synthetic_embedding_df_stsb_roberta_large)):
    embedding_original = synthetic_embedding_df_stsb_roberta_large.iloc[0, :][1:].values.tolist()
    embedding_comparison = synthetic_embedding_df_stsb_roberta_large.iloc[row_id, :][1:].values.tolist()
    cosine_score = util.pytorch_cos_sim(embedding_original,
                                        embedding_comparison)
    cosine_score = cosine_score.squeeze().tolist()

    cosine_ls.append(cosine_score)

# %%
synthetic_df = synthetic_df.assign(cosine_sim=cosine_ls)

# %%
print(synthetic_df)
