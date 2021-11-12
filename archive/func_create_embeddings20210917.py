from sentence_transformers import SentenceTransformer
import pickle
import pandas as pd


# %% function to create embeddings from survey questions
def create_embeddings(sentences, question_ids, model_name, save_name, save=False) -> object:
    # load model
    model = SentenceTransformer(model_name)

    # encode sentences
    embeddings = model.encode(sentences,
                              batch_size=32,
                              show_progress_bar=True,
                              output_value='sentence_embedding',
                              convert_to_numpy=True,
                              convert_to_tensor=False,
                              device='cpu')

    # turn embeddings into a data frame
    embedding_df = pd.DataFrame(data=embeddings,
                                columns=["dim%d" % (i + 1) for i in range(embeddings.shape[1])])

    # add the name of the sentence as the first column
    embedding_df.insert(loc=0, column='question_id', value=question_ids)

    # save or not
    if save:
        save_path = './data/embeddings_' + save_name + model_name + '.pkl'
        embedding_df.to_pickle(save_path,
                               protocol=pickle.HIGHEST_PROTOCOL)

    return embedding_df


# %% try out the function
# sentences = ['This framework generates embeddings for each input sentence',
#              'Sentences are passed as a list of string.',
#              'The quick brown fox jumps over the lazy dog.']
#
# embedding_df = create_embeddings(sentences=sentences,
#                                  question_ids=['id1', 'id2', 'id3'],
#                                  model_name='stsb-roberta-large',
#                                  save=True)
#
# print(embedding_df)


# %% Input sequence length
# print("Max Sequence Length:", model.max_seq_length)

# Change the length to 200
# model.max_seq_length = 200
