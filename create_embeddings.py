from sentence_transformers import SentenceTransformer
import pickle
import pandas as pd
import argparse
import fasttext.util
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# function to create random embeddings
def make_random_embeddings(questions_ls, size, seed):
    questions_vocab = []
    for question in questions_ls:
        question_tokenized = question.lower().split()
        questions_vocab = questions_vocab + question_tokenized

    questions_vocab = [s.translate(str.maketrans('', '', string.punctuation)) for s in questions_vocab]
    questions_vocab = set(questions_vocab)
    questions_vocab = sorted(questions_vocab)

    # assign random embedding to each word
    random_word_embeddings = {}

    rng = np.random.default_rng(seed=seed)
    for word in questions_vocab:
        random_word_embeddings[word] = rng.uniform(-1, 1, size)

    # make random sentence embeddings
    random_sent_embeddings_ls = []
    for rfa in questions_ls:
        rfa_tokenized = rfa.lower().split()
        rfa_ready = [s.translate(str.maketrans('', '', string.punctuation)) for s in rfa_tokenized]
        rfa_word_embeddings = []
        for word in rfa_ready:
            rfa_word_embeddings.append(random_word_embeddings[word])
        rfa_word_embeddings = np.array(rfa_word_embeddings)
        rfa_sent_embeddings = np.mean(rfa_word_embeddings, axis=0)
        random_sent_embeddings_ls.append(rfa_sent_embeddings)

    # make the data frame
    embeddings_df = pd.DataFrame(data=random_sent_embeddings_ls,
                                 columns=["dim%d" % (i + 1) for i in range(size)])

    return(embeddings_df)

# %% function to make count or tf-idf features based dfs
def simple_vectorizer(questions_ls, type):
    if type == "count":
        vectorizer = CountVectorizer(encoding='utf-8',
                                     lowercase=True,
                                     analyzer='word',
                                     stop_words='english',
                                     max_df=1.0,
                                     min_df=1,
                                     max_features=None)
    else:
        vectorizer = TfidfVectorizer(encoding='utf-8',
                                     lowercase=True,
                                     analyzer='word',
                                     stop_words='english',
                                     max_df=1.0,
                                     min_df=1,
                                     max_features=None)

    vectorizer.fit(questions_ls)
    feature_names = vectorizer.get_feature_names()
    for i in range(len(feature_names)):
        feature_names[i] = "dim_" + feature_names[i]

    embeddings_df = pd.DataFrame(vectorizer.transform(questions_ls).toarray(),
                                 columns=feature_names)

    return(embeddings_df)

# %% function to create embeddings from survey questions
def create_embeddings(sentences, question_ids, model_name, save_name, save=True, size=None, seed=None):
    if model_name == 'fasttext':
        # download and load model
        fasttext.util.download_model('en', if_exists='ignore')
        ft = fasttext.load_model('cc.en.300.bin')

        embeddings = []
        for sent in sentences:
            # tokenize the sentences
            sent = sent.translate(str.maketrans('', '', string.punctuation))
            sent = sent.split(" ")

            word_vec_ls = []
            for word in sent:
                word_vec_ls.append(ft.get_word_vector(word))

            # compute average sentence embedding
            sent_vec = np.mean(word_vec_ls, axis=0)
            embeddings.append(sent_vec)

        embedding_df = pd.DataFrame(data=embeddings,
                                    columns=["dim%d" % (i + 1) for i in range(len(embeddings[0]))])

    elif model_name == 'random':
        embedding_df = make_random_embeddings(sentences, size, seed)

    elif model_name == 'tf_idf':
        embedding_df = simple_vectorizer(sentences, type='tf_idf')

    elif model_name == 'count':
        embedding_df = simple_vectorizer(sentences, type='count')

    else:
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
        if model_name == 'random':
            save_path = './data/embeddings/' + save_name + '_' + model_name.replace("-", "_") + str(size) + '.pkl'
        else:
            save_path = './data/embeddings/' + save_name + '_' + model_name.replace("-", "_") + '.pkl'
        embedding_df.to_pickle(save_path,
                               protocol=pickle.HIGHEST_PROTOCOL)

    return embedding_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile",
                        type=str,
                        default=None)
    parser.add_argument("--model",
                        type=str,
                        default=None)
    parser.add_argument("--size",
                        type=int,
                        default=None)
    parser.add_argument("--seed",
                        type=int,
                        default=None)
    parser.add_argument("--savename",
                        type=str,
                        default=None)
    parser.add_argument("--save",
                        type=str,
                        default=True)


    args = parser.parse_args()

    model = args.model
    save = args.save
    savename = args.savename
    size = args.size
    seed = args.seed

    if "synthetic" in savename or "Synthetic" in savename:
        datafile = './data/synthetic/' + args.datafile

    elif "ESS" in savename or "ess" in savename:
        datafile = './data/ESS/' + args.datafile

    else:
        print("Wrong data file name. Should contain 'synthetic'")
        exit()

    if "xlsx" in datafile:
        questions_df = pd.read_excel(datafile)
    else:
        print("Only excel data files are supported.")
        exit()

    if "synthetic" in savename or "Synthetic" in savename:
        questions = questions_df.rfa.to_list()
        question_names = questions_df.row_id.to_list()

    elif "ESS" in savename or "ess" in savename:
        questions = questions_df.question_UK.to_list()
        question_names = questions_df.name.to_list()

    create_embeddings(sentences=questions,
                      question_ids=question_names,
                      model_name=model,
                      save_name=savename,
                      save=save,
                      size=size,
                      seed=seed)

if __name__ == '__main__':
    main()