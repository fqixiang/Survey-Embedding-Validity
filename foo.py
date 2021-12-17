from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# %%
corpus_train = ["How happy would you say you are?"]
vectorizer = TfidfVectorizer(smooth_idf=False)
X = vectorizer.fit_transform(corpus_train)
print(X)

# %%
feature_names = vectorizer.get_feature_names()

# %%
corpus_new = ["How is your health in general?"]
print(vectorizer.transform(corpus_new))

# %%
pd.DataFrame(vectorizer.transform(corpus_train).toarray(),
             columns=feature_names)



# %%
import numpy as np
import string

# %%
questions_ls = ['hi you', 'hey it is me', 'nioi']

questions_vocab = []
for question in questions_ls:
    question_tokenized = question.lower().split()
    questions_vocab = questions_vocab + question_tokenized

questions_vocab = [s.translate(str.maketrans('', '', string.punctuation)) for s in questions_vocab]
questions_vocab = set(questions_vocab)
questions_vocab = sorted(questions_vocab)
print(questions_vocab)

# %%
embeddings_dict = {}
with open("glove.840B.300d.txt", "r", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = ''.join(values[:-300])
        vector = np.asarray(values[-300:], "float32")
        embeddings_dict[word] = vector

# %%
import torchtext.vocab as vocab
import string
import numpy as np
import pandas as pd

# %%
glove = vocab.GloVe(name='840B', dim=300)

# %%
glove.vectors[glove.stoi['nioi-123']]

# %%
len(embeddings_dict.keys())

# %%
sentences = ['hi you', 'hey it is me', 'nioi']
embeddings = []
for sent in sentences:
    # tokenize the sentences
    sent = sent.translate(str.maketrans('', '', string.punctuation))
    sent = sent.split(" ")

    word_vec_ls = []
    for word in sent:
        vector = glove.vectors[glove.stoi[word]].numpy()
        word_vec_ls.append(vector)

    # compute average sentence embedding
    sent_vec = np.mean(word_vec_ls, axis=0)
    embeddings.append(sent_vec)

embedding_df = pd.DataFrame(data=embeddings,
                            columns=["dim%d" % (i + 1) for i in range(len(embeddings[0]))])

# %%
embedding_df

# %%
try:
    vector = embeddings_dict['nioi']
except KeyError:
    vector = 0

print(vector)


# %%
sent = "my on-the-job experience."
sent = sent.translate(str.maketrans('', '', ".,!?"))
sent = sent.split(" ")
print(sent)


# %%
import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')
ft = fasttext.load_model('cc.en.300.bin')

# %%
import tensorflow_hub as hub
import pandas as pd
# %%
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
embeddings = embed([
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"])


# %%
embedding_df = pd.DataFrame(data=embeddings.numpy(),
                            columns=["dim%d" % (i + 1) for i in range(len(embeddings[0]))])
print(embedding_df)

# %%
import numpy as np
import pandas as pd
# %%
number_ls_1 = [1,2,3,4]
number_ls_2 = [1,3,5,7]
number_ls_3 = [1,5,9,13]

var_group = np.var(number_ls_1) + np.var(number_ls_2) + np.var(number_ls_1)
print(var_group)

# %%
np.var(sum([number_ls_1, number_ls_2, number_ls_3], []))

# %%
pd.json_normalize([{'a':[1,2], 'b':[1,2]},
              {'a':[1,2], 'b':[3,4]}])

# %%
pd.DataFrame.from_dict({'a':[1,2], 'b':[1,2]}, orient='index')

# %%
df_all = pd.DataFrame()
df1 = pd.DataFrame({'a':[1,2], 'b':[1,2]})
df_all = pd.concat([df_all, df1])
print(df_all)