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