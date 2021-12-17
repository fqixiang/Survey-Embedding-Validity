import pandas as pd
import string
import numpy as np
from sklearn.linear_model import LogisticRegression
import argparse
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
import os

# %% count sentence length
def count_length(input):
    input_tokenized = input.split(' ')
    input_len = len(input_tokenized)
    return(input_len)

# %% make a random embedding df
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
    embeddings_df.insert(loc=0, column='question_id', value=embeddings_df.index)
    return(embeddings_df)

# %%
def init_logistic_model():
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='none', max_iter=10000,
                               class_weight='balanced')
    return(model)

# %% probe sentence length
def probe_length(data_questions, data_embeddings):
    train_ids = set(data_questions.row_id[(data_questions.basic_concept != 'rights') & (data_questions.form_request != 'imp_int')].to_list())
    test_ids = set(data_questions.row_id[(data_questions.basic_concept == 'rights') & (data_questions.form_request == 'imp_int')].to_list())

    test_x = data_embeddings.iloc[list(test_ids), 1:]
    test_y = data_questions.length_binned[data_questions.row_id.isin(test_ids)].to_list()

    train_x = data_embeddings.iloc[list(train_ids), 1:]
    train_y = data_questions.length_binned[data_questions.row_id.isin(train_ids)].to_list()

    assert data_embeddings.iloc[list(train_ids), :].question_id.to_list() == data_questions.row_id[data_questions.row_id.isin(train_ids)].to_list()

    model = init_logistic_model()
    model.fit(X = train_x, y = train_y)
    prediction = model.predict(test_x)
    acc_score = accuracy_score(y_true=test_y, y_pred=prediction)

    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(train_x, train_y)
    dummy_acc = dummy_clf.score(test_x, test_y)

    n_train = len(train_y)
    n_test = len(test_y)

    return(acc_score, dummy_acc, n_train, n_test)

def probe_basic_concepts(data_questions, data_embeddings):
    train_ids = set(data_questions.row_id[(data_questions.length_binned != '15-25') & (
            data_questions.similarity != 'high')].to_list())
    test_ids = set(data_questions.row_id[(data_questions.length_binned == '15-25') & (
            data_questions.similarity == 'high')].to_list())

    test_x = data_embeddings.iloc[list(test_ids), 1:]
    test_y = data_questions.basic_concept[data_questions.row_id.isin(test_ids)].to_list()

    train_x = data_embeddings.iloc[list(train_ids), 1:]
    train_y = data_questions.basic_concept[data_questions.row_id.isin(train_ids)].to_list()

    assert data_embeddings.iloc[list(train_ids), :].question_id.to_list() == data_questions.row_id[data_questions.row_id.isin(train_ids)].to_list()

    model = init_logistic_model()
    model.fit(X = train_x, y = train_y)
    prediction = model.predict(test_x)
    acc_score = accuracy_score(y_true=test_y, y_pred=prediction)

    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(train_x, train_y)
    dummy_acc = dummy_clf.score(test_x, test_y)

    n_train = len(train_y)
    n_test = len(test_y)

    return(acc_score, dummy_acc, n_train, n_test)


# %%concrete concepts
def probe_concrete_concepts(data_questions, data_embeddings, control_length = False):
    concrete_concept_df = pd.read_excel('./data/synthetic/Synthetic_Questions_Reference.xlsx')
    data_questions = pd.merge(left=data_questions,
                              right=concrete_concept_df,
                              left_on='question_id',
                              right_on='question_id')

    data_questions.sort_values('row_id', inplace=True)

    if control_length:
        data_embeddings['length'] = data_questions['length']

    train_ids = set(data_questions.row_id[(data_questions.similarity != 'high')].to_list())
    test_ids = set(data_questions.row_id[((data_questions.similarity == 'high'))].to_list())

    train_x = data_embeddings.iloc[list(train_ids), 1:]
    train_y = data_questions.concrete_concept[data_questions.row_id.isin(train_ids)].to_list()

    test_x = data_embeddings.iloc[list(test_ids), 1:]
    test_y = data_questions.concrete_concept_new[data_questions.row_id.isin(test_ids)].to_list()

    assert data_embeddings.iloc[list(train_ids), :].question_id.to_list() == data_questions.row_id[data_questions.row_id.isin(train_ids)].to_list()

    model = init_logistic_model()
    model.fit(X = train_x, y = train_y)
    prediction = model.predict(test_x)
    acc_score = accuracy_score(y_true=test_y, y_pred=prediction)

    dummy_clf = DummyClassifier(strategy="constant", constant='state_health_services')
    dummy_clf.fit(train_x, train_y)
    dummy_acc = dummy_clf.score(test_x, test_y)

    n_train = len(train_y)
    n_test = len(test_y)

    return(acc_score, dummy_acc, n_train, n_test)


#form of requests
def probe_form(data_questions, data_embeddings):
    train_ids = set(data_questions.row_id[(data_questions.length_binned != '0-10')].to_list())
    test_ids = set(data_questions.row_id[(data_questions.length_binned == '0-10')].to_list())

    test_x = data_embeddings.iloc[list(test_ids), 1:]
    test_y = data_questions.form_request[data_questions.row_id.isin(test_ids)].to_list()

    train_x = data_embeddings.iloc[list(train_ids), 1:]
    train_y = data_questions.form_request[data_questions.row_id.isin(train_ids)].to_list()

    assert data_embeddings.iloc[list(train_ids), :].question_id.to_list() == data_questions.row_id[data_questions.row_id.isin(train_ids)].to_list()

    model = init_logistic_model()
    model.fit(X = train_x, y = train_y)
    prediction = model.predict(test_x)
    acc_score = accuracy_score(y_true=test_y, y_pred=prediction)

    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(train_x, train_y)
    dummy_acc = dummy_clf.score(test_x, test_y)

    n_train = len(train_y)
    n_test = len(test_y)

    pred_correct = [x == y for x,y in zip(test_y, prediction)]

    return(acc_score, dummy_acc, n_train, n_test, pred_correct)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_data",
                        type=str,
                        default=None)
    parser.add_argument("--questions_data",
                        type=str,
                        default=None)
    parser.add_argument("--embeddings_data",
                        type=str,
                        default='random')
    parser.add_argument("--embeddings_size",
                        type=int,
                        default=768)

    args = parser.parse_args()

    features_path = './data/synthetic/' + args.features_data
    questions_path = './data/synthetic/' + args.questions_data

    features_df = pd.read_excel(features_path)
    questions_df = pd.read_excel(questions_path)
    questions_ls = questions_df.rfa.to_list()

    # create new features
    questions_df['length'] = questions_df['rfa'].apply(lambda rfa: count_length(rfa))
    bins = [0, 10, 12, 15, 25]
    labels = ['0-10', '10-12', '12-15', '15-25']
    questions_df['length_binned'] = pd.cut(questions_df['length'], bins=bins, labels=labels)

    # merge the two dataframes
    df_merged = pd.merge(questions_df[['row_id', 'question_id', 'form_request', 'length', 'length_binned']],
                         features_df[['question_id', 'basic_concept', 'concrete_concept', 'similarity']],
                         left_on='question_id',
                         right_on='question_id')
    df_merged.sort_values('row_id', inplace=True)

    if args.embeddings_data != 'random':
        embeddings_path = './data/embeddings/' + args.embeddings_data
        embeddings_df = pd.read_pickle(embeddings_path)
    else:
        embeddings_df = make_random_embeddings(questions_ls, args.embeddings_size, seed=42)

    #define properties to probe
    probe_var_ls = ['length_binned', 'basic_concept', 'concrete_concept', 'form_request']

    results_dict = {}
    results_dict['embeddings_type'] = []
    results_dict['acc_score'] = []
    results_dict['target_var'] = []
    results_dict['acc_dummy'] = []
    results_dict['n_train'] = []
    results_dict['n_test'] = []

    # probe length_binned
    probe_acc_score_length, probe_acc_dummy, n_train, n_test = probe_length(df_merged, embeddings_df)
    if args.embeddings_data != 'random':
        results_dict['embeddings_type'].append(args.embeddings_data)
    else:
        results_dict['embeddings_type'].append(args.embeddings_data + str(args.embeddings_size))
    results_dict['acc_score'].append(probe_acc_score_length)
    results_dict['acc_dummy'].append(probe_acc_dummy)
    results_dict['target_var'].append('length_binned')
    results_dict['n_train'].append(n_train)
    results_dict['n_test'].append(n_test)

    # probe basic concepts
    probe_acc_basic_concept, probe_acc_dummy, n_train, n_test = probe_basic_concepts(df_merged, embeddings_df)
    if args.embeddings_data != 'random':
        results_dict['embeddings_type'].append(args.embeddings_data)
    else:
        results_dict['embeddings_type'].append(args.embeddings_data + str(args.embeddings_size))
    results_dict['acc_score'].append(probe_acc_basic_concept)
    results_dict['acc_dummy'].append(probe_acc_dummy)
    results_dict['target_var'].append('basic_concept')
    results_dict['n_train'].append(n_train)
    results_dict['n_test'].append(n_test)

    # probe concrete concepts
    probe_acc_concrete_concept, probe_acc_dummy, n_train, n_test = probe_concrete_concepts(df_merged, embeddings_df)
    if args.embeddings_data != 'random':
        results_dict['embeddings_type'].append(args.embeddings_data)
    else:
        results_dict['embeddings_type'].append(args.embeddings_data + str(args.embeddings_size))
    results_dict['acc_score'].append(probe_acc_concrete_concept)
    results_dict['acc_dummy'].append(probe_acc_dummy)
    results_dict['target_var'].append('concrete_concept')
    results_dict['n_train'].append(n_train)
    results_dict['n_test'].append(n_test)

    # probe concrete concepts (with control)
    if args.embeddings_data == 'random':
        probe_acc_concrete_concept_control, probe_acc_dummy, n_train, n_test = probe_concrete_concepts(df_merged, embeddings_df, control_length=True)
        if args.embeddings_data != 'random':
            results_dict['embeddings_type'].append(args.embeddings_data)
        else:
            results_dict['embeddings_type'].append(args.embeddings_data + str(args.embeddings_size))
        results_dict['acc_score'].append(probe_acc_concrete_concept_control)
        results_dict['acc_dummy'].append(probe_acc_dummy)
        results_dict['target_var'].append('concrete_concept_controlled_length')
        results_dict['n_train'].append(n_train)
        results_dict['n_test'].append(n_test)

    # probe form of requests
    probe_acc_form, probe_acc_dummy, n_train, n_test, pred_correct = probe_form(df_merged, embeddings_df)
    if args.embeddings_data != 'random':
        results_dict['embeddings_type'].append(args.embeddings_data)
    else:
        results_dict['embeddings_type'].append(args.embeddings_data + str(args.embeddings_size))
    results_dict['acc_score'].append(probe_acc_form)
    results_dict['acc_dummy'].append(probe_acc_dummy)
    results_dict['target_var'].append('form_request')
    results_dict['n_train'].append(n_train)
    results_dict['n_test'].append(n_test)

    results_df = pd.DataFrame(results_dict)

    pred_df = pd.DataFrame(pred_correct, columns=['pred_binary'])

    if os.path.exists('./probing_results.csv'):
        results_df.to_csv('probing_results.csv', index=None, header=None, mode='a')
    else:
        results_df.to_csv('probing_results.csv', index=None, mode='a')

    if os.path.exists('./probing_pred.csv'):
        df = pd.read_csv('./probing_pred.csv')
        pred_df = pd.concat([df, pred_df], axis=1)
        pred_df.to_csv('probing_pred.csv', index=None)
    else:
        pred_df.to_csv('probing_pred.csv', index=None)

if __name__ == '__main__':
    main()

# # %%
# embeddings_df = pd.read_pickle('./data/embeddings/synthetic_bert_base_uncased_20210929185434.pkl')
#
# # %%
# embeddings_df = make_random_embeddings(questions_ls, 768)
#
# # %%
# data_questions = df_merged
# data_embeddings = embeddings_df
#
# train_ids = set(data_questions.row_id[(data_questions.length_binned != '15-25') & (
#     data_questions.similarity != 'high')].to_list())
# test_ids = set(data_questions.row_id[(data_questions.length_binned == '15-25') & (
#     data_questions.similarity == 'high')].to_list())
#
# test_x = data_embeddings.iloc[list(test_ids), 1:]
# test_y = data_questions.basic_concept[data_questions.row_id.isin(test_ids)].to_list()
#
# train_x = data_embeddings.iloc[list(train_ids), 1:]
# train_y = data_questions.basic_concept[data_questions.row_id.isin(train_ids)].to_list()
#
# assert data_embeddings.iloc[list(train_ids), :].question_id.to_list() == data_questions.row_id[
#     data_questions.row_id.isin(train_ids)].to_list()
#
# model = LogisticRegression(multi_class='multinomial', solver='newton-cg', penalty='none', max_iter=200,
#                            class_weight='balanced')
# model.fit(X=train_x, y=train_y)
# prediction = model.predict(test_x)
# acc_score = accuracy_score(y_true=test_y, y_pred=prediction)
#
# dummy_clf = DummyClassifier(strategy="most_frequent")
# dummy_clf.fit(train_x, train_y)
# dummy_acc = dummy_clf.score(test_x, test_y)
# print(acc_score, dummy_acc)
#
# # %%
# from collections import Counter
# # %%
# Counter(train_y)
#
# # %%
# test_ids

# %%
from statsmodels.stats.contingency_tables import mcnemar
import pandas as pd

df = pd.read_csv('./probing_pred.csv')

# %%
crossTab = pd.crosstab(df.pred_binary1, df.pred_binary2)

# %%
print(mcnemar(crossTab, exact=False))