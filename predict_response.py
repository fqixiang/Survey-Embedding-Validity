import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
import string
import os


# function to make vocab dictionary
def make_dictionary(questions_ls):
    questions_vocab = []
    for question in questions_ls:
        question_tokenized = question.lower().split()
        questions_vocab = questions_vocab + question_tokenized

    questions_vocab = [s.translate(str.maketrans('', '', string.punctuation)) for s in questions_vocab]
    questions_vocab = set(questions_vocab)
    questions_vocab = sorted(questions_vocab)
    return(questions_vocab)

# %% function to make random embeddings dictionary for lookup
def make_random_embeddings_dict(questions_vocab, size, seed):
    # assign random embedding to each word
    random_word_embeddings_dict = {}
    size = size

    rng = np.random.default_rng(seed)
    for word in questions_vocab:
        random_word_embeddings_dict[word] = rng.uniform(-1, 1, size)

    return(random_word_embeddings_dict)

# function to make random embeddings for training testing loop
# %%
def make_random_embeddings(random_word_embeddings_dict, questions_ls_train, questions_ls_all, size):
    questions_vocab_train = make_dictionary(questions_ls_train)

    #zero_embedding = np.zeros(size)

    # make random sentence embeddings
    random_sent_embeddings_ls = []
    for rfa in questions_ls_all:
        rfa_tokenized = rfa.lower().split()
        rfa_ready = [s.translate(str.maketrans('', '', string.punctuation)) for s in rfa_tokenized]

        rfa_word_embeddings = []
        for word in rfa_ready:
            if word in questions_vocab_train:
                rfa_word_embeddings.append(random_word_embeddings_dict[word])
            #else: rfa_word_embeddings.append(zero_embedding)

        rfa_word_embeddings = np.array(rfa_word_embeddings)
        rfa_sent_embeddings = np.mean(rfa_word_embeddings, axis=0)
        random_sent_embeddings_ls.append(rfa_sent_embeddings)

    # make the data frame
    embeddings_df = pd.DataFrame(data=random_sent_embeddings_ls,
                                 columns=["dim%d" % (i + 1) for i in range(size)])

    return(embeddings_df)


# %% function to split data properly
def data_split_pretrained_embeddings(survey_data, question_features, question_ids, seed, dummy, fold_n):
    # merge demographics (which contains the variable column) with question features
    df_combined = pd.merge(survey_data,
                           question_features,
                           left_on='variable',
                           right_on='question_id')

    # make dummy or not
    if dummy:
        df_combined = pd.get_dummies(df_combined,
                                     columns=['region', 'gender', 'edu', 'hh_income', 'religion', 'citizen_UK',
                                              'born_UK', 'language', 'minority', 'ever_spouse', 'married_ever'],
                                     drop_first=False)
    else:
        # label encode the region column
        df_combined["region"] = df_combined["region"].astype('category').cat.codes
        df_combined["language"] = df_combined["language"].astype('category').cat.codes

    results_ls = []
    # train-test split of question names (set seed)
    kf_overall = KFold(n_splits=fold_n,
                       shuffle=True,
                       random_state=seed)

    for train_index, test_index in kf_overall.split(question_ids):
        questions_train = [question_ids[i] for i in train_index]
        questions_test = [question_ids[i] for i in test_index]

        # train-test split of data
        x_train = df_combined.loc[df_combined['variable'].isin(questions_train)]
        x_train = x_train.reset_index(drop=True)

        # create index for CV
        kf_cv = KFold(n_splits=fold_n,
                      shuffle=True,
                      random_state=seed)

        cv_index = list()
        for train_index, test_index in kf_cv.split(questions_train):
            questions_train_CV = [questions_train[i] for i in train_index]
            questions_test_CV = [questions_train[i] for i in test_index]

            train_CV_index = x_train[x_train['variable'].isin(questions_train_CV)].index
            test_CV_index = x_train[x_train['variable'].isin(questions_test_CV)].index

            cv_index.append((train_CV_index, test_CV_index))

        y_train = x_train.score
        y_train = y_train.reset_index(drop=True)

        train_pp_id = x_train.pp_id

        x_train = x_train.drop(columns=['variable', 'score', 'pp_id', 'question_id'], axis=1)

        x_test = df_combined.loc[df_combined['variable'].isin(questions_test)]
        x_test = x_test.reset_index(drop=True)

        y_test = x_test.score
        y_test = y_test.reset_index(drop=True)

        test_pp_id = x_test.pp_id

        x_test = x_test.drop(columns=['variable', 'score', 'pp_id', 'question_id'], axis=1)

        result_dict = {'x_train': x_train,
                       'y_train': y_train,
                       'x_test': x_test,
                       'y_test': y_test,
                       'cv_index': cv_index,
                       'train_pp_id': train_pp_id,
                       'test_pp_id': test_pp_id}

        results_ls.append(result_dict)

    return(results_ls)


# %%
def data_split_baseline(survey_data, questions_data, seed, dummy, fold_n, type='tf_idf', size=None):

    question_ids = questions_data.name.to_list()
    questions = questions_data.question_UK.to_list()
    results_ls = []

    # train-test split of question names (set seed)
    kf_overall = KFold(n_splits=fold_n,
                       shuffle=True,
                       random_state=seed)

    if type == "random":
        questions_vocab = make_dictionary(questions)
        random_word_embeddings_dict = make_random_embeddings_dict(questions_vocab, size, seed=seed)


    for train_index, test_index in kf_overall.split(question_ids):
        questions_train = [question_ids[i] for i in train_index]
        questions_test = [question_ids[i] for i in test_index]

        questions_text_train = [questions[i] for i in train_index]

        if type == "random":
            question_features = make_random_embeddings(random_word_embeddings_dict, questions_text_train, questions, size)

        else:
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

            vectorizer.fit(questions_text_train)
            feature_names = vectorizer.get_feature_names()
            for i in range(len(feature_names)):
                feature_names[i] = feature_names[i] + "_"

            question_features = pd.DataFrame(vectorizer.transform(questions).toarray(),
                                             columns=feature_names)

        question_features['question_id'] = question_ids

        # merge demographics (which contains the variable column) with question features
        df_combined = pd.merge(survey_data,
                               question_features,
                               left_on='variable',
                               right_on='question_id')

        # make dummy or not
        if dummy:
            df_combined = pd.get_dummies(df_combined,
                                         columns=['region', 'gender', 'edu', 'hh_income', 'religion', 'citizen_UK',
                                                  'born_UK', 'language', 'minority', 'ever_spouse', 'married_ever'],
                                         drop_first=False)
        else:
            # label encode the region column
            df_combined["region"] = df_combined["region"].astype('category').cat.codes
            df_combined["language"] = df_combined["language"].astype('category').cat.codes

        # train-test split of data
        x_train = df_combined.loc[df_combined['variable'].isin(questions_train)]
        x_train = x_train.reset_index(drop=True)

        # create index for CV
        kf_cv = KFold(n_splits=fold_n,
                   shuffle=True,
                   random_state=seed)

        cv_index = list()
        for train_index, test_index in kf_cv.split(questions_train):
            questions_train_CV = [questions_train[i] for i in train_index]
            questions_test_CV = [questions_train[i] for i in test_index]

            train_CV_index = x_train[x_train['variable'].isin(questions_train_CV)].index
            test_CV_index = x_train[x_train['variable'].isin(questions_test_CV)].index

            cv_index.append((train_CV_index, test_CV_index))

        y_train = x_train.score
        y_train = y_train.reset_index(drop=True)

        train_pp_id = x_train.pp_id

        x_train = x_train.drop(columns=['variable', 'score', 'pp_id', 'question_id'], axis=1)

        x_test = df_combined.loc[df_combined['variable'].isin(questions_test)]
        x_test = x_test.reset_index(drop=True)

        y_test = x_test.score
        y_test = y_test.reset_index(drop=True)

        test_pp_id = x_test.pp_id

        x_test = x_test.drop(columns=['variable', 'score', 'pp_id', 'question_id'], axis=1)

        result_dict = {'x_train': x_train,
                       'y_train': y_train,
                       'x_test': x_test,
                       'y_test': y_test,
                       'cv_index': cv_index,
                       'train_pp_id': train_pp_id,
                       'test_pp_id': test_pp_id}

        results_ls.append(result_dict)

    return(results_ls)

# %% function lasso CV
def myLasso(features_train, target_train, features_test, target_test, seed, cv_index, type):
    if type == 'ridge':
        model_CV = RidgeCV(alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1],
                           fit_intercept=True,
                           normalize=False,
                           cv=cv_index,
                           scoring='neg_mean_squared_error')

    elif type == 'lasso':
        model_CV = LassoCV(eps=1e-3,
                           n_alphas=100,
                           alphas=None,
                           fit_intercept=True,
                           normalize=False,
                           precompute='auto',
                           max_iter=1000,
                           tol=1e-4,
                           cv=cv_index,
                           verbose=True,
                           n_jobs=8,
                           positive=False,
                           random_state=seed,
                           selection='random')


    features_scalar = StandardScaler().fit(features_train)
    features_train_array = features_scalar.transform(features_train)
    features_test_array = features_scalar.transform(features_test)

    features_train = pd.DataFrame(features_train_array, columns=features_train.columns)
    features_test = pd.DataFrame(features_test_array, columns=features_test.columns)

    model_CV.fit(features_train, target_train)

    y_pred = model_CV.predict(features_test)

    return y_pred

# %% function random forest CV
def myRandomForest(features_train, target_train, features_test, target_test, seed, cv_index):
    # Create the random grid
    random_grid = {'n_estimators': [int(x) for x in np.linspace(start=100, stop=500, num=5)],
                   'max_features': ['log2', 'sqrt'],
                   'max_depth': [int(x) for x in np.linspace(10, 100, num=10)]}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf_search = RandomForestRegressor(bootstrap=True,
                                      criterion='mse',
                                      n_jobs=4,
                                      random_state=seed)

    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf_search,
                                   param_distributions=random_grid,
                                   n_iter=10,
                                   cv=cv_index,
                                   verbose=2,
                                   random_state=seed,
                                   n_jobs=4)

    rf_random.fit(features_train, target_train)

    parameters = rf_random.best_params_

    RF_model_finetuned = RandomForestRegressor(**parameters,
                         random_state=seed,
                         bootstrap=True,
                         criterion='mse')

    RF_model_finetuned.fit(features_train, target_train)
    y_pred = RF_model_finetuned.predict(features_test)

    return y_pred


# the main model function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_data",
                        type=str,
                        default="./data/ESS/ESS_UK_scaled_questions.csv")
    parser.add_argument("--embeddings_data",
                        type=str,
                        default='None')
    parser.add_argument("--other_feature",
                        type=str,
                        default='None')
    parser.add_argument("--model",
                        type=str,
                        default='None')
    parser.add_argument("--dim_size",
                        type=int)
    parser.add_argument("--output_name",
                        type=str)

    args = parser.parse_args()

    response_data = args.response_data
    embedding_data = args.embeddings_data
    other_feature = args.other_feature
    model = args.model
    dim_size = args.dim_size
    output_name = args.output_name

    # read the response data
    ESS09_Responses_UK = pd.read_csv(response_data,
                                     encoding='utf-8')

    if embedding_data == 'None':
        # split data
        questions_data = pd.read_excel('./data/ESS/ESS09_Ordinal_20210623.xlsx')
        if other_feature == "random":
            split_results = data_split_baseline(survey_data=ESS09_Responses_UK,
                                                questions_data=questions_data,
                                                seed=42,
                                                dummy=True,
                                                fold_n=10,
                                                type = other_feature,
                                                size=dim_size)
        else:
            split_results = data_split_baseline(survey_data=ESS09_Responses_UK,
                                                questions_data=questions_data,
                                                seed=42,
                                                dummy=True,
                                                fold_n=10,
                                                type = other_feature,
                                                size=None)

    else:
        # read the embedding data
        embedding_data_path = "./data/embeddings/" + embedding_data + ".pkl"
        ESS09_embeddings = pd.read_pickle(embedding_data_path)

        # split data
        ESS09_question_names = ESS09_Responses_UK.variable.unique().tolist()

        split_results = data_split_pretrained_embeddings(survey_data=ESS09_Responses_UK,
                                                         question_features=ESS09_embeddings,
                                                         question_ids=ESS09_question_names,
                                                         seed=42,
                                                         dummy=True,
                                                         fold_n=10)

    prediction_results_df = pd.DataFrame()

    #go over the folds
    for split_id in range(len(split_results)):
        x_train = split_results[split_id]['x_train']
        y_train = split_results[split_id]['y_train']
        x_test = split_results[split_id]['x_test']
        y_test = split_results[split_id]['y_test']
        cv_index = split_results[split_id]['cv_index']
        train_id = split_results[split_id]['train_pp_id']
        test_id = split_results[split_id]['test_pp_id']

        train_pp_mean = pd.concat([train_id, y_train], axis=1).groupby('pp_id').mean().reset_index()
        test_pp_pred = pd.merge(test_id,
                                train_pp_mean,
                                how='left',
                                on=['pp_id']).score

        prediction_results = {'fold_id': [split_id] * len(y_test),
                              'y_true': y_test,
                              'ave_all_pred': [np.mean(y_train)] * len(y_test),
                              'ave_person_pred': test_pp_pred}

        if model in ["lasso", "ridge"]:
            y_pred = myLasso(features_train=x_train,
                             target_train=y_train,
                             features_test=x_test,
                             target_test=y_test,
                             seed=42,
                             cv_index=cv_index,
                             type=model)
            prediction_results['model'] = [model] * len(y_test)
            prediction_results['y_pred'] = y_pred

            if embedding_data == 'None':
                if other_feature == 'random':
                    prediction_results['feature'] = [other_feature + str(dim_size)] * len(y_test)
                else:
                    prediction_results['feature'] = [other_feature] * len(y_test)
            else:
                prediction_results['feature'] = [embedding_data] * len(y_test)

            prediction_results = pd.DataFrame(prediction_results)
            prediction_results_df = pd.concat([prediction_results_df, prediction_results])

        elif model == "rf":
            y_pred = myRandomForest(features_train=x_train,
                                    target_train=y_train,
                                    features_test=x_test,
                                    target_test=y_test,
                                    seed=42,
                                    cv_index=cv_index)

            prediction_results['model'] = ["rf"] * len(y_test)
            prediction_results['y_pred'] = y_pred

            if embedding_data == 'None':
                if other_feature == 'random':
                    prediction_results['feature'] = [other_feature + str(dim_size)] * len(y_test)
                else:
                    prediction_results['feature'] = [other_feature] * len(y_test)
            else:
                prediction_results['feature'] = [embedding_data] * len(y_test)

            prediction_results = pd.DataFrame(prediction_results)
            prediction_results_df = pd.concat([prediction_results_df, prediction_results])

        else:
            print("Model has to be either lasso or rf.")
            exit()

    output_path = './' + output_name + '.csv'
    output_file = output_name + '.csv'

    if os.path.exists(output_path):
        prediction_results_df.to_csv(output_file, index=None, header=None, mode='a')
    else:
        prediction_results_df.to_csv(output_file, index=None, mode='a')

if __name__ == '__main__':
    main()