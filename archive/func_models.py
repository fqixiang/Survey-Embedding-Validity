from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold


# %% function to split data properly
def data_split(survey_data, question_features, question_ids, split, seed, dummy, fold_n):
    # train-test split of question names (set seed)
    questions_train, questions_test = train_test_split(question_ids,
                                                       test_size=split,
                                                       random_state=seed)

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
    kf = KFold(n_splits=fold_n,
               shuffle=True,
               random_state=42)

    cv_index = list()
    for train_index, test_index in kf.split(questions_train):
        questions_train_CV = [questions_train[i] for i in train_index]
        questions_test_CV = [questions_train[i] for i in test_index]

        train_CV_index = x_train[x_train['variable'].isin(questions_train_CV)].index
        test_CV_index = x_train[x_train['variable'].isin(questions_test_CV)].index

        cv_index.append((train_CV_index, test_CV_index))

    y_train = x_train.score
    y_train = y_train.reset_index(drop=True)

    x_train = x_train.drop(columns=['variable', 'score', 'pp_id', 'question_id'], axis=1)

    x_test = df_combined.loc[df_combined['variable'].isin(questions_test)]
    x_test = x_test.reset_index(drop=True)

    y_test = x_test.score
    y_test = y_test.reset_index(drop=True)

    x_test = x_test.drop(columns=['variable', 'score', 'pp_id', 'question_id'], axis=1)

    return x_train, y_train, x_test, y_test, cv_index


# %% function lasso CV
def myLasso(features_train, target_train, features_test, target_test, seed, cv_index):
    lasso_model_CV = LassoCV(eps=1e-3,
                             n_alphas=100,
                             alphas=None,
                             fit_intercept=True,
                             normalize=True,
                             precompute='auto',
                             max_iter=1000,
                             tol=1e-4,
                             cv=cv_index,
                             verbose=True,
                             n_jobs=4,
                             positive=False,
                             random_state=seed,
                             selection='cyclic')

    lasso_model_CV.fit(features_train, target_train)

    df_coef = pd.DataFrame({'feature': features_train.columns.to_list(),
                            'coef': lasso_model_CV.coef_}).sort_values(by=['coef'], ascending=False)

    y_pred_model_Lasso_fine_tuned = lasso_model_CV.predict(features_test)
    MSE_score = mean_squared_error(target_test, y_pred_model_Lasso_fine_tuned)

    return df_coef, y_pred_model_Lasso_fine_tuned, MSE_score, lasso_model_CV


# %% try
# get survey data
from prepare_survey_data import ESS09_Responses_UK_valid_scaled, ESS09_question_names



# %%
# get embeddings data
embeddings_stsb_roberta_large = pd.read_pickle('./data/embeddings_stsb-roberta-large.pkl')

# %% try the data split function
x_train, y_train, x_test, y_test, cv_index = data_split(survey_data=ESS09_Responses_UK_valid_scaled,
                                                        question_features=embeddings_stsb_roberta_large,
                                                        question_ids=ESS09_question_names,
                                                        split=0.2,
                                                        seed=42,
                                                        dummy=True,
                                                        fold_n=5)


# %% try lasso with CV
df_coef, y_pred_model_Lasso_fine_tuned, MSE_score, lasso_model_CV = myLasso(features_train=x_train,
                                                            target_train=y_train,
                                                            features_test=x_test,
                                                            target_test=y_test,
                                                            seed=42,
                                                            cv_index=cv_index)

# %%
print(MSE_score)  # 0.07266513808161434 without customised cv, 0.07727466380323482 with customised cv

# %% baseline midpoint
mean_squared_error(y_test, [0.5] * len(y_test))
# 0.08060956808742818

# %% baseline average of responses
mean_squared_error(y_test, [np.mean(y_train)] * len(y_test))
# 0.07867001724447076

# %% baseline, average of domains

# %% baseline, model with domains

# %% baseline, model with only demographics
demographic_indices = [i for i, s in enumerate(x_train.columns) if 'dim' not in s]
x_train.iloc[:, demographic_indices]
# 0.07842059924900482

# %% baseline, model with bow


# %%
print(synthetic_embedding_df_stsb_roberta_large)

# %%
data_trstlgl_demographics = ESS09_Responses_UK_valid_scaled.loc[ESS09_Responses_UK_valid_scaled.variable == 'trstlgl']


# %%
df_combined = pd.merge(data_trstlgl_demographics,
                       synthetic_embedding_df_stsb_roberta_large,
                       how='cross')

# %%
true_scores = df_combined.score
# %%
true_scores

# %%
person_ids = df_combined.pp_id

# %%
person_ids

# %%
question_type = df_combined.question_id
# %%
question_type

# %%
df_combined = df_combined.drop(columns=['variable', 'score', 'pp_id', 'question_id'], axis=1)

# %%
df_combined = pd.get_dummies(df_combined,
                                           columns=['region', 'gender', 'edu', 'hh_income', 'religion', 'citizen_UK',
                                                    'born_UK', 'language', 'minority', 'ever_spouse', 'married_ever'],
                                           drop_first=False)




# %%
y_pred_synthetic = lasso_model_CV.predict(df_combined)

# %%
question_type
# %%
synthetic_model_df = pd.DataFrame({'y_pred': y_pred_synthetic,
              'question_type': question_type,
              'original_score': true_scores,
                                   'pp_id': person_ids})

# %%
synthetic_model_df = synthetic_model_df.assign(score_diff=synthetic_model_df['original_score'] - synthetic_model_df['y_pred'])

# %%
print(synthetic_model_df)

# %%
MSE_score = mean_squared_error(target_test, y_pred_model_Lasso_fine_tuned)

# %%
synthetic_model_df = synthetic_model_df.astype({'pp_id': 'int64'})

# %%
synthetic_model_df.loc[synthetic_model_df.pp_id == 22452]

# %%
synthetic_model_df.groupby('question_type')['score_diff'].var()

# %%
len(ESS09_question_names)

# %%
import seaborn as sns
import matplotlib

# %%
plot = sns.pairplot(x_vars=['pp_id'],
             y_vars=['score_diff'],
             data=synthetic_model_df.loc[synthetic_model_df.pp_id == 22452],
             hue="question_type")

# %%
plot

# %%
ESS09_Responses_UK_valid_scaled.loc[ESS09_Responses_UK_valid_scaled.variable == 'netifr']

# %% a random forest model
def model_rf():
    # the model, hyper-parameter search, set seed
    print('haha')
    # the model, fit best parameter values from hyper-parameter search

    # apply model to test partition

    # calculate model prediction error (MSE)

    # calculate baseline prediction error (MSE) based on the midpoint

    # calculate baseline prediction error (MSE) based on the mean of the training set

    # calculate baseline prediction error (MSE) based on the mean of each respondent's scores

    # return y_true, y_pred, mse_rf, mse_base_midpoint, mse_base_mean_train, mse_base_respondent


# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# %%
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]

# Number of features to consider at every split
max_features = ['log2', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 100, num=10)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
print(random_grid)

# %%
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf_search = RandomForestRegressor(bootstrap=True,
                                  criterion='mse',
                                  random_state=42)

# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf_search,
                               param_distributions=random_grid,
                               n_iter=100,
                               cv=5,
                               verbose=2,
                               random_state=42,
                               n_jobs=4)

# %%
# Fit the random search model
rf_random.fit(x_train.iloc[random_sample], y_train.iloc[random_sample])

# %%
parameters = rf_random.best_params_
parameters

# %%


# %% time elapse
import time

# %%
start = time.time()
RF_model_finetuned = RandomForestRegressor(  # **parameters,
    random_state=42,
    bootstrap=True,
    criterion='mse')

# RF_model_finetuned.fit(x_train.iloc[random_sample], y_train.iloc[random_sample])
RF_model_finetuned.fit(x_train, y_train)

RF_predictions = RF_model_finetuned.predict(x_test)

score = mean_squared_error(y_test, RF_predictions)
print(score)
# 0.07431804650182634, with a training size of 2000
# 0.07398332039773972, with a training size of 20000
# 0.07468657066309285, with a full training size, takes almost 40 min (2358.78 secs)

end = time.time()
print(end - start)

# %% a logistic regression model
from sklearn.linear_model import Lasso
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV

# %%
start = time.time()
model_Lasso = Lasso(fit_intercept=True,
                    normalize=True,
                    random_state=42,
                    selection='cyclic')

params = {'alpha': uniform(1e-5, 1)}

search_model_Lasso = RandomizedSearchCV(model_Lasso,
                                        param_distributions=params,
                                        random_state=42,
                                        n_iter=200,
                                        cv=5,
                                        verbose=1,
                                        n_jobs=4,
                                        scoring='neg_mean_squared_error',
                                        return_train_score=True)

search_model_Lasso.fit(x_train, y_train)
end = time.time()
print(end - start)

# %%
search_model_Lasso.best_score_

# %%
start = time.time()
model_Lasso_fine_tuned = Lasso(alpha=0.005532117123602399,
                               fit_intercept=True,
                               normalize=False,
                               random_state=42,
                               selection='cyclic')

model_Lasso_fine_tuned.fit(x_train, y_train)

y_pred_model_Lasso_fine_tuned = model_Lasso_fine_tuned.predict(x_test)

end = time.time()
print(end - start)

# %%
print(mean_squared_error(y_test, y_pred_model_Lasso_fine_tuned))
# 0.07177951505880328, normalize=False, selection='cyclic'

# %%
pd.DataFrame({'feature': x_train.columns.to_list(),
              'coef': model_Lasso_fine_tuned.coef_}).sort_values(by=['coef'], ascending=False)

# %%
start

# %%
end

# %% a function to make plots of the predictions (MSE, actual pred)


# check whether model understands survey questions
# example: how much trust do you have in the police?
# how much trust would you give to the police?
# how much


# %% try out lassoCV
from sklearn.linear_model import LassoCV

lasso_model_CV = LassoCV(eps=1e-3,
                         n_alphas=100,
                         alphas=None,
                         fit_intercept=True,
                         normalize=True,
                         precompute='auto',
                         max_iter=1000,
                         tol=1e-4,
                         cv=5,
                         verbose=True,
                         n_jobs=4,
                         positive=False,
                         random_state=42,
                         selection='cyclic')

lasso_model_CV.fit(x_train, y_train)

# %%
pd.DataFrame({'feature': x_train.columns.to_list(),
              'coef': lasso_model_CV.coef_}).sort_values(by=['coef'], ascending=False)

# %%
y_pred_model_Lasso_fine_tuned = lasso_model_CV.predict(x_test)
print(mean_squared_error(y_test, y_pred_model_Lasso_fine_tuned))
# 0.07266513808161434

# %% a function to generate synthetic survey questions
# two factors: topic change and wording change

# condition 1: original sentences
# condition 2: modified sentences with only topic change
# condition 3: modified sentences with only wording change
# condition 4: modified sentences with both topic and wording changes
# condition 5: negate the original sentences

# we can examine 1. the variance of the scores within each condition; 2. diff between the synthetic and the original
# we can examine only one sentence and all of its variations
# or, we can have multiple
