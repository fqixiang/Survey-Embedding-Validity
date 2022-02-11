import pandas as pd
import numpy as np

def main():
    # import response data
    ESS09_Responses = pd.read_csv('./data/ESS/ESS9e03.0_F1.csv', low_memory=False)

    # UK only
    ESS09_Responses_UK = ESS09_Responses.loc[ESS09_Responses.cntry == 'GB']

    # relevant background columns names
    # region: Region
    # gndr: Gender
    # agea: Age of respondent, calculated
    # edubgb2: Highest level of education, United Kingdom: Up to Ph.D or equivalent
    # wkhtot: Total hours normally worked per week in main job overtime included
    # isco08: Occupation, ISCO08
    # hinctnta: Household's total net income, all sources
    # rlgdngb: which religion or denomination (UK)?
    # ctzcntr: citizen of country (are you)?
    # ctzshipd: citizenship
    # brncntr: were you born in UK?
    # cntbrthd: country of birth
    # lnghom1: language most spoken at home
    # blgetmg: Belong to minority ethnic group in country
    # fbrncntc: Country of birth, father
    # mbrncntc: Country of birth, mother
    # evlvptn: Ever lived with a spouse or partner for 3 months or more
    # evmar: Are or ever been married
    # nbthcld: Number of children ever given birth to/ fathered
    # hhmmb: Number of people living regularly as member of household
    # edufbgb2: Father's highest level of education, United Kingdom: Up to Ph.D or equivalent
    # edumbgb2: Mother's highest level of education, United Kingdom: Up to Ph.D or equivalent

    # notes on these variables
    # -region: region, cate
    # -gndr: gender, cate
    # -agea: age, numeric
    # -edubgb2: edu, cate or numeric, 5555.0 (other), 7777.0 (refusal), 8888.0 (don't know), 9999.0 (no answer)
    # -wkhtot: work_hours, numeric, 666 (not applicable),777 (refusal), 888 (don't know), 999 (no answer)
    # isco08: too many categories, don't use, 66666 (not applicable), 77777 (refusal), 88888 (don't know), 99999 (no answer)
    # -hinctnta: hh_income, numeric, 77, 88, 99
    # -rlgdngb: religion, cate, 6666 (not applicable, probably means non-religious), 7777, 9999
    # -ctzcntr: citizen_UK, cate, 8 (don't know)
    # ctzshipd: citizenship, cate, 6666 (not applicable, meaning British), 8888 (don't know), 9999 (no answer)
    # -brncntr: born_UK, cate
    # cntbrthd: cntry_birth, cate, 6666 (meaning born in UK), 8888 (don't know)
    # -lnghom1: language, cate, 999 (no answer)
    # -blgetmg: minority, cate, 8 (don't know)
    # fbrncntc: cntry_birth_father, cate, 6666 (meaning father born in UK), 8888 (don't know)
    # -evlvptn: ever_spouse, cate
    # -evmar: married_ever, cate
    # -nbthcld: num_kids, numeric, 66 (not applicable, meaning 0)
    # -hhmmb: hh_size, numeric
    # edufbgb2: father_edu, cate or numeric, 5555.0 (other), 7777.0 (refusal), 8888.0 (don't know), 9999.0 (no answer)
    # edumbgb2: mother_edu, cate or numeric, 5555.0 (other), 7777.0 (refusal), 8888.0 (don't know), 9999.0 (no answer)

    # %% just a short list of variables to try
    background_var = ['region', 'gndr', 'agea', 'edubgb2', 'wkhtot', 'hinctnta', 'rlgdngb', 'ctzcntr', 'brncntr', 'lnghom1',
                      'blgetmg', 'evlvptn', 'evmar', 'nbthcld', 'hhmmb']

    # %% use this to check out the variables
    # print(ESS09_Responses_UK.wkhtot.value_counts(ascending=False, dropna=False).to_string())

    # %% list of variables of relevant questions
    ESS09_Ordinal = pd.read_excel('./data/ESS/ESS09_Ordinal.xlsx')
    ordinal_var = ESS09_Ordinal.name.to_list()

    # %% keep only the relevant background and ordinal variables
    ESS09_UK_Data = ESS09_Responses_UK[background_var + ordinal_var]
    ESS09_UK_Data.index.name = 'pp_id'
    ESS09_UK_Data.reset_index(inplace=True)

    # print(ESS09_UK_Data)

    # how to handle the invalid responses in the background variables
    # -region: region, make cate
    # -gndr: gender, make cate
    # -agea: age, make int
    # -edubgb2: edu, make cate, turn 8 (HE access), 5555.0 (other), 7777.0 (refusal), 8888.0 (don't know), 9999.0 (no answer) into 11 (other)
    # -wkhtot: work_hours, make int, turn 666 (not applicable), 777 (refusal), 888 (don't know), 999 (no answer) into 0
    # -hinctnta: hh_income, make cat, turn 77, 88, 99 into 77
    # -rlgdngb: religion, make cate, turn 7777, 9999 into 6666 (not applicable, probably means non-religious)
    # -ctzcntr: citizen_UK, make cate, 8 (don't know)
    # -brncntr: born_UK, cate
    # -lnghom1: language, cate, 999 (no answer)
    # -blgetmg: minority, cate, turn 8 (don't know) into 2
    # -evlvptn: ever_spouse, cate
    # -evmar: married_ever, cate
    # -nbthcld: num_kids, numeric, 66 (not applicable, meaning 0)
    # -hhmmb: hh_size, numeric

    # %% from wide to long format
    ESS09_UK_Data_Long = ESS09_UK_Data.melt(id_vars=['pp_id'] + background_var,
                                            value_vars=ordinal_var,
                                            var_name='variable',
                                            value_name='score')

    # print(ESS09_UK_Data_Long)


    # %% clean out the NAs and what not in the target variables
    def remove_invalid_responses(var_ls, datafile, docu):
        for var in var_ls:
            # retrieve the target variable column, with the weight and country and valid range
            valid_range = docu[['range_min', 'range_max']].loc[docu['name'] == var].values.tolist()[0]

            # amend the df so that only valid responses are retained
            datafile = datafile.loc[~((datafile['variable'] == var) &
                                      (~datafile['score'].isin(range(int(valid_range[0]), int(valid_range[1] + 1))))), :]

        return datafile


    ESS09_Responses_UK_valid = remove_invalid_responses(var_ls=ordinal_var,
                                                        datafile=ESS09_UK_Data_Long,
                                                        docu=ESS09_Ordinal)

    # print(ESS09_Responses_UK_valid)

    # %% rename the background variables
    ESS09_Responses_UK_valid = ESS09_Responses_UK_valid.rename(
        columns=dict(gndr='gender', agea='age', edubgb2='edu', wkhtot='work_hours',
                     hinctnta='hh_income', rlgdngb='religion', ctzcntr='citizen_UK', brncntr='born_UK',
                     lnghom1='language', blgetmg='minority', evlvptn='ever_spouse',
                     evmar='married_ever', nbthcld='num_kids', hhmmb='hh_size'))

    # %% check variable types
    # ESS09_Responses_UK_valid.dtypes

    # %% set float64 variable types to be int64
    ESS09_Responses_UK_valid = ESS09_Responses_UK_valid.astype({'edu': 'int64',
                                                                'work_hours': 'int64',
                                                                'religion': 'int64'})

    # %% clean out the NAs and what not in the background variables
    # -edubgb2: edu, make cate, turn 8 (HE access), 5555.0 (other), 7777.0 (refusal), 8888.0 (don't know), 9999.0 (no answer) into 11 (other)
    # -wkhtot: work_hours, make int, turn 666 (not applicable), 777 (refusal), 888 (don't know), 999 (no answer) into 0
    # -hinctnta: hh_income, make cat, turn 77, 88, 99 into 77
    # -rlgdngb: religion, make cate, turn 7777.0, 9999.0 into 6666.0 (not applicable, probably means non-religious)
    # -ctzcntr: citizen_UK, make cate, turn 8 (don't know) into 2 (no)
    # -lnghom1: language, cate, 999 (no answer), 888 (don't know)
    # -blgetmg: minority, cate, turn 8 (don't know) into 1 (yes)
    # -nbthcld: num_kids, numeric, 66 (not applicable, meaning 0)

    ESS09_Responses_UK_valid = ESS09_Responses_UK_valid.replace(
        {'edu': {8: 11, 5555: 11, 7777: 11, 8888: 11},
         'work_hours': {666: 0, 777: 0, 888: 0, 999: 0},
         'hh_income': {88: 77, 99: 77},
         'religion': {7777: 6666, 9999: 6666},
         'citizen_UK': {8: 2},
         'minority': {8: 1},
         'num_kids': {66: 0}})

    # %% convert certain columns to categorical
    ESS09_Responses_UK_valid = ESS09_Responses_UK_valid.astype({'pp_id': 'string',
                                                                'region': 'string',
                                                                'gender': 'string',
                                                                'edu': 'string',
                                                                'hh_income': 'string',
                                                                'religion': 'string',
                                                                'citizen_UK': 'string',
                                                                'born_UK': 'string',
                                                                'language': 'string',
                                                                'minority': 'string',
                                                                'ever_spouse': 'string',
                                                                'married_ever': 'string',
                                                                'variable': 'string'})

    # %% all responses to between 0 and 1
    ESS09_Responses_UK_valid_scaled = pd.merge(ESS09_Responses_UK_valid,
                                               ESS09_Ordinal[['name', 'range_min', 'range_max']],
                                               left_on='variable',
                                               right_on='name')

    ESS09_Responses_UK_valid_scaled = \
        ESS09_Responses_UK_valid_scaled.assign(score=lambda x: (x["score"] - x['range_min'])/
                                                               (x['range_max']-x['range_min']))

    ESS09_Responses_UK_valid_scaled = ESS09_Responses_UK_valid_scaled.drop(columns=['range_min', 'range_max', 'name'], axis=1)

    # %% some questions still need to be reversed (because of the direction of questioning)
    var_neg_scale_ls = ESS09_Ordinal.name[ESS09_Ordinal.scale_direction == "neg"].to_list()
    ESS09_Responses_UK_valid_scaled['score'] = np.where(ESS09_Responses_UK_valid_scaled['variable'].isin(var_neg_scale_ls),
                                                        1 - ESS09_Responses_UK_valid_scaled['score'],
                                                        ESS09_Responses_UK_valid_scaled['score'])

    # %%
    # ESS09_Responses_UK_valid_scaled.to_csv("./data/ESS_UK_scaled_questions.csv", index = False)
    ESS09_Responses_UK_valid_scaled.to_excel("./data/ESS/ESS_UK_scaled_questions.xlsx", index = False)


if __name__ == '__main__':
    main()