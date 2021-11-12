# Choice of Sentence Bert Models
1. paraphrase identification
* paraphrase-distilroberta-base-v1 - Trained on large scale paraphrase data.
* paraphrase-xlm-r-multilingual-v1 - Multilingual version of distilroberta-base-paraphrase-v1, trained on parallel data for 50+ languages.

2. semantic textual similarity
* stsb-roberta-large - STSb performance: 86.39
* stsb-roberta-base - STSb performance: 85.44
* stsb-bert-large - STSb performance: 85.29
* stsb-distilbert-base - STSb performance: 85.16

3. duplicate question detection
* quora-distilbert-base - Model first tuned on NLI+STSb data, then fine-tune for Quora Duplicate Questions detection retrieval.
* quora-distilbert-multilingual - Multilingual version of distilbert-base-nli-stsb-quora-ranking. Fine-tuned with parallel data for 50+ languages.

4. more multi-lingual models
* distiluse-base-multilingual-cased-v2: Multilingual knowledge distilled version of multilingual Universal Sentence Encoder. While the original mUSE model only supports 16 languages, this multilingual knowledge distilled version supports 50+ languages.
* T-Systems-onsite/cross-en-de-roberta-sentence-transformer - Multilingual model for English and German.

# Input Sequence Length
By default, a limit of 128 word pieces. Max is about 300-400 English words (or 512 word pieces).

# Background Variables
* cntry: country
* rlgblg: belonging to a religion or denomination
* rlgdngb: which one (UK)?
* rlgblge: ever belong to a religion or denomination
* rlgdegb: which one (UK)?
* ctzcntr: citizen of country (are you)?
* ctzshipd: citizenship
* brncntr: were you born in UK?
* cntbrthd: country of birth
* lnghom1: language most spoken at home
* blgetmg: Belong to minority ethnic group in country
* facntr: Father born in country
* fbrncntc: Country of birth, father
* mocntr: Mother born in country
* mbrncntc: Country of birth, mother
* evpdemp: Paid employment or apprenticeship at least 3 months 20 hours weekly
* evlvptn: Ever lived with a spouse or partner for 3 months or more
* evmar: Are or ever been married
* nbthcld: Number of children ever given birth to/ fathered
* ngchld: Number of grandchildren
* hhmmb: Number of people living regularly as member of household
* gndr: Gender
* agea: Age of respondent, calculated
* marstgb: Legal marital status, United Kingdom
* educgb1: Highest level of education, United Kingdom
* edubgb2: Highest level of education, United Kingdom: Up to Ph.D or equivalent
* eduyrs: Years of full-time education completed
* wkhtot: Total hours normally worked per week in main job overtime included
* isco08: Occupation, ISCO08
* pdwrk: Doing last 7 days: paid work
* pdjobev: Ever had a paid job
* hinctnta: Household's total net income, all sources
* edufbgb2: Father's highest level of education, United Kingdom: Up to Ph.D or equivalent
* edumbgb2: Mother's highest level of education, United Kingdom: Up to Ph.D or equivalent
* region: Region

# Just a helpful function to print all
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

# 