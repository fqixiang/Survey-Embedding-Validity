library(tidyverse)
library(haven)
library(readxl)


#### Get variants of questions (without caring for quality estimates) ####
#1. Question text data
header_codings = read_csv('questions_header.csv') %>% 
  names()

question_data = read_csv('questions.csv',
                    col_names = header_codings,
                    na = "\\N",
                    col_types = cols(.default = col_character())) %>% 
  filter(country_id == "GB") %>% 
  arrange(item_name) %>% 
  select(item_id, item_name, introduction_text, rfa_text, answer_text)


question_data %>% 
  filter(item_name == "ginveco")



#2. Question coding information
# not sure if these reliability, validity and quality scores are estimated from the prediction models or from experimental studies
question_coding = read_csv('qixiang.csv',
                           col_types = cols(.default = col_character())) %>% 
  filter(cntry == "GB") %>% 
  mutate(item_name = tolower(var_names)) %>% 
  select(-X1, -rounds, -cntry, -lang, -var_names, -country_id, -lang_iso, -lang_iso2, -admin_letter, -admin_number,
         -dom_backgrou, -dom_consumer, -dom_european, -dom_family, -dom_health, -dom_leisure, -dom_other, -dom_personal, -dom_work,
         -domain, -intpoldomain, -natpoldomain) %>% 
  select(study_name, item_id, exp_name, item_name, item_concept, mt, year, user_id, username, conc_simple, everything()) 

question_data %>% 
  select(rfa_text)

#3. Merge the two
question_coding %>% 
  left_join(question_data) %>% 
  select(-study_name, -item_id, -mt, -year, -user_id, -reliability_coefficient, -validity_coefficient, -quality, -rel, -val, -rel_lo, -rel_hi, -relz, -relz_se, -val_lo, -val_hi, -valz, -valz_se) %>% 
  select(exp_name, item_name, item_concept, username, conc_simple, introduction_text, rfa_text, answer_text, everything()) %>% 
  filter(!is.na(rfa_text)) %>% 
  write_csv("ess_question_coding_20210321.csv")





#### Get questions whose quality estimates are available ####
# New estimates, directly from UPF
estimates <- read_xlsx("estimate_df_20200518-final-cleaned.xlsx")

# Same transformations as performed for SQL dump,
#   also rename "cntry" to country_id as in MySQL db
estimates <- estimates %>%
  mutate(study_name = paste0("ESS", rounds),
         item_name = tolower(var_names),
         country_id = cntry)



estimates_joined %>% 
  filter(cntry == "GB") %>% 
  select(-rounds, -cntry, -lang, -var_names, -country_id, -lang_iso, -lang_iso2, -admin_letter, -admin_number,
         -dom_backgrou, -dom_consumer, -dom_european, -dom_family, -dom_health, -dom_leisure, -dom_other, -dom_personal, -dom_work,
         -domain, -intpoldomain, -natpoldomain) %>% 
  select(study_name, item_id, exp_name, item_name, item_concept, mt, year, user_id, username, conc_simple, everything()) %>% 
  arrange(item_id) 

#write_csv("ess_question_coding_20210310.csv")




