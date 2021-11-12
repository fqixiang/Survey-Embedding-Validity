library(tidyverse)
library(lsa)
library(plotly)
library(corpus)
library(readxl)

# import synthetic question data and embedding data
question_supplementary = read_excel('./data/Synthetic_Questions_Controlled_20210611.xlsx')
question_data = read_excel('./data/Synthetic_Questions_Controlled_Variants_20210614.xlsx')
embedding_data = read_csv('./data/synthetic_embedding_df_paraphrase_mpnet_base_v2_20210614.csv') %>% 
  rename(row_id = question_id)

# combine the two data sets
full_data = question_data %>% 
  left_join(question_supplementary) %>% 
  left_join(embedding_data)

# rearrange the dataframe
full_data = full_data %>% 
  arrange(group_id, form_request, question_id) %>% 
  group_by(group_id, form_request, question_id, similarity) %>% 
  mutate(id_x = row_number()) %>% 
  ungroup() %>% 
  arrange(group_id, form_request, id_x) %>% 
  #select(group_id, form_request, id_x, similarity, rfa)
  mutate(final_id = paste(group_id, form_request, id_x, sep = '_')) %>% 
  select(-id_x) %>% 
  select(final_id, everything())

# 


# 1812 groups of comparison

# calculate cosine similarity for sentence pairs
cosine_df = tibble(final_id = character(),
                   similarity = character(),
                   cosine_score = numeric())

n_groups = full_data %>% 
  select(final_id) %>% 
  n_distinct()

final_ids = full_data %>% 
  distinct(final_id) %>% 
  pull()


for (i in 1:n_groups){
  data_foo = full_data %>% 
    filter(final_id == final_ids[i])
  
  embedding_reference = data_foo[1, ] %>% 
    select(contains('dim')) %>% 
    as.numeric()
  
  embedding_similar = data_foo[2, ] %>% 
    select(contains('dim')) %>% 
    as.numeric()
  
  embedding_dissimilar = data_foo[3, ] %>% 
    select(contains('dim')) %>% 
    as.numeric()
  
  cosine_value_similar = cosine(embedding_reference,
                                embedding_similar) %>% 
    as.numeric()
  
  cosine_value_dissimilar = cosine(embedding_reference,
                                   embedding_dissimilar) %>% 
    as.numeric()
  
  cosine_df = cosine_df %>% 
    add_row(final_id = final_ids[i],
            similarity = 'high',
            cosine_score = cosine_value_similar) %>% 
    add_row(final_id = final_ids[i],
            similarity = 'low',
            cosine_score = cosine_value_dissimilar) 
}


cosine_df %>% 
  left_join(select(full_data, final_id, question_id, similarity), by = c("final_id", "similarity")) %>% 
  left_join(select(question_supplementary, -similarity), by = c("question_id")) %>% 
  ggplot(aes(x = similarity, y = cosine_score, label=concrete_concept)) +
  geom_point() +
  #geom_text(size = 4,
  #          position=position_jitter(width=0,height=1)) +
  #facet_wrap(~interaction(basic_concept, group_id)) +
  facet_wrap(~basic_concept)


cosine_df %>% 
  left_join(select(full_data, final_id, question_id, similarity), by = c("final_id", "similarity")) %>% 
  left_join(select(question_supplementary, -similarity), by = c("question_id")) %>% 
  group_by(final_id) %>% 
  mutate(cos_dif = cosine_score - lead(cosine_score, n = 1L, default = NA)) %>% 
  ungroup() %>% 
  select(final_id, similarity, cosine_score, cos_dif, everything()) %>% 
  drop_na() %>% 
  ggplot(aes(x = factor(concrete_concept), 
             y = cos_dif, 
             label=concrete_concept)) +
  geom_point() +
  geom_hline(yintercept = 0) +
  facet_wrap(~basic_concept, scales = "free_x")
  

# plot = final_data %>%
#   filter(cosine_score != 1,
#          form_request != 'direct instruction'
#          ) %>%
#   mutate(similarity = factor(similarity, levels=c('same','similar','dissimilar','random'))) %>%
#   ggplot(aes(x = similarity, y = cosine_score)) +
#   geom_point(aes(text = rfa)) +
#   facet_wrap(~basic_concept)
# 
# ggplotly(plot, tooltip="text")




# jacaard similarity
jaccard_index = function(sent1, sent2) {
  
  clean_string = function(string) {
    string_lower = tolower(string)
    string_clean = gsub('[[:punct:] ]+',' ',string_lower)
  }
  
  sent1_clean = clean_string(sent1)
  sent2_clean = clean_string(sent2)
  
  sent1_vec = text_tokens(sent1_clean, stemmer = "en")[[1]]
  sent2_vec = text_tokens(sent2_clean, stemmer = "en")[[1]]
  
  sent_union = union(sent1_vec, sent2_vec)
  sent_intersect = intersect(sent1_vec, sent2_vec)
  
  jaccard_score = length(sent_intersect) / length(sent_union)
  
  return(jaccard_score)
}


sent1 = "Hi I like you you"
sent2 = "hi how are you?"


jaccard_index(sent1, sent2)

# run it
jaccard_df = tibble(variable_id = numeric(),
                    question_id = numeric(),
                    jaccard_score = numeric())

for (i in 1:21) {
  data_foo = full_data %>% 
    filter(variable_id == i)
  
  concept_concrete_id_all = 1:63
  
  concept_concrete_id = data_foo %>% 
    select(concept_concrete_id) %>% 
    distinct() %>% 
    pull()
  
  concept_concrete_id_rest = concept_concrete_id_all[!(concept_concrete_id_all %in% concept_concrete_id)]
  
  set.seed(123)
  concept_concrete_id_random = sample(concept_concrete_id_rest, 1)
  
  data_random = full_data %>% 
    filter(concept_concrete_id == concept_concrete_id_random)
  
  data_combined = bind_rows(data_foo, data_random)
  
  for (j in 1:24) {
    # the embedding of the direct request
    rfa_main = data_combined[2, ] %>% 
      select(rfa) %>% 
      as.character()
    
    rfa_reference = data_combined[j, ] %>% 
      select(rfa) %>% 
      as.character()
    
    rfa_reference_question_id = data_combined[j, ] %>% 
      select(question_id) %>% 
      as.numeric()
    
    jaccard_value = jaccard_index(rfa_main,
                                  rfa_reference) %>% 
      as.numeric()
    
    
    jaccard_df = jaccard_df %>% 
      add_row(variable_id = i,
              question_id = rfa_reference_question_id,
              jaccard_score = jaccard_value)
    
  }
}


# check overall correlation between cosine and jaccard
final_data %>% 
  left_join(jaccard_df, by=c("variable_id", "question_id")) %>% 
  filter(cosine_score != 1,
         form_request != 'direct instruction'
  ) %>% 
  ggplot(aes(x = cosine_score,
             y = jaccard_score)) +
  geom_point() +
  geom_smooth(method="lm")


# check correlation between cosine and jaccard per basic variable type
final_data %>% 
  left_join(jaccard_df, by=c("variable_id", "question_id")) %>% 
  filter(cosine_score != 1,
         form_request != 'direct instruction'
  ) %>% 
  ggplot(aes(x = cosine_score,
             y = jaccard_score)) +
  geom_point() +
  facet_wrap(~basic_concept) + 
  geom_smooth(method="lm")


# check cosine and jaccard in parallel
final_data %>% 
  left_join(jaccard_df, by=c("variable_id", "question_id")) %>% 
  filter(cosine_score != 1,
         form_request != 'direct instruction'
  ) %>% 
  mutate(similarity = factor(similarity, levels=c('same','similar','dissimilar','random'))) %>% 
  select(basic_concept, cosine_score, jaccard_score, similarity, rfa) %>% 
  gather(measure, score, -similarity, -rfa, -basic_concept) %>% 
  ggplot(aes(x = similarity, y = score, color = measure)) +
  geom_point(aes(text = rfa)) +
  facet_wrap(~basic_concept)

# observation 1: cosine tends to be higher than jaccard
# observation 2: jaccard has a wider spread


# compare quantitative differences across similarity conditions
# the similarity scores are grouped within each basic variable type
library(lme4)
library(lmerTest)
data_similarity = final_data %>% 
  left_join(jaccard_df, by=c("variable_id", "question_id")) %>% 
  filter(cosine_score != 1,
         form_request != 'direct instruction'
  ) %>% 
  mutate(similarity = factor(similarity, levels=c('same','similar','dissimilar','random'))) %>% 
  select(basic_concept, cosine_score, jaccard_score, similarity, rfa)

model_cosine = lmer(jaccard_score ~ similarity + (1 | basic_concept),
                    data = mutate(data_similarity,
                                  similarity = relevel(similarity, ref = "same")))

summary(model_cosine)
#all conditions differ significantly in cosine similarity scores
#same observation for jaccard similarity scores


