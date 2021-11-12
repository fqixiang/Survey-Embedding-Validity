library(tidyverse)
library(lsa)
library(plotly)
library(corpus)

# import synthetic question data and embedding data
question_data = read_csv('./data/Synthetic_Questions_Simple_20210422.csv')
embedding_data = read_csv('sentence_embeddings_synthetic_20210422.csv')


# combine the two data sets
full_data = question_data %>% 
  left_join(embedding_data) %>% 
  group_by(concept_concrete) %>% 
  mutate(concept_concrete_id = cur_group_id()) %>% 
  ungroup()


# calculate cosine similarity for sentence pairs
cosine_df = tibble(variable_id = numeric(),
                   question_id = numeric(),
                   cosine_score = numeric())

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
    embedding_main = data_combined[2, ] %>% 
      select(contains('dim')) %>% 
      as.numeric()
    
    embedding_reference = data_combined[j, ] %>% 
      select(contains('dim')) %>% 
      as.numeric()
    
    embedding_reference_question_id = data_combined[j, ] %>% 
      select(question_id) %>% 
      as.numeric()
    
    cosine_value = cosine(embedding_main,
                          embedding_reference) %>% 
      as.numeric()
  
    
    cosine_df = cosine_df %>% 
      add_row(variable_id = i,
              question_id = embedding_reference_question_id,
              cosine_score = cosine_value)

  }
}

cosine_df %>% 
  filter(variable_id == 1) %>% 
  print(n=Inf)

final_data = cosine_df %>% 
  left_join(question_data) %>% 
  replace_na(list(similarity = "random")) %>% 
  fill(concept_type, basic_concept)
  

final_data = final_data %>% 
  left_join(select(question_data, question_id, concept_general, concept_concrete, assertion, form_request, rfa),
            by = "question_id") %>% 
  mutate(concept_general = coalesce(concept_general.x, concept_general.y),
         concept_concrete = coalesce(concept_concrete.x, concept_concrete.y),
         assertion = coalesce(assertion.x, assertion.y),
         rfa = coalesce(rfa.x, rfa.y),
         form_request = coalesce(form_request.x, form_request.y)) %>% 
  select(-concept_general.x, -concept_general.y, 
         -concept_concrete.x, -concept_concrete.y,
         -assertion.x, -assertion.y,
         -rfa.x, -rfa.y,
         -form_request.x, -form_request.y)

final_data %>% 
  distinct(variable_id, question_id)

final_data %>% 
  filter(variable_id == 2) %>% 
  print(n=Inf)

final_data %>% 
  distinct(form_request)

plot = final_data %>% 
  filter(cosine_score != 1,
         form_request != 'direct instruction'
         ) %>% 
  mutate(similarity = factor(similarity, levels=c('same','similar','dissimilar','random'))) %>% 
  ggplot(aes(x = similarity, y = cosine_score)) +
  geom_point(aes(text = rfa)) +
  facet_wrap(~basic_concept)

ggplotly(plot, tooltip="text")




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




# check which word is most important in determining the cosine similarity
# permute every sentence by leaving out one word at a time
# compute the embeddings
# compute the cosine similarity scores between 1) the original 2) the permutation




# try negation

# import synthetic question data and embedding data
question_negation_data = read_csv('./data/Synthetic_Questions_Simple_Negation_20210503.csv')
embedding_negation_data = read_csv('sentence_embeddings_synthetic_negation_20210503.csv')


# combine the two data sets
full_negation_data = question_negation_data %>% 
  left_join(embedding_negation_data) %>% 
  group_by(concept_concrete) %>% 
  mutate(concept_concrete_id = cur_group_id()) %>% 
  ungroup()


# calculate cosine similarity for sentence pairs
cosine_negation_df = tibble(concept_concrete_id = numeric(),
                            question_id = numeric(),
                            cosine_score = numeric())

full_negation_data %>% 
  select(concept_concrete_id) %>% 
  distinct()

for (i in 1:39) {
  data_foo = full_negation_data %>% 
    filter(concept_concrete_id == i)
  
  for (j in 2:3) {
    # the embedding of the direct request
    embedding_main = data_foo[1, ] %>% 
      select(contains('dim')) %>% 
      as.numeric()
    
    embedding_reference = data_foo[j, ] %>% 
      select(contains('dim')) %>% 
      as.numeric()
    
    embedding_reference_question_id = data_foo[j, ] %>% 
      select(question_id) %>% 
      as.numeric()
    
    cosine_value = cosine(embedding_main,
                          embedding_reference) %>% 
      as.numeric()
    
    
    cosine_negation_df = cosine_negation_df %>% 
      add_row(concept_concrete_id = i,
              question_id = embedding_reference_question_id,
              cosine_score = cosine_value)
    
  }
}

negation_plot = cosine_negation_df %>% 
  left_join(question_negation_data) %>% 
  mutate(negation = factor(negation, levels=c('no','yes','or'))) %>% 
  ggplot(aes(x = negation, y = cosine_score)) +
  geom_point(aes(text = rfa)) +
  facet_wrap(~concept_concrete)

ggplotly(negation_plot)
