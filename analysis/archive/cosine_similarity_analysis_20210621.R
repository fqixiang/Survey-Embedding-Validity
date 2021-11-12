library(tidyverse)
library(lsa)
#library(plotly)
library(corpus)
library(readxl)
library(lme4)
library(lmerTest)
library(doParallel)
library(doSNOW)
library(showtext)
font.add("Latex", "C:/Users/6161138/AppData/Local/Microsoft/Windows/Fonts/lmroman12-regular.otf")
showtext_auto()


# cosine function
cosine <- function(x,y) {
  return(crossprod(x,y)/sqrt(crossprod(x)*crossprod(y)))
}

# jaccard function
clean_string = function(string) {
  string_lower = tolower(string)
  string_clean = gsub('[[:punct:] ]+', ' ', string_lower)
}

compute_jaccard = function(sent1, sent2) {

  sent1_clean = clean_string(sent1)
  sent2_clean = clean_string(sent2)
  
  sent1_vec = text_tokens(sent1_clean, stemmer = "en")[[1]]
  sent2_vec = text_tokens(sent2_clean, stemmer = "en")[[1]]
  
  sent_union = union(sent1_vec, sent2_vec)
  sent_intersect = intersect(sent1_vec, sent2_vec)
  
  jaccard_score = length(sent_intersect) / length(sent_union)
  
  return(jaccard_score)
}


# function to read data sets (from multiple embedding models)
read_data = function(question_data_path,
                     question_supplementary_path,
                     embedding_data_path) {
  
  question_data = read_excel(question_data_path)
  question_supplementary = read_excel(question_supplementary_path)
  embedding_data = read_csv(embedding_data_path) %>% 
    rename(row_id = question_id)
  
  # combine the data sets
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
  
  return(full_data)
}

# function to calculate cosine similarity scores for survey questions of varying degrees of similarity
compute_cosine_h1 = function(input_data){
  
  # make an empty cosine df to store the data
  cosine_df = tibble(final_id = character(),
                     similarity = character(),
                     cosine_score = numeric(),
                     jaccard_score = numeric())
  
  # number of trios
  n_groups = input_data %>% 
    select(final_id) %>% 
    n_distinct()
  
  # unique trio ids
  final_ids = input_data %>% 
    distinct(final_id) %>% 
    pull()
  
  for (i in 1:n_groups){
    data_foo = input_data %>% 
      filter(final_id == final_ids[i])
    
    embedding_reference = data_foo %>%
      filter(similarity == "reference") %>% 
      select(contains('dim')) %>% 
      as.numeric()
    
    embedding_similar = data_foo %>% 
      filter(similarity == "high") %>% 
      select(contains('dim')) %>% 
      as.numeric()
    
    embedding_dissimilar = data_foo %>% 
      filter(similarity == "low") %>% 
      select(contains('dim')) %>% 
      as.numeric()
    
    rfa_reference = data_foo %>%
      filter(similarity == "reference") %>% 
      pull(rfa)
    
    rfa_similar = data_foo %>%
      filter(similarity == "high") %>% 
      pull(rfa)
    
    rfa_dissimilar = data_foo %>%
      filter(similarity == "low") %>% 
      pull(rfa)
    
    cosine_value_similar = cosine(embedding_reference,
                                  embedding_similar) %>% 
      as.numeric()
    
    cosine_value_dissimilar = cosine(embedding_reference,
                                     embedding_dissimilar) %>% 
      as.numeric()
    
    jaccard_value_similar = compute_jaccard(rfa_reference,
                                            rfa_similar)
    
    jaccard_value_dissimilar = compute_jaccard(rfa_reference,
                                               rfa_dissimilar)
    
    cosine_df = cosine_df %>% 
      add_row(final_id = final_ids[i],
              similarity = 'high',
              cosine_score = cosine_value_similar,
              jaccard_score = jaccard_value_similar) %>% 
      add_row(final_id = final_ids[i],
              similarity = 'low',
              cosine_score = cosine_value_dissimilar,
              jaccard_score = jaccard_value_dissimilar) 
    
    if (i %% 100 == 0) {
      cat(sprintf("Progress: %d/%d\n", i, n_groups))
    }
  }
  
  return(cosine_df)
}



# function to calculate cosine similarity scores for h2
compute_cosine_h2 = function(input_data) {
  # make an empty cosine df to store the data
  cosine_df = tibble(final_id = character(),
                     similarity = character(),
                     form_request = character(),
                     cosine_score = numeric(),
                     jaccard_score = numeric())
  
  # number of trios
  n_groups = input_data %>% 
    select(final_id) %>% 
    n_distinct()
  
  # unique trio ids
  final_ids = input_data %>% 
    distinct(final_id) %>% 
    pull()
  
  for (i in 1:n_groups){
    data_between = input_data %>% 
      filter(final_id == final_ids[i],
             similarity != "high")
    
    embedding_reference = data_between %>%
      filter(similarity == "reference") %>% 
      select(contains('dim')) %>% 
      as.numeric()
    
    rfa_reference = data_between %>% 
      filter(similarity == "reference") %>% 
      pull(rfa)
    
    embedding_dissimilar = data_between %>% 
      filter(similarity == "low") %>% 
      select(contains('dim')) %>% 
      as.numeric()
    
    rfa_dissimilar = data_between %>% 
      filter(similarity == "low") %>% 
      pull(rfa)
    
    
    cosine_value_dissimilar = cosine(embedding_reference,
                                     embedding_dissimilar) %>% 
      as.numeric()
    
    jaccard_dissimilar = compute_jaccard(rfa_reference, rfa_dissimilar)
    
    form_request_reference = data_between %>%
      filter(similarity == "reference") %>% 
      pull(form_request)
    
    cosine_df = cosine_df %>% 
      add_row(final_id = final_ids[i],
              similarity = 'dissimilar',
              form_request = form_request_reference,
              cosine_score = cosine_value_dissimilar,
              jaccard_score = jaccard_dissimilar)
    
    
    question_id_reference = data_between %>% 
      filter(similarity == "reference") %>% 
      pull(question_id)
    
    row_id_reference = data_between %>% 
      filter(similarity == "reference") %>% 
      pull(row_id)
    
    data_within = input_data %>% 
      filter(similarity == "reference") %>% 
      filter(question_id == question_id_reference) %>% 
      filter(row_id != row_id_reference)
    
    n_forms = nrow(data_within)
    
    j = 1
    for (j in 1:n_forms){
      embedding_similar = data_within[j, ] %>% 
        select(contains('dim')) %>% 
        as.numeric()
      
      rfa_similar = data_within[j, ] %>% 
        pull(rfa)
      
      cosine_value_similar = cosine(embedding_reference,
                                    embedding_similar) %>% 
        as.numeric()
      
      
      jaccard_similar = compute_jaccard(rfa_reference, rfa_similar)
      
      cosine_df = cosine_df %>% 
        add_row(final_id = final_ids[i],
                similarity = 'reference',
                form_request = pull(data_within[j, ], form_request),
                cosine_score = cosine_value_similar,
                jaccard_score = jaccard_similar)
      
    }
  }
  return(cosine_df)
}


# import synthetic question data and embedding data from multiple embedding models
embedding_data_paths = c('../data/embeddings/synthetic_count.csv',
                         '../data/embeddings/synthetic_tf_idf.csv',
                         '../data/embeddings/synthetic_random300.csv',
                         '../data/embeddings/synthetic_random768.csv',
                         '../data/embeddings/synthetic_random1024.csv',
                         '../data/embeddings/synthetic_fasttext.csv',
                         '../data/embeddings/synthetic_bert_base_uncased.csv',
                         '../data/embeddings/synthetic_bert_large_uncased.csv',
                         '../data/embeddings/synthetic_paraphrase_mpnet_base_v2.csv',
                         '../data/embeddings/synthetic_stsb_mpnet_base_v2.csv',
                         '../data/embeddings/synthetic_stsb_roberta_base_v2.csv',
                         '../data/embeddings/synthetic_stsb_roberta_large.csv')

full_data_ls = lapply(embedding_data_paths,
                      read_data,
                      question_data_path = '../data/synthetic/Synthetic_Questions_Controlled_Variants.xlsx',
                      question_supplementary_path = '../data/synthetic/Synthetic_Questions_Controlled.xlsx')

lapply(c(1,2,3,4),
       as.character)

question_short_concept_names = read_excel('../data/synthetic/Synthetic_Questions_Controlled_Shorter_Concept_Names.xlsx') %>% 
  select(concrete_concept, concrete_concept_reference)

# make a df for the cosine scores for h1
cl <- makeCluster(detectCores()[1]-1)
registerDoParallel(cl)
iterations = length(full_data_ls)

cosine_df_h1_ls <- foreach(i = 1:iterations, .packages=c("dplyr", "tidyr", "tibble", "corpus")) %dopar% {
  compute_cosine_h1(full_data_ls[[i]])
}

stopCluster(cl)
saveRDS(cosine_df_h1_ls, file = "cosine_df_h1_ls.rds")

# make a df for the cosine scores for h2
cl <- makeCluster(detectCores()[1]-1)
#registerDoParallel(cl)
registerDoSNOW(cl)
iterations = length(full_data_ls)
pb <- txtProgressBar(max = iterations, style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

cosine_df_h2_ls <- foreach(i = 1:iterations, 
                           .packages=c("dplyr", "tidyr", "tibble", "corpus"),
                           .options.snow = opts) %dopar% {
  compute_cosine_h2(full_data_ls[[i]])
}

close(pb)
stopCluster(cl)
#ans <- Reduce("cbind", results)
saveRDS(cosine_df_h2_ls, file = "cosine_df_h2_ls.rds")





cl <- makeCluster(detectCores()[1]-1)
registerDoParallel(cl)
iterations = 3

foo_data_ls = list("a" = 1, "b" = 2, "c" = 3)

foo_result_ls <- foreach(i = 1:3, .packages=c("dplyr", "tidyr", "tibble", "corpus")) %dopar% {
  as.character(foo_data_ls[[i]])
}

stopCluster(cl)

foo_data_ls[[1]]







# read the previously saved cosine data
cosine_df_h1_ls = read_rds("cosine_df_h1_ls.rds")
cosine_df_h2_ls = read_rds("cosine_df_h2_ls.rds")

model_ls = c("BOW",
             "TF-IDF",
             "Random-300",
             "Random-768",
             "Random-1024",
             "FastText",
             "BERT-Base-Uncased",
             "BERT-Large-Uncased",
             "Paraphrase-MPNET-Base",
             "STSB-MPNET-BASE",
             "STSB-ROBERTA-BASE",
             "STSB-ROBERTA-Large")

for (i in 1:length(model_ls)) {
  cosine_df_h1_ls[[i]] <- cosine_df_h1_ls[[i]] %>% 
    mutate(Model = model_ls[[i]])
  
  cosine_df_h2_ls[[i]] <- cosine_df_h2_ls[[i]] %>% 
    mutate(Model = model_ls[[i]])
  
  full_data_ls[[i]] <- full_data_ls[[i]] %>% 
    mutate(Model = model_ls[[i]])
}


# show cosine scores of high vs. low similarity rfas
model_id = 4
cosine_df_ls[[model_id]] %>% 
  left_join(select(full_data_ls[[model_id]], final_id, question_id, similarity), by = c("final_id", "similarity")) %>% 
  left_join(select(question_supplementary, -similarity), by = c("question_id")) %>% 
  ggplot(aes(x = similarity, y = cosine_score, label=concrete_concept)) +
  geom_point() +
  #geom_text(size = 4,
  #          position=position_jitter(width=0,height=1)) +
  #facet_wrap(~interaction(basic_concept, group_id)) +
  facet_wrap(~basic_concept)

cosine_df_ls %>% 
  bind_rows() %>% 
  left_join(select(bind_rows(full_data_ls), final_id, question_id, similarity), by = c("final_id", "similarity")) %>% 
  left_join(select(question_supplementary, -similarity), by = c("question_id")) %>% 
  ggplot(aes(x = similarity, y = cosine_score, label=concrete_concept)) +
  geom_point() +
  #geom_text(size = 4,
  #          position=position_jitter(width=0,height=1)) +
  #facet_wrap(~interaction(basic_concept, group_id)) +
  facet_wrap(~basic_concept)



question_data = read_excel('../data/synthetic/Synthetic_Questions_Controlled_Variants.xlsx')
question_supplementary = read_excel('../data/synthetic/Synthetic_Questions_Controlled.xlsx')


color_pallete = c(rep("#F8766D", 2),
                  rep("#A3A500", 3),
                  rep("#00BF7D", 1),
                  rep("#00B0F6", 2),
                  rep("#E76BF3", 4))

color_pallete = c(rep("#F8766D", 2),
                  rep("#7CAE00", 1),
                  rep("#00BFC4", 2),
                  rep("#C77CFF", 4))

set.seed(123)
fig_h1 = cosine_df_h1_ls %>% 
  bind_rows() %>% 
  left_join(select(bind_rows(full_data_ls), final_id, question_id, similarity), by = c("final_id", "similarity")) %>% 
  left_join(select(question_supplementary, -similarity), by = c("question_id")) %>% 
  left_join(question_short_concept_names) %>% 
  #filter(!Model %in% c("Random-300", "Random-768", "Random-1024")) %>% 
  mutate(Model = factor(Model, levels = model_ls)) %>% 
  group_by(final_id, Model) %>% 
  mutate(cos_dif = cosine_score - cosine_score[similarity == "low"]) %>% 
  ungroup() %>%
  filter(similarity == "high") %>% 
  select(final_id, similarity, cosine_score, cos_dif, everything()) %>% 
  sample_frac(0.1) %>%
  ggplot() +
  geom_point(aes(x = factor(concrete_concept_reference), 
                 y = cos_dif,
                 color = Model,
                 group = Model),
             size = 1,
             alpha = 1/10,
             position = position_dodge(width = 0.4)) +
  geom_hline(yintercept = 0) +
  #scale_x_discrete(guide = guide_axis(n.dodge=2)) + 
  #ylim(-1,1) +
  theme_bw() +
  theme(text=element_text(family="Latex", size = 12),
        legend.position = "top",
        legend.text = element_text(size = 9), 
        #legend.direction = "horizontal",
        #legend.justification = c(1,0),
        #legend.position = c(1,0),
        #legend.box = "horizontal",
        axis.text=element_text(size=5),
        axis.title.x = element_blank()) + 
  scale_color_manual(values = color_pallete) + 
  #guides(shape = guide_legend(title.position = "top")) + 
  facet_wrap(~basic_concept, scales = "free_x", ncol = 2) +
  guides(color = guide_legend(override.aes= list(alpha = 1))) +
  labs(#title= "Figure H1: Distribution of cos(ref, similar) - cos(ref, dissimilar) scores, across 13 basic concepts.",
    y="Difference in cosine scores") 

ggsave("fig_h1.pdf", fig_h1, device = "pdf", width = 170, height = 225, units = "mm", dpi = 300)

#library(scales)
#show_col(hue_pal()(5))

# show difference in cosine scores between high and low similarity rfas
fig_h1 = cosine_df_h1_ls %>% 
  bind_rows() %>% 
  left_join(select(bind_rows(full_data_ls), final_id, question_id, similarity), by = c("final_id", "similarity")) %>% 
  left_join(select(question_supplementary, -similarity), by = c("question_id")) %>% 
  left_join(question_short_concept_names) %>% 
  group_by(final_id) %>% 
  mutate(cos_dif = cosine_score - cosine_score[similarity == "low"]) %>% 
  ungroup() %>%
  filter(similarity == "high") %>% 
  select(final_id, similarity, cosine_score, cos_dif, everything()) %>% 
  ggplot() +
  geom_point(aes(x = factor(concrete_concept_reference), 
                 y = cos_dif,
                 shape = Model,
                 group = Model),
             size = 1,
             alpha = 1/10,
             position = position_dodge(width = 0.4)) +
  geom_hline(yintercept = 0) +
  scale_x_discrete(guide = guide_axis(n.dodge=2)) + 
  #ylim(-1,1) +
  theme_bw() +
  theme(text=element_text(family="Latex", size = 12),
        legend.position = "top",
        #legend.direction = "horizontal",
        #legend.justification = c(1,0),
        #legend.position = c(1,0),
        #legend.box = "horizontal",
        axis.text=element_text(size=5),
        axis.title.x = element_blank()) + 
  #guides(shape = guide_legend(title.position = "top")) + 
  facet_wrap(~basic_concept, scales = "free_x") +
  guides(shape = guide_legend(override.aes= list(alpha = 1))) +
  labs(#title= "Figure H1: Distribution of cos(ref, similar) - cos(ref, dissimilar) scores, across 13 basic concepts.",
       y="Difference in cosine scores")

fig_h1

ggsave("fig_h1.pdf", fig_h1, device = "pdf", width = 8.5, dpi = 500)
#ggsave("fig_h1.png", fig_h1, device = "png", width = 8.5, dpi = 500)


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



# compare quantitative differences across similarity conditions
# the similarity scores are grouped within each basic variable type
model_id = 5
cos_diff_df = cosine_df_h1_ls[[model_id]] %>% 
  left_join(select(full_data_ls[[model_id]], final_id, question_id, similarity, form_request), by = c("final_id", "similarity")) %>% 
  left_join(select(question_supplementary, -similarity), by = c("question_id")) %>% 
  group_by(final_id) %>% 
  mutate(cos_dif = cosine_score - cosine_score[similarity == "low"]) %>% 
  ungroup() %>%
  filter(similarity == "high") %>% 
  select(final_id, similarity, cosine_score, cos_dif, everything())

cos_diff_df %>% 
  mutate(cosine_diff_binary = if_else(cos_dif > 0, 1, 0)) %>% 
  summarise(prop_positive = mean(cosine_diff_binary))
# model 1: 0.967
# model 2: 0.962
# model 3: 0.939
# model 4: 0.986

model_cosine = lmer(cos_dif ~ 1 + (1 | concrete_concept) + (1 | form_request),
                    data = cos_diff_df)

summary(model_cosine)
#



# comparing conceptually different but syntactically similar questions with conceptually similar but syntactically different questions
cosine_df_h2_ls[[4]] %>% 
  group_by(final_id, form_request, similarity) %>% 
  summarise(cosine_score_ave = mean(cosine_score)) %>% 
  ungroup() %>% 
  ggplot(aes(x = final_id,
             y = cosine_score_ave,
             color = similarity)) +
  geom_point() +
  # theme(axis.title.x=element_blank(),
  #       axis.text.x=element_blank(),
  #       axis.ticks.x=element_blank()
  #       ) +
  scale_x_discrete(labels = NULL, 
                   breaks = NULL
                   )

final_id_to_concepts = full_data_ls[[1]] %>% 
  select(final_id, basic_concept, concrete_concept) %>% 
  left_join(question_short_concept_names)

fig_h2 = cosine_df_h2_ls %>%
  bind_rows() %>% 
  group_by(Model, final_id, form_request, similarity) %>% 
  summarise(cosine_score_ave = mean(cosine_score)) %>% 
  ungroup() %>% 
  group_by(final_id) %>% 
  mutate(cosine_diff = cosine_score_ave - cosine_score_ave[similarity == "dissimilar"]) %>% 
  ungroup() %>% 
  filter(similarity == "reference") %>% 
  left_join(final_id_to_concepts) %>% 
  ggplot(aes(x = concrete_concept_reference,
             y = cosine_diff)) +
  geom_point(aes(shape = Model,
                 group = Model),
             size = 1,
             alpha = 1/50,
             position = position_dodge(width = 0.4)) +
  geom_hline(yintercept = 0) +
  scale_x_discrete(guide = guide_axis(n.dodge=2)) + 
  theme_bw() +
  theme(text=element_text(family="Latex", size = 12),
        legend.position = "top",
        axis.text=element_text(size=5),
        axis.title.x = element_blank()) + 
  #scale_x_discrete(labels = NULL, 
  #                 breaks = NULL
  #) +
  facet_wrap(~basic_concept,
             scales = "free_x") +
  guides(shape = guide_legend(override.aes= list(alpha = 1))) +
  labs(#title= "Figure H2: Distribution of cos(ref, identical) - cos(ref, dissimilar) scores, across 13 basic concepts.",
       y="Difference in cosine scores")

fig_h2

#ggsave("fig_h2.png", fig_h2, device = "png", width = 8.5, dpi = 500)
ggsave("fig_h2.pdf", fig_h2, device = "pdf", width = 8.5, dpi = 500)

# check each model's performance (as in differentiating conceptually similar from disimilar questions)
cosine_df_h2_ls[[4]] %>% 
  group_by(final_id, form_request, similarity) %>% 
  summarise(cosine_score_ave = mean(cosine_score)) %>% 
  ungroup() %>% 
  group_by(final_id) %>% 
  mutate(cosine_diff = cosine_score_ave - cosine_score_ave[similarity == "dissimilar"]) %>% 
  ungroup() %>% 
  filter(similarity == "reference") %>% 
  mutate(cosine_diff_binary = if_else(cosine_diff > 0, 1, 0)) %>% 
  summarise(prop_positive = mean(cosine_diff_binary))
  
# model 1: 0.977
# model 2: 0.959
# model 3: 0.956
# model 4: 0.993




# correlation between jaccard and cosine
cosine_df_h2_ls[[4]] %>% 
  group_by(final_id, form_request, similarity) %>% 
  summarise(cosine_score_ave = mean(cosine_score),
            jaccard_score_ave = mean(jaccard_score)) %>% 
  ungroup() %>% 
  ggplot(aes(x = cosine_score_ave,
             y = jaccard_score_ave)) +
  geom_point() +
  geom_smooth(method = "gam")

cosine_jaccard_df = cosine_df_h2_ls[[4]] %>% 
  group_by(final_id, form_request, similarity) %>% 
  summarise(cosine_score_ave = mean(cosine_score),
            jaccard_score_ave = mean(jaccard_score)) %>% 
  ungroup() %>% 
  select(cosine_score_ave, jaccard_score_ave)

cor.test(pull(cosine_jaccard_df, cosine_score_ave), pull(cosine_jaccard_df, jaccard_score_ave))
# Pearson's correlation: -0.3837725