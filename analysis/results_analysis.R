library(tidyverse)

results <- read_csv('prediction_results.csv')


results %>% 
  group_by(model, feature) %>% 
  summarise(mean_baseline = mean(baseline_average_response_within),
            sd_baseline = sd(baseline_average_response_within), 
            mean_MSE = mean(MSE),
            sd_MSE = sd(MSE)
            ) %>% 
  print(n = Inf)


results %>% 
  replace_na(list(Pearson_R = 0)) %>% 
  group_by(model, feature) %>% 
  summarise(mean_r = mean(Pearson_R),
            baseline_r = mean(baseline_best_pearson_r)) %>% 
  mutate(delta = (mean_r - baseline_r)/baseline_r * 100) %>% 
  print(n = Inf)

results_summary <- results %>% 
  group_by(model, feature) %>% 
  summarise(mean_baseline = mean(baseline_average_response_within),
            sd_baseline = sd(baseline_average_response_within), 
            mean_MSE = mean(MSE),
            sd_MSE = sd(MSE))

results_summary %>% 
  gather(model_name, ave_MSE, mean_MSE, -model, -feature, -sd_baseline, -sd_MSE)
  
?gather
results_summary %>% 
  ggplot(aes(shape = model,
             color = feature,
             y = mean_MSE,
             x = model)) +
  geom_point() +
  #geom_errorbar(aes(ymin=mean_MSE-sd_MSE, ymax=mean_MSE+sd_MSE), width=.2, position=position_dodge(0.5)) +
  geom_label(label = results_summary$feature, alpha = 0.1) +
  geom_hline(yintercept  = 0.0838) +
  theme(legend.position = "none")
?geom_hline



# analysis for probing tasks
dataset = read_csv('merged.csv')


# probe sentence length
dataset %>% 
  ggplot(aes(x = length)) +
  geom_histogram() +
  facet_wrap(~basic_concept)
#leave out int_int, evaluation; int_int, feelings; int_int, importance; int_int cognitive judgment, int_int belief
#we need form and concepts not seen in training set


dataset %>% 
  ggplot(aes(x = length_binned)) +
  geom_bar() +
  facet_wrap(~concrete_concept)
# 15-25
# lose jobs identity

dataset %>% 
  filter(length_binned == "15-25") %>% 
  ggplot(aes(x = length_binned)) +
  geom_bar() +
  facet_wrap(~basic_concept)

# probe basic concept:
dataset %>% 
  ggplot(aes(x = concrete_concept)) +
  geom_bar() +
  facet_wrap(~concrete_concept)
# we need sentence length and concrete concept not seen in the training set

# probe concrete concept:
# we need sentence length not seen in the training set (but can't avoid the same basic concepts)

# probe form of request
# we need sentence length not seen in the training set




# check correlations
tbl = table(dataset[c('length_binned', 'form_request')])
chi2 = chisq.test(tbl, correct=F)
c(chi2$statistic, chi2$p.value)
sqrt(chi2$statistic / sum(tbl))

# form of request, highly correlated with length
# basic concept, highly correlated with length
# concrete concept, highly correlated with length
# concrete concept highly correlated with basic concept

dataset %>% 
  filter(concrete_concept == "lose_jobs_gender_identity")

dataset %>% 
  distinct(length_binned)


library(readxl)
questions_df = read_excel('./data/synthetic/Synthetic_Questions_Controlled_Variants_20210614.xlsx') %>% 
  select(row_id, rfa)

set.seed(123)
dataset %>% 
  filter(length_binned == '0-10') %>% 
  select(row_id, concrete_concept) %>% 
  sample_n(100) %>% 
  left_join(questions_df) %>% 
  write_csv('sample_ids_concrete_concept.csv')


features_df = read_excel('./data/synthetic/Synthetic_Questions_Controlled_20210611.xlsx') 
