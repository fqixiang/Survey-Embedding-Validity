library(tidyverse)
library(Metrics)

results1 <- read_csv('results1.csv')
results2 <- read_csv('results2.csv')
results3 <- read_csv('results3.csv')
results4 <- read_csv('results4.csv')
results5 <- read_csv('results5.csv')

results = bind_rows(results1, results2, results3, results4, results5)


results %>% 
  distinct(feature)

compute_baselines <- function(results) {
  
  midpoint_mae_vec = as.numeric()
  ave_all_mae_vec = as.numeric()
  ave_person_mae_vec = as.numeric()
  ave_person_r_vec = as.numeric()
  weight_vec = as.numeric()
  
  sample_size = nrow(results)

  for (i in 0:9) {
    data = results %>% 
      filter(fold_id == i)
    
    #fold size
    weight = dim(data)[1]
    weight_vec = c(weight_vec, weight)
    
    #baseline midpoint, MAE
    midpoint_mae = mae(data$y_true, rep(0.5, weight))
    midpoint_mae_vec = c(midpoint_mae_vec, midpoint_mae)
    
    #baseline ave_all, MAE
    ave_all_mae = mae(data$y_true, data$ave_all_pred)
    ave_all_mae_vec = c(ave_all_mae_vec, ave_all_mae)
    
    #baseline ave_person, MAE and r
    ave_person_mae = mae(data$y_true, data$ave_person_pred)
    ave_person_mae_vec = c(ave_person_mae_vec, ave_person_mae)
    
    ave_person_r_ojb = cor.test(data$y_true, data$ave_person_pred)
    ave_person_r = as.vector(ave_person_r_ojb$estimate)
    ave_person_r_vec = c(ave_person_r_vec, ave_person_r)
  }
  
  r_within_person_ave = weighted.mean(ave_person_r_vec, weight_vec)
  r_se = sqrt((1-r_within_person_ave^2)/(sample_size-2))
  
  print(paste("MAE based on midpoint is:", weighted.mean(midpoint_mae_vec, weight_vec)))
  print(paste("MAE based on average is:", weighted.mean(ave_all_mae_vec, weight_vec)))
  print(paste("MAE based on within-person average is:", weighted.mean(ave_person_mae_vec, weight_vec)))
  print(paste("r based on within-person average is:", r_within_person_ave))
  print(paste("Lower 95% CI of r is:", r_within_person_ave - 1.96*r_se))
  print(paste("Upper 95% CI of r is:", r_within_person_ave + 1.96*r_se))
}

results %>% 
  filter(feature == "count") %>% 
  compute_baselines()


compute_model_performance <- function(results, model_name, feature_name) {
  data_all = results %>% 
    filter(model == model_name,
           feature == feature_name)
  
  
  mae_score_vec = as.numeric()
  r_vec = as.numeric()
  weight_vec = as.numeric()
  sample_size = nrow(data_all)
  
  for (i in 0:9) {
    data = data_all %>% 
      filter(fold_id == i)
    
    #fold size
    weight = dim(data)[1]
    weight_vec = c(weight_vec, weight)
    
    #MAE and r
    mae_score = mae(data$y_true, data$y_pred)
    mae_score_vec = c(mae_score_vec, mae_score)
    
    r_ojb = cor.test(data$y_true, data$y_pred)
    r = as.vector(r_ojb$estimate)
    r_vec = c(r_vec, r)
  }
  
  r = weighted.mean(r_vec, weight_vec, na.rm = TRUE)
  r_se = sqrt((1-r^2)/(sample_size-2))
    
  print(paste(model_name, feature_name))
  print(paste("MAE:", weighted.mean(mae_score_vec, weight_vec, na.rm = TRUE)))
  print(paste("r:", r))
  print(paste("Change is:", (r - 0.187)/0.187*100))
  print(paste("SE is:", r_se))
  print(paste("Lower 95% CI of r is:", r - 1.96*r_se))
  print(paste("Upper 95% CI of r is:", r + 1.96*r_se))
}

compute_model_performance(results, model = "lasso", feature = "count")
compute_model_performance(results, model = "lasso", feature = "tf_idf")
compute_model_performance(results, model = "lasso", feature = "random300")
compute_model_performance(results, model = "lasso", feature = "random768")
compute_model_performance(results, model = "lasso", feature = "random1024")
compute_model_performance(results, model = "lasso", feature = "ESS_fasttext")
compute_model_performance(results, model = "lasso", feature = "ESS_glove")
compute_model_performance(results, model = "lasso", feature = "ESS_bert_base_uncased")
compute_model_performance(results, model = "lasso", feature = "ESS_bert_large_uncased")
compute_model_performance(results, model = "lasso", feature = "ESS_all_distilroberta_v1")
compute_model_performance(results, model = "lasso", feature = "ESS_all_mpnet_base_v2")
compute_model_performance(results, model = "lasso", feature = "ESS_USE")

compute_model_performance(results, model = "rf", feature = "count")
compute_model_performance(results, model = "rf", feature = "tf_idf")
compute_model_performance(results, model = "rf", feature = "random300")
compute_model_performance(results, model = "rf", feature = "random768")
compute_model_performance(results, model = "rf", feature = "random1024")
compute_model_performance(results, model = "rf", feature = "ESS_fasttext")
compute_model_performance(results, model = "rf", feature = "ESS_glove")
compute_model_performance(results, model = "rf", feature = "ESS_bert_base_uncased")
compute_model_performance(results, model = "rf", feature = "ESS_bert_large_uncased")
compute_model_performance(results, model = "rf", feature = "ESS_all_distilroberta_v1")
compute_model_performance(results, model = "rf", feature = "ESS_all_mpnet_base_v2")
compute_model_performance(results, model = "rf", feature = "ESS_USE")


compute_r <- function(data, fold){
  data = data %>% 
    filter(fold_id == fold)
  
  r_obj = cor.test(data$y_true, data$y_pred)
  print(as.vector(r_obj$estimate))
}

foo <- results %>% 
  filter(model == "rf",
         feature == "ESS_bert_large_uncased")


foo_r_ls = lapply(seq(0,9,1), compute_r, data = foo) %>% 
  unlist()

mean(foo_r_ls)
sd(foo_r_ls) * 1.96


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
tbl = table(dataset[c('form_request', 'basic_concept')])
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
