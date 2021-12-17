library(ggthemes)

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
  #filter(Model %in% c("GloVe", "USE")) %>% 
  #filter(basic_concept %in% c("belief", "importance")) %>% 
  ggplot() +
  geom_violin(aes(x = factor(concrete_concept_reference), 
                  y = cos_dif,
                  group = interaction(factor(concrete_concept_reference), Model)),
              trim = FALSE,
              outlier.shape = NA) +
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
        axis.text=element_text(size=6),
        axis.title.x = element_blank(),
        panel.grid.major.x = element_blank()) + 
  #scale_color_manual(values = color_pallete) + 
  #scale_shape_manual(values = shape_pallete) +
  #guides(shape = guide_legend(title.position = "top")) + 
  facet_wrap(~basic_concept, scales = "free_x", ncol = 2) +
  guides(color = guide_legend(override.aes= list(alpha = 1))) +
  labs(#title= "Figure H1: Distribution of cos(ref, similar) - cos(ref, dissimilar) scores, across 13 basic concepts.",
    y="Difference in cosine scores") 

ggsave("fig_h1.pdf", fig_h1, device = "pdf", width = 170, height = 225, units = "mm", dpi = 300)

