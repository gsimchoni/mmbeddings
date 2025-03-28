library(tidyverse)
library(patchwork)
library(extrafont)

font_import()
loadfonts(device="win") 


# Time versus increasing-q simulation viz ---------------------------------------

increase_q <- read_csv("results/res_continuous_single_feature.csv")

pQ <- increase_q %>%
  filter(q0 > 100, !exp_type %in% c("hashing", "mean-encoding")) %>%
  mutate(exp_type = case_when(
    exp_type == "embeddings" ~ "Embed.",
    exp_type == "ignore" ~ "Ignore",
    exp_type == "mmbeddings" ~ "MMbed.",
    exp_type == "embeddings-l2" ~ "EmbedL2",
    exp_type == "regbeddings" ~ "REbed.",
    exp_type == "tabtransformer" ~ "TabTran."
  ),
  exp_type = fct_relevel(exp_type, c("Ignore", "Embed.", "EmbedL2", "REbed.", "TabTran.", "MMbed."))) %>%
  mutate(time_per_epoch = time / n_epochs) %>%
  group_by(q0, exp_type) %>%
  summarise(time_per_epoch = mean(time_per_epoch)) %>%
  ggplot(aes(log10(q0), log10(time_per_epoch), shape = exp_type, color = exp_type)) +
  geom_point(size=4, alpha=0.8) +
  geom_line(linetype=2) +
  labs(x = "q", y = "Mean runtime per epoch (sec)") +
  scale_x_continuous(breaks = log10(c(1000, 10000, 100000)),
                     labels = c(expression(10^3), expression(10^4), expression(10^5))) +
  scale_y_continuous(breaks = -2:4,
                     labels = c(expression(10^-2), expression(10^-1), expression(10^0), expression(10^1), expression(10^2), expression(10^3), expression(10^4)),
                     limits = c(-2, 4)) +
  scale_shape_manual("", values=c(20:15)) +
  scale_color_manual("", values=c("brown", "purple", "orange", "red", "blue", "black")) +
  theme_bw() +
  theme(text = element_text(family = "Century", size=14), legend.position=c(.25,.70), legend.title=element_blank())

# Validation loss plot ---------------------------------------

# Read the CSV file
val_loss <- read_csv("results/val_loss_q10K_epochs500.csv")  # replace with the correct filename

# Normalize both loss columns to start at 100%
df_norm <- val_loss %>%
  mutate(epoch = row_number(),
         Embed_norm = 100 * `Embed.` / first(`Embed.`),
         MMbed_norm = 100 * `MMbed.` / first(`MMbed.`)) %>%
  select(epoch, Embed_norm, MMbed_norm) %>%
  pivot_longer(cols = -epoch, names_to = "Model", values_to = "Loss")

# Plot
pLoss <- ggplot(df_norm, aes(x = epoch, y = Loss, color = Model)) +
  geom_line(lwd = 1) +
  scale_color_manual(values = c("Embed_norm" = "purple", "MMbed_norm" = "black"),
                     labels = c("Embed.", "MMbed."),
                     name = "") +
  scale_y_continuous(labels = scales::label_percent(scale = 1)) +
  labs(y = "Validation Loss (normalized)") +
  theme_bw() +
  theme(legend.position = c(0.75, 0.51),
        text = element_text(family = "Century", size=14),
        legend.title=element_blank())

p <- pQ | pLoss

ggsave("images/sim_increase_q_val_loss.png", p, device = "png", width = 8, height = 3.5, dpi = 300)
