library(tidyverse)
library(patchwork)
library(extrafont)

font_import()
loadfonts(device="win") 


# B_hat versus B_true ---------------------------------------

# Read the datasets
b_true <- read_csv("results/b_true_embed.csv")
b_pred <- read_csv("results/b_pred_embed.csv")

# Print for example the best correlation with column j
# j <- 1
# for (i in seq(1, 10)) {
#   print(cor(b_true[, j], b_pred[, i]))
# }

# Ensure column names are consistent
colnames(b_true) <- sprintf("b%02d", 1:ncol(b_true))
colnames(b_pred) <- sprintf("b%02d", 1:ncol(b_pred))

# Compute correlations and match each true column to the best predicted column
best_matches <- colnames(b_true) %>% map_chr(function(col_true) {
  cors <- sapply(colnames(b_pred), function(col_pred) {
    abs(cor(b_true[[col_true]], b_pred[[col_pred]]))
  })
  if (length(cors) == 0) {
    stop(glue::glue("No columns in b_pred to correlate with {col_true}"))
  }
  best_col <- names(which.max(cors))
  if (length(best_col) != 1) {
    stop(glue::glue("Could not find a unique best match for {col_true}"))
  }
  best_col
}) %>% set_names(colnames(b_true))

print(best_matches)

# Prepare data for plotting
plot_data <- map_dfr(names(best_matches), function(col_true) {
  col_pred <- best_matches[[col_true]]
  tibble(
    True = b_true[[col_true]],
    Pred = b_pred[[col_pred]],
    Component = col_true
  )
})

# Convert facet labels to parsed math expressions like b[1]
plot_data <- plot_data %>%
  mutate(Component = factor(Component, levels = sprintf("b%02d", 1:10)),
         Component = fct_relabel(Component, ~ str_replace(., "b", "b[")),
         Component = fct_relabel(Component, ~ paste0(., "]")))

# Shared axis limits
axis_lim <- range(c(plot_data$True, plot_data$Pred))

# Plot
ggplot(plot_data, aes(x = True, y = Pred)) +
  geom_point(alpha = 0.6) +
  facet_wrap(~ Component, nrow = 2, labeller = label_parsed) +
  coord_fixed(xlim = axis_lim) +
  theme_bw() +
  theme(
    aspect.ratio = 1,
    text = element_text(family = "Century", size = 16),
    strip.background = element_blank(),
    strip.text = element_text(face = "bold")
  )

# MMbeddings
# Read the datasets
b_true <- read_csv("results/b_true_mmbed.csv")
b_pred <- read_csv("results/b_pred_mmbed.csv")

# Print for example the best correlation with column j
# j <- 1
# for (i in seq(1, 10)) {
#   print(cor(b_true[, j], b_pred[, i]))
# }

# Ensure column names are consistent
colnames(b_true) <- sprintf("b%02d", 1:ncol(b_true))
colnames(b_pred) <- sprintf("b%02d", 1:ncol(b_pred))

# Compute correlations and match each true column to the best predicted column
best_matches <- colnames(b_true) %>% map_chr(function(col_true) {
  cors <- sapply(colnames(b_pred), function(col_pred) {
    abs(cor(b_true[[col_true]], b_pred[[col_pred]]))
  })
  if (length(cors) == 0) {
    stop(glue::glue("No columns in b_pred to correlate with {col_true}"))
  }
  best_col <- names(which.max(cors))
  if (length(best_col) != 1) {
    stop(glue::glue("Could not find a unique best match for {col_true}"))
  }
  best_col
}) %>% set_names(colnames(b_true))

print(best_matches)

# Prepare data for plotting
plot_data <- map_dfr(names(best_matches), function(col_true) {
  col_pred <- best_matches[[col_true]]
  tibble(
    True = b_true[[col_true]],
    Pred = b_pred[[col_pred]],
    Component = col_true
  )
})

# Convert facet labels to parsed math expressions like b[1]
plot_data <- plot_data %>%
  mutate(Component = factor(Component, levels = sprintf("b%02d", 1:10)),
         Component = fct_relabel(Component, ~ str_replace(., "b", "b[")),
         Component = fct_relabel(Component, ~ paste0(., "]")))

# Shared axis limits
axis_lim <- range(c(plot_data$True, plot_data$Pred))

# Plot
p <- ggplot(plot_data, aes(x = True, y = Pred)) +
  geom_point(alpha = 0.6) +
  facet_wrap(~ Component, nrow = 2, labeller = label_parsed) +
  coord_fixed(xlim = axis_lim) +
  theme_bw() +
  theme(
    aspect.ratio = 1,
    text = element_text(family = "Century", size = 16),
    strip.background = element_blank(),
    strip.text = element_text(face = "bold")
  )

ggsave("images/sim_b_pred_vs_true.png", p, device = "png", width = 8, dpi = 300)
