library(jsonlite)
library(lme4)
library(lattice)
library(lmerTest)
library(MuMIn)
library(performance)
library(rstudioapi)
library(emmeans)
library(car)
library(tibble)
library(dplyr)
library(ggplot2)


# ----------------------------------------------------------------------
# Configuration and data loading

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
config <- fromJSON("config.json")

dataframes_dir <- config$dataframes_dir
tables_dir <- config$tables_dir
if (!dir.exists(tables_dir)) {
  dir.create(tables_dir, recursive = TRUE)
}
scores_filepath <- file.path(dataframes_dir, config$scores_filename)
table_outpath <- file.path(tables_dir, config$T1_lmm_scores_filename)

data <- read.csv(scores_filepath)


# ----------------------------------------------------------------------
# Data preparation

# Preparation 1: Model to factor then sum coding
n_models <- length(unique(data$model))
model_order <- names(config$models_dict)
data$model <- factor(data$model, levels = model_order)
contrasts(data$model) <- contr.sum(n_models)

# Preparation 2: MoCA_group
n_groups <- length(unique(data$MoCA_group))
data$MoCA_group <- factor(data$MoCA_group, levels = c("normal", "low"))
contrasts(data$MoCA_group) <- contr.sum(n_groups)

# Preparation 3: Subject ID
data$subject_id <- factor(data$subject_id)

# Preparation 4: z-score continuous variables
data$PTA_z <- scale(data$PTA_dB)

# Fit my final model
final_model <- lmer(
  score_r ~ 1 + MoCA_group * PTA_z * model +
    (1 | subject_id),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)


# ----------------------------------------------------------------------
# Post-hoc analysis on main effect of model

main_results <- emmeans(final_model, ~ model)
main_pairwise <- pairs(main_results)
 summary(main_pairwise)

# Convert the results to a data frame
main_pairwise_df <- as.data.frame(main_pairwise)
main_pairwise_df$lower.CL <- main_pairwise_df$estimate - 1.96 * main_pairwise_df$SE
main_pairwise_df$upper.CL <- main_pairwise_df$estimate + 1.96 * main_pairwise_df$SE

# Plot the pairwise comparisons
ggplot(main_pairwise_df, aes(x = contrast, y = estimate, ymin = lower.CL, ymax = upper.CL)) +
  geom_pointrange() +
  coord_flip() +
  labs(title = "Pairwise Comparisons of Model Effects",
       x = "Comparison",
       y = "Estimated Difference in Marginal Means") +
  theme_minimal()


# ----------------------------------------------------------------------
# Post-hoc analysis on PTA_z and model interaction

# Create grid of PTA_z values
mean_PTA_z <- mean(data$PTA_z)
sd_PTA_z <- sd(data$PTA_z)
PTA_z_values <- c(mean_PTA_z - sd_PTA_z, mean_PTA_z, mean_PTA_z + sd_PTA_z)

# Obtain the estimated marginal means for the interaction
interaction_results <- emmeans(
  final_model, ~ model | PTA_z,
  at = list(PTA_z = PTA_z_values)
)

# Plot interactions
emmip(interaction_results, model ~ PTA_z, CIs = TRUE) +
  labs(title = "Interaction between PTA_z and Model on Score",
       x = "PTA_z",
       y = "Estimated Marginal Means (Score)",
       color = "Model") +
  theme_minimal()

