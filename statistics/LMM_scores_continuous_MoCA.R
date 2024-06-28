library(jsonlite)
library(lme4)
library(lattice)
library(lmerTest)
library(MuMIn)
library(performance)
library(rstudioapi)
library(xtable)
library(emmeans)
library(car)
library(tibble)
library(dplyr)
library(ggplot2)


# ----------------------------------------------------------------------
# Configuration and data loading

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
config <- fromJSON("config.json")
source("helpers.R")

dataframes_dir <- config$dataframes_dir
tables_dir <- config$tables_dir
if (!dir.exists(tables_dir)) {
  dir.create(tables_dir, recursive = TRUE)
}
scores_filepath <- file.path(dataframes_dir, config$scores_filename)
table_outpath <- file.path(tables_dir, config$TS1_lmm_scores_continous_filename)

data <- read.csv(scores_filepath)


# ----------------------------------------------------------------------
# Data preparation

# Preparation 1: mTRF model to factor then sum coding
n_models <- length(unique(data$model))
model_order <- names(config$models_dict)
data$model <- factor(data$model, levels = model_order)
contrasts(data$model) <- contr.sum(n_models)

# Preparation 3: Subject ID
data$subject_id <- factor(data$subject_id)

# Preparation 4: z-score continuous variables
data$MoCA_z <- scale(data$MoCA_score)
data$PTA_z <- scale(data$PTA_dB)

# Check what contrasts look like
data$model
data$MoCA_group

# Check factor variable
data$subject_id

# Check z-scored continuous variables
data$MoCA_z
data$PTA_z


# ----------------------------------------------------------------------
# Fit same model as in binary MoCA analysis

final_model <- lmer(
  score_r ~ MoCA_z * PTA_z * model + (1 | subject_id),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)
summary(final_model)


# ----------------------------------------------------------------------
# Draw inferences and create nice LaTeX tables

t_values <- summary(final_model)$coefficients[, "t value"]
df_values <- summary(final_model)$coefficients[, "df"]
p_values <- summary(final_model)$coefficients[, "Pr(>|t|)"]

conf_intervals <- confint(
  final_model,
  parm = "beta_",
  method = "Wald",
  nsim = 5000,
  seed = 2025
)

ci_tab <- as.data.frame(cbind(
  estimate = fixef(final_model),
  conf_intervals,
  df_values,
  t_values,
  p_values
))


new_coeff_names <- c(
  "Intercept",
  "MoCA score ($z$)",
  "PTA ($z$)",
  "mTRF model (Seg. word-level)",
  "mTRF model (Seg. phoneme-level)",
  "mTRF model (Lin. word-level)",
  "mTRF model (Lin. phoneme-level)",
  "MoCA score * PTA",
  "MoCA score * mTRF model (Seg. word-level)",
  "MoCA score * mTRF model (Seg. phoneme-level)",
  "MoCA score * mTRF model (Lin. word-level)",
  "MoCA score * mTRF model (Lin. phoneme-level)",
  "PTA * mTRF model (Seg. word-level)",
  "PTA * mTRF model (Seg. phoneme-level)",
  "PTA * mTRF model (Lin. word-level)",
  "PTA * mTRF model (Lin. phoneme-level)",
  "MoCA score * PTA * mTRF model (Seg. word-level)",
  "MoCA score * PTA * mTRF model (Seg. phoneme-level)",
  "MoCA score * PTA * mTRF model (Lin. word-level)",
  "MoCA score * PTA * mTRF model (Lin. phoneme-level)"
)
rownames(ci_tab) <- new_coeff_names

# Add significance stars
ci_tab$significance <- ifelse(
  ci_tab$p_values < 0.001, "***",
  ifelse(
    ci_tab$p_values < 0.01, "**",
    ifelse(
      ci_tab$p_values < 0.05, "*",
      ""
    )
  )
)

# Apply formatting
ci_tab$estimate <- sapply(ci_tab$estimate, format_and_wrap)
ci_tab$`2.5 %` <- sapply(ci_tab$`2.5 %`, format_and_wrap)
ci_tab$`97.5 %` <- sapply(ci_tab$`97.5 %`, format_and_wrap)
ci_tab$df_values <- sapply(ci_tab$df_values, function(x) format_and_wrap(x, digits = 0))
ci_tab$t_values <- sapply(ci_tab$t_values, function(x) format_and_wrap(x, digits = 1))
ci_tab$p_values <- sapply(ci_tab$p_values, function(x) format_and_wrap(x, digits = 3, is_p_value = TRUE))
ci_tab <- rownames_to_column(ci_tab, var = "Coefficient")

colnames(ci_tab) <- c("Coefficient", "Estimate", "LL", "UL", "df", "t", "p", "")

write.csv(ci_tab, table_outpath, row.names = FALSE)


# ----------------------------------------------------------------------
# Post-hoc analysis on MoCA_z, PTA_z and model interaction

mean_MoCA_z <- mean(data$MoCA_z)
sd_MoCA_z <- sd(data$MoCA_z)
MoCA_z_values <- c(mean_MoCA_z - sd_MoCA_z, mean_MoCA_z, mean_MoCA_z + sd_MoCA_z)

mean_PTA_z <- mean(data$PTA_z)
sd_PTA_z <- sd(data$PTA_z)
PTA_z_values <- c(mean_PTA_z - sd_PTA_z, mean_PTA_z, mean_PTA_z + sd_PTA_z)

emm <- emmeans(
  final_model,
  ~ MoCA_z * PTA_z | model,
  at = list(MoCA_z = MoCA_z_values, PTA_z = PTA_z_values)
)

emm_df <- as.data.frame(emm)
emm_model3 <- subset(emm_df, model == "linguistic word")

# Conduct pairwise comparisons for model3
pairwise_results <- pairs(
  emmeans(
    final_model, ~ MoCA_z * PTA_z | model,
    at = list(MoCA_z = MoCA_z_values, PTA_z = PTA_z_values)
  ), which = "model3"
)

# Plot the interaction
ggplot(
  emm_model3,
  aes(
    x = PTA_z, y = emmean, color = factor(MoCA_z), group = factor(MoCA_z)
  )
) +
  geom_line() +
  geom_point() +
  geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL), width = 0.2) +
  labs(title = "Three-Way Interaction between MoCA_z, PTA_z, and Model3",
       x = "PTA_z",
       y = "Estimated Marginal Means (Score)",
       color = "MoCA_z") +
  theme_minimal() +
  theme(legend.position = "right")

print(pairwise_results)