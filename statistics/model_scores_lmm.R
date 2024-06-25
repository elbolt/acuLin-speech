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

# Set working directory and load configurations
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
config <- fromJSON("config.json")
source("helpers.R")

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

# Check what contrasts look like
data$model
data$MoCA_group

# Check factor variable
data$subject_id

# Check z-scored continuous variables
data$PTA_z


# ----------------------------------------------------------------------
# Define fixed effects structure
# (I only include random intercepts in random effects structure)

max_model <- lmer(
  score_r ~ 1 + MoCA_group * PTA_z * model +
    (1 | subject_id),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)

red1_model <- lmer(
  score_r ~ 1 + MoCA_group * PTA_z + model +
    (1 | subject_id),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)

red2_model <- lmer(
  score_r ~ 1 + MoCA_group * model +
    (1 | subject_id),
  data = data,
  control = lmerControl(optimizer = "bobyqa")
)

anova(max_model, red1_model, red2_model)

final_model <- max_model

summary(final_model)

# ----------------------------------------------------------------------
# Model diagnostics

# Residuals vs Fitted
plot(fitted(final_model), residuals(final_model), main = "Residuals vs Fitted")
abline(h = 0, col = "red")

# QQ Plot of residuals
qqnorm(residuals(final_model))
qqline(residuals(final_model), col = "red")

# Histogram of residuals
hist(residuals(final_model), breaks = 30, main = "Histogram of Residuals")

# Extract random effects
ranef(final_model)

# Cook's Distance
cooksd <- cooks.distance(final_model)
plot(cooksd, main = "Cook's Distance", type = "h")
abline(h = 4 / (nrow(data) - length(fixef(final_model))), col = "red")

# Goodness-of-Fit
r.squaredGLMM(final_model)
AIC(final_model)
BIC(final_model)

# Multicollinearity
vif(lm(score_r ~ MoCA_group * PTA_z * model, data = data))


# ----------------------------------------------------------------------
# Post-hoc analysis

emm_interaction <- emmeans(final_model, ~ PTA_z * model)
pairs(emm_interaction)

# Plot interaction
emm_data <- as.data.frame(emm_interaction)
ggplot(emm_data, aes(x = model, y = emmean, color = as.factor(PTA_z))) +
  geom_point(position = position_dodge(width = 0.5)) +
  geom_errorbar(
    aes(ymin = emmean - SE, ymax = emmean + SE),
    width = 0.2,
    position = position_dodge(width = 0.5)
  ) +
  labs(title = "Interaction between PTA_z and Model on Scores",
       x = "Model",
       y = "Estimated Marginal Means",
       color = "PTA_z") +
  theme_minimal()


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
  "MoCA group (low)",
  "PTA ($z$)",
  "Model (Segment. word-level)",
  "Model (Segment. phoneme-level)",
  "Model (Linguistic word-level)",
  "Model (Linguistic phoneme-level)",
  "MoCA group (low) * PTA ($z$)",
  "MoCA group (low) * Model (Segment. word-level)",
  "MoCA group (low) * Model (Segment. phoneme-level)",
  "MoCA group (low) * Model (Linguistic word-level)",
  "MoCA group (low) * Model (Linguistic phoneme-level)",
  "PTA ($z$) * Model (Segment. word-level)",
  "PTA ($z$) * Model (Segmentation phoneme-level)",
  "PTA ($z$) * Model (Linguistic word-level)",
  "PTA ($z$) * Model (Linguistic phoneme-level)",
  "MoCA group (low) * PTA ($z$) * Model (Segment. word-level)",
  "MoCA group (low) * PTA ($z$) * Model (Segment. phoneme-level)",
  "MoCA group (low) * PTA ($z$) * Model (Linguistic word-level)",
  "MoCA group (low) * PTA ($z$) * Model (Linguistic phoneme-level)"
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

colnames(ci_tab) <- c("Coefficient", "Estimate", "$LL$", "$UL$", "$df$", "$t$", "$p$", "")

write.csv(ci_tab, table_outpath, row.names = FALSE)
