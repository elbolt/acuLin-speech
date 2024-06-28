library(jsonlite)
library(mediation)

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
table_outpath <- file.path(tables_dir, config$T1_lmm_scores_filename)

data <- read.csv(scores_filepath)


# ----------------------------------------------------------------------
# Data preparation

# Preparation 1: Model to factor then sum coding
n_models <- length(unique(data$model))
model_order <- names(config$models_dict)
data$model <- factor(data$model, levels = model_order)

# Preparation 2: MoCA_group
n_groups <- length(unique(data$MoCA_group))
data$MoCA_group <- factor(data$MoCA_group, levels = c("normal", "low"))

# Preparation 3: Subject ID
data$subject_id <- factor(data$subject_id)

# Check what contrasts look like
data$model
data$MoCA_group

# Check factor variable
data$subject_id

# Check continuous variables
data$PTA
data$age


# ----------------------------------------------------------------------
# Fit a mediation model

# Mediator model: MoCA_score as the mediator
mediator_model <- lm(MoCA_score ~ PTA_dB * age, data = data)

# Outcome model: score_r as the outcome
outcome_model <- lm(score_r ~ PTA_dB + MoCA_score + age + model, data = data)

med_fit <- mediate(mediator_model, outcome_model, treat = "PTA_dB", mediator = "MoCA_score", boot = TRUE)
summary(med_fit)
