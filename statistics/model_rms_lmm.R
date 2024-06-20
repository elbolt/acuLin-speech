# Load required libraries
library(jsonlite)
library(lme4)
library(lattice)
library(lmerTest)
library(MuMIn)
library(performance)
library(rstudioapi)
library(sjPlot)
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
rms_table_file <- config$rms_table_file
rms_filename <- paste(dataframes_dir, config$rms_filename, sep = "/")
channel_cluster_dummy_dict <- config$channel_cluster_dummy_dict

data <- read.csv(rms_filename)


# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
# # IMPORTANT: Model!
# model_no <- 4
models <- unique(data$model)
# current_model <- models[model_no]
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +


# ----------------------------------------------------------------------
# Data subset and preparation

# Preparation 1: Add cluster information to data
data$cluster <- sapply(
  data$electrode_id,
  find_cluster,
  cluster_dict = channel_cluster_dummy_dict
)

# Initialize list to store final models
final_models <- list()
ci_tabs <- list()

for (current_model in models) {
  print(paste("========= Current model is:", current_model, "========="))
  
  # Preparation 2: Subset data
  data_subset <- subset(data, model == current_model)
  # rm(data)
  
  # Preparation 3: Set response order for each model
  if (current_model == "acoustic") {
    response_order <- c("envelope", "envelope onsets")
  } else if (current_model == "segmentation word") {
    response_order <- c("word onset")
  } else if (current_model == "segmentation phone") {
    response_order <- c("phoneme onset")
  } else if (current_model == "linguistic word") {
    response_order <- c("word surprisal", "word frequency")
  } else if (current_model == "linguistic phone") {
    response_order <- c("phoneme surprisal", "phoneme entropy")
  }
  
  data_subset$response <- factor(data_subset$response, levels = response_order)
  if (current_model != "segmentation word" && current_model != "segmentation phone") {
    n_responses <- length(unique(data_subset$response))
    contrasts(data_subset$response) <- contr.treatment(n_responses)
  }
  
  # Preparation 4: MoCA_group
  n_groups <- length(unique(data_subset$MoCA_group))
  data_subset$MoCA_group <- factor(data_subset$MoCA_group, levels = c("normal", "low"))
  contrasts(data_subset$MoCA_group) <- contr.treatment(n_groups)

  # Preparation 5: Subject ID
  data_subset$subject_id <- factor(data_subset$subject_id)
  
  # Preparation 5: z-score continuous variables
  data_subset$PTA_z <- scale(data_subset$PTA_dB)
  
  # Preparation 6: Cluster
  data_subset$cluster <- factor(data_subset$cluster, levels = c("F", "C", "P"))
  contrasts(data_subset$cluster) <- contr.treatment(3)

  # Check what contrasts look like
  data_subset$response
  data_subset$MoCA_group
  
  # Check factor variable
  data_subset$subject_id
  
  # Check z-scored continuous variables
  data_subset$PTA_z
  
  
  # ----------------------------------------------------------------------
  # Define fixed effects structure
  # (I only include random intercepts in random effects structure)
  
  # Segmentation models have one response variable only.
  if (current_model != "segmentation word" && current_model != "segmentation phone") {
    max_model <- lmer(
      RMS ~ 1 + MoCA_group * PTA_z * response * cluster +
        (1 | subject_id),
      data = data_subset,
      control = lmerControl(optimizer = "bobyqa")
    )
    
    red1_model <- lmer(
      RMS ~ 1 + MoCA_group * PTA_z * response + cluster +
        (1 | subject_id),
      data = data_subset,
      control = lmerControl(optimizer = "bobyqa")
    )

    red2_model <- lmer(
      RMS ~ 1 + MoCA_group * PTA_z + response + cluster +
        (1 | subject_id),
      data = data_subset,
      control = lmerControl(optimizer = "bobyqa")
    )

    final_model <- red2_model
    final_models <- c(final_models, final_model)

  } else {
    max_model <- lmer(
      RMS ~ 1 + MoCA_group * PTA_z * cluster +
        (1 | subject_id),
      data = data_subset,
      control = lmerControl(optimizer = "bobyqa")
    )
    
    red1_model <- lmer(
      RMS ~ 1 + MoCA_group * PTA_z + cluster +
        (1 | subject_id),
      data = data_subset,
      control = lmerControl(optimizer = "bobyqa")
    )

    final_model <- red1_model
    final_models <- c(final_models, final_model)
  }
  
  if (current_model == "acoustic") {
    response_name <- "Speech feature (envelope onsets)"
  } else if (current_model == "linguistic word") {
    response_name <- "Speech feature (word frequency)"
  } else if (current_model == "linguistic phone") {
    response_name <- "Speech feature (phoneme entropy)"
  }
  
  # ----------------------------------------------------------------------
  # Draw inferences and create nice LaTeX tables
  
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
    p_values
  ))
  
  if (current_model != "segmentation word" && current_model != "segmentation phone") {
    new_coeff_names <- c(
      "Intercept",
      "MoCA group (low)",
      "PTA ($z$)",
      response_name,
      "Cluster (C)",
      "Cluster (P)",
      "MoCA group (low) * PTA ($z$)"
    )
  } else {
    new_coeff_names <- c(
      "Intercept",
      "MoCA group (low)",
      "PTA ($z$)",
      "Cluster (C)",
      "Cluster (P)",
      "MoCA group (low) * PTA ($z$)"
    )
  }
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
  ci_tab$p_values <- sapply(ci_tab$p_values, format_and_wrap)
  ci_tab <- rownames_to_column(ci_tab, var = "Coefficient")
  
  # Rename columns for confidence intervals
  colnames(ci_tab) <- c("Coefficient", "Estimate", "$LL$", "$UL$", "$p$", "")
  
  ci_tabs <- c(ci_tabs, ci_tab)
  
  print(ci_tab)
}

ci_tabs
