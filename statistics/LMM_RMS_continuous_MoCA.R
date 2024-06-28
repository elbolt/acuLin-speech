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
library(tidyr)
library(ggplot2)
library(xtable)


# ----------------------------------------------------------------------
# Configuration and data loading

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
config <- fromJSON("config.json")
source("helpers.R")

dataframes_dir <- config$dataframes_dir
tables_dir <- config$tables_dir
if (!dir.exists(tables_dir)) {
  dir.create(tables_dir)
}
rms_filename <- paste(dataframes_dir, config$rms_filename, sep = "/")
rms_table_file <- config$TS2_lmm_rms_continous_filename

data <- read.csv(rms_filename)
models <- unique(data$model)


# ----------------------------------------------------------------------
# Data subset and preparation

# Initialize list to store final models
final_models <- list()
ci_tabs <- list()

for (current_model in models) {

  # Preparation 1: Subset data
  data_subset <- subset(data, model == current_model)

  # Preparation 2: Set response order for each model
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

  # Preparation 3: Subject ID
  data_subset$subject_id <- factor(data_subset$subject_id)

  # Preparation 4: z-score continuous variables
  data_subset$MoCA_z <- scale(data_subset$MoCA_score)
  data_subset$PTA_z <- scale(data_subset$PTA_dB)

  # Preparation 6: Cluster
  data_subset$cluster <- factor(data_subset$cluster, levels = c("F", "C", "P"))
  contrasts(data_subset$cluster) <- contr.treatment(3)

  # Check what contrasts look like
  data_subset$response

  # Check factor variable
  data_subset$subject_id

  # Check z-scored continuous variables
  data_subset$MoCA_z
  data_subset$PTA_z


  # ----------------------------------------------------------------------
  # # Fit same model as in binary MoCA analysis
  
  if (current_model != "segmentation word" && current_model != "segmentation phone") {
    final_model <- lmer(
      RMS ~ 1 + MoCA_z * PTA_z + response + cluster +
        (1 | subject_id),
      data = data_subset,
      control = lmerControl(optimizer = "bobyqa")
    )
  } else {
    final_model <- lmer(
      RMS ~ 1 + MoCA_z * PTA_z + cluster +
        (1 | subject_id),
      data = data_subset,
      control = lmerControl(optimizer = "bobyqa")
    )
  }

  if (current_model == "acoustic") {
    response_name <- "Speech feature (envelope onsets)"
  } else if (current_model == "linguistic word") {
    response_name <- "Speech feature (word frequency)"
  } else if (current_model == "linguistic phone") {
    response_name <- "Speech feature (phoneme entropy)"
  }


  # ----------------------------------------------------------------------
  # Draw inferences and create data frame for table

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

  if (current_model != "segmentation word" && current_model != "segmentation phone") {
    new_coeff_names <- c(
      "Intercept",
      "MoCA ($z$)",
      "PTA ($z$)",
      response_name,
      "Cluster (C)",
      "Cluster (P)",
      "MoCA * PTA"
    )
  } else {
    new_coeff_names <- c(
      "Intercept",
      "MoCA ($z$)",
      "PTA ($z$)",
      "Cluster (C)",
      "Cluster (P)",
      "MoCA * PTA"
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
  ci_tab$df_values <- sapply(ci_tab$df_values, function(x) format_and_wrap(x, digits = 1))
  ci_tab$t_values <- sapply(ci_tab$t_values, function(x) format_and_wrap(x, digits = 1))
  ci_tab$p_values <- sapply(ci_tab$p_values, function(x) format_and_wrap(x, digits = 3, is_p_value = TRUE))
  ci_tab <- rownames_to_column(ci_tab, var = "Coefficient")

  # Rename columns for confidence intervals
  colnames(ci_tab) <- c("Coefficient", "$\beta$", "LL", "UL", "df", "$t$", "$p$", "")

  ci_tabs <- c(ci_tabs, ci_tab)
  print(summary(final_model))
}


# ----------------------------------------------------------------------
# Create nice LaTeX table

# Combined data frame
model_names <- c(
  "Acoustic",
  "Segmentation word-level",
  "Segmentation phoneme-level",
  "Linguistic word-level",
  "Linguistic phoneme-level"
)

combined_df <- combine_tables(ci_tabs, model_names)
combined_df <- combined_df[, c("Model", setdiff(names(combined_df), "Model"))]

colnames(combined_df) <- c(
  "Model",
  "Coefficient",
  "$\beta$",
  "LL",
  "UL",
  "df",
  "$t$",
  "$p$",
  ""
)

write.csv(combined_df, file.path(tables_dir, rms_table_file), row.names = FALSE)