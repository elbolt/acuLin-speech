# Load necessary libraries
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

peaks_filepath <- file.path(dataframes_dir, config$peak_window_filename)
peaks_stats_filepath <- file.path(tables_dir, config$peak_pecentages_filename)
table_review_outpath <- file.path(tables_dir, config$peakstats_review_filename)
table_outpath <- file.path(tables_dir, config$T3_peakstats_filename)

data <- read.csv(peaks_filepath)
data_percentages <- read.csv(peaks_stats_filepath)

# ----------------------------------------------------------------------
# Data preparation

# Remove peaks with negative lags
data <- data %>% filter(latency_ms >= 0)

subject_ids <- unique(data$subject_id)
windows <- unique(data$window)
responses <- unique(data$response)
clusters <- unique(data$cluster)

analyses <- c("amplitude", "latency")

# ----------------------------------------------------------------------
# Run multiple tests over all windows, responses, and clusters

run_tests <- function(current_analysis) {
  results <- tibble(window = character(), response = character(), cluster = character(), 
                    U = numeric(), p = numeric(), r = numeric())
  participants_no <- tibble(window = character(), response = character(), cluster = character(), 
                            n_normal = numeric(), n_low = numeric())

  for (window in windows) {
    for (response in responses) {
      for (cluster in clusters) {
        data_subset <- data %>%
          filter(window == !!window, response == !!response, cluster == !!cluster)

        U_test <- if (current_analysis == "amplitude") {
          wilcox.test(amplitude ~ MoCA_group, data = data_subset)
        } else {
          wilcox.test(latency_ms ~ MoCA_group, data = data_subset)
        }

        U <- U_test$statistic
        p <- U_test$p.value

        n_1 <- sum(data_subset$MoCA_group == "normal")
        n_2 <- sum(data_subset$MoCA_group == "low")
        r <- (U - n_1 * (n_1 + 1) / 2) / (n_1 * n_2)

        results <- results %>%
          add_row(
            window = window,
            response = response,
            cluster = cluster,
            U = U,
            p = p,
            r = r
          )

        n_normal <- data_subset %>%
          filter(MoCA_group == "normal") %>%
          pull(subject_id) %>% unique() %>% length()
        n_low <- data_subset %>%
          filter(MoCA_group == "low") %>%
          pull(subject_id) %>% unique() %>% length()

        participants_no <- participants_no %>% add_row(
          window = window,
          response = response,
          cluster = cluster,
          n_normal = n_normal,
          n_low = n_low
        )
      }
    }
  }

  results <- results %>%
    mutate(sign = case_when(
      p < 0.001 ~ "***",
      p < 0.01 ~ "**",
      p < 0.05 ~ "*",
      TRUE ~ ""
    ))

  if (!is_review) {
    filtered_clusters <- data_percentages %>%
      filter(Percentage.of.subjects >= 75.0) %>%
      select(window, cluster, response)

    results <- results %>%
      semi_join(filtered_clusters, by = c("window", "cluster", "response"))
  }

  results_corr <- results %>%
    group_by(response) %>%
    mutate(p_corr = holm_bonferroni(p)) %>%
    ungroup() %>%
    mutate(sign_corr = case_when(
      p_corr < 0.001 ~ "***",
      p_corr < 0.01 ~ "**",
      p_corr < 0.05 ~ "*",
      TRUE ~ ""
    )) %>%
    select(window, response, cluster, U, p, sign, p_corr, sign_corr, r)

  list(results = results_corr, participants_no = participants_no)
}

is_review <- FALSE

amplitude_results <- run_tests("amplitude")$results
amplitude_subject_no <- run_tests("amplitude")$participants_no
latency_results <- run_tests("latency")$results
latency_subject_no <- run_tests("latency")$participants_no

# ----------------------------------------------------------------------
# Check results

print(paste0("Data frames identical: ", identical(amplitude_subject_no, latency_subject_no)))

final_df <- amplitude_results %>%
  left_join(latency_results, by = c("window", "response", "cluster"), suffix = c("_amp", "_lat")) %>%
  left_join(amplitude_subject_no)

# ----------------------------------------------------------------------
# Create nice LaTeX tables

# Formatting functions for each column
final_df$n_normal <- sapply(final_df$n_normal, function(x) format_and_wrap(x, digits = 0))
final_df$n_low <- sapply(final_df$n_low, function(x) format_and_wrap(x, digits = 0))
final_df$U_amp <- sapply(final_df$U_amp, function(x) format_and_wrap(x, digits = 1))
final_df$p_amp <- sapply(final_df$p_amp, function(x) format_and_wrap(x, digits = 3, is_p_value = TRUE))
final_df$p_corr_amp <- sapply(final_df$p_corr_amp, function(x) format_and_wrap(x, digits = 3, is_p_value = TRUE))
final_df$r_amp <- sapply(final_df$r_amp, function(x) format_and_wrap(x, digits = 2))
final_df$U_lat <- sapply(final_df$U_lat, function(x) format_and_wrap(x, digits = 1))
final_df$p_lat <- sapply(final_df$p_lat, function(x) format_and_wrap(x, digits = 3, is_p_value = TRUE))
final_df$p_corr_lat <- sapply(final_df$p_corr_lat, function(x) format_and_wrap(x, digits = 3, is_p_value = TRUE))
final_df$r_lat <- sapply(final_df$r_lat, function(x) format_and_wrap(x, digits = 2))

# Formatting window and response columns
final_df$window <- sapply(final_df$window, function(x) {
  x <- tolower(x)
  x <- paste0(toupper(substr(x, 1, 1)), substr(x, 2, nchar(x)))
})
final_df$response <- sapply(final_df$response, function(x) {
  x <- tolower(x)
  x <- paste0(toupper(substr(x, 1, 1)), substr(x, 2, nchar(x)))
})

# Renaming columns
final_df <- final_df %>%
  rename("Speech feature" = response) %>%
  rename("Window" = window) %>%
  rename("Cluster" = cluster)

if (is_review) {
  write.csv(final_df, table_review_outpath, row.names = FALSE)
} else {
  relevant_cols <- c(
    "Window",
    "Speech feature",
    "Cluster",
    "U_amp",
    "p_corr_amp",
    "sign_corr_amp",
    "r_amp",
    "U_lat",
    "p_corr_lat",
    "sign_corr_lat",
    "r_lat",
    "n_normal",
    "n_low"
  )

  final_df <- final_df %>%
    select(relevant_cols)

  write.csv(table_outpath, row.names = FALSE)
}
