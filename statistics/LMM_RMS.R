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

# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
is_review <- FALSE
# + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +


# ----------------------------------------------------------------------
# Data preparation

# Remove peaks with negative lags
data <- data[data$latency_ms >= 0, ]

subject_ids <- unique(data$subject_id)
windows <- unique(data$window)
responses <- unique(data$response)
clusters <- unique(data$cluster)

results_tables <- list()
participants_tables <- list()
analyses <- c("amplitude", "latency")


# ----------------------------------------------------------------------
# Run multiple tests over all windows, responses and clusters

for (current_analysis in analyses) {
  
  results <- data.frame(
    window = character(),
    response = character(),
    cluster = character(),
    U = numeric(),
    p = numeric(),
    r = numeric()
  )
  participants_no <- data.frame(
    window = character(),
    response = character(),
    cluster = character(),
    n_normal = numeric(),
    n_low = numeric()
  )

  for (window in windows) {
    for (response in responses) {
      for (cluster in clusters) {
        data_subset <- data[
          data$window == window & data$response == response & data$cluster == cluster,
        ]

        # Wilcoxon rank-sum test
        if (current_analysis == "amplitude") {
          U_test <- wilcox.test(amplitude ~ MoCA_group, data = data_subset)
        } else if (current_analysis == "latency") {
          U_test <- wilcox.test(latency_ms ~ MoCA_group, data = data_subset)
        }

        U <- U_test$statistic
        p <- U_test$p.value

        # Rank-biserial correlation (effect size)
        n_1 <- sum(data_subset$MoCA_group == "normal")
        n_2 <- sum(data_subset$MoCA_group == "low")
        r <- (U - n_1 * (n_1 + 1) / 2) / (n_1 * n_2)

        results <- rbind(results, data.frame(
          window = window,
          response = response,
          cluster = cluster,
          U = U,
          p = p,
          r = r
        ))

        n_normal <- length(
          unique(data_subset[data_subset$MoCA_group == "normal", ]$subject_id)
        )
        n_low <- length(
          unique(data_subset[data_subset$MoCA_group == "low", ]$subject_id)
        )

        participants_no <- rbind(participants_no, data.frame(
          window = window,
          response = response,
          cluster = cluster,
          n_normal = n_normal,
          n_low = n_low
        ))
      }
    }
  }
  
  results$sign <- ifelse(
    results$p < 0.001, "***",
    ifelse(
      results$p < 0.01, "**",
      ifelse(
        results$p < 0.05, "*", ""
      )
    )
  )


  # ----------------------------------------------------------------------
  # Dataframe for review

  if (!is_review) {
    # In final paper, I only want to show results with at least 75% of subjects
    filtered_clusters <- data_percentages %>%
      filter(Percentage.of.subjects >= 75.0) %>%
      select(window, cluster, response)

    # Subset results dataframe while retaining original order
    results_filtered <- results %>%
      semi_join(filtered_clusters, by = c("window", "cluster", "response"))

    results <- results_filtered
  }

  # Apply Holm-Bonferroni correction by response type
  results_corr <- results %>%
    group_by(response) %>%
    mutate(p_corr = holm_bonferroni(p))

  results_corr$sign_corr <- ifelse(
    results_corr$p_corr < 0.001, "***",
    ifelse(
      results_corr$p_corr < 0.01, "**",
      ifelse(
        results_corr$p_corr < 0.05, "*", ""
      )
    )
  )

  # Reorder column
  results_corr <- results_corr %>%
    select(window, response, cluster, U, p, sign, p_corr, sign_corr, r)
  
  results_corr <- as.data.frame(results_corr)
  
  if (current_analysis == "amplitude") {
    amplitude_results <- results_corr
    amplitude_subject_no <- participants_no
  } else if (current_analysis == "latency") {
    latency_results <- results_corr
    latency_subject_no <- participants_no
  }
}


# ----------------------------------------------------------------------
# Check results

print(
  paste0(
    "Data frames identical: ",
    identical(amplitude_subject_no, latency_subject_no)
  )
)
merged_df <- amplitude_results %>%
  left_join(
    latency_results,
    by = c("window", "response", "cluster"),
    suffix = c("_amp", "_lat")
  )
final_df <- merged_df %>%
  left_join(
    amplitude_subject_no
  )


# ----------------------------------------------------------------------
# Create nice LaTeX tables

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

final_df$window <- sapply(final_df$window, function(x) {
  x <- tolower(x)
  x <- paste0(toupper(substr(x, 1, 1)), substr(x, 2, nchar(x)))
})
final_df$response <- sapply(final_df$response, function(x) {
  x <- tolower(x)
  x <- paste0(toupper(substr(x, 1, 1)), substr(x, 2, nchar(x)))
})
final_df <- final_df %>%
  rename("Speech feature" = response)
final_df <- final_df %>%
  rename("Window" = window)
final_df <- final_df %>%
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
    "p_lat",
    "sign_corr_lat",
    "r_lat",
    "n_normal",
    "n_low"
  )
  final_df <- final_df %>%
    select(relevant_cols)
  
  write.csv(final_df, table_outpath, row.names = FALSE)
}