library(jsonlite)
library(rstudioapi)
library(dplyr)
library(tidyr)
library(tools)


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
table_outpath <- file.path(tables_dir, config$TS1_peak_desctats_filename)
response_levels <- names(config$responses_dict)
cluster_levels <- names(config$channel_cluster_dict)
window_levels <- config$window_levels

data <- read.csv(peaks_filepath)
data$MoCA_group <- factor(data$MoCA_group, levels = c("normal", "low"))
data_percentages <- read.csv(peaks_stats_filepath)


# ----------------------------------------------------------------------
# Create summary of peak statistics table

# Step 1: Filter out clusters with less than 75% of subjects
valid_clusters <- data_percentages %>%
  filter(Percentage.of.subjects >= 75.0) %>%
  select(window, cluster, response)
data <- data %>%
  inner_join(valid_clusters, by = c("window", "cluster", "response"))

# Step 2: Calculate the summary statistics
summary_table <- data %>%
  group_by(window, response, cluster, MoCA_group, subject_id) %>%
  summarise(
    amplitude = mean(amplitude, na.rm = TRUE),
    latency_ms = mean(latency_ms, na.rm = TRUE)
  ) %>%
  ungroup() %>%
  group_by(window, response, cluster, MoCA_group) %>%
  summarise(
    n = n(),
    M_amp = mean(amplitude),
    SD_amp = sd(amplitude),
    M_lat = mean(latency_ms),
    SD_lat = sd(latency_ms)
  ) %>%
  ungroup() %>%
  mutate(
    cluster = factor(cluster, levels = cluster_levels),
    response = factor(response, levels = response_levels),
    window = factor(window, levels = window_levels),
  ) %>%
  arrange(window, response, cluster, MoCA_group)

# ----------------------------------------------------------------------
# Create LaTeX table

summary_table <- summary_table %>%
  pivot_wider(
    names_from = MoCA_group,
    values_from = c(n, M_amp, SD_amp, M_lat, SD_lat),
    names_glue = "{MoCA_group}_{.value}"
  ) %>%
  arrange(window, response, cluster)

col_oder <- c(
  "window",
  "response",
  "cluster",
  "normal_n",
  "normal_M_amp",
  "normal_SD_amp",
  "normal_M_lat",
  "normal_SD_lat",
  "low_n",
  "low_M_amp",
  "low_SD_amp",
  "low_M_lat",
  "low_SD_lat"
)
summary_table <- summary_table[, col_oder]

summary_table$response <- as.character(summary_table$response)
summary_table$response <- sapply(summary_table$response, capitalize_first)

summary_table$normal_n <- sapply(summary_table$normal_n, function(x) format_and_wrap(x, digits = 0))
summary_table$normal_M_amp <- sapply(summary_table$normal_M_amp, function(x) format_and_wrap(x, digits = 3))
summary_table$normal_SD_amp <- sapply(summary_table$normal_SD_amp, function(x) format_and_wrap(x, digits = 3))
summary_table$normal_M_lat <- sapply(summary_table$normal_M_lat, function(x) format_and_wrap(x, digits = 1))
summary_table$normal_SD_lat <- sapply(summary_table$normal_SD_lat, function(x) format_and_wrap(x, digits = 1))

summary_table$low_n <- sapply(summary_table$low_n, function(x) format_and_wrap(x, digits = 0))
summary_table$low_M_amp <- sapply(summary_table$low_M_amp, function(x) format_and_wrap(x, digits = 3))
summary_table$low_SD_amp <- sapply(summary_table$low_SD_amp, function(x) format_and_wrap(x, digits = 3))
summary_table$low_M_lat <- sapply(summary_table$low_M_lat, function(x) format_and_wrap(x, digits = 1))
summary_table$low_SD_lat <- sapply(summary_table$low_SD_lat, function(x) format_and_wrap(x, digits = 1))

col_names <- c(
  "",
  "Speech features",
  "Cluster",
  "n",
  "M",
  "SD",
  "M",
  "SD",
  "n",
  "M",
  "SD",
  "M",
  "SD"
)
colnames(summary_table) <- col_names

write.csv(summary_table, table_outpath, row.names = FALSE)
