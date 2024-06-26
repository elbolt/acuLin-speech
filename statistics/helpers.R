#' Formats a value as scientific notation if it is less than 0.0005, otherwise rounds to three decimals.
#'
#' @param value Numeric value to be formatted.
#' @return A character string of the formatted value.
#' @examples
#' format_value(0.0001)  # returns "1.0e-04"
#' format_value(0.005)   # returns "0.005"
format_value <- function(value, digits = 3) {
  if (abs(value) < 0.0005) {
    return(formatC(value, format = "e", digits = 1))
  } else {
    return(formatC(value, format = "f", digits = digits))
  }
}


#' Formats and wraps a numeric value in dollar signs.
#'
#' @param x Numeric value to be formatted and wrapped.
#' @param digits Number of decimal places to round to.
#' @param is_p_value Logical indicating if the value is a p-value.
#' @return A character string with the formatted value wrapped in dollar signs.
#' @examples
#' format_and_wrap(0.0001)  # returns "$1.0e-04$"
#' format_and_wrap(0.005)   # returns "$0.005$"
#' format_and_wrap(0.0005, is_p_value = TRUE)  # returns "$< .001$"
format_and_wrap <- function(x, digits = 3, is_p_value = FALSE) {
  if (is_p_value && x < 0.001) {
    return("$< 0.001$")
  } else {
    return(paste0("$", format_value(x, digits), "$"))
  }
}


#' Custom sanitization function that returns the input unchanged.
#'
#' @param x A character string to be sanitized.
#' @return The input character string unchanged.
#' @examples
#' sanitize("$example$")  # returns "$example$"
sanitize <- function(x) x


#' Custom function to combine multiple tables into a single data frame.
#' 
#' @param table_list A list of data frames to be combined.
#' @param model_names A character vector of model names to be added to the combined data frame.
#' @return A single data frame with all tables combined.
combine_tables <- function(table_list, model_names) {
  combined <- data.frame()
  for (i in seq(1, length(table_list), by = 8)) {
    model_df <- data.frame(
      Coefficient = table_list[[i]],
      Estimate = table_list[[i + 1]],
      LL = table_list[[i + 2]],
      UL = table_list[[i + 3]],
      df = table_list[[i + 4]],
      t = table_list[[i + 5]],
      p = table_list[[i + 6]],
      Significance = table_list[[i + 7]]
    )
    model_df$Model <- ""  # Initialize Model column with empty strings
    model_name_index <- ceiling(i / 8)
    separator <- data.frame(
      Coefficient = NA,
      Estimate = NA,
      LL = NA,
      UL = NA,
      df = NA,
      t = NA,
      p = NA,
      Significance = NA,
      Model = model_names[model_name_index]
    )
    combined <- bind_rows(combined, separator, model_df)
  }
  return(combined)
}

#' Function to apply Holm-Bonferroni correction
#' 
#' @param pvals A numeric vector of p-values to be corrected.
#' @return A numeric vector of corrected p-values.
holm_bonferroni <- function(pvals) {
  m <- length(pvals)
  ordered_pvals <- sort(pvals, index.return = TRUE)
  adjusted_pvals <- ordered_pvals$x * (m - seq_along(ordered_pvals$x) + 1)
  adjusted_pvals[adjusted_pvals > 1] <- 1
  result <- rep(NA, m)
  result[ordered_pvals$ix] <- adjusted_pvals
  return(result)
}


#' Function to capitalize the first letter of a string
#' 
#' @param x A character string to be capitalized.
#' @return A character string with the first letter capitalized.
capitalize_first <- function(x) {
  paste0(toupper(substr(x, 1, 1)), tolower(substr(x, 2, nchar(x))))
}
