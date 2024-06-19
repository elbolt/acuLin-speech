#' Formats a value as scientific notation if it is less than 0.0005, otherwise rounds to three decimals.
#'
#' @param value Numeric value to be formatted.
#' @return A character string of the formatted value.
#' @examples
#' format_value(0.0001)  # returns "1.0e-04"
#' format_value(0.005)   # returns "0.005"
format_value <- function(value) {
  if (abs(value) < 0.0005) {
    return(formatC(value, format = "e", digits = 1))
  } else {
    return(formatC(value, format = "f", digits = 3))
  }
}

#' Formats and wraps a numeric value in dollar signs.
#'
#' @param x Numeric value to be formatted and wrapped.
#' @return A character string with the formatted value wrapped in dollar signs.
#' @examples
#' format_and_wrap(0.0001)  # returns "$1.0e-04$"
#' format_and_wrap(0.005)   # returns "$0.005$"
format_and_wrap <- function(x) {
  paste0("$", format_value(x), "$")
}

#' Custom sanitization function that returns the input unchanged.
#'
#' @param x A character string to be sanitized.
#' @return The input character string unchanged.
#' @examples
#' sanitize("$example$")  # returns "$example$"
sanitize <- function(x) x