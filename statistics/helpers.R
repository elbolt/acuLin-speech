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


#' Finds the cluster to which an electrode belongs.
#' 
#' @param electrode_id Character string representing the electrode ID.
#' @param cluster_dict A named list where each element is cluster of electrodes.
#' @return Name of cluster to which the electrode belongs, or NA.
find_cluster <- function(electrode_id, cluster_dict) {
  for (cluster in names(cluster_dict)) {
    if (electrode_id %in% cluster_dict[[cluster]]) {
      return(cluster)
    }
  }
  return(NA)
}