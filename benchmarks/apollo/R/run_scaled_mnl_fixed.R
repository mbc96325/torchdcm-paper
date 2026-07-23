#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag) {
  index <- match(flag, args)
  if (is.na(index) || index == length(args)) {
    stop(paste("Missing required argument", flag))
  }
  args[[index + 1]]
}

data_path <- get_arg("--data")
spec_path <- get_arg("--spec")
output_path <- get_arg("--output")

suppressPackageStartupMessages(library(jsonlite))

spec <- fromJSON(spec_path, simplifyVector = FALSE)
database <- read.csv(data_path)
params <- unlist(spec$parameters)

n_obs <- nrow(database)
alternatives <- spec$alternatives
utilities <- matrix(0, nrow = n_obs, ncol = length(alternatives))
availability <- matrix(FALSE, nrow = n_obs, ncol = length(alternatives))

for (j in seq_along(alternatives)) {
  alt <- alternatives[[j]]
  item <- spec$utility[[alt]]
  utility <- rep(0, n_obs)
  if (!is.null(item$asc)) {
    utility <- utility + params[[item$asc]]
  }
  utility <- utility + params[["B_TIME"]] * database[[item$time]]
  utility <- utility + params[["B_COST"]] * database[[item$cost]]
  scale_value <- spec$scales[[alt]]
  if (is.character(scale_value)) {
    scale_value <- params[[scale_value]]
  }
  utilities[, j] <- utility / scale_value
  availability[, j] <- as.logical(database[[item$availability]])
}

utilities[!availability] <- -Inf
max_utility <- apply(utilities, 1, max)
exp_utility <- exp(utilities - max_utility)
exp_utility[!is.finite(exp_utility)] <- 0
probabilities <- exp_utility / rowSums(exp_utility)
chosen_prob <- probabilities[cbind(seq_len(n_obs), database[[spec$choice_col]])]

out <- list(
  backend = "apollo_r_fixed",
  model = "scaled_mnl",
  loglike = as.numeric(sum(log(pmax(chosen_prob, .Machine$double.xmin)))),
  probabilities = as.vector(t(probabilities))
)

write_json(out, output_path, pretty = TRUE, auto_unbox = TRUE, digits = 16)
