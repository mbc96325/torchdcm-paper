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
n_classes <- length(spec$classes)

membership_logits <- matrix(0, nrow = n_obs, ncol = n_classes)
if (n_classes > 1) {
  for (class_index in 2:n_classes) {
    class_spec <- spec$classes[[class_index]]
    values <- rep(0, n_obs)
    for (term in class_spec$membership_terms) {
      variable_values <- rep(1, n_obs)
      if (!is.null(term$variable)) {
        variable_values <- database[[term$variable]]
      }
      values <- values + params[[term$param]] * variable_values
    }
    membership_logits[, class_index] <- values
  }
}
max_membership <- apply(membership_logits, 1, max)
membership_probs <- exp(membership_logits - max_membership)
membership_probs <- membership_probs / rowSums(membership_probs)

prob_sum <- matrix(0, nrow = n_obs, ncol = length(alternatives))
chosen_prob <- rep(0, n_obs)
availability <- matrix(FALSE, nrow = n_obs, ncol = length(alternatives))
for (j in seq_along(alternatives)) {
  alt <- alternatives[[j]]
  availability[, j] <- as.logical(database[[spec$availability[[alt]]]])
}

for (class_index in seq_len(n_classes)) {
  class_spec <- spec$classes[[class_index]]
  utilities <- matrix(0, nrow = n_obs, ncol = length(alternatives))
  for (j in seq_along(alternatives)) {
    alt <- alternatives[[j]]
    item <- class_spec$utility[[alt]]
    utility <- rep(0, n_obs)
    if (!is.null(item$asc)) {
      utility <- utility + params[[item$asc]]
    }
    utility <- utility + params[[item$time_param]] * database[[item$time]]
    utility <- utility + params[[item$cost_param]] * database[[item$cost]]
    utilities[, j] <- utility
  }
  utilities[!availability] <- -Inf
  max_utility <- apply(utilities, 1, max)
  exp_utility <- exp(utilities - max_utility)
  exp_utility[!is.finite(exp_utility)] <- 0
  probabilities <- exp_utility / rowSums(exp_utility)
  class_weight <- membership_probs[, class_index]
  prob_sum <- prob_sum + class_weight * probabilities
  chosen_code <- database[[spec$choice_col]]
  chosen_prob <- chosen_prob + class_weight * probabilities[cbind(seq_len(n_obs), chosen_code)]
}

out <- list(
  backend = "apollo_r_fixed",
  model = "latent_class_logit",
  loglike = as.numeric(sum(log(pmax(chosen_prob, .Machine$double.xmin)))),
  probabilities = as.vector(t(prob_sum)),
  class_probabilities = as.vector(t(membership_probs))
)

write_json(out, output_path, pretty = TRUE, auto_unbox = TRUE, digits = 16)
