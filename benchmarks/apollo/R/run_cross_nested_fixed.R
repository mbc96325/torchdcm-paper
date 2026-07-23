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
nests <- spec$nests
probabilities <- matrix(0, nrow = n_obs, ncol = length(alternatives))
availability <- matrix(FALSE, nrow = n_obs, ncol = length(alternatives))
utilities <- matrix(0, nrow = n_obs, ncol = length(alternatives))

for (j in seq_along(alternatives)) {
  alt <- alternatives[[j]]
  item <- spec$utility[[alt]]
  utility <- rep(0, n_obs)
  if (!is.null(item$asc)) {
    utility <- utility + params[[item$asc]]
  }
  utility <- utility + params[["B_TIME"]] * database[[item$time]]
  utility <- utility + params[["B_COST"]] * database[[item$cost]]
  utilities[, j] <- utility
  availability[, j] <- as.logical(database[[item$availability]])
}

log_s <- list()
log_g_terms <- list()
for (nest_name in names(nests)) {
  nest <- nests[[nest_name]]
  lambda <- params[[nest$lambda_param]]
  terms <- matrix(0, nrow = n_obs, ncol = length(alternatives))
  for (j in seq_along(alternatives)) {
    alpha <- nest$allocations[[alternatives[[j]]]]
    if (is.null(alpha) || alpha <= 0) {
      terms[, j] <- 0
    } else {
      terms[, j] <- alpha * availability[, j] * exp(utilities[, j] / lambda)
    }
  }
  s <- rowSums(terms)
  log_s[[nest_name]] <- log(pmax(s, .Machine$double.xmin))
  log_g_terms[[nest_name]] <- lambda * log_s[[nest_name]]
}

log_g_matrix <- do.call(cbind, log_g_terms)
max_log_g <- apply(log_g_matrix, 1, max)
log_g <- max_log_g + log(rowSums(exp(log_g_matrix - max_log_g)))

for (j in seq_along(alternatives)) {
  alt <- alternatives[[j]]
  log_num_terms <- list()
  for (nest_name in names(nests)) {
    nest <- nests[[nest_name]]
    alpha <- nest$allocations[[alt]]
    if (is.null(alpha) || alpha <= 0) {
      log_num_terms[[nest_name]] <- rep(-Inf, n_obs)
    } else {
      lambda <- params[[nest$lambda_param]]
      log_num_terms[[nest_name]] <- log(alpha) + utilities[, j] / lambda + (lambda - 1) * log_s[[nest_name]]
      log_num_terms[[nest_name]][!availability[, j]] <- -Inf
    }
  }
  log_num_matrix <- do.call(cbind, log_num_terms)
  max_log_num <- apply(log_num_matrix, 1, max)
  probabilities[, j] <- exp(max_log_num + log(rowSums(exp(log_num_matrix - max_log_num))) - log_g)
  probabilities[!is.finite(probabilities[, j]), j] <- 0
}

chosen_prob <- probabilities[cbind(seq_len(n_obs), database[[spec$choice_col]])]

out <- list(
  backend = "apollo_r_fixed",
  model = "cross_nested_logit",
  loglike = as.numeric(sum(log(pmax(chosen_prob, .Machine$double.xmin)))),
  probabilities = as.vector(t(probabilities))
)

write_json(out, output_path, pretty = TRUE, auto_unbox = TRUE, digits = 16)
