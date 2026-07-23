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
draws_path <- get_arg("--draws")
output_path <- get_arg("--output")

suppressPackageStartupMessages(library(jsonlite))

spec <- fromJSON(spec_path, simplifyVector = FALSE)
database <- read.csv(data_path)
draws <- as.matrix(read.csv(draws_path, check.names = FALSE))
params <- unlist(spec$parameters)

n_obs <- nrow(database)
n_draws <- nrow(draws)
alternatives <- spec$alternatives

prob_sum <- matrix(0, nrow = n_obs, ncol = length(alternatives))
chosen_prob_by_draw <- matrix(0, nrow = n_obs, ncol = n_draws)

for (r in seq_len(n_draws)) {
  wtp_time <- params[["WTP_TIME"]] + params[["SIGMA_WTP_TIME"]] * draws[r, "WTP_TIME"]
  b_cost <- params[["B_COST"]]
  utilities <- matrix(0, nrow = n_obs, ncol = length(alternatives))
  availability <- matrix(FALSE, nrow = n_obs, ncol = length(alternatives))

  for (j in seq_along(alternatives)) {
    alt <- alternatives[[j]]
    item <- spec$utility[[alt]]
    utility <- rep(0, n_obs)
    if (!is.null(item$asc)) {
      utility <- utility + params[[item$asc]]
    }
    utility <- utility + b_cost * database[[item$cost]] + b_cost * wtp_time * database[[item$time]]
    utilities[, j] <- utility
    availability[, j] <- as.logical(database[[item$availability]])
  }

  utilities[!availability] <- -Inf
  max_utility <- apply(utilities, 1, max)
  exp_utility <- exp(utilities - max_utility)
  exp_utility[!is.finite(exp_utility)] <- 0
  probabilities <- exp_utility / rowSums(exp_utility)
  prob_sum <- prob_sum + probabilities
  chosen_prob_by_draw[, r] <- probabilities[cbind(seq_len(n_obs), database[[spec$choice_col]])]
}

avg_prob <- prob_sum / n_draws
if (isTRUE(spec$panel)) {
  ids <- database[[spec$panel_id_col]]
  loglike <- 0
  for (id in unique(ids)) {
    rows <- which(ids == id)
    draw_log_prob <- colSums(log(pmax(chosen_prob_by_draw[rows, , drop = FALSE], .Machine$double.xmin)))
    loglike <- loglike + log(mean(exp(draw_log_prob - max(draw_log_prob)))) + max(draw_log_prob)
  }
} else {
  chosen_avg_prob <- rowMeans(chosen_prob_by_draw)
  loglike <- sum(log(pmax(chosen_avg_prob, .Machine$double.xmin)))
}

out <- list(
  backend = "apollo_r_fixed",
  model = "wtp_mixed_logit",
  loglike = as.numeric(loglike),
  probabilities = as.vector(t(avg_prob)),
  n_draws = n_draws,
  panel = isTRUE(spec$panel)
)

write_json(out, output_path, pretty = TRUE, auto_unbox = TRUE, digits = 16)
