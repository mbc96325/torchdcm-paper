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
code_by_alt <- setNames(seq_along(alternatives), alternatives)
random_specs <- spec$random_coefficients
if (is.null(random_specs)) {
  random_specs <- list(list(name = "B_TIME", distribution = "normal"))
}
n_random <- length(random_specs)
correlated <- isTRUE(spec$correlated)
error_components <- spec$error_components

prob_sum <- matrix(0, nrow = n_obs, ncol = length(alternatives))
chosen_prob_by_draw <- matrix(0, nrow = n_obs, ncol = n_draws)

random_value <- function(base_value, distribution) {
  if (distribution == "normal") {
    return(base_value)
  }
  if (distribution == "lognormal") {
    return(exp(base_value))
  }
  if (distribution == "negative_lognormal") {
    return(-exp(base_value))
  }
  stop(paste("Unsupported distribution", distribution))
}

drawn_parameters <- function(r) {
  values <- list()
  if (n_random == 0) {
    return(values)
  }
  latent_noise <- rep(0, n_random)
  for (i in seq_len(n_random)) {
    row_spec <- random_specs[[i]]
    row_name <- row_spec$name
    for (j in seq_len(i)) {
      if (!correlated && i != j) {
        next
      }
      col_name <- random_specs[[j]]$name
      if (i == j) {
        param_name <- paste0("SIGMA_", row_name)
      } else {
        param_name <- paste0("CHOL_", row_name, "__", col_name)
      }
      latent_noise[[i]] <- latent_noise[[i]] + params[[param_name]] * draws[r, col_name]
    }
  }
  for (i in seq_len(n_random)) {
    item <- random_specs[[i]]
    values[[item$name]] <- random_value(params[[item$name]] + latent_noise[[i]], item$distribution)
  }
  return(values)
}

for (r in seq_len(n_draws)) {
  random_params <- drawn_parameters(r)
  b_time <- if (!is.null(random_params[["B_TIME"]])) random_params[["B_TIME"]] else params[["B_TIME"]]
  b_cost <- if (!is.null(random_params[["B_COST"]])) random_params[["B_COST"]] else params[["B_COST"]]
  utilities <- matrix(0, nrow = n_obs, ncol = length(alternatives))
  availability <- matrix(FALSE, nrow = n_obs, ncol = length(alternatives))

  for (j in seq_along(alternatives)) {
    alt <- alternatives[[j]]
    item <- spec$utility[[alt]]
    utility <- rep(0, n_obs)
    if (!is.null(item$asc)) {
      utility <- utility + params[[item$asc]]
    }
    utility <- utility + b_time * database[[item$time]] + b_cost * database[[item$cost]]
    if (!is.null(error_components)) {
      for (component in error_components) {
        value <- if (!is.null(random_params[[component$parameter]])) random_params[[component$parameter]] else params[[component$parameter]]
        loading <- component$loadings[[alt]]
        if (!is.null(loading)) {
          utility <- utility + value * loading
        }
      }
    }
    utilities[, j] <- utility
    availability[, j] <- as.logical(database[[item$availability]])
  }

  utilities[!availability] <- -Inf
  max_utility <- apply(utilities, 1, max)
  exp_utility <- exp(utilities - max_utility)
  exp_utility[!is.finite(exp_utility)] <- 0
  probabilities <- exp_utility / rowSums(exp_utility)
  prob_sum <- prob_sum + probabilities

  chosen_code <- database[[spec$choice_col]]
  chosen_prob_by_draw[, r] <- probabilities[cbind(seq_len(n_obs), chosen_code)]
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
  model = "mixed_logit",
  loglike = as.numeric(loglike),
  probabilities = as.vector(t(avg_prob)),
  n_draws = n_draws,
  panel = isTRUE(spec$panel)
)

write_json(out, output_path, pretty = TRUE, auto_unbox = TRUE, digits = 16)
