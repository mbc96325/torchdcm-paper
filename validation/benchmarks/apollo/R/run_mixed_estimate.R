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

suppressPackageStartupMessages(library(apollo))
suppressPackageStartupMessages(library(jsonlite))
options(expressions = 500000)

spec <- fromJSON(spec_path, simplifyVector = FALSE)
database <- read.csv(data_path)
panel_mode <- isTRUE(spec$panel)
if (!panel_mode) {
  database$apollo_row_id <- seq_len(nrow(database))
}

apollo_initialise()

apollo_control <- list(
  modelName = spec$model_name,
  modelDescr = "TorchDCM MMNL benchmark",
  indivID = if (panel_mode) spec$panel_id_col else "apollo_row_id",
  mixing = TRUE,
  panelData = panel_mode,
  nCores = 1,
  analyticGrad = FALSE,
  noDiagnostics = TRUE
)

apollo_beta <- unlist(spec$parameters)
apollo_fixed <- c()

if (panel_mode) {
  apollo_draws <- list(
    interDrawsType = "halton",
    interNDraws = spec$n_draws,
    interUnifDraws = c(),
    interNormDraws = c("draws_time")
  )
} else {
  apollo_draws <- list(
    intraDrawsType = "halton",
    intraNDraws = spec$n_draws,
    intraUnifDraws = c(),
    intraNormDraws = c("draws_time")
  )
}

apollo_randCoeff <- function(apollo_beta, apollo_inputs) {
  apollo_attach(apollo_beta, apollo_inputs)
  on.exit(apollo_detach(apollo_beta, apollo_inputs))
  randcoeff <- list()
  randcoeff[["B_TIME_RND"]] <- B_TIME + SIGMA_B_TIME * draws_time
  return(randcoeff)
}

apollo_probabilities <- function(apollo_beta, apollo_inputs, functionality = "estimate") {
  apollo_attach(apollo_beta, apollo_inputs)
  on.exit(apollo_detach(apollo_beta, apollo_inputs))

  P <- list()
  V <- list()
  V[["TRAIN"]] <- ASC_TRAIN + B_TIME_RND * time_train + B_COST * cost_train
  V[["SM"]] <- B_TIME_RND * time_sm + B_COST * cost_sm
  V[["CAR"]] <- ASC_CAR + B_TIME_RND * time_car + B_COST * cost_car

  mnl_settings <- list(
    alternatives = c(TRAIN = 1, SM = 2, CAR = 3),
    avail = list(TRAIN = avail_train, SM = avail_sm, CAR = avail_car),
    choiceVar = choice_code,
    V = V
  )

  P[["model"]] <- apollo_mnl(mnl_settings, functionality)
  if (panel_mode) {
    P <- apollo_panelProd(P, apollo_inputs, functionality)
  }
  P <- apollo_avgInterDraws(P, apollo_inputs, functionality)
  P <- apollo_prepareProb(P, apollo_inputs, functionality)
  return(P)
}

apollo_inputs <- apollo_validateInputs()
estimate_start <- proc.time()[["elapsed"]]
model <- apollo_estimate(
  apollo_beta,
  apollo_fixed,
  apollo_probabilities,
  apollo_inputs,
  list(writeIter = FALSE)
)
estimate_seconds <- proc.time()[["elapsed"]] - estimate_start

covariance_start <- proc.time()[["elapsed"]]
covariance <- unname(model$varcov)
robust_covariance <- unname(model$robvarcov)
covariance_seconds <- proc.time()[["elapsed"]] - covariance_start

out <- list(
  backend = "apollo",
  model = "mixed_logit",
  apollo_version = as.character(packageVersion("apollo")),
  loglike = as.numeric(model$maximum),
  estimates = as.list(model$estimate),
  parameter_names = names(model$estimate),
  se = as.list(model$se),
  robust_se = as.list(model$robse),
  covariance = covariance,
  robust_covariance = robust_covariance,
  timing = list(
    estimate_seconds = estimate_seconds,
    covariance_seconds = covariance_seconds
  ),
  convergence = list(
    status = model$code,
    message = model$message
  )
)

write_json(out, output_path, pretty = TRUE, auto_unbox = TRUE, digits = 16)
