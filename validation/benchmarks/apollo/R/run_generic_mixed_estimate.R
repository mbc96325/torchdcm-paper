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
database$apollo_row_id <- seq_len(nrow(database))

apollo_initialise()

apollo_control <- list(
  modelName = spec$model_name,
  modelDescr = "TorchDCM generic mixed logit benchmark",
  indivID = "apollo_row_id",
  mixing = TRUE,
  nCores = 1,
  analyticGrad = FALSE,
  noDiagnostics = TRUE,
  outputDirectory = dirname(output_path)
)

apollo_beta <- unlist(spec$parameters)
apollo_fixed <- c()

draw_names <- paste0("draw_", spec$random_coefficients)
apollo_draws <- list(
  interDrawsType = "halton",
  interNDraws = spec$n_draws,
  interUnifDraws = c(),
  interNormDraws = draw_names
)

apollo_randCoeff <- function(apollo_beta, apollo_inputs) {
  randcoeff <- list()
  for (param in spec$random_coefficients) {
    randcoeff[[paste0(param, "_RND")]] <- get(param) + get(paste0("SIGMA_", param)) * get(paste0("draw_", param))
  }
  return(randcoeff)
}

apollo_probabilities <- function(apollo_beta, apollo_inputs, functionality = "estimate") {
  apollo_attach(apollo_beta, apollo_inputs)
  on.exit(apollo_detach(apollo_beta, apollo_inputs))

  P <- list()
  V <- list()
  av <- list()
  alternatives <- list()

  for (alt in spec$alternatives) {
    item <- spec$utility[[alt]]
    utility <- rep(0, length(get(spec$choice_col)))
    for (term in item$terms) {
      coeff_name <- term$parameter
      if (coeff_name %in% spec$random_coefficients) {
        coeff_name <- paste0(coeff_name, "_RND")
      }
      if (is.null(term$variable)) {
        utility <- utility + get(coeff_name)
      } else {
        utility <- utility + get(coeff_name) * get(term$variable)
      }
    }
    V[[alt]] <- utility
    av[[alt]] <- get(item$availability)
    alternatives[[alt]] <- item$code
  }

  mnl_settings <- list(
    alternatives = alternatives,
    avail = av,
    choiceVar = get(spec$choice_col),
    V = V
  )

  P[["model"]] <- apollo_mnl(mnl_settings, functionality)
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
covariance <- model$varcov
robust_covariance <- model$robvarcov
covariance_seconds <- proc.time()[["elapsed"]] - covariance_start

out <- list(
  backend = "apollo",
  model = "generic_mixed_logit",
  apollo_version = as.character(packageVersion("apollo")),
  loglike = as.numeric(model$maximum),
  estimates = as.list(model$estimate),
  parameter_names = names(model$estimate),
  se = as.list(model$se),
  robust_se = as.list(model$robse),
  covariance = unname(covariance),
  covariance_names = names(model$estimate),
  robust_covariance = unname(robust_covariance),
  robust_covariance_names = names(model$estimate),
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
