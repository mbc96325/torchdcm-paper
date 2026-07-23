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

spec <- fromJSON(spec_path, simplifyVector = FALSE)
database <- read.csv(data_path, check.names = FALSE)
kind <- spec$kind
shared_draws <- as.numeric(unlist(spec$draws))
n_draws <- length(shared_draws)
n_repeats <- as.integer(spec$n_repeats)

apollo_initialise()

apollo_control <- list(
  modelName = spec$model_name,
  modelDescr = paste("TorchDCM aligned fixed-likelihood replay for", kind),
  indivID = spec$id_col,
  mixing = kind %in% c("hybrid_choice", "panel_likelihood"),
  nCores = 1,
  analyticGrad = FALSE,
  noDiagnostics = TRUE,
  outputDirectory = dirname(output_path)
)

apollo_beta <- unlist(spec$parameters)
# Apollo's input validator requires at least one formally free parameter even
# though this runner performs no optimization. The supplied beta vector is
# replayed unchanged by apollo_llCalc.
apollo_fixed <- names(apollo_beta)[-1]

if (apollo_control$mixing) {
  apollo_draws <- list(
    interDrawsType = "halton",
    interNDraws = n_draws,
    interUnifDraws = c(),
    interNormDraws = c("unused_draw"),
    intraDrawsType = "",
    intraNDraws = 0,
    intraUnifDraws = c(),
    intraNormDraws = c()
  )

  apollo_randCoeff <- function(apollo_beta, apollo_inputs) {
    randcoeff <- list()
    eta <- matrix(
      rep(shared_draws, each = nrow(database)),
      nrow = nrow(database),
      ncol = n_draws
    )
    if (kind == "hybrid_choice") {
      randcoeff[["LV"]] <- G_Q * q + SIGMA_LV * eta
    } else {
      randcoeff[["B_X_RND"]] <- B_X + SIGMA_B_X * eta
    }
    randcoeff
  }
}

apollo_probabilities <- function(apollo_beta, apollo_inputs, functionality = "estimate") {
  apollo_attach(apollo_beta, apollo_inputs)
  on.exit(apollo_detach(apollo_beta, apollo_inputs))
  P <- list()

  if (kind == "latent_class") {
    V1 <- list(
      A = B_X_C1 * x_A,
      B = ASC_B_C1 + B_X_C1 * x_B,
      C = ASC_C_C1 + B_X_C1 * x_C
    )
    V2 <- list(
      A = B_X_C2 * x_A,
      B = ASC_B_C2 + B_X_C2 * x_B,
      C = ASC_C_C2 + B_X_C2 * x_C
    )
    common <- list(
      alternatives = c(A = 1, B = 2, C = 3),
      avail = list(A = av_A, B = av_B, C = av_C),
      choiceVar = choice_code
    )
    class_1_settings <- c(common, list(utilities = V1, componentName = "class_1"))
    class_2_settings <- c(common, list(utilities = V2, componentName = "class_2"))
    in_class <- list(
      class_1 = apollo_mnl(class_1_settings, functionality),
      class_2 = apollo_mnl(class_2_settings, functionality)
    )

    V_class <- list(class_1 = 0, class_2 = CLASS_2 + CLASS_2_Z * z)
    allocation_settings <- list(
      alternatives = c(class_1 = 1, class_2 = 2),
      avail = 1,
      choiceVar = NA,
      utilities = V_class,
      componentName = "class_allocation"
    )
    class_prob <- apollo_mnl(allocation_settings, functionality = "raw")
    lc_settings <- list(
      inClassProb = in_class,
      classProb = class_prob,
      componentName = "model"
    )
    P[["model"]] <- apollo_lc(lc_settings, apollo_inputs, functionality)
  }

  if (kind == "hybrid_choice") {
    indicator_1_settings <- list(
      outcomeNormal = y1,
      xNormal = LV,
      mu = 0,
      sigma = SIGMA_Y1,
      componentName = "indicator_1"
    )
    indicator_2_settings <- list(
      outcomeNormal = y2,
      xNormal = A2 + L2 * LV,
      mu = 0,
      sigma = SIGMA_Y2,
      componentName = "indicator_2"
    )
    P[["indicator_1"]] <- apollo_normalDensity(indicator_1_settings, functionality)
    P[["indicator_2"]] <- apollo_normalDensity(indicator_2_settings, functionality)

    V <- list(A = 0, B = ASC_B + B_X * x + B_ATT * LV)
    choice_settings <- list(
      alternatives = c(A = 1, B = 2),
      avail = list(A = 1, B = 1),
      choiceVar = choice_code,
      utilities = V,
      componentName = "choice"
    )
    P[["choice"]] <- apollo_mnl(choice_settings, functionality)
    P <- apollo_combineModels(P, apollo_inputs, functionality)
    P <- apollo_avgInterDraws(P, apollo_inputs, functionality)
  }

  if (kind == "panel_likelihood") {
    V <- list(
      A = B_X_RND * x_A,
      B = ASC_B + B_X_RND * x_B,
      C = ASC_C + B_X_RND * x_C
    )
    alternatives <- c(A = 1, B = 2, C = 3)
    availability <- list(A = 1, B = 1, C = 1)
    if ("x_D" %in% names(database)) {
      V[["D"]] <- ASC_D + B_X_RND * x_D
      alternatives <- c(alternatives, D = 4)
      availability[["D"]] <- 1
    }
    choice_settings <- list(
      alternatives = alternatives,
      avail = availability,
      choiceVar = choice_code,
      utilities = V,
      componentName = "choice"
    )
    P[["model"]] <- apollo_mnl(choice_settings, functionality)
    P <- apollo_panelProd(P, apollo_inputs, functionality)
    P <- apollo_avgInterDraws(P, apollo_inputs, functionality)
  }

  P <- apollo_prepareProb(P, apollo_inputs, functionality)
  P
}

apollo_inputs <- apollo_validateInputs(silent = TRUE)

calculate_loglike <- function() {
  components <- apollo_llCalc(
    apollo_beta,
    apollo_probabilities,
    apollo_inputs,
    silent = TRUE
  )
  if (is.list(components)) {
    if (!is.null(components[["model"]])) {
      return(as.numeric(components[["model"]]))
    }
    return(as.numeric(components[[1]]))
  }
  as.numeric(components)
}

# Warm up validation/preprocessed settings before timing the repeated replay.
loglike <- calculate_loglike()
timings <- numeric(n_repeats)
for (i in seq_len(n_repeats)) {
  start <- proc.time()[["elapsed"]]
  current <- calculate_loglike()
  timings[[i]] <- proc.time()[["elapsed"]] - start
  if (!isTRUE(all.equal(current, loglike, tolerance = 1e-10))) {
    stop("Apollo likelihood replay changed across repeated evaluations")
  }
}

out <- list(
  backend = "apollo",
  apollo_version = as.character(packageVersion("apollo")),
  kind = kind,
  loglike = loglike,
  eval_seconds = median(timings),
  repeats = n_repeats,
  n_rows = nrow(database),
  n_draws = n_draws
)

write_json(out, output_path, pretty = TRUE, auto_unbox = TRUE, digits = NA)
