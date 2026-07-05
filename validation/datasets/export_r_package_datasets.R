suppressPackageStartupMessages({
  if (!requireNamespace("utils", quietly = TRUE)) {
    stop("The utils package is required.")
  }
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript export_r_package_datasets.R <raw_output_dir>")
}

raw_output_dir <- normalizePath(args[[1]], mustWork = FALSE)
dir.create(raw_output_dir, recursive = TRUE, showWarnings = FALSE)

safe_id <- function(x) {
  x <- gsub("[^A-Za-z0-9_]+", "_", x)
  tolower(gsub("_+", "_", x))
}

write_dataset <- function(dataset_id, package_name, object_name, frame) {
  out_dir <- file.path(raw_output_dir, dataset_id)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  data_path <- file.path(out_dir, "data.csv")
  utils::write.csv(frame, data_path, row.names = FALSE, na = "")
  data.frame(
    dataset_id = dataset_id,
    source_package = package_name,
    object_name = object_name,
    status = "downloaded",
    rows = nrow(frame),
    columns = ncol(frame),
    file = data_path,
    message = "",
    stringsAsFactors = FALSE
  )
}

load_package_data <- function(package_name) {
  if (!requireNamespace(package_name, quietly = TRUE)) {
    return(list(objects = character(), error = paste("R package not installed:", package_name)))
  }
  package_data <- utils::data(package = package_name)$results
  if (is.null(package_data) || length(package_data) == 0) {
    return(list(objects = character(), error = "No package data objects found."))
  }
  list(objects = package_data[, "Item"], error = "")
}

load_object <- function(package_name, object_name) {
  env <- new.env(parent = emptyenv())
  utils::data(list = object_name, package = package_name, envir = env)
  if (!exists(object_name, envir = env, inherits = FALSE)) {
    stop(paste("Data object was not loaded:", object_name))
  }
  obj <- get(object_name, envir = env, inherits = FALSE)
  if (!is.data.frame(obj)) {
    obj <- as.data.frame(obj)
  }
  obj
}

export_named <- function(dataset_id, package_name, candidates = character(), pattern = NULL) {
  package_info <- load_package_data(package_name)
  if (length(package_info$error) && nzchar(package_info$error)) {
    return(data.frame(
      dataset_id = dataset_id,
      source_package = package_name,
      object_name = "",
      status = "missing",
      rows = NA_integer_,
      columns = NA_integer_,
      file = "",
      message = package_info$error,
      stringsAsFactors = FALSE
    ))
  }
  objects <- package_info$objects
  object_name <- NA_character_
  for (candidate in candidates) {
    hit <- objects[tolower(objects) == tolower(candidate)]
    if (length(hit) > 0) {
      object_name <- hit[[1]]
      break
    }
  }
  if (is.na(object_name) && !is.null(pattern)) {
    hit <- grep(pattern, objects, ignore.case = TRUE, value = TRUE)
    if (length(hit) > 0) {
      object_name <- hit[[1]]
    }
  }
  if (is.na(object_name)) {
    return(data.frame(
      dataset_id = dataset_id,
      source_package = package_name,
      object_name = "",
      status = "missing",
      rows = NA_integer_,
      columns = NA_integer_,
      file = "",
      message = paste("Could not find data object. Available:", paste(objects, collapse = ", ")),
      stringsAsFactors = FALSE
    ))
  }
  tryCatch(
    write_dataset(dataset_id, package_name, object_name, load_object(package_name, object_name)),
    error = function(e) data.frame(
      dataset_id = dataset_id,
      source_package = package_name,
      object_name = object_name,
      status = "failed",
      rows = NA_integer_,
      columns = NA_integer_,
      file = "",
      message = conditionMessage(e),
      stringsAsFactors = FALSE
    )
  )
}

exports <- list(
  export_named("mlogit_fishing", "mlogit", candidates = c("Fishing")),
  export_named("mlogit_modecanada", "mlogit", candidates = c("ModeCanada")),
  export_named("mlogit_electricity", "mlogit", candidates = c("Electricity")),
  export_named("mlogit_train", "mlogit", candidates = c("Train")),
  export_named("mlogit_car", "mlogit", candidates = c("Car")),
  export_named("mlogit_risky_transport", "mlogit", candidates = c("RiskyTransport")),
  export_named("mlogit_catsup", "mlogit", candidates = c("Catsup")),
  export_named("mlogit_cracker", "mlogit", candidates = c("Cracker")),
  export_named("mlogit_game", "mlogit", candidates = c("Game")),
  export_named("mlogit_game2", "mlogit", candidates = c("Game2")),
  export_named("mlogit_hc", "mlogit", candidates = c("HC")),
  export_named("mlogit_heating", "mlogit", candidates = c("Heating")),
  export_named("mlogit_japanese_fdi", "mlogit", candidates = c("JapaneseFDI")),
  export_named("mlogit_mode", "mlogit", candidates = c("Mode")),
  export_named("mlogit_nox", "mlogit", candidates = c("NOx")),
  export_named(
    "apollo_mode_choice",
    "apollo",
    candidates = c("apollo_modeChoiceData", "modeChoiceData", "apollo_mode_choice"),
    pattern = "mode.*choice|choice.*mode"
  ),
  export_named(
    "apollo_drug_choice",
    "apollo",
    candidates = c("apollo_drugChoiceData", "drugChoiceData", "apollo_drug_choice"),
    pattern = "drug"
  ),
  export_named(
    "apollo_swiss_route_choice",
    "apollo",
    candidates = c("apollo_swissRouteChoiceData", "swissRouteChoiceData", "apollo_swiss_route_choice"),
    pattern = "swiss.*route|route.*swiss"
  ),
  export_named(
    "apollo_time_use",
    "apollo",
    candidates = c("apollo_timeUseData", "timeUseData", "apollo_time_use"),
    pattern = "time.*use|use.*time"
  )
)

manifest <- do.call(rbind, exports)
utils::write.csv(manifest, file.path(raw_output_dir, "_r_export_manifest.csv"), row.names = FALSE, na = "")

cat("R package dataset export summary\n")
for (i in seq_len(nrow(manifest))) {
  row <- manifest[i, ]
  cat(sprintf(
    "%-28s %-10s %-28s rows=%s cols=%s status=%s\n",
    row$dataset_id,
    row$source_package,
    row$object_name,
    as.character(row$rows),
    as.character(row$columns),
    row$status
  ))
  if (nzchar(row$message)) {
    cat("  message:", row$message, "\n")
  }
}
