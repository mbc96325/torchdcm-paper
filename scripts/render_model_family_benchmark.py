from __future__ import annotations

import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUITE_JSON = ROOT / "generated" / "estimator_benchmark_suite_full.json"
PUBLIC_MNL_JSON = ROOT / "generated" / "public_mnl_battery_full.json"
OUTPUT = ROOT / "generated" / "model-family-benchmark-comparison.md"

BACKEND_RE = re.compile(
    r"^(torchdcm|torchdcm_fit|torchdcm_fixed|scipy_bfgs|biogeme|biogeme_fixed|apollo|apollo_r_fixed|mlogit|gmnl|xlogit)\s+"
)


def main() -> None:
    suite = json.loads(SUITE_JSON.read_text(encoding="utf-8"))
    public_mnl = json.loads(PUBLIC_MNL_JSON.read_text(encoding="utf-8"))

    lines = [
        "# Model-Family Benchmark Comparison",
        "",
        "This note groups the remote benchmark results by model family for the IJOC",
        "software-paper experiments. All rows use real/public data aligned with",
        "Biogeme, Apollo, or R package examples. Parameter-estimation time and",
        "covariance/Hessian time are reported separately when the backend exposes",
        "both values.",
        "",
        "Source result files:",
        "",
        "- `generated/estimator_benchmark_suite_full.json`",
        "- `generated/public_mnl_battery_full.json`",
        "",
        "## Cross-Model Summary",
        "",
        "| Case | Model | Dataset | Mode | n | Torch total (s) | Torch est. (s) | Torch cov. (s) | References | Best ref total (s) | Max LL diff | Max beta diff | Max prob diff | Max cov diff | Max SE diff | Status |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for result in suite:
        lines.append(render_suite_row(result))

    lines.extend(
        [
            "",
            "## Public MNL Full-Estimation Battery",
            "",
            "These cases broaden MNL validation beyond Swissmetro. They use shared",
            "zero starts, matched specifications, and classic inverse observed",
            "information covariance.",
            "",
            "| Case | Dataset | n | k | Torch est. (s) | Torch cov. (s) | Biogeme est. (s) | LL diff | beta diff | prob diff | cov diff | SE diff |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for case in public_mnl:
        lines.append(render_public_mnl_row(case))

    lines.extend(
        [
            "",
            "## Reading The Table",
            "",
            "- `full_estimation` means every available backend estimates parameters on",
            "  the aligned public dataset.",
            "- `torch_fit_then_replay` means TorchDCM estimates parameters and reference",
            "  packages replay the same fitted parameters to verify likelihood and",
            "  probability kernels.",
            "- `fixed_replay_shared_draws` means all backends use identical fixed",
            "  parameters and simulation draws. This validates mixed-logit probability",
            "  and log-likelihood kernels, but it is not yet a full estimation",
            "  benchmark.",
            "",
            "## Current Interpretation",
            "",
            "- MNL is the strongest completed family: Swissmetro plus public Biogeme",
            "  data and R community data all match at numerical tolerance, including",
            "  covariance, standard errors, WTP, and elasticity where defined.",
            "- Nested logit full estimation matches Biogeme/Apollo in likelihood,",
            "  parameters, probabilities, and WTP. Biogeme's nested covariance differs",
            "  more noticeably and should be audited before the manuscript table is",
            "  frozen.",
            "- Cross-nested logit now has full-estimation comparison against Biogeme.",
            "  Likelihood and probabilities align tightly; covariance/t-statistic",
            "  differences are currently the main item to explain or fix.",
            "- Mixed logit and WTP-space mixed logit have shared-draw fixed replay",
            "  parity against Biogeme and Apollo. The next required step is full",
            "  simulated maximum-likelihood estimation with shared random streams or",
            "  deterministic draws.",
            "- Ordered logit/probit full estimation on Optima aligns closely with",
            "  Biogeme for likelihood, parameters, probabilities, covariance, and",
            "  standard errors.",
            "",
            "## Next Benchmark Work",
            "",
            "1. Add full mixed-logit estimation on Swissmetro with deterministic",
            "   antithetic draws and compare against Biogeme/Apollo.",
            "2. Add mixed-logit comparisons with `gmnl` and `xlogit` on Fishing or",
            "   Electricity once the data shape and draw conventions are aligned.",
            "3. Extend nested-logit validation beyond Swissmetro using another public",
            "   Biogeme/Apollo example with a nontrivial nest structure.",
            "4. Decide how manuscript tables should report covariance for constrained",
            "   nest parameters, because that is currently the largest non-runtime",
            "   discrepancy.",
            "",
        ]
    )

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT)


def render_suite_row(result: dict) -> str:
    tables = parse_tables(result["stdout"])
    primary = tables[0] if tables else []
    secondary = tables[1] if len(tables) > 1 else []
    rows = {row["backend"]: row for row in primary}
    extra = {row["backend"]: row for row in secondary}

    torch_name = next((name for name in rows if name.startswith("torchdcm")), "")
    torch_row = rows.get(torch_name, {})
    ref_rows = [row for name, row in rows.items() if not name.startswith("torchdcm") and row.get("available") != "False"]
    refs = ", ".join(row["backend"] for row in ref_rows) or "NA"

    n_obs = extract_value(result["stdout"], r"^n_obs:\s+([0-9]+)")
    best_ref_total = min_float(row.get("total_s") for row in ref_rows)

    max_ll = max_abs(row.get("ll_diff") for row in ref_rows)
    max_beta = max_abs(first_present(row, ["max_param_diff", "param_diff"]) for row in ref_rows)
    max_prob = max_abs(first_present(row, ["prob_diff", "max_prob_diff"]) for row in ref_rows)
    max_cov = max_abs(first_present(row, ["cov_diff", "max_cov_diff"]) for row in ref_rows)
    max_se = max_abs(first_present(row, ["se_diff", "max_se_diff"]) for row in ref_rows)

    for row in ref_rows:
        erow = extra.get(row["backend"], {})
        max_prob = max_with_nan(max_prob, abs_float(erow.get("prob_diff")))
        max_cov = max_with_nan(max_cov, abs_float(erow.get("cov_diff")))
        max_se = max_with_nan(max_se, abs_float(erow.get("se_diff")))

    return (
        f"| {result['name']} | {result['model']} | {result['dataset']} | {result['mode']} | "
        f"{fmt_int(n_obs)} | {fmt_float(torch_row.get('total_s'))} | "
        f"{fmt_float(torch_row.get('estimate_s'))} | {fmt_float(first_present(torch_row, ['cov_s', 'covariance_s']))} | "
        f"{refs} | {fmt_float(best_ref_total)} | {fmt_sci(max_ll)} | {fmt_sci(max_beta)} | "
        f"{fmt_sci(max_prob)} | {fmt_sci(max_cov)} | {fmt_sci(max_se)} | {status_note(result)} |"
    )


def render_public_mnl_row(case: dict) -> str:
    result = case["result"]
    backends = {row["backend"]: row for row in result["backends"]}
    torch = backends["torchdcm"]
    biogeme = backends["biogeme"]
    return (
        f"| {result['case']} | {result['dataset_id']} | {result['n_obs']} | {result['n_parameters']} | "
        f"{fmt_float(torch.get('estimate_s'))} | {fmt_float(torch.get('covariance_s'))} | "
        f"{fmt_float(biogeme.get('estimate_s'))} | {fmt_sci(biogeme.get('ll_diff'))} | "
        f"{fmt_sci(biogeme.get('max_param_diff'))} | {fmt_sci(biogeme.get('max_prob_diff'))} | "
        f"{fmt_sci(biogeme.get('max_cov_diff'))} | {fmt_sci(biogeme.get('max_se_diff'))} |"
    )


def parse_tables(text: str) -> list[list[dict]]:
    tables: list[list[dict]] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line.startswith("backend"):
            index += 1
            continue
        headers = split_columns(line)
        rows: list[dict] = []
        index += 1
        while index < len(lines):
            row_line = lines[index].strip()
            if not row_line:
                break
            if not BACKEND_RE.match(row_line):
                break
            values = split_columns(row_line)
            row = dict(zip(headers, values))
            for key, value in list(row.items()):
                row[key] = cast_value(value)
            rows.append(row)
            index += 1
        tables.append(rows)
        index += 1
    return tables


def split_columns(line: str) -> list[str]:
    return re.split(r"\s{2,}", line.strip())


def cast_value(value: str):
    if value in {"NA", "True", "False"}:
        return value
    try:
        return float(value)
    except ValueError:
        return value


def extract_value(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1) if match else ""


def first_present(row: dict, keys: list[str]):
    for key in keys:
        if key in row:
            return row[key]
    return None


def abs_float(value) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return abs(float(value))
    return None


def max_abs(values) -> float:
    vals = [abs_float(value) for value in values]
    vals = [value for value in vals if value is not None]
    return max(vals) if vals else float("nan")


def min_float(values) -> float:
    vals = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    return min(vals) if vals else float("nan")


def max_with_nan(left: float, right: float | None) -> float:
    values = [value for value in [left, right] if isinstance(value, (int, float)) and math.isfinite(value)]
    return max(values) if values else float("nan")


def fmt_float(value) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "NA"
    return f"{value:.3f}" if abs(value) >= 0.001 else f"{value:.2e}"


def fmt_sci(value) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "NA"
    return f"{value:.2e}"


def fmt_int(value: str) -> str:
    return value or "NA"


def status_note(result: dict) -> str:
    mode = result["mode"]
    if mode == "full_estimation":
        return "full estimation"
    if mode == "torch_fit_then_replay":
        return "Torch fit plus reference replay"
    if mode == "fixed_replay_shared_draws":
        return "shared-draw fixed replay"
    return mode


if __name__ == "__main__":
    main()
