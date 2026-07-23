from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
GENERATED = ROOT / "generated"


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    command: list[str]
    dataset: str
    model: str
    mode: str


def smoke_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            "swissmetro_mnl_estimate",
            [
                "compare_mnl_estimators.py",
                "--case",
                "swissmetro",
                "--n-obs",
                "500",
                "--initial",
                "zero",
            ],
            "Biogeme Swissmetro",
            "MNL",
            "full_estimation",
        ),
        BenchmarkCase(
            "swissmetro_nested_estimate",
            [
                "compare_nested_logit_estimators.py",
                "--n-obs",
                "500",
                "--initial",
                "zero",
                "--lambda-init",
                "0.8",
            ],
            "Biogeme Swissmetro",
            "Nested Logit",
            "full_estimation",
        ),
        BenchmarkCase(
            "swissmetro_cross_nested_estimate",
            [
                "compare_cross_nested_logit_estimators.py",
                "--n-obs",
                "500",
                "--mode",
                "full-estimation",
                "--max-iter",
                "80",
            ],
            "Biogeme Swissmetro",
            "Cross-Nested Logit",
            "full_estimation",
        ),
        BenchmarkCase(
            "swissmetro_mixed_panel_fixed",
            [
                "compare_mixed_logit_estimators.py",
                "--n-obs",
                "500",
                "--n-draws",
                "32",
                "--panel",
            ],
            "Biogeme Swissmetro",
            "Mixed Logit",
            "fixed_replay_shared_draws",
        ),
        BenchmarkCase(
            "swissmetro_wtp_mixed_panel_fixed",
            [
                "compare_wtp_mixed_logit_estimators.py",
                "--n-obs",
                "500",
                "--n-draws",
                "32",
                "--panel",
            ],
            "Biogeme Swissmetro",
            "WTP Mixed Logit",
            "fixed_replay_shared_draws",
        ),
        BenchmarkCase(
            "swissmetro_latent_class_fit_replay",
            [
                "compare_latent_class_estimators.py",
                "--n-obs",
                "500",
                "--mode",
                "fit-replay",
                "--membership",
                "ga",
                "--max-iter",
                "40",
            ],
            "Biogeme Swissmetro",
            "Latent Class Logit",
            "torch_fit_then_replay",
        ),
        BenchmarkCase(
            "optima_ordered_logit_estimate",
            [
                "compare_ordered_estimators.py",
                "--kind",
                "logit",
                "--indicator",
                "Envir01",
                "--n-obs",
                "500",
                "--mode",
                "full-estimation",
                "--max-iter",
                "80",
            ],
            "Biogeme Optima",
            "Ordered Logit",
            "full_estimation",
        ),
        BenchmarkCase(
            "optima_ordered_probit_estimate",
            [
                "compare_ordered_estimators.py",
                "--kind",
                "probit",
                "--indicator",
                "Envir01",
                "--n-obs",
                "500",
                "--mode",
                "full-estimation",
                "--max-iter",
                "80",
            ],
            "Biogeme Optima",
            "Ordered Probit",
            "full_estimation",
        ),
        BenchmarkCase(
            "mlogit_fishing_mnl_estimate",
            ["compare_mlogit_estimators.py", "--dataset", "fishing"],
            "R mlogit Fishing",
            "MNL",
            "full_estimation",
        ),
        BenchmarkCase(
            "mlogit_modecanada_mnl_estimate",
            ["compare_mlogit_estimators.py", "--dataset", "modecanada"],
            "R mlogit ModeCanada",
            "MNL",
            "full_estimation",
        ),
    ]


def full_cases() -> list[BenchmarkCase]:
    cases = smoke_cases()
    converted: list[BenchmarkCase] = []
    for case in cases:
        command = list(case.command)
        for index, token in enumerate(command):
            if command[index - 1] == "--n-obs" and token == "500":
                command[index] = "100000"
            if command[index - 1] == "--n-draws" and token == "32":
                command[index] = "64"
        converted.append(BenchmarkCase(case.name.replace("_500", "_full"), command, case.dataset, case.model, case.mode))
    return converted


def run_case(case: BenchmarkCase, python: str) -> dict:
    command = [python, str(BENCHMARKS / case.command[0]), *case.command[1:]]
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=BENCHMARKS, text=True, capture_output=True)
    wall = time.perf_counter() - start
    stdout = proc.stdout
    stderr = proc.stderr
    return {
        "name": case.name,
        "dataset": case.dataset,
        "model": case.model,
        "mode": case.mode,
        "command": command,
        "returncode": proc.returncode,
        "wall_seconds": wall,
        "stdout": stdout,
        "stderr": stderr,
        "backends": parse_backend_rows(stdout),
        "params": parse_param_blocks(stdout),
        "diffs": parse_secondary_rows(stdout),
    }


def parse_backend_rows(text: str) -> list[dict]:
    rows: list[dict] = []
    for line in text.splitlines():
        if not re.match(r"^(torchdcm|torchdcm_fit|torchdcm_fixed|scipy_bfgs|biogeme|biogeme_fixed|apollo|apollo_r_fixed|mlogit|gmnl|xlogit)\s+", line):
            continue
        parts = line.split()
        if len(parts) < 3 or parts[1] not in {"True", "False"}:
            continue
        row = {"backend": parts[0], "available": parts[1] == "True", "raw": line}
        if row["available"]:
            numeric_tokens = [_to_float(token) for token in parts[2:]]
            numeric_tokens = [value for value in numeric_tokens if value is not None]
            if numeric_tokens:
                row["numeric"] = numeric_tokens
        rows.append(row)
    return rows


def parse_secondary_rows(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        if not re.match(r"^(torchdcm|torchdcm_fit|torchdcm_fixed|scipy_bfgs|biogeme|biogeme_fixed|apollo|apollo_r_fixed|mlogit|gmnl|xlogit)\s+", line):
            continue
        if "prob_diff" in line or "params:" in line:
            continue
        if any(key in line for key in ["e-", "e+", "NA"]):
            rows.append({"raw": line})
    return rows


def parse_param_blocks(text: str) -> dict[str, dict[str, float]]:
    blocks: dict[str, dict[str, float]] = {}
    current: str | None = None
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z0-9_]+) params:$", line.strip())
        if match:
            current = match.group(1)
            blocks[current] = {}
            continue
        if current is None:
            continue
        param_match = re.match(r"^\s+([A-Za-z0-9_]+):\s+([-+0-9.eE]+)", line)
        if param_match:
            blocks[current][param_match.group(1)] = float(param_match.group(2))
        elif line and not line.startswith(" "):
            current = None
    return blocks


def _to_float(token: str) -> float | None:
    if token == "NA":
        return None
    try:
        return float(token)
    except ValueError:
        return None


def write_outputs(results: list[dict], profile: str) -> tuple[Path, Path]:
    GENERATED.mkdir(parents=True, exist_ok=True)
    json_path = GENERATED / f"estimator_benchmark_suite_{profile}.json"
    md_path = GENERATED / f"estimator_benchmark_suite_{profile}.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(results, profile), encoding="utf-8")
    return json_path, md_path


def render_markdown(results: list[dict], profile: str) -> str:
    lines = [
        f"# TorchDCM Estimator Benchmark Suite ({profile})",
        "",
        "All commands were run on the remote benchmark machine. Timing columns reported by the underlying scripts split parameter estimation and covariance calculation where the backend exposes both.",
        "",
        "| case | dataset | model | mode | status | wall_s | backends |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for result in results:
        status = "ok" if result["returncode"] == 0 else f"fail({result['returncode']})"
        backend_names = ", ".join(row["backend"] for row in result["backends"])
        lines.append(
            f"| {result['name']} | {result['dataset']} | {result['model']} | {result['mode']} | "
            f"{status} | {result['wall_seconds']:.3f} | {backend_names} |"
        )
    lines.append("")
    for result in results:
        lines.extend(
            [
                f"## {result['name']}",
                "",
                "```text",
                result["stdout"].strip() or result["stderr"].strip(),
                "```",
                "",
            ]
        )
        if result["stderr"].strip() and result["returncode"] == 0:
            lines.extend(["stderr:", "", "```text", result["stderr"].strip(), "```", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    cases = smoke_cases() if args.profile == "smoke" else full_cases()
    results = []
    for case in cases:
        print(f"[benchmark] running {case.name}", flush=True)
        result = run_case(case, args.python)
        results.append(result)
        status = "ok" if result["returncode"] == 0 else f"failed ({result['returncode']})"
        print(f"[benchmark] {case.name}: {status}, wall={result['wall_seconds']:.3f}s", flush=True)
    json_path, md_path = write_outputs(results, args.profile)
    print(f"json: {json_path}")
    print(f"markdown: {md_path}")
    if any(result["returncode"] != 0 for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
