from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
GENERATED = ROOT / "generated"


DEFAULT_CASES = ["airline", "parking", "telephone", "lpmc"]


def run_case(case: str, python: str, n_obs: int | None) -> dict:
    json_path = GENERATED / f"public_mnl_{case}.json"
    command = [
        python,
        str(BENCHMARKS / "compare_biogeme_public_mnl.py"),
        "--case",
        case,
        "--json-output",
        str(json_path),
    ]
    if n_obs is not None:
        command.extend(["--n-obs", str(n_obs)])
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=BENCHMARKS, text=True, capture_output=True)
    wall_s = time.perf_counter() - start
    payload = {
        "case": case,
        "returncode": proc.returncode,
        "wall_s": wall_s,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if proc.returncode == 0 and json_path.exists():
        payload["result"] = json.loads(json_path.read_text(encoding="utf-8"))
    return payload


def backend(result: dict, name: str) -> dict:
    return next(row for row in result["backends"] if row["backend"] == name)


def render_markdown(results: list[dict]) -> str:
    lines = [
        "# Public MNL Full-Estimation Battery",
        "",
        "This table follows the IJOC software-paper benchmark style: public data, aligned model specification, shared zero starts, runtime split, and numerical parity metrics.",
        "",
        "| Case | Dataset | n | k | Torch est. (s) | Torch cov. (s) | Biogeme est. (s) | Biogeme cov. (s) | LL diff | beta diff | prob diff | cov diff | SE diff |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        if item["returncode"] != 0:
            lines.append(f"| {item['case']} | failed |  |  |  |  |  |  |  |  |  |  |  |")
            continue
        result = item["result"]
        torch = backend(result, "torchdcm")
        bio = backend(result, "biogeme")
        lines.append(
            "| {case} | {dataset} | {n_obs} | {n_parameters} | {torch_est} | {torch_cov} | {bio_est} | {bio_cov} | {ll_diff} | {param_diff} | {prob_diff} | {cov_diff} | {se_diff} |".format(
                case=result["case"],
                dataset=result["dataset_id"],
                n_obs=result["n_obs"],
                n_parameters=result["n_parameters"],
                torch_est=_fmt(torch["estimate_s"]),
                torch_cov=_fmt(torch["covariance_s"]),
                bio_est=_fmt(bio["estimate_s"]),
                bio_cov=_fmt(bio["covariance_s"]),
                ll_diff=_sci(bio["ll_diff"]),
                param_diff=_sci(bio["max_param_diff"]),
                prob_diff=_sci(bio["max_prob_diff"]),
                cov_diff=_sci(bio["max_cov_diff"]),
                se_diff=_sci(bio["max_se_diff"]),
            )
        )
    lines.extend(
        [
            "",
            "## Per-Case Logs",
            "",
        ]
    )
    for item in results:
        lines.append(f"### {item['case']}")
        lines.append("")
        if item["stdout"]:
            lines.append("```text")
            lines.append(item["stdout"].strip())
            lines.append("```")
        if item["stderr"]:
            lines.append("")
            lines.append("stderr:")
            lines.append("```text")
            lines.append(item["stderr"].strip())
            lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.3f}"


def _sci(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.2e}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--n-obs", type=int, default=None, help="Optional cap for smoke runs.")
    parser.add_argument("--profile", default="full")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    GENERATED.mkdir(parents=True, exist_ok=True)
    results = [run_case(case, args.python, args.n_obs) for case in args.cases]
    json_path = GENERATED / f"public_mnl_battery_{args.profile}.json"
    md_path = GENERATED / f"public_mnl_battery_{args.profile}.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(results), encoding="utf-8")
    print(f"json: {json_path}")
    print(f"markdown: {md_path}")
    for item in results:
        print(f"{item['case']}: returncode={item['returncode']} wall_s={item['wall_s']:.3f}")


if __name__ == "__main__":
    main()
