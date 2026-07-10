from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from torchdcm import MixedLogit, RandomCoefficient

import compare_generated_choice_battery as generated
import compare_real_mixed_logit_battery as mixed_real


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "generated"


@dataclass(frozen=True)
class DeviceStressSpec:
    case: str
    n_obs: int
    n_alternatives: int
    n_variables: int
    rho: float
    random_coefficients: int
    n_draws: int
    max_iter: int


def profile_specs(profile: str) -> list[DeviceStressSpec]:
    if profile == "smoke":
        return [
            DeviceStressSpec("torch_device_smoke_mixl", 1000, 4, 4, 0.2, 3, 32, 20),
        ]
    if profile == "calibration":
        return [
            DeviceStressSpec("torch_device_calib_mixl_50k", 50000, 12, 12, 0.5, 8, 256, 80),
            DeviceStressSpec("torch_device_calib_mixl_100k", 100000, 12, 12, 0.5, 8, 512, 80),
        ]
    return [
        DeviceStressSpec("torch_device_stress_mixl", 100000, 12, 12, 0.5, 8, 512, 80),
    ]


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / 1024**2)


def run_worker(args: argparse.Namespace) -> None:
    if args.cpu_threads and args.worker_device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.worker_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(json.dumps({"available": False, "status": "Unavailable", "message": "CUDA is not available."}))
        return
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    spec = DeviceStressSpec(
        case=args.case,
        n_obs=args.n_obs,
        n_alternatives=args.n_alternatives,
        n_variables=args.n_variables,
        rho=args.rho,
        random_coefficients=args.random_coefficients,
        n_draws=args.n_draws,
        max_iter=args.max_iter,
    )
    build_start = time.perf_counter()
    meta = generated.GeneratedSpec(
        spec.case,
        "mixl",
        spec.n_obs,
        spec.n_alternatives,
        spec.n_variables,
        spec.rho,
        spec.random_coefficients,
    )
    mnl_case = generated.build_mnl_case(meta, args.seed)
    case = generated.build_mixed_case(mnl_case)
    draws = mixed_real.make_draws(spec.n_draws, args.seed + generated.stable_case_offset(spec.case), len(case.random_names))
    build_s = time.perf_counter() - build_start

    setup_start = time.perf_counter()
    model = MixedLogit(
        case.spec,
        [RandomCoefficient(name, sigma_init=case.sigma_init) for name in case.random_names],
        draws=draws,
        panel=False,
        device=device,
        max_iter=spec.max_iter,
        dtype=torch.float64,
    )
    data = case.data.to(device=model.device, dtype=model.dtype)
    compiled = model.compile(data)
    internal_initial = torch.cat(
        [
            compiled.free_initial,
            model._sigma_to_internal(compiled.sigma_initial[~compiled.sigma_is_fixed]),
            compiled.chol_offdiag_initial,
        ]
    )
    internal_params = internal_initial.clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [internal_params],
        max_iter=spec.max_iter,
        tolerance_grad=model.tolerance_grad,
        line_search_fn=model.line_search_fn,
    )
    setup_s = time.perf_counter() - setup_start

    closures = {"count": 0}

    def closure():
        optimizer.zero_grad(set_to_none=True)
        natural = model._internal_to_natural(internal_params, compiled)
        loss = -model.loglike(natural, data, compiled)
        loss.backward()
        closures["count"] += 1
        return loss

    sync_device(device)
    estimate_start = time.perf_counter()
    optimizer.step(closure)
    sync_device(device)
    estimate_s = time.perf_counter() - estimate_start

    final_internal = internal_params.detach().clone()
    final_natural = model._internal_to_natural(final_internal, compiled)
    sync_device(device)
    loglike_start = time.perf_counter()
    loglike = float(model.loglike(final_natural, data, compiled).detach().cpu())
    sync_device(device)
    loglike_s = time.perf_counter() - loglike_start

    params = {name: float(value) for name, value in zip(compiled.free_names, final_natural.detach().cpu().numpy())}
    payload = {
        "available": True,
        "status": "Completed",
        "case": spec.case,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "n_obs": spec.n_obs,
        "n_alternatives": spec.n_alternatives,
        "n_variables": spec.n_variables,
        "rho": spec.rho,
        "random_coefficients": len(case.random_names),
        "n_draws": spec.n_draws,
        "max_iter": spec.max_iter,
        "n_parameters": len(compiled.free_names),
        "parameter_names": compiled.free_names,
        "build_s": build_s,
        "setup_s": setup_s,
        "estimate_s": estimate_s,
        "loglike_s": loglike_s,
        "total_model_s": setup_s + estimate_s + loglike_s,
        "total_worker_s": build_s + setup_s + estimate_s + loglike_s,
        "closure_evaluations": closures["count"],
        "loglike": loglike,
        "params": params,
        "peak_cuda_memory_mb": peak_memory_mb(device),
        "torch_version": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
    }
    print(json.dumps(payload, allow_nan=False))


def run_child(spec: DeviceStressSpec, args: argparse.Namespace, device: str, timeout_s: int) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-device",
        device,
        "--case",
        spec.case,
        "--n-obs",
        str(spec.n_obs),
        "--n-alternatives",
        str(spec.n_alternatives),
        "--n-variables",
        str(spec.n_variables),
        "--rho",
        str(spec.rho),
        "--random-coefficients",
        str(spec.random_coefficients),
        "--n-draws",
        str(spec.n_draws),
        "--max-iter",
        str(spec.max_iter),
        "--seed",
        str(args.seed),
    ]
    if args.cpu_threads:
        command.extend(["--cpu-threads", str(args.cpu_threads)])
    start = time.perf_counter()
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        return {
            "available": False,
            "status": "Timeout",
            "device": device,
            "timeout_s": timeout_s,
            "total_worker_s": time.perf_counter() - start,
            "message": f"Timed out after {timeout_s} seconds.",
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
        }
    if completed.returncode != 0:
        return {
            "available": False,
            "status": "Failed",
            "device": device,
            "total_worker_s": time.perf_counter() - start,
            "message": (completed.stderr or completed.stdout).strip()[-4000:],
        }
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {
            "available": False,
            "status": "Failed",
            "device": device,
            "total_worker_s": time.perf_counter() - start,
            "message": f"Could not parse worker JSON: {exc}; stdout={completed.stdout[-2000:]} stderr={completed.stderr[-2000:]}",
        }
    payload["supervisor_wall_s"] = time.perf_counter() - start
    return payload


def compare_cpu_gpu(results: dict[str, dict]) -> dict:
    cpu = results.get("cpu", {})
    gpu = results.get("cuda", {})
    comparison: dict[str, object] = {}
    if cpu.get("available") and gpu.get("available"):
        comparison["objective_diff"] = abs(float(cpu["loglike"]) - float(gpu["loglike"]))
        cpu_params = cpu.get("params", {})
        gpu_params = gpu.get("params", {})
        common = sorted(set(cpu_params) & set(gpu_params))
        comparison["max_param_diff"] = max((abs(float(cpu_params[name]) - float(gpu_params[name])) for name in common), default=None)
        comparison["speedup_total_model"] = float(cpu["total_model_s"]) / float(gpu["total_model_s"])
        comparison["speedup_estimate"] = float(cpu["estimate_s"]) / float(gpu["estimate_s"])
        comparison["consistent"] = bool(
            comparison["objective_diff"] <= 1e-5 * max(1.0, abs(float(gpu["loglike"])))
            and (comparison["max_param_diff"] is None or comparison["max_param_diff"] <= 1e-4)
        )
    elif cpu.get("status") == "Timeout" and gpu.get("available"):
        comparison["consistent"] = "CPU timeout; GPU completed"
        comparison["speedup_total_model"] = f">{cpu.get('timeout_s', 0) / float(gpu['total_model_s']):.1f}x"
    else:
        comparison["consistent"] = "Not evaluated"
    return comparison


def fmt_time(result: dict | None, key: str = "total_model_s") -> str:
    if not result:
        return "NA"
    if result.get("status") == "Timeout":
        return "Timeout"
    if not result.get("available"):
        return "Fail"
    value = result.get(key)
    return f"{float(value):.3f}" if isinstance(value, (int, float)) and np.isfinite(value) else "NA"


def render_markdown(rows: list[dict], profile: str) -> str:
    lines = [
        f"# TorchDCM CPU/GPU Device Stress ({profile})",
        "",
        "Rows use pure synthetic mixed-logit cases with identical initialization and antithetic normal draws on CPU and CUDA. CPU workers are capped by the requested timeout; GPU workers use the same TorchDCM model code with `device='cuda'`.",
        "",
        "| case | N | J | K | rho | RC | draws | CPU total model s | GPU total model s | GPU estimate s | GPU memory MB | Result |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        cpu = row["devices"].get("cpu")
        gpu = row["devices"].get("cuda")
        comparison = row["comparison"]
        result = comparison.get("consistent")
        memory = gpu.get("peak_cuda_memory_mb") if gpu else None
        lines.append(
            "| {case} | {N} | {J} | {K} | {rho} | {rc} | {draws} | {cpu_total} | {gpu_total} | {gpu_estimate} | {memory} | {result} |".format(
                case=row["case"],
                N=row["n_obs"],
                J=row["n_alternatives"],
                K=row["n_variables"],
                rho=row["rho"],
                rc=row["random_coefficients"],
                draws=row["n_draws"],
                cpu_total=fmt_time(cpu),
                gpu_total=fmt_time(gpu),
                gpu_estimate=fmt_time(gpu, "estimate_s"),
                memory=f"{float(memory):.0f}" if isinstance(memory, (int, float)) else "NA",
                result=result,
            )
        )
    lines.extend(["", "## Diagnostics", ""])
    for row in rows:
        lines.append(f"- `{row['case']}` comparison: {row['comparison']}")
        for device, result in row["devices"].items():
            lines.append(
                f"  - {device}: status={result.get('status')}, total_model_s={fmt_time(result)}, "
                f"estimate_s={fmt_time(result, 'estimate_s')}, loglike={result.get('loglike')}"
            )
    return "\n".join(lines) + "\n"


def write_outputs(rows: list[dict], profile: str) -> tuple[Path, Path]:
    GENERATED.mkdir(parents=True, exist_ok=True)
    json_path = GENERATED / f"torch_device_stress_{profile}.json"
    md_path = GENERATED / f"torch_device_stress_{profile}.md"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(rows, profile), encoding="utf-8")
    return json_path, md_path


def run_supervisor(args: argparse.Namespace) -> None:
    rows: list[dict] = []
    for spec in profile_specs(args.profile):
        if args.case and spec.case != args.case:
            continue
        print(
            f"[device-stress] case={spec.case} N={spec.n_obs} J={spec.n_alternatives} K={spec.n_variables} "
            f"RC={spec.random_coefficients} draws={spec.n_draws}",
            flush=True,
        )
        devices = {
            "cpu": run_child(spec, args, "cpu", args.cpu_timeout),
            "cuda": run_child(spec, args, "cuda", args.gpu_timeout),
        }
        row = {
            "case": spec.case,
            "model": "Mixed logit",
            "n_obs": spec.n_obs,
            "n_alternatives": spec.n_alternatives,
            "n_variables": spec.n_variables,
            "rho": spec.rho,
            "random_coefficients": spec.random_coefficients,
            "n_draws": spec.n_draws,
            "max_iter": spec.max_iter,
            "devices": devices,
            "comparison": compare_cpu_gpu(devices),
            "timeout_policy": {"cpu_timeout_s": args.cpu_timeout, "gpu_timeout_s": args.gpu_timeout},
        }
        rows.append(row)
        write_outputs(rows, args.profile)
        print(
            f"[device-stress] {spec.case}: cpu={fmt_time(devices['cpu'])} "
            f"gpu={fmt_time(devices['cuda'])} result={row['comparison'].get('consistent')}",
            flush=True,
        )
    json_path, md_path = write_outputs(rows, args.profile)
    print(f"json: {json_path}")
    print(f"markdown: {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["smoke", "calibration", "stress"], default="stress")
    parser.add_argument("--case")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--cpu-timeout", type=int, default=300)
    parser.add_argument("--gpu-timeout", type=int, default=900)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--worker-device", choices=["cpu", "cuda"])
    parser.add_argument("--n-obs", type=int)
    parser.add_argument("--n-alternatives", type=int)
    parser.add_argument("--n-variables", type=int)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--random-coefficients", type=int)
    parser.add_argument("--n-draws", type=int)
    parser.add_argument("--max-iter", type=int)
    if "--worker-device" in sys.argv:
        args = parser.parse_args()
        required = ["case", "n_obs", "n_alternatives", "n_variables", "rho", "random_coefficients", "n_draws", "max_iter"]
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise ValueError(f"Missing worker arguments: {missing}")
        run_worker(args)
        return
    args = parser.parse_args()
    run_supervisor(args)


if __name__ == "__main__":
    main()
