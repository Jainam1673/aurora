"""AURORA One-Click Scientific Reproducibility Pipeline.

Orchestrates complete reproduction of:
1. Native C++23 High-Precision Systems Benchmarking Engine.
2. Python 3.14 Systems Flamegraph Profiling & Amdahl Decomposition.
3. Cross-Language Empirical Speedup and Parity Comparison.
4. Publication LaTeX Tables Compilation (Ablations, Native Systems, Cross-Language).
5. Publication Vector Figures Compilation (Performance Profiles, Execution Breakdown).
6. Comprehensive SHA-256 Checksum Manifest and Artifact Verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
PAPER_DIR = REPO_ROOT / "paper"
BUILD_RELEASE_DIR = REPO_ROOT / "build" / "release"


@dataclass
class StageResult:
    name: str
    status: str
    duration_sec: float
    details: dict[str, Any]


def get_git_commit() -> str:
    """Retrieve current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_system_environment() -> dict[str, Any]:
    """Capture host architecture and compiler environment."""
    env_info: dict[str, Any] = {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "git_commit": get_git_commit(),
    }

    # GCC version
    try:
        gcc_res = subprocess.run(["gcc", "--version"], capture_output=True, text=True)
        if gcc_res.returncode == 0:
            env_info["gcc_version"] = gcc_res.stdout.splitlines()[0]
    except Exception:
        env_info["gcc_version"] = "unavailable"

    # Clang version
    try:
        clang_res = subprocess.run(["clang", "--version"], capture_output=True, text=True)
        if clang_res.returncode == 0:
            env_info["clang_version"] = clang_res.stdout.splitlines()[0]
    except Exception:
        env_info["clang_version"] = "unavailable"

    return env_info


def run_command(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    """Run a subprocess command and return exit code, stdout, stderr."""
    proc = subprocess.run(
        cmd,
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def stage_cpp_benchmark(quick: bool = False) -> StageResult:
    """Stage 1: Execute Native C++23 Throughput Benchmark."""
    t0 = time.perf_counter()
    bin_path = BUILD_RELEASE_DIR / "aurora_benchmark_throughput"

    if not bin_path.exists():
        print(f"Building release target: aurora_benchmark_throughput at {BUILD_RELEASE_DIR}...")
        code, out, err = run_command(
            ["cmake", "--build", str(BUILD_RELEASE_DIR), "--target", "aurora_benchmark_throughput"]
        )
        if code != 0:
            return StageResult(
                name="cpp_benchmark",
                status="FAILED",
                duration_sec=time.perf_counter() - t0,
                details={"error": f"Build failed: {err}"},
            )

    output_json = RESULTS_DIR / "cpp_benchmark_results.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Running C++23 native systems benchmark...")
    cmd = [str(bin_path), "--json", str(output_json)]
    code, out, err = run_command(cmd)

    duration = time.perf_counter() - t0
    if code != 0:
        return StageResult(
            name="cpp_benchmark",
            status="FAILED",
            duration_sec=duration,
            details={"error": err, "stdout": out},
        )

    return StageResult(
        name="cpp_benchmark",
        status="SUCCESS",
        duration_sec=duration,
        details={"output_json": str(output_json), "exists": output_json.exists()},
    )


def stage_profile_python(quick: bool = False) -> StageResult:
    """Stage 2: Execute Python Systems Profiler & Flamegraph Tracing."""
    t0 = time.perf_counter()
    output_json = RESULTS_DIR / "python_systems_profile.json"
    prof_file = RESULTS_DIR / "aurora_profile.prof"

    env_steps = 30 if quick else 60
    cmd = [
        sys.executable,
        "benchmarks/profile_aurora.py",
        "--env-steps",
        str(env_steps),
        "--output-json",
        str(output_json),
        "--prof-file",
        str(prof_file),
    ]

    print(f"Running Python systems profiler ({env_steps} env steps)...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)

    duration = time.perf_counter() - t0
    if proc.returncode != 0:
        return StageResult(
            name="profile_python",
            status="FAILED",
            duration_sec=duration,
            details={"error": proc.stderr, "stdout": proc.stdout},
        )

    return StageResult(
        name="profile_python",
        status="SUCCESS",
        duration_sec=duration,
        details={
            "output_json": str(output_json),
            "prof_file": str(prof_file),
            "env_steps": env_steps,
        },
    )


def stage_cross_language() -> StageResult:
    """Stage 3: Execute Cross-Language Benchmark."""
    t0 = time.perf_counter()
    cpp_json = RESULTS_DIR / "cpp_benchmark_results.json"
    output_json = RESULTS_DIR / "cross_language_comparison.json"

    cmd = [
        sys.executable,
        "benchmarks/benchmark_cross_language.py",
        "--cpp-json",
        str(cpp_json),
        "--output-json",
        str(output_json),
    ]

    print("Running cross-language parity & speedup benchmark...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)

    duration = time.perf_counter() - t0
    if proc.returncode != 0:
        return StageResult(
            name="cross_language",
            status="FAILED",
            duration_sec=duration,
            details={"error": proc.stderr, "stdout": proc.stdout},
        )

    return StageResult(
        name="cross_language",
        status="SUCCESS",
        duration_sec=duration,
        details={"output_json": str(output_json)},
    )


def stage_figures_and_tables() -> StageResult:
    """Stage 4: Generate Publication Figures and LaTeX Tables."""
    t0 = time.perf_counter()
    script = PAPER_DIR / "generate_figures_and_tables.py"

    print("Compiling publication tables and figures...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, str(script)], cwd=REPO_ROOT, env=env, capture_output=True, text=True
    )

    duration = time.perf_counter() - t0
    if proc.returncode != 0:
        return StageResult(
            name="figures_and_tables",
            status="FAILED",
            duration_sec=duration,
            details={"error": proc.stderr, "stdout": proc.stdout},
        )

    return StageResult(
        name="figures_and_tables",
        status="SUCCESS",
        duration_sec=duration,
        details={"stdout": proc.stdout},
    )


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash for a given file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def stage_verify_and_checksums() -> StageResult:
    """Stage 5: Verify all target paper artifacts and compute SHA-256 checksums."""
    t0 = time.perf_counter()

    expected_files = [
        PAPER_DIR / "main.tex",
        PAPER_DIR / "references.bib",
        PAPER_DIR / "table_ablations.tex",
        PAPER_DIR / "table_systems.tex",
        PAPER_DIR / "table_cross_language.tex",
        PAPER_DIR / "fig_performance_profiles.pdf",
        PAPER_DIR / "fig_performance_profiles.png",
        PAPER_DIR / "fig_systems_breakdown.pdf",
        PAPER_DIR / "fig_systems_breakdown.png",
        RESULTS_DIR / "ablation" / "ablation_summary.json",
        RESULTS_DIR / "cpp_benchmark_results.json",
        RESULTS_DIR / "python_systems_profile.json",
        RESULTS_DIR / "cross_language_comparison.json",
    ]

    missing = []
    checksums: dict[str, str] = {}

    for file_path in expected_files:
        if not file_path.exists():
            missing.append(str(file_path.relative_to(REPO_ROOT)))
        else:
            rel = str(file_path.relative_to(REPO_ROOT))
            checksums[rel] = compute_sha256(file_path)

    manifest_path = PAPER_DIR / "manifest_checksums.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "checksums": checksums,
            },
            f,
            indent=2,
        )

    duration = time.perf_counter() - t0
    if missing:
        return StageResult(
            name="verify_and_checksums",
            status="FAILED",
            duration_sec=duration,
            details={"missing_files": missing, "checksum_manifest": str(manifest_path)},
        )

    return StageResult(
        name="verify_and_checksums",
        status="SUCCESS",
        duration_sec=duration,
        details={
            "verified_count": len(checksums),
            "manifest_file": str(manifest_path),
            "checksums": checksums,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AURORA One-Click Scientific Reproducibility Pipeline"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run faster iteration with reduced profiling steps",
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Skip heavy benchmark reruns and regenerate artifacts from existing manifests",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default="results/reproduction_report.json",
        help="Path to write consolidated reproduction report JSON",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("      AURORA SCIENTIFIC REPRODUCIBILITY & ARTIFACT COMPILATION PIPELINE      ")
    print("=" * 80)

    env_info = get_system_environment()
    print(f"Host OS:         {env_info['os']}")
    print(f"Python:          {env_info['python_version']}")
    print(f"GCC:             {env_info.get('gcc_version', 'N/A')}")
    print(f"Clang:           {env_info.get('clang_version', 'N/A')}")
    print(f"Git Commit:      {env_info['git_commit']}")
    print(f"CPU Cores:       {env_info['cpu_count']}")
    print("-" * 80)

    stages_executed: list[StageResult] = []

    if not args.skip_benchmarks:
        # Stage 1: C++ Native Benchmarks
        res_cpp = stage_cpp_benchmark(quick=args.quick)
        stages_executed.append(res_cpp)
        print(f"[{res_cpp.status}] C++ Benchmarks ({res_cpp.duration_sec:.2f}s)")
        if res_cpp.status == "FAILED":
            print(f"Error: {res_cpp.details.get('error')}")

        # Stage 2: Python Profiling
        res_prof = stage_profile_python(quick=args.quick)
        stages_executed.append(res_prof)
        print(f"[{res_prof.status}] Python Profiling ({res_prof.duration_sec:.2f}s)")
        if res_prof.status == "FAILED":
            print(f"Error: {res_prof.details.get('error')}")

        # Stage 3: Cross-Language Parity
        res_cross = stage_cross_language()
        stages_executed.append(res_cross)
        print(f"[{res_cross.status}] Cross-Language Benchmark ({res_cross.duration_sec:.2f}s)")
        if res_cross.status == "FAILED":
            print(f"Error: {res_cross.details.get('error')}")
    else:
        print("[SKIPPED] Native and Python benchmarks skipped via --skip-benchmarks.")

    # Stage 4: Figures and Tables Generation
    res_fig = stage_figures_and_tables()
    stages_executed.append(res_fig)
    print(f"[{res_fig.status}] Publication Figures & Tables ({res_fig.duration_sec:.2f}s)")
    if res_fig.status == "FAILED":
        print(f"Error: {res_fig.details.get('error')}")

    # Stage 5: Verification & Checksums
    res_ver = stage_verify_and_checksums()
    stages_executed.append(res_ver)
    print(f"[{res_ver.status}] Artifact Verification & Checksums ({res_ver.duration_sec:.2f}s)")
    if res_ver.status == "FAILED":
        print(f"Error: Missing files: {res_ver.details.get('missing_files')}")

    # Export Report
    report = {
        "environment": env_info,
        "stages": [asdict(s) for s in stages_executed],
        "all_passed": all(s.status == "SUCCESS" for s in stages_executed),
    }

    report_path = REPO_ROOT / args.report_file
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 80)
    print(f"Consolidated Reproduction Report saved to: {report_path}")
    if report["all_passed"]:
        print("ALL REPRODUCTION STAGES COMPLETED SUCCESSFULLY.")
        print("=" * 80)
        return 0
    else:
        print("WARNING: One or more reproduction stages failed.")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
