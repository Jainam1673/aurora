"""Systematic component ablation study for the AURORA algorithm."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.metrics import compute_statistical_summary
from evaluation.plotting import plot_performance_profiles
from evaluation.profiles import probability_of_improvement
from evaluation.significance import welch_t_test
from experiments.runner import run_experiment


def run_ablation_suite(
    seeds: list[int] | None = None,
    output_dir: str | Path = "results/ablation",
    verbose: bool = True,
) -> dict[str, Any]:
    """Execute systematic ablation evaluation across multiple seeds."""
    if seeds is None:
        seeds = [42, 101, 202]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    conditions = [
        ("Full AURORA", "configs/pendulum_aurora.json"),
        ("No Adaptive Horizon", "configs/ablation_no_adaptive_horizon.json"),
        ("No Dynamic Blending", "configs/ablation_no_dynamic_blending.json"),
        ("No Pessimism", "configs/ablation_no_pessimism.json"),
    ]

    all_results: dict[str, list[float]] = {}

    for cond_name, cfg_path in conditions:
        if verbose:
            print("\n==========================================")
            print(f"Condition: {cond_name} ({len(seeds)} seeds)")
            print("==========================================")

        cond_returns: list[float] = []
        for s in seeds:
            manifest = run_experiment(
                config=cfg_path,
                seed=s,
                output_dir=out_path,
                verbose=False,
            )
            rets = manifest["metrics"]["final_returns"]
            cond_returns.extend(rets)
            mean_ret = float(np.mean(rets))
            if verbose:
                print(f"  Seed {s:4d} | Mean Final Return: {mean_ret:8.2f}")

        all_results[cond_name] = cond_returns

    # Statistical Analysis
    reference_scores = all_results["Full AURORA"]
    summary_report: dict[str, Any] = {}

    if verbose:
        print("\n" + "=" * 90)
        hdr = (
            f"{'Condition':<25} | {'IQM':<10} | {'95% Bootstrap CI':<22} | "
            f"{'P(AURORA > X)':<14} | {'p-value':<8}"
        )
        print(hdr)
        print("-" * 90)

    for cond_name, rets in all_results.items():
        stat_summary = compute_statistical_summary(rets, seed=42)
        p_improve = probability_of_improvement(reference_scores, rets)
        t_res = welch_t_test(reference_scores, rets)

        summary_report[cond_name] = {
            "iqm": stat_summary.iqm,
            "ci_lower": stat_summary.ci_lower,
            "ci_upper": stat_summary.ci_upper,
            "mean": stat_summary.mean,
            "std": stat_summary.std,
            "prob_improvement_over_this": p_improve,
            "welch_p_value": t_res.p_value,
            "count": len(rets),
        }

        if verbose:
            ci_str = f"[{stat_summary.ci_lower:7.2f}, {stat_summary.ci_upper:7.2f}]"
            p_imp_str = f"{p_improve:6.2%}" if cond_name != "Full AURORA" else "---"
            p_val_str = f"{t_res.p_value:8.4f}" if cond_name != "Full AURORA" else "---"
            row = (
                f"{cond_name:<25} | {stat_summary.iqm:<10.2f} | {ci_str:<22} | "
                f"{p_imp_str:<14} | {p_val_str:<8}"
            )
            print(row)

    if verbose:
        print("=" * 90)

    # Save summary report to JSON
    summary_file = out_path / "ablation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    thresholds = np.linspace(-2000.0, -800.0, 50).tolist()
    plot_performance_profiles(
        scores_dict=all_results,
        thresholds=thresholds,
        output_path=out_path / "performance_profiles.png",
        title="AURORA Component Ablation: Performance Profiles",
    )

    return summary_report


if __name__ == "__main__":
    run_ablation_suite(seeds=[42, 101, 202], verbose=True)
