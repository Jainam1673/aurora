"""Automated Publication Table & Figure Generation Script for AURORA Paper.

Ingests raw experiment manifests from `results/` and compiles:
1. Publication-ready LaTeX tables (`paper/table_*.tex`)
2. Vector PDF and high-resolution PNG figures (`paper/fig_*.pdf`, `paper/fig_*.png`)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
PAPER_DIR = REPO_ROOT / "paper"


def generate_benchmark_and_ablation_tables() -> None:
    """Generate LaTeX tables for empirical benchmarks and ablation studies."""
    ablation_summary_file = RESULTS_DIR / "ablation" / "ablation_summary.json"
    ablation_manifest_file = RESULTS_DIR / "ablation" / "ablation_manifest.json"

    summaries: dict[str, Any] = {}
    significance: dict[str, Any] = {}

    if ablation_summary_file.exists():
        with open(ablation_summary_file) as f:
            summaries = json.load(f)
    elif ablation_manifest_file.exists():
        with open(ablation_manifest_file) as f:
            data = json.load(f)
            summaries = data.get("summaries", {})
            significance = data.get("significance_vs_full", {})
    else:
        print("Warning: Neither ablation_summary.json nor ablation_manifest.json found.")
        return

    # 1. Ablation Study Table (table_ablations.tex)
    table_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Systematic Component Ablation Study on Continuous Control (Pendulum).} "
        r"Evaluated across multiple seeds with 200 interaction steps each. "
        r"IQM trims the top and bottom 25\% of evaluation returns. "
        r"Bootstrap CIs represent 95\% confidence intervals over $R=2000$ resamples. "
        r"$P(\text{Full} > \text{Ablation})$ denotes the empirical probability of improvement.}",
        r"\label{tab:ablation_results}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Method / Variant} & \textbf{IQM} $\uparrow$ & \textbf{95\% Bootstrap CI} & "
        r"\textbf{Mean} $\pm$ \textbf{Std} & \textbf{Median} & $P(\text{Full} > \text{Var})$ & "
        r"$p$-value \\",
        r"\midrule",
    ]

    order = [
        (["Full AURORA", "AURORA (Full Model)"], "AURORA (Full)"),
        (["No Adaptive Horizon"], "w/o Adaptive Horizon ($H=4$)"),
        (["No Dynamic Blending"], r"w/o Dynamic Blending ($\eta=0.5$)"),
        (["No Pessimism", "No Pessimistic Penalty"], r"w/o Pessimistic Penalty ($\beta=0$)"),
    ]

    for aliases, label in order:
        cond_key = next((k for k in aliases if k in summaries), None)
        if cond_key is None:
            continue
        s = summaries[cond_key]
        iqm = s["iqm"]
        ci_low = s["ci_lower"]
        ci_high = s["ci_upper"]
        mean = s["mean"]
        std = s["std"]
        median = s.get("median", mean)

        if "Full" in cond_key:
            p_imp_str = "---"
            pval_str = "---"
            name_formatted = r"\textbf{" + label + r"}"
        else:
            p_imp = s.get("prob_improvement_over_this")
            if p_imp is None:
                p_imp = significance.get(cond_key, {}).get("probability_of_improvement", 0.5)
            pval = s.get("welch_p_value")
            if pval is None:
                pval = significance.get(cond_key, {}).get("welch_t_test", {}).get("p_value", 1.0)
            p_imp_str = f"{p_imp:.2f}"
            pval_str = f"{pval:.4f}" if pval >= 0.001 else r"$< 0.001$"
            name_formatted = label

        line = (
            f"{name_formatted} & \\textbf{{{iqm:.2f}}} & [{ci_low:.2f}, {ci_high:.2f}] & "
            f"{mean:.2f} $\\pm$ {std:.2f} & {median:.2f} & {p_imp_str} & {pval_str} \\\\"
        )
        table_lines.append(line)

    table_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    out_table = PAPER_DIR / "table_ablations.tex"
    with open(out_table, "w") as f:
        f.write("\n".join(table_lines) + "\n")
    print(f"Generated LaTeX ablation table: {out_table}")


def generate_systems_tables() -> None:
    """Generate LaTeX tables for C++23 native systems throughput and cross-language speedups."""
    cpp_file = RESULTS_DIR / "cpp_benchmark_results.json"
    cross_file = RESULTS_DIR / "cross_language_comparison.json"

    if cpp_file.exists():
        with open(cpp_file) as f:
            cpp_data = json.load(f)

        benchmarks = cpp_data.get("benchmarks", [])
        table_lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{\textbf{Native C++23 High-Throughput Engine Microbenchmarks.} "
            r"Evaluated with GCC 16.2.1 (-O3 -march=native) on an Intel Core i5-8265U "
            r"CPU (AVX2/FMA). Timings report mean, median ($p_{50}$), and 99th-percentile "
            r"($p_{99}$) latencies over $\ge 20$ trials.}",
            r"\label{tab:native_systems_performance}",
            r"\begin{tabular}{llcccc}",
            r"\toprule",
            r"\textbf{Subsystem} & \textbf{Benchmark Workload} & \textbf{Mean} ($\mu$s) & "
            r"$p_{50}$ ($\mu$s) & $p_{99}$ ($\mu$s) & \textbf{Throughput} \\",
            r"\midrule",
        ]

        for b in benchmarks:
            cat = b["category"].replace("&", r"\&")
            name = b["name"].replace("&", r"\&")
            mean_us = b["mean_us"]
            p50_us = b["median_us"]
            p99_us = b["p99_us"]
            tput = b["throughput"]
            unit = b["throughput_unit"]

            if tput >= 1e9:
                tput_str = f"{tput / 1e9:.2f} G"
            elif tput >= 1e6:
                tput_str = f"{tput / 1e6:.2f} M"
            elif tput >= 1e3:
                tput_str = f"{tput / 1e3:.2f} k"
            else:
                tput_str = f"{tput:.1f}"

            tput_full = f"{tput_str} {unit}"
            table_lines.append(
                f"{cat} & {name} & {mean_us:.2f} & {p50_us:.2f} & {p99_us:.2f} & {tput_full} \\\\"
            )

        table_lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        out_table = PAPER_DIR / "table_systems.tex"
        with open(out_table, "w") as f:
            f.write("\n".join(table_lines) + "\n")
        print(f"Generated LaTeX native systems table: {out_table}")

    if cross_file.exists():
        with open(cross_file) as f:
            cross_data = json.load(f)

        comps = cross_data.get("comparisons", [])
        table_lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{\textbf{Cross-Language Systems Throughput Comparison "
            r"(Python 3.14 vs. C++23 Native).} "
            r"Evaluated across identical computational tasks on the same host hardware.}",
            r"\label{tab:cross_language_comparison}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"\textbf{Computational Workload} & \textbf{Unit} & "
            r"\textbf{Python Mean} ($\mu$s) & \textbf{C++23 Mean} ($\mu$s) & "
            r"\textbf{Speedup} ($S_{\text{C++}} / S_{\text{Py}}$) \\",
            r"\midrule",
        ]

        for c in comps:
            bench = c["benchmark"].replace("&", r"\&")
            unit = c["unit"]
            py_lat = c["python_mean_us"]
            cpp_lat = c["cpp_mean_us"]
            speedup = c["speedup_cpp_vs_python"]

            speedup_str = f"\\textbf{{{speedup:.2f}x}}" if speedup >= 1.0 else f"{speedup:.2f}x"
            table_lines.append(
                f"{bench} & {unit} & {py_lat:.2f} & {cpp_lat:.2f} & {speedup_str} \\\\"
            )

        table_lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        out_table = PAPER_DIR / "table_cross_language.tex"
        with open(out_table, "w") as f:
            f.write("\n".join(table_lines) + "\n")
        print(f"Generated LaTeX cross-language table: {out_table}")


def generate_publication_figures() -> None:
    """Generate vector PDF and PNG figures for the paper manuscript."""
    scores_by_cond: dict[str, list[float]] = {}
    ablation_dir = RESULTS_DIR / "ablation"
    if ablation_dir.exists():
        for manifest_path in ablation_dir.glob("*/manifest.json"):
            try:
                with open(manifest_path) as f:
                    m = json.load(f)
                parent_name = manifest_path.parent.name
                if "pendulum_aurora" in parent_name:
                    friendly = "AURORA (Full Model)"
                elif "no_adaptive_horizon" in parent_name:
                    friendly = "No Adaptive Horizon"
                elif "no_dynamic_blending" in parent_name:
                    friendly = "No Dynamic Blending"
                elif "no_pessimism" in parent_name:
                    friendly = "No Pessimistic Penalty"
                else:
                    friendly = parent_name

                rets = m.get("metrics", {}).get("final_returns", [])
                if friendly not in scores_by_cond:
                    scores_by_cond[friendly] = []
                scores_by_cond[friendly].extend(rets)
            except Exception:
                pass

    if scores_by_cond:
        all_scores = [s for runs in scores_by_cond.values() for s in runs]
        if all_scores:
            min_score = min(all_scores) - 10.0
            max_score = max(all_scores) + 10.0
            thresholds = np.linspace(min_score, max_score, 200)

            fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
            colors = {
                "AURORA (Full Model)": "#1f77b4",
                "No Adaptive Horizon": "#ff7f0e",
                "No Dynamic Blending": "#2ca02c",
                "No Pessimistic Penalty": "#d62728",
            }
            styles = {
                "AURORA (Full Model)": ("-", 2.5),
                "No Adaptive Horizon": ("--", 1.8),
                "No Dynamic Blending": ("-.", 1.8),
                "No Pessimistic Penalty": (":", 1.8),
            }

            for cond_name, scores in scores_by_cond.items():
                profile = [np.mean([1.0 if s >= t else 0.0 for s in scores]) for t in thresholds]
                ls, lw = styles.get(cond_name, ("-", 1.5))
                c = colors.get(cond_name, "#333333")
                ax.plot(thresholds, profile, label=cond_name, linestyle=ls, linewidth=lw, color=c)

            ax.set_title(
                "Empirical Performance Profiles on Continuous Control",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_xlabel(r"Evaluation Return Threshold ($\tau$)", fontsize=11)
            ax.set_ylabel(
                r"Fraction of Runs with Score $\geq \tau$ (Higher is Better)", fontsize=11
            )
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="lower left", framealpha=0.9, fontsize=9.5)
            fig.tight_layout()

            pdf_path = PAPER_DIR / "fig_performance_profiles.pdf"
            png_path = PAPER_DIR / "fig_performance_profiles.png"
            fig.savefig(pdf_path, bbox_inches="tight")
            fig.savefig(png_path, bbox_inches="tight", dpi=300)
            plt.close(fig)
            print(f"Saved performance profile figure: {pdf_path} and {png_path}")

    # 2. Systems Breakdown Figure
    prof_file = RESULTS_DIR / "python_systems_profile.json"
    if prof_file.exists():
        with open(prof_file) as f:
            prof_data = json.load(f)

        comps = prof_data.get("components", {})
        labels = []
        shares = []
        palette = ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2", "#59a14f"]

        name_map = {
            "policy_update": "Policy Optimization",
            "adaptive_imagination": "Adaptive Imagination",
            "env_step": "Environment Step",
            "world_model_train": "Model Training",
            "evaluation": "Evaluation",
        }

        for k, v in comps.items():
            if v["fraction_pct"] > 0:
                labels.append(f"{name_map.get(k, k)} ({v['fraction_pct']:.1f}%)")
                shares.append(v["fraction_pct"])

        if shares:
            fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
            _wedges, _ = ax.pie(
                shares,
                labels=labels,
                colors=palette[: len(shares)],
                startangle=140,
                wedgeprops=dict(width=0.4, edgecolor="w", linewidth=1.5),
            )
            ax.set_title(
                "Amdahl Execution Time Breakdown (Online Training)", fontsize=12, fontweight="bold"
            )
            fig.tight_layout()

            pdf_path = PAPER_DIR / "fig_systems_breakdown.pdf"
            png_path = PAPER_DIR / "fig_systems_breakdown.png"
            fig.savefig(pdf_path, bbox_inches="tight")
            fig.savefig(png_path, bbox_inches="tight", dpi=300)
            plt.close(fig)
            print(f"Saved systems breakdown figure: {pdf_path} and {png_path}")


def main() -> None:
    print("Compiling publication tables and figures from experiment results...")
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    generate_benchmark_and_ablation_tables()
    generate_systems_tables()
    generate_publication_figures()
    print("Figure and table compilation complete.")


if __name__ == "__main__":
    main()
