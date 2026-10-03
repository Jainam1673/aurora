"""Automated scientific visualization: learning curves and performance profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from evaluation.profiles import performance_profile

matplotlib.use("Agg")


def plot_performance_profiles(
    scores_dict: dict[str, list[float]],
    thresholds: list[float] | np.ndarray,
    output_path: str | Path,
    title: str = "Performance Profiles",
) -> None:
    """Plot performance profile curves across evaluation thresholds."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    thresh_arr = np.asarray(thresholds, dtype=np.float64)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, (label, scores) in enumerate(scores_dict.items()):
        prof = performance_profile(scores, thresh_arr)
        color = colors[idx % len(colors)]
        ax.plot(thresh_arr, prof, label=label, color=color, linewidth=2)

    ax.set_xlabel("Evaluation Return Threshold", fontsize=12)
    ax.set_ylabel("Fraction of Runs >= Threshold", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower left", fontsize=10)

    fig.tight_layout()
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    plt.close(fig)


def format_ascii_table(summary_dict: dict[str, dict[str, Any]]) -> str:
    """Format dictionary of statistical summaries into a publication-ready ASCII table."""
    header = (
        f"{'Algorithm / Condition':<25} | {'IQM':<10} | {'Mean ± Std':<18} | "
        f"{'95% Bootstrap CI':<22}"
    )
    separator = "-" * len(header)
    lines = [header, separator]

    for name, stats in summary_dict.items():
        iqm = stats.get("iqm", 0.0)
        mean = stats.get("mean", 0.0)
        std = stats.get("std", 0.0)
        ci_l = stats.get("ci_lower", 0.0)
        ci_u = stats.get("ci_upper", 0.0)

        mean_str = f"{mean:7.2f} ± {std:5.2f}"
        ci_str = f"[{ci_l:7.2f}, {ci_u:7.2f}]"
        lines.append(f"{name:<25} | {iqm:<10.2f} | {mean_str:<18} | {ci_str:<22}")

    lines.append(separator)
    return "\n".join(lines)
