import argparse
import numpy as np
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from core.utils import *
from core.eval.eval_utils import *
from core.eval.visualization import *
from core.eval.across_exp import *

# from plot_utils import *

METRIC_SPECS = [
    ("avg_saregret_01", "SARegret@0.1", True),
    ("avg_saregret_all", "SARegret@All", True),
    ("avg_interval_width_aa", "IntervalWidth", True),
    ("calibration_score_hc", "CalibrationScore", True),
]

DATASET_DISPLAY_NAMES = {
    "elec": "electricity",
    "hosp": "flu hospitalization",
}

def paired_improvement(base_v, plus_v, lower_is_better, as_percent):
    if np.isnan(base_v) or np.isnan(plus_v):
        return np.nan
    delta = (base_v - plus_v) if lower_is_better else (plus_v - base_v)
    if not as_percent:
        return delta
    if np.isclose(base_v, 0.0):
        return np.nan
    return delta / abs(base_v) * 100.0


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 16,
    })

    # Parse arguments (if needed)
    parser = argparse.ArgumentParser(description="Evaluate ACI results.")
    parser.add_argument(
        '--configs', '-c',
        type=str,
        nargs='+',
        default=['0'],
        help='One or more configuration IDs for the experiment'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default='results/kdd_plots',
        help='Directory to save generated plots'
    )
    args = parser.parse_args()
    configs = args.configs
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cov_frames = []
    summary_frames = []
    
    # Merge all the csv files into one DataFrame
    for config in configs:
        csv_path = Path("results") / f"eval_{config}" / "csv"
        current_cov_90_df = pd.read_csv(csv_path / "cov90.csv")
        current_cov_90_df["config"] = str(config)
        current_summary_metrics_df = pd.read_csv(csv_path / "summary_metrics.csv")
        cov_frames.append(current_cov_90_df)
        summary_frames.append(current_summary_metrics_df)

    cov_90_df = pd.concat(cov_frames, ignore_index=True)
    summary_metrics_df = pd.concat(summary_frames, ignore_index=True)

    # Last 3 columns from the original cov file (not helper columns like config/dataset)
    y_cols = list(cov_frames[0].columns[-4:-1])
    datasets = sorted(cov_90_df["dataset"].dropna().unique())
    n_datasets = len(datasets)

    for y_col in y_cols:
        panel_w = 3.9
        panel_h = 5.6
        fig, axes = plt.subplots(
            1, n_datasets,
            figsize=(max(8.0, panel_w * n_datasets), panel_h),
            sharex=False
        )
        axes = np.atleast_1d(axes).ravel()
        all_methods = sorted(cov_90_df["method"].dropna().unique())
        pair_roots = sorted(
            m for m in all_methods
            if not str(m).endswith("+") and (str(m) + "+") in all_methods
        )
        color_map = {root: plt.cm.tab10(i % 10) for i, root in enumerate(pair_roots)}

        for ds_i, (ax, dataset) in enumerate(zip(axes, datasets)):
            sub = cov_90_df[cov_90_df["dataset"] == dataset].copy()
            methods = set(sub["method"].dropna().unique())
            valid_roots = [root for root in pair_roots if root in methods and (root + "+") in methods]
            for root in valid_roots:
                base = root
                plus = root + "+"
                color = color_map[root]

                cur_base = sub[sub["method"] == base].sort_values("horizon")
                ax.plot(
                    cur_base["horizon"], cur_base[y_col],
                    linewidth=2.0, linestyle="--", alpha=0.45, color=color, label=base
                )

                cur_plus = sub[sub["method"] == plus].sort_values("horizon")
                ax.plot(
                    cur_plus["horizon"], cur_plus[y_col],
                    linewidth=2.2, linestyle="-", alpha=1.0, color=color, label=plus
                )
            ax.set_title(DATASET_DISPLAY_NAMES.get(dataset, dataset), pad=8)
            if ds_i == 0:
                ax.set_ylabel(y_col.replace("_", " "))
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)
            ax.set_xlabel("horizon")
            ax.grid(alpha=0.22, linewidth=0.8)
            ax.tick_params(axis="both", which="major", length=3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        legend_handles = []
        for root in pair_roots:
            color = color_map[root]
            legend_handles.append(Line2D([0], [0], color=color, lw=2.0, ls="--", alpha=0.45, label=root))
            legend_handles.append(Line2D([0], [0], color=color, lw=2.2, ls="-", alpha=1.0, label=f"{root}+"))

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=min(len(legend_handles), 8),
            frameon=False,
            handlelength=2.8,
            columnspacing=1.2,
        )
        fig.subplots_adjust(left=0.08, right=0.995, top=0.90, bottom=0.20, wspace=0.10)
        save_path = outdir / f"horizon_vs_{y_col}.pdf"
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {save_path}")

    # Relative improvements of + methods over corresponding baselines.
    summary_agg = (
        summary_metrics_df
        .groupby(["dataset", "method"], as_index=False)
        .mean(numeric_only=True)
    )
    all_summary_methods = sorted(summary_agg["method"].dropna().unique())
    pair_roots = sorted(
        m for m in all_summary_methods
        if not str(m).endswith("+") and (str(m) + "+") in all_summary_methods
    )

    if pair_roots:
        metric_specs = [m for m in METRIC_SPECS if m[0] in summary_agg.columns]
        metric_colors = {m[1]: plt.cm.Set2(i % 8) for i, m in enumerate(metric_specs)}
        width = 0.18 if len(metric_specs) >= 4 else 0.22
        panel_w = 3.9
        panel_h = 5.6

        plot_modes = [
            {
                "as_percent": True,
                "ylabel": "Relative improvement (%)",
                "title": "Relative improvement of + methods over baselines by dataset",
                "filename": "relative_improvement_bar_pct.png",
            },
            {
                "as_percent": False,
                "ylabel": "Improvement (actual value)",
                "title": "Improvement of + methods over baselines by dataset",
                "filename": "relative_improvement_bar_abs.png",
            },
        ]

        for mode in plot_modes:
            fig, axes = plt.subplots(
                1, n_datasets,
                figsize=(max(8.0, panel_w * n_datasets), panel_h),
                sharey=False
            )
            axes = np.atleast_1d(axes).ravel()

            for ds_i, (ax, dataset) in enumerate(zip(axes, datasets)):
                sub = summary_agg[summary_agg["dataset"] == dataset].copy()
                methods = set(sub["method"].dropna().unique())
                valid_roots = [root for root in pair_roots if root in methods and (root + "+") in methods]
                y = np.arange(len(valid_roots))

                for m_i, (metric_col, metric_label, lower_is_better) in enumerate(metric_specs):
                    vals = []
                    for root in valid_roots:
                        base_row = sub[sub["method"] == root]
                        plus_row = sub[sub["method"] == f"{root}+"]
                        if base_row.empty or plus_row.empty:
                            vals.append(np.nan)
                            continue
                        base_v = float(base_row.iloc[0][metric_col]) if pd.notna(base_row.iloc[0][metric_col]) else np.nan
                        plus_v = float(plus_row.iloc[0][metric_col]) if pd.notna(plus_row.iloc[0][metric_col]) else np.nan
                        vals.append(
                            paired_improvement(
                                base_v,
                                plus_v,
                                lower_is_better=lower_is_better,
                                as_percent=mode["as_percent"],
                            )
                        )

                    offset = (m_i - (len(metric_specs) - 1) / 2.0) * width
                    ax.barh(
                        y + offset,
                        vals,
                        height=width,
                        color=metric_colors[metric_label],
                        alpha=0.92,
                        label=metric_label
                    )

                ax.axvline(0.0, color="0.3", linewidth=1.0, alpha=0.8)
                ax.set_title(DATASET_DISPLAY_NAMES.get(dataset, dataset), pad=8)
                ax.set_xlabel(mode["ylabel"])
                if ds_i == 0:
                    ax.set_ylabel("Method pair")
                else:
                    ax.set_ylabel("")
                if len(valid_roots) > 0:
                    ax.set_yticks(y)
                    if ds_i == 0:
                        ax.set_yticklabels(valid_roots)
                    else:
                        ax.tick_params(axis="y", labelleft=False)
                else:
                    ax.set_yticks([])
                ax.grid(axis="x", alpha=0.22, linewidth=0.8)
                ax.tick_params(axis="both", which="major", length=3)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

            legend_handles = [
                Line2D([0], [0], color=metric_colors[m[1]], lw=6, alpha=0.92, label=m[1])
                for m in metric_specs
            ]
            fig.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.03),
                ncol=min(len(legend_handles), 4),
                frameon=False,
                handlelength=2.0,
                columnspacing=1.2,
            )
            fig.subplots_adjust(left=0.08, right=0.995, top=0.90, bottom=0.20, wspace=0.10)
            save_path = outdir / mode["filename"].replace(".png", ".pdf")
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved plot: {save_path}")
