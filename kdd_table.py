import argparse
import math
from pathlib import Path

import pandas as pd


PLUS_METHODS = ["ACI+", "DtACI+", "CPID+"]
BASELINE_METHODS = ["ACI", "DtACI", "CPID"]
OTHER_METHOD_ORDER = ["ACMCP", "CFRNN"]
PLUS_BASELINE_PAIRS = [("ACI+", "ACI"), ("DtACI+", "DtACI"), ("CPID+", "CPID")]
METHOD_DISPLAY_NAMES = {
    "CopulaCP": "CF-RNN",
}

METRICS = [
    ("SARegret@0.1", "avg_saregret_01", "avg_saregret_01_std", True),
    ("SARegret@All", "avg_saregret_all", "avg_saregret_all_std", True),
    ("IntervalWidth", "avg_interval_width_aa", "avg_interval_width_aa_std", True),
    ("CalibrationScore", "calibration_score_hc", "calibration_score_hc_std", True),
]


def fmt_num(x: float) -> str:
    x = float(x)
    ax = abs(x)
    if ax >= 1000:
        return f"{x:.0f}"
    if ax >= 100:
        return f"{x:.1f}"
    if ax >= 1:
        return f"{x:.3f}"
    return f"{x:.4f}"


def rank_labels(vals_by_method: dict, lower_is_better: bool) -> tuple[str | None, str | None]:
    valid = [(m, v[0]) for m, v in vals_by_method.items() if v[0] is not None and not math.isnan(v[0])]
    if not valid:
        return None, None
    valid.sort(key=lambda x: x[1], reverse=not lower_is_better)
    best = valid[0][0]
    second = valid[1][0] if len(valid) > 1 else None
    return best, second


def build_latex_tabular(df: pd.DataFrame, show_std: bool = True) -> str:
    methods_present = sorted(df["method"].unique().tolist())
    plus_methods = [m for m in PLUS_METHODS if m in methods_present]
    baseline_methods = [m for m in BASELINE_METHODS if m in methods_present]
    other_methods = [m for m in OTHER_METHOD_ORDER if m in methods_present]
    other_methods += sorted([m for m in methods_present if m not in plus_methods + baseline_methods + other_methods])
    methods = plus_methods + baseline_methods + other_methods
    if methods:
        methods = methods[:-1]

    datasets = sorted(df["dataset"].unique().tolist())

    lines = []
    col_spec = "l|l|"
    if plus_methods:
        col_spec += "c" * len(plus_methods)
        if baseline_methods or other_methods:
            col_spec += "|"
    col_spec += "c" * len(baseline_methods)
    if baseline_methods and other_methods:
        col_spec += "|"
    col_spec += "c" * len(other_methods)

    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    method_headers = [METHOD_DISPLAY_NAMES.get(m, m) for m in methods]
    lines.append("Dataset & Metric & " + " & ".join(method_headers) + " \\\\")
    lines.append("\\midrule")

    for d_i, dataset in enumerate(datasets):
        dsub = df[df["dataset"] == dataset]
        for i, (metric_label, mean_col, std_col, lower_is_better) in enumerate(METRICS):
            vals_by_method = {}
            for method in methods:
                msub = dsub[dsub["method"] == method]
                if msub.empty:
                    vals_by_method[method] = (None, None)
                    continue
                row = msub.iloc[0]
                mean_v = float(row[mean_col]) if pd.notna(row[mean_col]) else None
                std_v = float(row[std_col]) if std_col in row and pd.notna(row[std_col]) else None
                vals_by_method[method] = (mean_v, std_v)

            best, second = rank_labels(vals_by_method, lower_is_better)
            green_plus_methods = set()
            for plus_method, baseline_method in PLUS_BASELINE_PAIRS:
                if plus_method not in methods or baseline_method not in methods:
                    continue
                plus_mean = vals_by_method[plus_method][0]
                baseline_mean = vals_by_method[baseline_method][0]
                if plus_mean is None or baseline_mean is None:
                    continue
                is_better = plus_mean < baseline_mean if lower_is_better else plus_mean > baseline_mean
                if is_better:
                    green_plus_methods.add(plus_method)

            cells = []
            for method in methods:
                mean_v, std_v = vals_by_method[method]
                if mean_v is None:
                    cells.append("--")
                    continue
                mean_s = fmt_num(mean_v)
                if (not show_std) or std_v is None:
                    core = mean_s
                else:
                    core = f"{mean_s}\\pm{fmt_num(std_v)}"

                styled = core
                if method == best:
                    styled = f"\\mathbf{{{styled}}}"
                elif method == second:
                    styled = f"\\underline{{{styled}}}"
                marker = ""
                if method in green_plus_methods:
                    # Keep main value black for bold/underline readability; encode wins via tint + marker.
                    marker = "^{\\textcolor{green!35!black}{\\scriptstyle\\uparrow}}"
                    cell_text = f"\\cellcolor{{green!10}}${styled}{marker}$"
                else:
                    cell_text = f"${styled}$"
                cells.append(cell_text)

            left = f"\\multirow{{{len(METRICS)}}}{{*}}{{{dataset}}}" if i == 0 else ""
            lines.append(f"{left} & {metric_label} & " + " & ".join(cells) + " \\\\")

        if d_i < len(datasets) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def build_latex_table_block(tabular_code: str) -> str:
    lines = []
    lines.append("% Requires: \\usepackage{booktabs}, \\usepackage{multirow}, and \\usepackage[table]{xcolor}")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Performance comparison across datasets and metrics. Best is bold; second-best is underlined. "
        "Green-tinted cells with an up-arrow indicate a + method outperforms its paired baseline.}"
    )
    lines.append("\\label{tab:kdd_main_results}")
    lines.append(tabular_code)
    lines.append("\\end{table*}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate KDD LaTeX table from summary_metrics.csv files.")
    parser.add_argument(
        "--configs",
        "-c",
        nargs="+",
        required=True,
        help="Config IDs, e.g. -c 34 35 36",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results/kdd_table/kdd_table.txt",
        help="Output txt file path for LaTeX table code.",
    )
    parser.add_argument(
        "--mean-only",
        action="store_true",
        help="Show only mean values in cells (omit ± std).",
    )
    args = parser.parse_args()

    frames = []
    for config in args.configs:
        csv_path = Path("results") / f"eval_{config}" / "csv" / "summary_metrics.csv"
        frames.append(pd.read_csv(csv_path))

    df = pd.concat(frames, ignore_index=True)
    tabular_code = build_latex_tabular(df, show_std=not args.mean_only)
    table_code = build_latex_table_block(tabular_code)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table_code)
    print(f"Saved LaTeX table to {output_path}")


if __name__ == "__main__":
    main()
