import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable

from core.utils import *
from core.eval.eval_utils import *
from core.eval.visualization import *
from core.eval.across_exp import *
from core.constants import *

def plot_time_varying_metrics(
    experiment_ids: List[Tuple[str, str, str]],
    display_names: List[str],
    get_metrics: Callable[[Tuple[str, str, str]], Dict],
    metric_name: str,
    theta: None,
    subset_idx: int,
    alpha_idx: int,
    start_t: int,
    end_t: int,
    save_path: Path
):
    for i, eid in enumerate(experiment_ids):
        res = get_metrics(eid)
        subsets = list(res.keys())
        cur_subset = subsets[subset_idx]
        res = res[cur_subset][metric_name]
        if metric_name == 'exponential_risk':
            res = res[theta]
        alphas = list(res.keys())
        all_time_series = []
        for alpha in alphas:
            time_series = res[alpha][start_t:end_t]
            all_time_series.append(time_series)
        mean_time_series = np.mean(all_time_series, axis=0)
        time_series = None
        if alpha_idx < 0:
            time_series = mean_time_series
        else:
            time_series = all_time_series[alpha_idx]
        idxes = np.arange(len(time_series))
        plt.plot(idxes, time_series, label=display_names[i])
    plt.xlabel("Time Step")
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / f'{metric_name}_subset{subset_idx}_alpha{alpha_idx}.png')
    plt.close()


def bar_plots_for_avg_metrics(experiment_ids, display_names, save_path):
    # Plot averaged saregret
    vals = []
    for eid in experiment_ids:
        res = get_metrics(eid)
        saregret_vals = res['saregret']
        current_vals = []
        for alpha in saregret_vals:
            alpha_vals = saregret_vals[alpha]
            alpha_vals = [alpha_vals[t] for t in alpha_vals]
            current_vals.append(np.mean(alpha_vals))
        vals.append(np.mean(current_vals))
    plt.bar(display_names, vals)
    plt.ylabel("Avg Saregret")
    plt.xticks(rotation=45, ha='right')
    # plt.ylim(y_lim)
    plt.tight_layout()
    plt.savefig(save_path / "avg_saregret.png")
    plt.close()
    
    # Plot averaged 0.1 saregret
    vals = []
    target_alpha = 0.1
    for eid in experiment_ids:
        res = get_metrics(eid)
        saregret_vals = res['saregret'][target_alpha]
        saregret_vals = [saregret_vals[t] for t in saregret_vals]
        vals.append(np.mean(saregret_vals))
    plt.bar(display_names, vals)
    plt.ylabel("Avg Saregret (alpha={})".format(target_alpha))
    plt.xticks(rotation=45, ha='right')
    # plt.ylim(y_lim)
    plt.tight_layout()
    plt.savefig(save_path / "avg_saregret_{}.png".format(target_alpha))
    plt.close()
    
    # Plot exponential risk
    for t in range(len(thetas_c)):
        vals = []
        thresholds = None
        for eid in experiment_ids:
            res = get_metrics(eid)
            exp_risk = res['exponential_risk']
            cur_vals = []
            for alpha in exp_risk:
                if thresholds is None:
                    thresholds = list(exp_risk[alpha].keys())
                    print('Thresholds:', thresholds)
                cur_vals.append(exp_risk[alpha][thresholds[t]])
            vals.append(np.mean(cur_vals))
        # second highest
        sorted_vals = sorted(vals)
        # y_lim = (sorted_vals[0]*0.9, sorted_vals[-2]*1.1)
        plt.bar(display_names, vals)
        plt.ylabel("Avg Exponential Risk (theta={})".format(thresholds[t]))
        plt.xticks(rotation=45, ha='right')
        # plt.ylim(y_lim)
        plt.tight_layout()
        plt.savefig(save_path / "avg_exponential_risk_{}.png".format(thresholds[t]))
        plt.close()
    
    # Plot average absolute risk across methods
    vals = []
    for eid in experiment_ids:
        res = get_metrics(eid)
        exp_risk = res['avg_risk_nrw']
        cur_vals = []
        for alpha in exp_risk:
            cur_vals.append(exp_risk[alpha])
        vals.append(np.mean(cur_vals))
    # second highest
    sorted_vals = sorted(vals)
    # y_lim = (sorted_vals[0]*0.9, sorted_vals[-2]*1.1)
    plt.bar(display_names, vals)
    plt.ylabel("Avg Risk")
    plt.xticks(rotation=45, ha='right')
    # plt.ylim(y_lim)
    plt.tight_layout()
    plt.savefig(save_path / "avg_risk.png")
    plt.close()
    
    # Plot number of violations
    for t in range(len(thresholds_c)):
        vals = []
        thresholds = None
        for eid in experiment_ids:
            res = get_metrics(eid)
            exp_risk = res['num_violations_nrw']
            cur_vals = []
            for alpha in exp_risk:
                if thresholds is None:
                    thresholds = list(exp_risk[alpha].keys())
                cur_vals.append(exp_risk[alpha][thresholds[t]])
            vals.append(np.mean(cur_vals))
        # second highest
        sorted_vals = sorted(vals)
        # y_lim = (sorted_vals[0]*0.9, sorted_vals[-2]*1.1)
        plt.bar(display_names, vals)
        plt.ylabel("Avg Number of Violations (threshold={})".format(thresholds[t]))
        plt.xticks(rotation=45, ha='right')
        # plt.ylim(y_lim)
        plt.tight_layout()
        plt.savefig(save_path / "avg_number_of_violations_{}.png".format(thresholds[t]))
        plt.close()
    
    # Plot exponential risk for NRW
    for t in range(len(thetas_c)):
        vals = []
        thresholds = None
        for eid in experiment_ids:
            res = get_metrics(eid)
            exp_risk = res['exponential_risk_nrw']
            cur_vals = []
            for alpha in exp_risk:
                if thresholds is None:
                    thresholds = list(exp_risk[alpha].keys())
                cur_vals.append(exp_risk[alpha][thresholds[t]])
            vals.append(np.mean(cur_vals))
        # second highest
        sorted_vals = sorted(vals)
        # y_lim = (sorted_vals[0]*0.9, sorted_vals[-2]*1.1)
        plt.bar(display_names, vals)
        plt.ylabel("Avg Exponential Risk NRW (theta={})".format(thresholds[t]))
        plt.xticks(rotation=45, ha='right')
        # plt.ylim(y_lim)
        plt.tight_layout()
        plt.savefig(save_path / "avg_exponential_risk_nrw_{}.png".format(thresholds[t]))
        plt.close()
    
    # Plot avg monotonicity loss
    vals = []
    for eid in experiment_ids:
        res = get_metrics(eid)
        exp_risk = res['monotonicity_score']
        cur_vals = []
        for k, v in exp_risk.items():
            cur_vals.append(v)
        vals.append(np.mean(cur_vals))
    # second highest
    sorted_vals = sorted(vals)
    # y_lim = (sorted_vals[0]*0.9, sorted_vals[-2]*1.1)
    plt.bar(display_names, vals)
    plt.ylabel("Avg Monotonicity Loss")
    plt.xticks(rotation=45, ha='right')
    # plt.ylim(y_lim)
    plt.tight_layout()
    plt.savefig(save_path / "avg_monotonicity_loss.png")
    plt.close()
    
    vals = []
    for eid in experiment_ids:
        res = get_metrics(eid)
        dcr = res['dcr']
        cur_vals = []
        for k, v in dcr.items():
            cur_vals.append(v)
        vals.append(np.mean(cur_vals))
    # second highest
    # sorted_vals = sorted(vals)
    # y_lim = (sorted_vals[0]*0.9, sorted_vals[-2]*1.1)
    plt.bar(display_names, vals)
    plt.ylabel("Avg Distribution Consistent Ratio")
    plt.xticks(rotation=45, ha='right')
    # plt.ylim(y_lim)
    plt.tight_layout()
    plt.savefig(save_path / "avg_distribution_consistent_ratio.png")
    plt.close()

def line_plot_for_metrics(experiment_ids, display_names, save_path):
    # Plot 0.1 saregret for each horizon
    target_alpha = 0.1
    for eid in experiment_ids:
        res = get_metrics(eid)
        saregret_vals = res['saregret'][target_alpha]
        x_ticks = list(saregret_vals.keys())
        saregret_vals = [saregret_vals[t] for t in saregret_vals]
        plt.plot(x_ticks, saregret_vals, label=display_names[experiment_ids.index(eid)])
    plt.legend()
    plt.xlabel("Horizon")
    plt.ylabel("Avg Saregret (alpha={})".format(target_alpha))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path / "saregret_{}.png".format(target_alpha))
    plt.close()
    
    # Plot saregret for each horizon averaged across alphas
    for eid in experiment_ids:
        res = get_metrics(eid)
        saregret_dict = res['saregret']
        vals2plot = []
        alphas = list(saregret_dict.keys())
        for h in saregret_dict[alphas[0]]:
            h_vals = []
            for alpha in alphas:
                h_vals.append(saregret_dict[alpha][h])
            vals2plot.append(np.mean(h_vals))
        x_ticks = list(saregret_dict[alphas[0]].keys())
        saregret_vals = vals2plot
        plt.plot(x_ticks, saregret_vals, label=display_names[experiment_ids.index(eid)])
    plt.legend()
    plt.xlabel("Horizon")
    plt.ylabel("Avg Saregret (averaged over alphas)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path / "saregret_averaged_over_alphas.png")
    plt.close()

def comparison_line_plot(experiment_ids, display_names, save_path, percentage=False, average=False):
    # Plot 0.1 saregret percentage improvement for each horizon (Compare with baselines)
    target_alpha = 0.1
    records = []
    for eid in experiment_ids:
        res = get_metrics(eid, filename='metrics_all_subsets.pkl')
        subsets = list(res.keys())
        for subset in subsets:
            if average:
                saregret_vals = res[subset]['saregret']
                for h in saregret_vals[list(saregret_vals.keys())[0]]:
                    h_vals = []
                    for alpha in saregret_vals:
                        h_vals.append(saregret_vals[alpha][h])
                    records.append({
                        'display_name': display_names[experiment_ids.index(eid)],
                        'subset': subset,
                        'horizon': h,
                        'saregret': np.mean(h_vals)
                    })
            else:
                saregret_vals = res[subset]['saregret'][target_alpha]
                for h in saregret_vals:
                    records.append({
                        'display_name': display_names[experiment_ids.index(eid)],
                        'subset': subset,
                        'horizon': h,
                        'saregret': saregret_vals[h]
                    })
    df = pd.DataFrame(records)
    for display_name in df['display_name'].unique():
        if display_name[-1] == '+':
            continue
        baseline_col = df[df['display_name'] == display_name]
        improved_col = df[df['display_name'] == display_name + '+']
        merged = pd.merge(
            baseline_col,
            improved_col,
            on=['subset', 'horizon'],
            suffixes=('_base', '_impv')
        )
        merged['percentage_improvement'] = (
            (merged['saregret_base'] - merged['saregret_impv']) if not percentage else
            (merged['saregret_base'] - merged['saregret_impv']) / (abs(merged['saregret_base']) + 1e-8) * 100
        )
        stats = merged.groupby('horizon')['percentage_improvement'].agg(['mean', 'std']).reset_index()
        plt.plot(
            stats['horizon'],
            stats['mean'],
            label=display_name
        )
        # plt.fill_between(
        #     stats['horizon'],
        #     stats['mean'] - stats['std'],
        #     stats['mean'] + stats['std'],
        #     alpha=0.2
        # )
    alpha_str = 'average' if average else target_alpha
    title = f'saregret_improv_{alpha_str}{"_percentage" if percentage else ""}'
    print(title)
    plt.legend()
    plt.xlabel("Horizon")
    plt.ylabel(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path / f"{title}.jpg")
    plt.close()

def retrieve_hcs_percentiles(
    experiment_ids: List[Tuple[str, str, str]],
    get_metrics: Callable[[Tuple[str, str, str]], Dict],
    alphas: List[float],
    percentiles: List[float] = [0.25, 0.5, 0.75]
) -> List[Dict[str, object]]:
    """
    Compute specified percentiles of the horizon coverage score error per experiment.
    Returns a list of dicts with keys:
      - 'experiment': (data_type, method, exp_num)
      - 'method': method name
      - 'percentiles': {p: value for each p in percentiles}
    """
    results = []
    for exp_id in experiment_ids:
        data_type, method, exp_num = exp_id
        metrics = get_metrics(exp_id)

        alpha2diff = []
        for alpha in alphas:
            # collect raw horizon coverage scores
            if data_type in ('lane', 'cyclone', 'markovar', 'flusight'):
                hc_vals = []
                for v in metrics['horizon_coverage_t'].values():
                    hc_vals.extend(v[alpha])
            else:
                hc_vals = metrics['horizon_coverage_t'][alpha]

            hc_vals = np.array(hc_vals)
            diff = np.abs(hc_vals - 1 + alpha)
            alpha2diff.append(diff)

        alpha2diff = np.vstack(alpha2diff)  # shape (len(alphas), steps)
        cs_per_step = np.mean(alpha2diff, axis=0)
        pct_dict = {p: np.quantile(cs_per_step, p) for p in percentiles}

        results.append({
            'data_type': data_type,
            'experiment': exp_id,
            'method': method,
            'percentiles': pct_dict
        })
    return results

def multi_plot_hcs_percentiles(
    all_hcs_percentiles: List[List[Dict[str, object]]],
    x_label: str = 'Dataset',
    y_label: str = 'Horizon Coverage Score',
    dataset_names: Optional[List[str]] = None,
    percentile: float = 0.5,
    display_names: Dict[Tuple[str, str], str] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
    font_size: int = 12
):
    # Build list of per-dataset dicts method->value
    kv_dict_list: List[Dict[str, float]] = []
    for hcs_list in all_hcs_percentiles:
        kv: Dict[str, float] = {}
        for entry in hcs_list:
            exp_id = entry['experiment']
            _, method, exp = exp_id
            if display_names is None:
                name = f"{method}-{exp_id}"
            else:
                name = display_names[(method, exp)]
            value = entry['percentiles'][percentile]
            kv[name] = value
        kv_dict_list.append(kv)

    # Default dataset names
    n = len(kv_dict_list)
    if dataset_names is None:
        dataset_names = [f"Dataset {i+1}" for i in range(n)]
    if len(dataset_names) != n:
        raise ValueError("Length of dataset_names must match number of datasets")

    # Determine all methods
    methods = sorted({m for kv in kv_dict_list for m in kv.keys()})
    m = len(methods)

    # Gather values per method across datasets
    values = []
    for method in methods:
        vals = [kv.get(method, 0.0) for kv in kv_dict_list]
        values.append(vals)

    # Plotting
    plt.rcParams.update({'font.size': font_size})
    x = np.arange(n)
    total_width = 0.8
    bar_width = total_width / m
    offsets = np.linspace(-total_width/2 + bar_width/2,
                           total_width/2 - bar_width/2,
                           m)

    fig, ax = plt.subplots(figsize=figsize)
    for i, (method, vals) in enumerate(zip(methods, values)):
        ax.bar(x + offsets[i], vals, width=bar_width, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(title='Method', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_hcs_percentiles_lines(
    all_hcs_percentiles: List[List[Dict[str, object]]],
    percentile_list: List[float],
    x_label: str = 'Percentile',
    y_label: str = 'Horizon Coverage Score',
    dataset_names: Optional[List[str]] = None,
    display_names: Dict[Tuple[str, str], str] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 4),
    font_size: int = 12
):
    """
    For each dataset, plots one subplot. Within each subplot, draws one line per (method, experiment),
    with markers at the values of the given percentiles.
    Legend is placed inside the last subplot.
    Each subplot auto-scales its own y-axis.
    """
    n = len(all_hcs_percentiles)
    if dataset_names is None:
        dataset_names = [f"Dataset {i+1}" for i in range(n)]
    if len(dataset_names) != n:
        raise ValueError("dataset_names must match number of datasets")

    plt.rcParams.update({'font.size': font_size})
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes[0]

    # store last-ax handles for legend
    last_handles, last_labels = [], []

    for i, hcs_list in enumerate(all_hcs_percentiles):
        ax = axes[i]
        for entry in hcs_list:
            exp_id = entry['experiment']
            _, method, exp = exp_id
            # fallback if display_names missing key
            if display_names and (method, exp) in display_names:
                label = display_names[(method, exp)]
            else:
                label = f"{method}-{exp_id}"
            vals = [entry['percentiles'][p] for p in percentile_list]
            line, = ax.plot(percentile_list, vals, marker='o', linewidth=2, label=label)
            # collect for legend only from last subplot
            if i == n-1:
                last_handles.append(line)
                last_labels.append(label)

        ax.set_title(dataset_names[i], fontsize=font_size)
        ax.set_xlabel(x_label)
        if i == 0:
            ax.set_ylabel(y_label)
        ax.set_xticks(percentile_list)
        ax.grid(False)
        ax.tick_params(axis='both', labelrotation=45)

    # draw legend inside last subplot
    ax_last = axes[-1]
    ax_last = axes[-1]
    ax_last.legend(
        loc='upper left',
        frameon=True,
        borderpad=0.3,
        labelspacing=0.2,
        handletextpad=0.4,
        handlelength=1.0,
        columnspacing=0.5
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    

def multi_save_2d_to_csv(
    x_data_list: List[Dict[str, Dict[str, List[float]]]],
    y_data_list: List[Dict[str, Dict[str, List[float]]]],
    x_std_err_list: Optional[List[Dict[str, Dict[str, List[float]]]]] = None,
    y_std_err_list: Optional[List[Dict[str, Dict[str, List[float]]]]] = None,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Collects all x/y (and optional x_err/y_err) values across four 2D scatter 'plots'
    into a single DataFrame and saves it to CSV if save_path is provided.

    Returns
    -------
    pd.DataFrame
        Columns: ['plot_idx', 'dataset', 'method', 'x', 'y', 'x_err', 'y_err']
    """
    n_plots = len(x_data_list)
    if n_plots != 4 or len(y_data_list) != 4:
        raise ValueError("This function expects exactly 4 entries in x_data_list and y_data_list.")

    if x_std_err_list is None:
        x_std_err_list = [None] * n_plots
    if y_std_err_list is None:
        y_std_err_list = [None] * n_plots

    rows = []
    for plot_idx in range(n_plots):
        x_data = x_data_list[plot_idx]
        y_data = y_data_list[plot_idx]
        x_err_data = x_std_err_list[plot_idx] or {}
        y_err_data = y_std_err_list[plot_idx] or {}

        # Expect exactly one dataset key per entry
        if len(x_data) != 1 or len(y_data) != 1:
            raise ValueError("Each x_data and y_data entry must contain exactly one dataset.")
        dataset = next(iter(x_data))
        if dataset not in y_data:
            raise ValueError(f"Dataset '{dataset}' not found in corresponding y_data.")

        methods = sorted(x_data[dataset].keys())
        if set(methods) != set(y_data[dataset].keys()):
            raise ValueError("Mismatch of methods between x_data and y_data for dataset '{dataset}'.")

        for method in methods:
            xs = x_data[dataset][method]
            ys = y_data[dataset][method]
            if len(xs) != len(ys):
                raise ValueError(f"Length mismatch for method '{method}' in plot {plot_idx}.")

            x_errs = x_err_data.get(dataset, {}).get(method, [None]*len(xs))
            y_errs = y_err_data.get(dataset, {}).get(method, [None]*len(ys))

            # pad errors if missing
            if x_errs is None:
                x_errs = [None] * len(xs)
            if y_errs is None:
                y_errs = [None] * len(ys)
            if len(x_errs) != len(xs) or len(y_errs) != len(ys):
                raise ValueError(f"Error-array length mismatch for method '{method}' in plot {plot_idx}.")

            for i, (x, y, xe, ye) in enumerate(zip(xs, ys, x_errs, y_errs)):
                rows.append({
                    'plot_idx': plot_idx,
                    'dataset': dataset,
                    'method': method,
                    'x': x,
                    'y': y,
                    'x_err': xe,
                    'y_err': ye
                })

    df = pd.DataFrame(rows, columns=['plot_idx', 'dataset', 'method', 'x', 'y', 'x_err', 'y_err'])

    if save_path:
        df.to_csv(save_path, index=False)
    return df

def multi_plot_2d(
    x_data_list: List[Dict[str, Dict[str, List[float]]]],
    y_data_list: List[Dict[str, Dict[str, List[float]]]],
    x_std_err_list: Optional[List[Dict[str, Dict[str, List[float]]]]] = None,
    y_std_err_list: Optional[List[Dict[str, Dict[str, List[float]]]]] = None,
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    titles: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (24, 6),
    save_path: Optional[Path] = None,
    marker_size: int = 100,
    legend_marker_size: int = 12,
    font_size: int = 12
) -> None:
    """
    Plot four 2D scatter plots in a row with error bars for each method.

    Parameters
    ----------
    x_data_list, y_data_list : List[Dict[str, Dict[str, List[float]]]]
        Each item should have only one dataset, mapping dataset -> method -> list of values.
    x_std_err_list, y_std_err_list : Optional[List[Dict[str, Dict[str, List[float]]]]]
        Same as x_data_list/y_data_list, values are std errors for error bars.
    x_labels, y_labels : Optional[List[str]]
        Axis labels for each subplot. Only left subplot will display y label.
    titles : Optional[List[str]]
        Titles for each subplot.
    figsize : Tuple[int, int]
        Figure size for all subplots together.
    save_path : Optional[Path]
        If set, save figure to this path.
    marker_size : int
        Size of scatter markers.
    legend_marker_size : int
        Size of legend markers.
    font_size : int
        Font size for all text in the plot.
    """
    plt.rcParams.update({'font.size': font_size})

    n_plots = len(x_data_list)
    if n_plots != 4 or len(y_data_list) != 4:
        raise ValueError("This function is designed to plot exactly 4 subplots.")

    if x_std_err_list is None:
        x_std_err_list = [None] * n_plots
    if y_std_err_list is None:
        y_std_err_list = [None] * n_plots
    if x_labels is None:
        x_labels = ["X"] * n_plots
    if y_labels is None:
        y_labels = ["Y"] * n_plots
    if titles is None:
        titles = [None] * n_plots

    fig, axes = plt.subplots(1, 4, figsize=figsize, squeeze=False)
    axes = axes[0]  # axes is 2d, shape (1, 4)

    # For legend sharing, collect all methods and colors from all plots
    all_methods = set()
    for xd in x_data_list:
        dataset = list(xd.keys())[0]
        all_methods.update(xd[dataset].keys())
    all_methods = sorted(list(all_methods))
    cmap = plt.get_cmap("tab10")
    colors = {method: cmap(i % cmap.N) for i, method in enumerate(all_methods)}
    marker = "o"

    method_handles = [
        plt.Line2D(
            [0], [0],
            marker=marker,
            color=colors[m],
            linestyle="",
            markersize=legend_marker_size,
            label=m
        )
        for m in all_methods
    ]

    for plot_idx in range(4):
        ax = axes[plot_idx]
        x_data = x_data_list[plot_idx]
        y_data = y_data_list[plot_idx]
        x_std_err = x_std_err_list[plot_idx]
        y_std_err = y_std_err_list[plot_idx]

        # Only one dataset should be present
        if len(x_data) != 1 or len(y_data) != 1:
            raise ValueError("Each x_data and y_data must contain exactly one dataset.")

        dataset = list(x_data.keys())[0]
        if dataset not in y_data:
            raise ValueError("Dataset key in x_data not found in y_data.")

        methods_x = set(x_data[dataset])
        methods_y = set(y_data[dataset])
        if methods_x != methods_y:
            raise ValueError("Methods in x_data and y_data do not match.")
        these_methods = sorted(methods_x)

        for method in these_methods:
            xs = x_data[dataset].get(method, [])
            ys = y_data[dataset].get(method, [])
            if len(xs) != len(ys):
                raise ValueError(
                    f"Length mismatch for method '{method}': {len(xs)} x's vs {len(ys)} y's."
                )
            # Get std errors if provided
            xerr = None
            yerr = None
            if x_std_err is not None:
                xerr = x_std_err.get(dataset, {}).get(method, None)
            if y_std_err is not None:
                yerr = y_std_err.get(dataset, {}).get(method, None)

            ax.errorbar(
                xs, ys,
                xerr=xerr,
                yerr=yerr,
                fmt=marker,
                color=colors[method],
                markersize=marker_size // 10,
                elinewidth=1.2,
                capsize=4,
                label=method,
                alpha=0.8
            )

        ax.set_xlabel(x_labels[plot_idx], fontsize=font_size)
        # Only display y label for the left subplot
        if plot_idx == 0:
            ax.set_ylabel(y_labels[plot_idx], fontsize=font_size)
        else:
            ax.set_ylabel("")  # Remove y label for other subplots

        if titles[plot_idx]:
            ax.set_title(titles[plot_idx], fontsize=font_size)
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.grid(False)

    # Put the combined method legend in the upper right of the last subplot
    axes[-1].legend(
        handles=method_handles,
        title="Method",
        loc="best",
        frameon=True,
        fontsize=font_size,
        title_fontsize=font_size
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf')
        plt.close()
    else:
        plt.show()


def plot_2d_v1(
    x_data: Dict[str, Dict[str, List[float]]],
    y_data: Dict[str, Dict[str, List[float]]],
    x_std_err: Optional[Dict[str, Dict[str, List[float]]]] = None,
    y_std_err: Optional[Dict[str, Dict[str, List[float]]]] = None,
    x_label: str = "X",
    y_label: str = "Y",
    figsize: Tuple[int, int] = (8, 6),
    title: str = None,
    save_path: Optional[Path] = None,
    marker_size: int = 100,
    legend_marker_size: int = 12,
    font_size: int = 12  # <-- Added argument
) -> None:
    """
    Plot 2D scatter for one dataset, error bars for each method.

    Parameters
    ----------
    x_data, y_data : Dict[str, Dict[str, List[float]]]
        Should have only one dataset, mapping dataset -> method -> list of values.
    x_std_err, y_std_err : Optional[Dict[str, Dict[str, List[float]]]]
        Same structure as x_data/y_data, values are std errors for error bars.
    x_label, y_label : str
        Axis labels.
    figsize : Tuple[int, int]
        Figure size.
    title : str, optional
        Plot title.
    save_path : Optional[Path]
        If set, save figure to this path.
    marker_size : int
        Size of scatter markers.
    legend_marker_size : int
        Size of legend markers.
    font_size : int
        Font size for all text in the plot.
    """
    # Set font size for all text (affects labels, legend, title, ticks)
    plt.rcParams.update({'font.size': font_size})

    # Only one dataset should be present
    if len(x_data) != 1 or len(y_data) != 1:
        raise ValueError("x_data and y_data must each contain exactly one dataset.")

    dataset = list(x_data.keys())[0]
    if dataset not in y_data:
        raise ValueError("Dataset key in x_data not found in y_data.")

    # Get methods
    methods_x = set(x_data[dataset])
    methods_y = set(y_data[dataset])
    if methods_x != methods_y:
        raise ValueError("Methods in x_data and y_data do not match.")
    all_methods = sorted(methods_x)

    # Assign one unique color per method
    cmap = plt.get_cmap("tab10")
    colors = {method: cmap(i % cmap.N) for i, method in enumerate(all_methods)}

    # Default marker for all methods
    marker = "o"

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot scatter points with error bars
    for method in all_methods:
        xs = x_data[dataset].get(method, [])
        ys = y_data[dataset].get(method, [])
        if len(xs) != len(ys):
            raise ValueError(
                f"Length mismatch for method '{method}': {len(xs)} x's vs {len(ys)} y's."
            )
        # Get std errors if provided
        xerr = None
        yerr = None
        if x_std_err is not None:
            xerr = x_std_err.get(dataset, {}).get(method, None)
        if y_std_err is not None:
            yerr = y_std_err.get(dataset, {}).get(method, None)

        ax.errorbar(
            xs, ys,
            xerr=xerr,
            yerr=yerr,
            fmt=marker,
            color=colors[method],
            markersize=marker_size // 10,
            elinewidth=1.2,
            capsize=4,
            label=method,
            alpha=0.8
        )

    # Only legend for methods (colors)
    method_handles = [
        plt.Line2D(
            [0], [0],
            marker=marker,
            color=colors[m],
            linestyle="",
            markersize=legend_marker_size,
            label=m
        )
        for m in all_methods
    ]
    ax.legend(
        handles=method_handles,
        title="Method",
        loc="upper right",
        frameon=True,
        fontsize=font_size,
        title_fontsize=font_size
    )

    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    if title:
        ax.set_title(title, fontsize=font_size)
    ax.grid(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf')
        plt.close()
    else:
        plt.show()


def get_std_data(experiment_id):
    data_type, method, exp_id = experiment_id
    # Construct the path to the results file
    results_path = Path(f"results/{data_type}/{method}/{exp_id}/metrics_std.pkl")
    metrics = load_pickle(results_path)
    if metrics is None:
        raise FileNotFoundError(f"No metrics found for {experiment_id}")
    return metrics


import matplotlib.pyplot as plt

def plot_alpha_metrics_v1(
    experiment_ids,
    metric_name,
    alphas,
    display_names=None,
    alpha_name='alpha',
    get_results=None,
    plot_path=None,
    title=None,
    font_size=12,
    x_as_int=True
):
    """
    Compare metrics that vary over alphas (but are scalars per alpha).
    
    Parameters
    ----------
    experiment_ids : list of pairs of (data_type, method, exp_id)
        Identifiers for each experiment.
    metric_name : str
        Key in each dict that maps to {alpha: float}, e.g. 'interval_width'.
    alphas : list of float / str
        The alpha values to plot (in order). Can als be other qualities like 'thresholds'.
    alpha_name : str
        Name of the alpha parameter for labeling axes (default: 'alpha').
    get_results : callable
        Function signature get_results(experiment_id) -> metrics_dict.
    display_names : list of str, optional
        Display names for each experiment (default: method-exp_id).
    plot_path : str, optional
        If not None, saves the plot as a PDF to this path.
    title : str, optional
        Title for the plot.
    font_size : int, optional
        Font size for all plot text.
    x_as_int : bool, optional
        If True, x axis ticks/labels are cast to integers.
    """
    if get_results is None:
        raise ValueError("get_results function must be provided.")

    plt.rcParams.update({'font.size': font_size})

    results_list = [get_results(eid) for eid in experiment_ids]
    plt.figure()
    for idx, (exp_id, res) in enumerate(zip(experiment_ids, results_list)):
        vals = [res[metric_name].get(alpha) for alpha in alphas]
        display_name = f"{exp_id[1]}-{exp_id[2]}" if display_names is None else display_names[idx]
        plt.plot(alphas, vals, marker='x', label=display_name)

    title = title or f'{metric_name} across {alpha_name} on {experiment_ids[0][0]}'
    plt.title(title, fontsize=font_size)
    plt.xlabel(alpha_name, fontsize=font_size)
    plt.ylabel(metric_name, fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.grid(True)
    plt.xticks(rotation=45, fontsize=font_size)
    if x_as_int:
        try:
            int_alphas = [int(float(a)) for a in alphas]
            plt.xticks(int_alphas, [str(a) for a in int_alphas])
        except Exception:
            pass  # fallback to default ticks if conversion fails

    plt.tight_layout()
    if plot_path is not None:
        # Always save as PDF
        if not plot_path.lower().endswith('.pdf'):
            plot_path = plot_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(plot_path, format='pdf')
    else:
        plt.show()
    plt.close()


def plot_across_horizon(
    experiment_ids,
    metric_name,
    alpha,
    x_label='Horizon',
    y_label='Coverage',
    display_names=None,
    get_results=None,
    plot_path=None,
    title=None,
    font_size=12,
    x_as_int=True
):
    if get_results is None:
        raise ValueError("get_results function must be provided.")

    plt.rcParams.update({'font.size': font_size})

    results_list = [get_results(eid) for eid in experiment_ids]
    plt.figure()
    for idx, (exp_id, res) in enumerate(zip(experiment_ids, results_list)):
        vals_dict = res[metric_name].get(alpha)
        horizons = sorted(vals_dict.keys())
        vals = [vals_dict[h] for h in horizons]
        display_name = f"{exp_id[1]}-{exp_id[2]}" if display_names is None else display_names[idx]
        plt.plot(horizons, vals, marker='x', label=display_name)

    title = title
    plt.title(title, fontsize=font_size)
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.grid(True)
    plt.xticks(rotation=45, fontsize=font_size)
    if x_as_int:
        try:
            int_horizons = [int(float(h)) for h in horizons]
            plt.xticks(int_horizons, [str(h) for h in int_horizons])
        except Exception:
            pass  # fallback to default ticks if conversion fails

    plt.tight_layout()
    if plot_path is not None:
        plt.savefig(plot_path, format='pdf')
    else:
        plt.show()
    plt.close()


import matplotlib.pyplot as plt


def calibration_plot(exp_id, save_dir, get_metrics, horizons):
    results = get_metrics(exp_id)
    avg_cov = results['coverage_avg']
    alphas = sorted(avg_cov.keys())
    target_cov = [1 - alpha for alpha in alphas]
    covs = np.linspace(0, 1, 100)
    for horizon in horizons:
        cov_vals = [avg_cov[alpha][horizon] for alpha in alphas]
        plt.plot(target_cov, cov_vals, marker='x')
        plt.plot(covs, covs, linestyle='--', color='gray', label='ideal coverage')
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.xlabel('Target Coverage')
        plt.ylabel('Empirical Coverage')
        plt.legend()
        plt.savefig(save_dir / f"{exp_id[1]}-{exp_id[2]}_{horizon}.png")
        plt.close()


def plot2_across_horizon(
    experiment_ids,
    metric_names,  # expects a list or tuple of exactly two metric names
    alpha,
    x_label='Horizon',
    y_labels=None,  # expects a list or tuple of two y labels
    display_names=None,
    get_results=None,
    plot_path=None,
    title=None,
    ourmethod=None,
    font_size=12,
    line_width=2,
    x_as_int=True
):
    if get_results is None:
        raise ValueError("get_results function must be provided.")
    if len(metric_names) != 2:
        raise ValueError("Please provide exactly two metric names in metric_names.")
    if y_labels is None:
        y_labels = ['Metric 1', 'Metric 2']
    if len(y_labels) != 2:
        raise ValueError("Please provide exactly two y_labels.")

    plt.rcParams.update({'font.size': font_size})

    # fetch all results once
    results_list = [get_results(eid) for eid in experiment_ids]

    # two plots stacked vertically, with extra width for each
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for idx, metric_name in enumerate(metric_names):
        ax = axes[idx]
        
        if idx == 0:
            ylim = (0, 1)
            ax.set_ylim(ylim)

        # if plotting coverage, draw the reference line
        if metric_name == 'coverage_avg':
            ax.axhline(y=1 - alpha, color='gray', linestyle='--', linewidth=line_width, label=f'{int((1-alpha)*100)}% Reference')

        # plot each experiment
        for exp_idx, (exp_id, res) in enumerate(zip(experiment_ids, results_list)):
            vals_dict = res[metric_name].get(alpha)
            horizons = sorted(vals_dict.keys())
            horizons_disp = [int(h+1) for h in horizons]  # ensure horizons are floats
            vals = [vals_dict[h] for h in horizons]
            name = (
                f"{exp_id[1]}-{exp_id[2]}"
                if display_names is None
                else display_names[exp_idx]
            )
            ax.plot(
                horizons_disp,
                vals,
                marker='x',
                linewidth=line_width,
                label=name
            )

        # subplot titles and labels
        # subtitle = f"{title} - {metric_name}" if title else metric_name
        # ax.set_title(subtitle, fontsize=font_size)
        ax.set_ylabel(y_labels[idx], fontsize=font_size)
        if idx == 1:
            ax.legend(fontsize=font_size)

        # ax.grid(True)
        ax.tick_params(axis='x', rotation=45, labelsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)

        # force integer x-ticks if desired
        if x_as_int:
            try:
                ix = [int(float(h)) for h in horizons_disp]
                ax.set_xticks(ix)
                ax.set_xticklabels([str(x) for x in ix])
            except Exception:
                pass

    # common x-label on the bottom plot
    axes[-1].set_xlabel(x_label, fontsize=font_size)

    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path, format='pdf')
    else:
        plt.show()
    plt.close()
