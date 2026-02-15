import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from core.utils import load_pickle

def get_metrics(experiment_id, get_std=False, filename='metrics.pkl'):
    """
    Fetch evaluated metrics for an experiment.
    
    Parameters
    ----------
    experiment_id : pair of (data_type, method, exp_id)
    
    Returns
    -------
    dict
        Dictionary containing metrics for the experiment.
    """
    data_type, method, exp_id = experiment_id
    # Construct the path to the results file
    results_path = Path(f"results/{data_type}/{method}/{exp_id}/{filename}")
    if get_std:
        results_path = Path(f"results/{data_type}/{method}/{exp_id}/metrics_std.pkl")
    metrics = load_pickle(results_path)
    if metrics is None:
        raise FileNotFoundError(f"No metrics found for {experiment_id}")
    return metrics

def get_time_varying_metrics(experiment_id):
    """
    Fetch evaluated time-varying metrics for an experiment.
    
    Parameters
    ----------
    experiment_id : pair of (data_type, method, exp_id)
    
    Returns
    -------
    dict
        Dictionary containing metrics for the experiment.
    """
    data_type, method, exp_id = experiment_id
    # Construct the path to the results file
    results_path = Path(f"results/{data_type}/{method}/{exp_id}/metrics_time_varying.pkl")
    metrics = load_pickle(results_path)
    if metrics is None:
        raise FileNotFoundError(f"No metrics found for {experiment_id}")
    return metrics

def plot_scalar_metrics(experiment_ids, metric_name, display_names=None, plot_path=None, title=None):
    """
    Compare scalar (float) metrics across experiments.
    
    Parameters
    ----------
    experiment_ids : list of pairs of (data_type, method, exp_id)
        Identifiers for each experiment.
    metric_name : str
        Key of the metric in each dict that map to float values 
        (e.g. 'calibration_score_hc').
    """
    # fetch all results
    results_list = [get_metrics(eid) for eid in experiment_ids]
    
    if display_names is None:
        x_ticks = [f"{eid[1]}-{eid[2]}" for eid in experiment_ids]
    else:
        x_ticks = display_names
    
    plt.figure()
    values = [res[metric_name] for res in results_list]
    plt.bar(x_ticks, values)
    # Annotate each bar with its value
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.3f}", ha='center', va='bottom')
    plt.xticks(rotation=45)
    title = title or f'{metric_name} on {experiment_ids[0][0]}'
    plt.title(title)
    plt.xlabel('Method - Experiment ID')
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.tight_layout()
    if plot_path is not None:
        plt.savefig(plot_path)
    else:
        plt.show()
    plt.close()


def plot_alpha_metrics(experiment_ids, metric_name, alphas, display_names=None, alpha_name='alpha', get_results=get_metrics, plot_path=None, title=None):
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
    """
    results_list = [get_results(eid) for eid in experiment_ids]
    plt.figure()
    for idx, (exp_id, res) in enumerate(zip(experiment_ids, results_list)):
        vals = [res[metric_name].get(alpha) for alpha in alphas]
        display_name = f"{exp_id[1]}-{exp_id[2]}" if display_names is None else display_names[idx]
        plt.plot(alphas, vals, marker='x', label=display_name)
    title = title or f'{metric_name} across {alpha_name} on {experiment_ids[0][0]}'
    plt.title(title)
    plt.xlabel(alpha_name)
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if plot_path is not None:
        plt.savefig(plot_path)
    else:
        plt.show()
    plt.close()


def plot_nested_alpha_metrics(experiment_ids, metric_name, alphas, subkeys, alpha_name='alpha', display_names=None, plot_path=None, title=None):
    """
    For each subkey, compare metrics that vary over alphas using plot_alpha_metrics.
    
    Parameters
    ----------
    experiment_ids : list of pairs of (data_type, method, exp_id)
        Identifiers for each experiment.
    metric_name : str
        Top-level key mapping to {alpha: {subkey: float}}.
    alphas : list of float
        The alpha values to plot.
    subkeys : list
        Inner-dict keys to plot (e.g. thresholds).
    """
    for subkey in subkeys:
        # helper to extract flattened metrics for this subkey
        def get_flat_results(eid):
            res = get_metrics(eid)
            nested = res[metric_name]
            return {metric_name: {alpha: nested.get(alpha, {}).get(subkey) for alpha in alphas}}

        # generate plots for this subkey
        plot_alpha_metrics(
            experiment_ids=experiment_ids,
            metric_name=f"{metric_name}",
            alphas=alphas,
            alpha_name=alpha_name,
            display_names=display_names,
            get_results=get_flat_results,
            plot_path=plot_path / f"{subkey}.png" if plot_path else None,
            title=title or f'{metric_name} - {subkey} across alphas on {experiment_ids[0][0]}'
        )


import matplotlib.pyplot as plt
from itertools import cycle
from typing import Dict, List, Tuple

def plot_2d(
    x_data: Dict[str, Dict[str, List[float]]],
    y_data: Dict[str, Dict[str, List[float]]],
    x_label: str = "X",
    y_label: str = "Y",
    figsize: Tuple[int, int] = (8, 6),
    title: str = None,
    save_path: Optional[Path] = None,
    marker_size: int = 100,
    legend_marker_size: int = 12
) -> None:
    # Ensure same datasets in both x and y
    datasets = set(x_data) | set(y_data)
    if set(x_data) != set(y_data):
        raise ValueError("Datasets in x_data and y_data must match exactly.")

    # Assign one marker per dataset
    markers = cycle(("o", "s", "^", "d", "v", "p", "X", "*", "+", "1"))
    dataset_markers = {ds: next(markers) for ds in sorted(datasets)}

    # Collect all methods across all datasets
    all_methods = set()
    for ds in datasets:
        methods_x = set(x_data[ds])
        methods_y = set(y_data[ds])
        if methods_x != methods_y:
            raise ValueError(
                f"Methods for dataset '{ds}' do not match between x_data and y_data."
            )
        all_methods.update(methods_x)
    all_methods = sorted(all_methods)

    # Assign one unique color per method
    cmap = plt.get_cmap("tab10")
    colors = {method: cmap(i % cmap.N) for i, method in enumerate(all_methods)}

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot scatter points
    for ds in sorted(datasets):
        marker = dataset_markers[ds]
        for method in all_methods:
            xs = x_data[ds].get(method, [])
            ys = y_data[ds].get(method, [])
            if len(xs) != len(ys):
                raise ValueError(
                    f"Length mismatch for dataset '{ds}', method '{method}': "
                    f"{len(xs)} x's vs {len(ys)} y's."
                )
            ax.scatter(
                xs, ys,
                color=colors[method],
                marker=marker,
                s=marker_size,
                edgecolors="w",
                linewidths=0.5,
                alpha=0.8
            )

    # Build method handles
    method_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color=colors[m],
            linestyle="",
            markersize=legend_marker_size,
            label=m
        )
        for m in all_methods
    ]
    # Build dataset handles
    dataset_handles = [
        plt.Line2D(
            [0], [0],
            marker=dataset_markers[d],
            color="black",
            linestyle="",
            markersize=legend_marker_size,
            label=d
        )
        for d in sorted(datasets)
    ]

    # Combine both sets of handles into a single legend block
    combined_handles = method_handles + dataset_handles
    ax.legend(
        handles=combined_handles,
        loc="upper right",
        ncol=2,
        frameon=True
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    # Remove grid/background lines
    ax.grid(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def _extract_metric_values(
    metrics: Dict[str, Any],
    metric_name: str,
    params: Optional[Dict[str, Any]] = None
) -> List[float]:
    """
    Extract a list of values from the metrics dict given a metric name
    and optional indexing parameters.

    Parameters
    ----------
    metrics : Dict[str, Any]
        The full metrics dictionary returned by get_metrics().
    metric_name : str
        One of the top‐level keys in metrics.
    params : Dict[str, Any], optional
        A mapping from parameter name to the key to index into nested dicts.
        e.g. {"alpha": 0.1, "h": 5} for metrics["interval_width"][0.1][5].

    Returns
    -------
    List[float]
        A flat list of numeric values for that metric.
    """
    if metric_name not in metrics:
        raise KeyError(f"Metric '{metric_name}' not found in metrics.")

    # Drill down into nested dicts/arrays if params provided
    value = metrics[metric_name]
    if params:
        for p_name, p_key in params.items():
            if value is None:
                raise KeyError(f"No data for parameter '{p_name}' in metric '{metric_name}'.")
            value = value[p_key]
    # if it's a mapping, average its values (handles nested dicts)
    if isinstance(value, dict):
        def _flatten(v):
            if isinstance(v, dict):
                out = []
                for sub in v.values():
                    out.extend(_flatten(sub))
                return out
            else:
                return [v]
        flat_vals = _flatten(value)
        return [float(np.mean(flat_vals))]
    return [float(value)]


def prepare_plot_data(
    experiment_ids: List[Tuple[str, str, str]],
    x_metric: str,
    y_metric: str,
    x_params: Optional[Dict[str, Any]] = None,
    y_params: Optional[Dict[str, Any]] = None,
    display_names = None,
    get_data_func=get_metrics,
) -> Tuple[
    Dict[str, Dict[str, List[float]]],
    Dict[str, Dict[str, List[float]]]
]:
    x_data: Dict[str, Dict[str, List[float]]] = {}
    y_data: Dict[str, Dict[str, List[float]]] = {}

    for dataset, method, exp_id in experiment_ids:
        # fetch metrics for this experiment
        metrics = get_data_func((dataset, method, exp_id))

        # extract x and y lists
        xs = _extract_metric_values(metrics, x_metric, x_params)
        ys = _extract_metric_values(metrics, y_metric, y_params)

        if len(xs) != len(ys):
            raise ValueError(
                f"Length mismatch for experiment ({dataset}, {method}, {exp_id}): "
                f"{len(xs)} x vs {len(ys)} y."
            )
        # use display names if provided, else use method-exp_id
        if display_names is not None:
            method_exp_id = display_names.get((method, exp_id), f"{method}-{exp_id}")
        else:
            method_exp_id = f"{method}-{exp_id}"
        x_data.setdefault(dataset, {})[method_exp_id] = xs
        y_data.setdefault(dataset, {})[method_exp_id] = ys

    return x_data, y_data