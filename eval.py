import argparse
import numpy as np
import os
import shutil
from pathlib import Path
from plot_utils import *
from core.utils import *
from core.eval.eval_utils import *
from core.eval.visualization import *
from core.eval.across_exp import *
from core.constants import *

def scale_data(
    data: Dict[str, Dict[str, List[float]]]
) -> Dict[str, Dict[str, List[float]]]:
    """
    Scale the values for all methods in each dataset to the range [0, 1].

    Parameters
    ----------
    data : Dict[str, Dict[str, List[float]]]
        Mapping from dataset -> method -> list of values.

    Returns
    -------
    scaled : Dict[str, Dict[str, List[float]]]
        New dict with the same structure, but each list of values for a given
        dataset is linearly scaled so that the minimum across all methods is 0
        and the maximum is 1.
    """
    scaled: Dict[str, Dict[str, List[float]]] = {}
    for dataset, methods in data.items():
        # Flatten all values across methods to find global min/max
        all_vals = [v for vals in methods.values() for v in vals]
        if not all_vals:
            scaled[dataset] = {m: [] for m in methods}
            continue
        min_val = min(all_vals)
        max_val = max(all_vals)
        span = max_val - min_val
        # Avoid division by zero: if all values are equal, map them to 0.5
        for method, vals in methods.items():
            if span > 0:
                scaled_vals = [(v - min_val) / span for v in vals]
            else:
                scaled_vals = [0.5 for _ in vals]
            scaled.setdefault(dataset, {})[method] = scaled_vals
    return scaled

if __name__ == "__main__":
    # Parse arguments (if needed)
    parser = argparse.ArgumentParser(description="Evaluate ACI results.")
    parser.add_argument(
        '--configs', '-c',
        type=str,
        nargs='+',
        default=['0'],
        help='One or more configuration IDs for the experiment'
    )
    parser.add_argument('--reevaluate', '-r', action='store_true', help='Re-evaluate the results even if they exist')
    parser.add_argument('--skip_method', '-s', type=str, nargs='+', default=None, help='Method name to skip during evaluation')
    args = parser.parse_args()
    configs = args.configs
    
    all_exp_ids = []
    all_display_names = {}
    
    for config in configs:
        save_path = Path("results") / f"eval_{config}"
        if save_path.exists():
            shutil.rmtree(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Load global setting parameters from YAML file
        global_params_path = Path("configs") / 'global.yaml'
        if global_params_path.exists():
            global_params = load_yaml(global_params_path)
        
        config_path = Path("configs") / "eval" / f"{config}.yaml"
        config = load_yaml(config_path)
        methods = config['methods']
        exp_ids = config['exp_ids']
        config_ids = config['config_ids']
        if 'data_types' in config:
            data_types = config['data_types']
            pretrain_fractions = config['pretrain_fractions']
        else:
            data_type = config['data_type']
            pretrain_fraction = config.get('pretrain_fraction', 0)
            data_types = [data_type]*len(methods)
            pretrain_fractions = [pretrain_fraction]*len(methods)
        
        # load data params
        data_config_path = Path("configs") / "data" / f"{data_types[0]}.yaml"
        data_params = load_yaml(data_config_path)
        H = data_params.get('H')
        horizons = np.arange(0, H).tolist()
        
        display_names = config.get('display_names', None)
        
        experiment_ids = []
        remove_indices = []
        for i in range(len(exp_ids)):
            if args.skip_method is not None and methods[i] in args.skip_method:
                print(f"Skipping method {methods[i]} as per user request.")
                remove_indices.append(i)
                continue
            experiment_ids.append((data_types[i], methods[i], exp_ids[i]))
        # Remove skipped methods from other lists
        for index in sorted(remove_indices, reverse=True):
            del methods[index]
            del exp_ids[index]
            del data_types[index]
            del pretrain_fractions[index]
            if display_names is not None:
                del display_names[index]
        # experiment_ids = [(data_types[i], methods[i], exp_ids[i]) for i in range(len(exp_ids))]
        
        if display_names is None:
            print('No display names found, using default names.')
        else:
            display_names_dict = {}
            for i in range(len(experiment_ids)):
                data_type, method, exp_id = experiment_ids[i]
                display_names_dict[(method, exp_id)] = display_names[i]

        all_exp_ids.extend(experiment_ids)
        all_display_names.update(display_names_dict)

        print(f"Experiment IDs: {experiment_ids}")
        
        #######################################
        # -- Re-evaluate results if needed -- #
        #######################################
        
        if args.reevaluate:
            print("Re-evaluating results...")
            # for data_type, method, exp_id in experiment_ids:
            for i in range(len(experiment_ids)):
                data_type, method, exp_id = experiment_ids[i]
                config_id = config_ids[i]
                # run the main.py script without running the simulation
                os.system(f"python main.py -d {data_type} -m {method} -e {exp_id} -c {config_id} -pf {pretrain_fractions[i]} -n")
            print("Re-evaluation complete.")
        
        # metrics = get_metrics(experiment_ids[0])
        # print(f"Metrics for {experiment_ids[0]}")
        # print(metrics['var'].keys())
        # print(metrics['var'][0.3])
        
        alphas = sorted(config['alphas'])
        
        # Plot scalar metrics
        scalar_metrics = ['calibration_score_hc']
        for metric in scalar_metrics:
            plot_scalar_metrics(
                experiment_ids=experiment_ids,
                metric_name=metric,
                display_names=display_names,
                plot_path=save_path / f"{metric}.png",
            )
        
        # Plot alpha metrics
        alpha_metrics = ['calibration_score_h', 'horizon_cov_overall', 'worst_case_risk', 'avg_risk_nrw']
        alpha_names = ['horizon', 'alpha', 'alpha', 'alpha']
        for i, metric in enumerate(alpha_metrics):
            try:
                alpha_vals = alphas if metric != 'calibration_score_h' else horizons
                plot_alpha_metrics(
                    experiment_ids=experiment_ids,
                    metric_name=metric,
                    alphas=alpha_vals,
                    display_names=display_names,
                    alpha_name=alpha_names[i],
                    plot_path=save_path / f"{metric}.png",
                )
            except Exception as e:
                print(f"Error plotting metric {metric}: {e}")

        # Plot nested alpha metrics
        nested_alpha_metrics = ['interval_width', 'var', 'exponential_risk', 'max_consecutive_violations', 'var_h', 'exponential_risk_h', 'max_consecutive_violations_h', 'num_violations_nrw', 'exponential_risk_nrw', 'saregret']
        subkeys_list = [
            horizons,
            risk_levels_c,
            thetas_c,
            thresholds_c,
            risk_levels_c,
            thetas_c,
            thresholds_c,
            thresholds_c,
            thetas_c,
            horizons,
        ]
        alphas_args = [alphas, alphas, alphas, alphas, horizons, horizons, horizons, alphas, alphas, alphas]
        alpha_names = ['alpha', 'alpha', 'alpha', 'alpha', 'horizon', 'horizon', 'horizon', 'alpha', 'alpha', 'alpha']
        for i, metric in enumerate(nested_alpha_metrics):
            try:
                plot_path = save_path / f"{metric}"
                plot_path.mkdir(parents=True, exist_ok=True)
                plot_nested_alpha_metrics(
                    experiment_ids=experiment_ids,
                    metric_name=metric,
                    alphas=alphas_args[i],
                    subkeys=subkeys_list[i],
                    display_names=display_names,
                    alpha_name=alpha_names[i],  
                    plot_path=plot_path,
                )
            except Exception as e:
                print(f"Error plotting nested metric {metric}: {e}")

        bar_plots_for_avg_metrics(experiment_ids=experiment_ids, display_names=display_names, save_path=save_path)
        line_plot_for_metrics(experiment_ids=experiment_ids, display_names=display_names, save_path=save_path)
        plot_time_varying_metrics(experiment_ids=experiment_ids, display_names=display_names, get_metrics=get_time_varying_metrics, metric_name='avg_risk', theta=None, subset_idx=0, alpha_idx=-1, start_t=0, end_t=-1, save_path=save_path)
        comparison_line_plot(experiment_ids=experiment_ids, display_names=display_names, save_path=save_path)
        comparison_line_plot(experiment_ids=experiment_ids, display_names=display_names, save_path=save_path, percentage=True)
        comparison_line_plot(experiment_ids=experiment_ids, display_names=display_names, save_path=save_path, percentage=False, average=True)
        comparison_line_plot(experiment_ids=experiment_ids, display_names=display_names, save_path=save_path, percentage=True, average=True)
        

    #######################
    # -- Make 2D plots -- #
    #######################
    save_path = Path("results") / f"eval_all"
    plot_2d_save_path = save_path / "2d_plots"
    plot_2d_save_path.mkdir(parents=True, exist_ok=True)
    
    # interval_width v.s. horizon_cov_overall
    x_metric = 'interval_width'
    x_params = {}
    
    y_pairs = [
        ('calibration_score_h', {}),
        ('calibration_score_hc', {}),
        ('var', {'alpha': 0.1, 'risk_level': 0.2}),
        ('worst_case_risk', {'alpha': 0.05}),
    ]

    for y_metric, y_params in y_pairs:
        x_data, y_data = prepare_plot_data(
            experiment_ids=all_exp_ids,
            x_metric=x_metric,
            y_metric=y_metric,
            x_params=x_params,
            y_params=y_params,
            display_names=all_display_names,
        )

        # scale x only
        x_data = scale_data(x_data)

        plot_2d(
            x_data=x_data,
            y_data=y_data,
            x_label=x_metric,
            y_label=y_metric,
            title=f"{x_metric} vs {y_metric}",
            save_path=plot_2d_save_path / f"{x_metric}_vs_{y_metric}.png",
        )
    
    x_metric = 'calibration_score_hc'
    x_params = {}
    
    y_pairs = [
        ('var', {'alpha': 0.1, 'risk_level': 0.2}),
        ('worst_case_risk', {'alpha': 0.05}),
    ]
    
    for y_metric, y_params in y_pairs:
        x_data, y_data = prepare_plot_data(
            experiment_ids=all_exp_ids,
            x_metric=x_metric,
            y_metric=y_metric,
            x_params=x_params,
            y_params=y_params,
            display_names=all_display_names,
        )

        plot_2d(
            x_data=x_data,
            y_data=y_data,
            x_label=x_metric,
            y_label=y_metric,
            title=f"{x_metric} vs {y_metric}",
            save_path=plot_2d_save_path / f"{x_metric}_vs_{y_metric}.png",
        )
    
    