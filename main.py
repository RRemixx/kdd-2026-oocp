import argparse
import os
from pathlib import Path
import yaml
import copy
import time
import shutil

from core.data import *
from core.utils import *
from core.eval.eval_utils import *
from core.method.cp_h1 import run_single_horizon_simulation
from core.method.cp_hm import run_iid_simulation, run_multi_horizon_simulation
from core.method.cp_ours import *
from core.method.cpid_loop import run_cpid_sim
from core.method.cp_ours_multialpha import run_algorithm_with_multiple_alphas
from core.eval.visualization import *
import matplotlib
matplotlib.use("Agg")


def _build_data_generator(data_name, T_obs, H, N, data_params):
    if data_name == 'markovar' or data_name == 'debug':
        return MarkovARData(T_obs=T_obs, H=H, N=N, data_args=data_params)
    if data_name == 'flusight':
        return FluSightDataProcessor(T_obs=T_obs, H=H, N=N, data_args=data_params)
    if data_name == 'iidlr':
        return LinearRegressionDataGenerator(T_obs=T_obs, H=H, N=N, data_args=data_params)
    if data_name == 'lane':
        return LanePredictionDataProcessor(T_obs=T_obs, H=H, N=N, data_args=data_params)
    if data_name == 'cyclone' or data_name == 'cycloneg':
        return CycloneDataset(T_obs=T_obs, H=H, N=N, data_args=data_params)
    if data_name == 'hosp':
        return HospDataProcessor(T_obs=T_obs, H=H, N=N, data_args=data_params)
    if data_name == 'weather':
        return WeatherDataProcessor(T_obs=T_obs, H=H, N=N, data_args=data_params)
    if data_name == 'elec':
        return WeatherDataProcessor(T_obs=T_obs, H=H, N=N, data_args=data_params)
    raise ValueError(f"Unsupported data type: {data_name}. Supported types are 'markovar' and 'flusight'.")
    
def main(args):
    if isinstance(args, dict):
        args = argparse.Namespace(**args)
    ########################################
    # --- Parse command line arguments --- #
    ########################################
    
    ####################################
    # --- Load configuration files --- #
    ####################################
    # Load global setting parameters from YAML file
    global_params_path = Path("configs") / 'global.yaml'
    if global_params_path.exists():
        global_params = load_yaml(global_params_path)
    
    # Load dataset-specific parameters from YAML file
    data_params_path = Path("configs/data") / f"{args.data}.yaml"
    if data_params_path.exists():
        data_params = load_yaml(data_params_path)
    print(f'Data parameters loaded from {data_params_path}')
    
    # Load model parameters from YAML file
    config_path = Path("configs") / args.method / f"{args.config_id}.yaml"
    if config_path.exists():
        params = load_yaml(config_path)
    
    params['alphas'] = global_params['alphas']
    params['data'] = args.data  # Add data type to parameters
    params['expid'] = args.expid  # Add experiment ID to parameters
    params['data_params'] = data_params  # Add data parameters to params
    params['dynamic_S_max'] = data_params.get('dynamic_S_max', False)  # Whether to use dynamic S_max
    # Override score_func_type with value from data_params if specified
    if 'score_func_type' in data_params:
        params['score_func_args']['type'] = data_params['score_func_type']
    if 'H' in data_params:
        params['H'] = data_params['H']  # Override H if specified in data_params
        params['N'] = data_params.get('N', None)  # Override N if specified in data_params, default to None
        # params['alphas'] = data_params.get('alphas', None)  # Override alphas if specified in data_params
    
    # Save parameters to yaml file
    save_path = Path("results") / args.data / args.method / args.expid
    # If the directory exists, remove it and recreate
    # if save_path.exists():
    #     print(f"Warning: The save path {save_path} already exists. Removing existing contents.")
    #     shutil.rmtree(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / 'params.yaml', 'w') as f:
        yaml.dump(params, f)
    
    # logger
    save_run_metadata(args=args, meta_path=save_path / 'log.txt')
    
    twodim = data_params.get('twodim', False)
    if twodim:
        print("Running in 2D mode")
    
    # Unpack parameters for use
    T_obs = params['data_params']['T_obs']
    H = params['H']
    N = params['N']
    params['start_t'] = data_params.get('start_t', -1)  # Default start_t is -1 if not specified
    S_max = data_params['S_max'][params['score_func_args']['type']]
    if 'optional_args' in params:
        params['optional_args']['S_max'] = S_max
    params['S_max'] = S_max
    if params.get('optim_arg', {}).get('Ss_size', None) is not None:
        params['optim_arg']['Ss_size'] = min(params['optim_arg']['Ss_size'], H)
    method = args.method
    
    # Initialize data generator
    data_generator = _build_data_generator(args.data, T_obs, H, N, data_params)
    
    if args.pretrain_fraction > 0 and len(data_generator.subsets) > 1:
        pretrain_subset, remaining_subsets = data_generator.pretrain_split(args.pretrain_fraction, total_fraction=params.get('total_fraction', 1.0))

    ##############################
    # --- Run the simulation --- #
    ##############################
    merged_df = None
    if args.run:
        # Initialize dictionaries to store results
        all_ground_truths = {}
        all_prediction_intervals = {}
        all_timestamps = {}
        all_q_preds = {
            'qs': {},
            'scores': {}
        }
        notplot = True  # Set to True if you want to skip plotting
        print(f"Running experiment {args.expid} for method={method}, T_obs={T_obs}, H={H}, N={N}, S_max={S_max}")
        # print("Experiment parameters:")
        # print(yaml.safe_dump(params, sort_keys=False))

        learned_scores_df = None
        additional_context = None
        learned_scores_record_list = []
        if args.pretrain_fraction > 0 and len(data_generator.subsets) > 1:
            for subset in pretrain_subset:
                data_generator.set_subset(subset)
                score_record_list = collect_scores(
                    data_generator=data_generator,
                    score_func_args=params['score_func_args'],
                    T_obs=data_generator.T_obs,
                    H=H,
                    N=N,
                    start_t=params['start_t'],
                    twodim=twodim,
                    subset_name=subset,
                )
                learned_scores_record_list.extend(score_record_list)
            learned_scores_df = pd.DataFrame(learned_scores_record_list)
            optim_arg = params.get('optim_arg', {})
            traj_value_col = optim_arg.get('traj_value_col', 'score')
            additional_source_df = get_additional_Fs(learned_scores_df, score_window=params.get('score_window', 100))
            additional_context = get_context_from_other_subsets(
                df=additional_source_df,
                optim_arg=optim_arg,
                start_t=params['start_t'],
            )

        seq_length = []
        runtimes = []
        print(f"Processing subsets: {data_generator.subsets}")

        all_ground_truths = {subset: {} for subset in data_generator.subsets}
        all_prediction_intervals = {subset: {} for subset in data_generator.subsets}
        all_timestamps = {subset: None for subset in data_generator.subsets}
        all_q_preds['qs'] = {subset: {} for subset in data_generator.subsets}
        all_q_preds['scores'] = {subset: {} for subset in data_generator.subsets}

        def run_one(subset, alpha):
            local_data_generator = _build_data_generator(args.data, T_obs, H, N, data_params)
            local_data_generator.set_subset(subset)
            local_params = copy.deepcopy(params)
            if method in {'cpt', 'dtaci', 'acmcp', 'cfrnn'}:
                timestamps, ground_truths, prediction_intervals, scores, qs = run_aci_simulation(
                    data_generator=local_data_generator,
                    score_func_args=local_params['score_func_args'],
                    alpha=alpha,
                    T_obs=local_data_generator.T_obs,
                    H=H,
                    N=N,
                    start_t=local_params['start_t'],
                    method_opt=method,
                    S_max=S_max,
                    save_path=save_path / f'{subset}' / f'alpha_{alpha}',
                    plot=args.plot,
                    print_max_score=False,
                    twodim=twodim,
                    learned_scores=learned_scores_df,
                    expand_boundaries=local_params['expand_boundaries_values'][0],
                    additional_context=additional_context,
                    params=local_params,
                )
            elif method == 'cpid':
                timestamps, ground_truths, prediction_intervals, scores, qs = run_cpid_sim(
                    data_generator=local_data_generator,
                    score_func_args=local_params['score_func_args'],
                    alpha=alpha,
                    T_obs=local_data_generator.T_obs,
                    H=H,
                    N=N,
                    start_t=local_params['start_t'],
                    method_opt=method,
                    S_max=S_max,
                    save_path=save_path / f'{subset}' / f'alpha_{alpha}',
                    plot=args.plot,
                    print_max_score=False,
                    twodim=twodim,
                    learned_scores=learned_scores_df,
                    expand_boundaries=local_params['expand_boundaries_values'][0],
                    additional_context=additional_context,
                    params=local_params,
                )
                
            elif method == 'copulacp':
                timestamps, ground_truths, prediction_intervals, scores, qs = run_multi_horizon_simulation(
                    method=method,
                    params=local_params,
                    data_generator=local_data_generator,
                    alpha=alpha,
                    T_obs=local_data_generator.T_obs,
                    H=H,
                    N=N,
                    start_t=local_params['start_t'],
                    twodim=twodim,
                    learned_scores_df=learned_scores_df,
                )
            else:
                raise ValueError(f"Unsupported method: {method}")
            return subset, alpha, timestamps, ground_truths, prediction_intervals, scores, qs

        max_workers = args.workers
        start_time = time.time()
        futures = []
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for subset in data_generator.subsets:
                seq_length.append(data_generator.T_obs)
                for alpha in params['alphas']:
                    futures.append(executor.submit(run_one, subset, alpha))
            for future in tqdm(as_completed(futures), total=len(futures)):
                subset, alpha, timestamps, ground_truths, prediction_intervals, scores, qs = future.result()

                def contains_invalid(arr):
                    arr_np = np.array(arr)
                    return np.any(np.isnan(arr_np)) or np.any(np.isinf(arr_np))

                if contains_invalid(ground_truths):
                    print(f"Warning: Ground truths contain invalid values for subset {subset} and alpha {alpha}. Skipping this alpha.")

                all_ground_truths[subset][alpha] = copy.deepcopy(ground_truths)
                all_prediction_intervals[subset][alpha] = copy.deepcopy(prediction_intervals)
                if all_timestamps[subset] is None:
                    all_timestamps[subset] = copy.deepcopy(timestamps)
                all_q_preds['qs'][subset][alpha] = copy.deepcopy(qs)
                all_q_preds['scores'][subset][alpha] = copy.deepcopy(scores)

        runtimes.append(time.time() - start_time)

        if args.plot and twodim:
            alpha_to_plot = 0.1
            for subset in data_generator.subsets:
                if alpha_to_plot not in all_prediction_intervals[subset]:
                    continue
                print(f"Plotting results for alpha={alpha_to_plot} in 2D")
                if notplot:
                    print("Skipping plotting due to notplot flag")
                    notplot = False
                    current_save_path = save_path / f'plot_{alpha_to_plot}'
                    current_save_path.mkdir(parents=True, exist_ok=True)
                    ground_truths = all_ground_truths[subset][alpha_to_plot]
                    prediction_intervals = all_prediction_intervals[subset][alpha_to_plot]
                    for t in range(T_obs):
                        samples, qs = prediction_intervals
                        plot_forecast_step_2d(
                            samples=samples[t],
                            ground_truth=ground_truths[t],
                            qs=qs[t],
                            sample_color='gray',
                            sample_alpha=0.4,
                            gt_inside_color='green',
                            gt_outside_color='red',
                            circle_alpha=0.3,
                            save_path=current_save_path / f'forecast_t{t}_alpha{alpha_to_plot}.png'
                        )
        
        # save runtime info
        # time_record = {
        #     'data': args.data,
        #     'method': method,
        #     'exp_id': args.expid,
        #     'runtime_mean': np.mean(runtimes),
        #     'runtime_std': np.std(runtimes),
        #     'num_subsets': len(data_generator.subsets),
        #     'num_alphas': len(params['alphas']),
        #     'avg_seq_length': np.mean(seq_length),
        #     'horizons': H,
        #     'num_samples': N,
        # }
        # time_records = pd.read_csv(Path("results") / "time_records.csv") if (Path("results") / "time_records.csv").exists() else pd.DataFrame()
        # time_records = pd.concat(
        #     [time_records, pd.DataFrame([time_record])],
        #     ignore_index=True
        # )
        # time_records.to_csv(Path("results") / "time_records.csv", index=False)

        ####################################
        # --- Save results and metrics --- #
        ####################################
        # print(f'Removed invalid subsets: {set(data_generator.subsets) - set(valid_subsets)}')
        # data_generator.subsets = valid_subsets  # Update the data generator's subsets to only include valid ones
        # print(f"Valid subsets after filtering: {data_generator.subsets}")
        all_df = {}
        for subset in data_generator.subsets:
            results_df = create_results_dataframe(
                alphas=params['alphas'],
                time=all_timestamps[subset],
                horizon=H,
                ground_truths_dict=all_ground_truths[subset],
                prediction_intervals_dict=all_prediction_intervals[subset],
                scores_dict=all_q_preds['scores'][subset],
                qs_dict=all_q_preds['qs'][subset],
            ) if not twodim else create_2dresults_dataframe(
                alphas=params['alphas'],
                time=all_timestamps[subset],
                horizon=H,
                ground_truths_dict=all_ground_truths[subset],
                prediction_intervals_dict=all_prediction_intervals[subset],
                scores_dict=all_q_preds['scores'][subset],
                qs_dict=all_q_preds['qs'][subset],
            )
            all_df[subset] = results_df
            # save_json(results_df, save_path / f'results_{subset}.json', strftime=True, time_column_name='time')
        # Merge all subsets' dataframes, adding a 'subset' column
        merged_df = pd.concat(
            [df.assign(subset=subset) for subset, df in all_df.items()],
            ignore_index=True
        )
        save_json(merged_df, save_path / f'results_merged.json', strftime=True, time_column_name='time')
        # save_json(merged_df, save_path / f'results_merged.json', strftime=True, time_column_name='time')
    
    ######################
    # --- Evaluation --- #
    ######################
    print("Evaluating results...")
    
    # Loading results
    if merged_df is None:
        merged_df = pd.read_json(save_path / f'results_merged.json', orient='records')
        all_df = {subset: merged_df[merged_df['subset'] == subset].reset_index(drop=True) for subset in merged_df['subset'].unique()}
    
    all_results = {}
    all_results_t = {}
    if len(data_generator.subsets) == 0:
        raise ValueError("No subsets found in the data generator. Please check the data generator configuration.")
    else:
        for subset in tqdm(data_generator.subsets):
            evaluq = EvalUQ(results_df=all_df[subset], window=global_params['window'], risk_type=global_params['risk_type'], twodim=twodim, score_params=params['score_func_args'], start_t=params['start_t'])
            results = evaluq.metrics()
            results_t = evaluq.compute_time_varying_metrics()
            # save_results_flat_csv(results, csv_path=save_path / f"metrics_{subset}.csv")
            all_results[subset] = results
            all_results_t[subset] = results_t
        # aggregate results across subsets
        aggregated_results_mean, aggregated_results_std = aggregate_results(all_results)
        return save_path, aggregated_results_mean, aggregated_results_std, all_results, all_results_t, merged_df

    # if args.plot:
    #     subset = data_generator.subsets[0]  # Assuming we plot for the first subset
    #     plot_path = save_path / 'plots'
    #     plot_path.mkdir(parents=True, exist_ok=True)
    #     for alpha in params['alphas']:
    #         for h in range(H):
    #             plot_predictions_w_coverage(
    #                 alpha=alpha,
    #                 time_indexes=np.arange(T_obs)[:],
    #                 horizon=h,
    #                 timestamps=all_timestamps[subset],
    #                 ground_truths=all_ground_truths[subset],
    #                 prediction_intervals=all_prediction_intervals[subset],
    #                 rolling_window=15,
    #                 save_path=plot_path / f'alpha_{alpha}_horizon_{h}.png'
    #             )
    # np_dates = np.array(timestamps, dtype='datetime64[D]')
    # np.save('data/flusight_dates', np_dates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', '-m', type=str, default='cpt', help='method to run, options: cpt, faci, nexcp')
    parser.add_argument('--config_id', '-c', type=str, default='0', help='parameter set ID')
    parser.add_argument('--expid', '-e', type=str, default='0', help='experiment ID')
    parser.add_argument('--data', '-d', type=str, default='markovar', help='data generator class to use, options: markovar, flusight')
    parser.add_argument('--pretrain_fraction', '-pf', type=float, default=0, help='fraction of data to use for pretraining')
    parser.add_argument('--plot', action='store_true', help='whether to plot results')
    parser.add_argument('--no-run', '-n', action='store_false', dest='run', help='do not run the simulation (default: run simulation)')
    parser.add_argument('--consistency_exp', '-ce', action='store_true', help='whether to run consistency experiment')
    parser.add_argument('--workers', '-w', type=int, default=max(1, min(4, (os.cpu_count() or 1))), help='number of worker threads for concurrent runs')
    
    args = parser.parse_args()
    
    save_path, aggregated_results_mean, aggregated_results_std, all_results, all_results_t, merged_df = main(args)
    
    print(f"Saving results to {save_path}")
    save_pickle(aggregated_results_mean, save_path / "metrics.pkl")
    save_results_flat_csv(aggregated_results_mean, csv_path=save_path / "metrics.csv")
    save_pickle(aggregated_results_std, save_path / "metrics_std.pkl")
    save_results_flat_csv(aggregated_results_std, csv_path=save_path / "metrics_std.csv")
    save_pickle(all_results_t, save_path / f'metrics_time_varying.pkl')
    save_pickle(all_results, save_path / f'metrics_all_subsets.pkl')