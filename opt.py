import numpy as np
import yaml
from pathlib import Path
import copy
import pickle
import argparse
from pathlib import Path
import subprocess
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from hyperopt import fmin, tpe, hp, space_eval, Trials
from hyperopt import base

from core.data import *
from core.utils import *
from core.eval.eval_utils import *
from core.method.cp_ours import *
from main import main


def calculate_score(results: dict, weights: dict = None) -> float:
    # Set default weights if none are provided by the user.
    if weights is None:
        weights = {
            'interval_width': 1.0,
            'calibration_score_hc': 1.0,
            'var': 1.0,
            'exponential_risk': 1.0,
            'worst_case_risk': 1.0,
        }

    # A dictionary to hold the averaged value for each metric.
    averaged_metrics = {}

    # --- Extract and Average Metrics ---
    averaged_metrics['avg_cs_aa'] = results['calibration_score_hc']
    averaged_metrics['avg_risk_aa'] = results['avg_risk_aa']
    averaged_metrics['saregret_90'] = np.mean([v for h, v in results['saregret'][0.1].items()])
    averaged_metrics['saregret_10'] = np.mean([v for h, v in results['saregret'][0.9].items()])
    averaged_metrics['saregret_50'] = np.mean([v for h, v in results['saregret'][0.5].items()])

    # Average 'interval_width' over all alphas and horizons (h).
    # Structure: {alpha: {h: float}}
    iw_data = results.get('interval_width', {})
    iw_values = [width for h_dict in iw_data.values() for width in h_dict.values()]
    averaged_metrics['interval_width'] = sum(iw_values) / len(iw_values) if iw_values else 0.0

    cov_data = results.get('coverage', {})
    for alpha, h_dict in cov_data.items():
        covs = np.array([h_val for h_val in h_dict.values()])
        cov_gaps = np.maximum(0, 1 - alpha - covs)  # Only consider under-coverage
        averaged_metrics[f'cov_{alpha}'] = np.mean(cov_gaps)

    # --- Calculate Final Weighted Score ---
    total_score = sum(weights.get(key, 0.0) * value for key, value in averaged_metrics.items())
    return total_score


def _run_experiment(sampled_params):
    # print(f"Running experiment with params: {sampled_params}")
    sampled_params = copy.deepcopy(sampled_params)
    run_id = sampled_params.pop('_run_id', None)
    outputid = global_vars["outputid"] if run_id is None else f"{global_vars['outputid']}_{run_id}"
    # prepare yaml config file
    yaml_path = Path("configs") / global_vars['method']
    exp_params = load_yaml(yaml_path / f"{global_vars['config']}.yaml")
    if 'optim_arg' not in exp_params:
        exp_params['optim_arg'] = {}
    if 'optim_arg' in sampled_params:
        for k, v in sampled_params['optim_arg'].items():
            exp_params['optim_arg'][k] = v
        sampled_params.pop('optim_arg')
    if 'hyperopt_manual_args' in sampled_params:
        # Manually set args for hyperopt
        for k, v in sampled_params['hyperopt_manual_args'].items():
            if k == 'use_score':
                exp_params['optim_arg']['data_options']['scores']['use'] = v
            elif k == 'score_context_size':
                exp_params['optim_arg']['data_options']['scores']['context_size'] = v
            elif k == 'pred_context_size':
                exp_params['optim_arg']['data_options']['preds']['context_size'] = v
        sampled_params.pop('hyperopt_manual_args')
    for k, v in sampled_params.items():
        exp_params[k] = v
    with open(yaml_path / f'{outputid}.yaml', 'w') as f:
        yaml.dump(exp_params, f)
    print('Wrote config file:', yaml_path / f'{outputid}.yaml')
    # run the experiment
    args = {
        'config_id': outputid,
        'method': global_vars['method'],
        'data': global_vars['data'],
        'expid': outputid,
        'pretrain_fraction': global_vars['pf'],
        'plot': False,
        'run': True,
        'consistency_exp': False,
        'workers': 4,
    }
    _, aggregated_results_mean, aggregated_results_std, all_results, all_results_t, merged_df = main(args)
    print(f"Experiment {outputid} completed.")
    # load results
    results = aggregated_results_mean
    weights = {
        'avg_interval_width_aa': 0.01,
        'avg_cs_aa': 60,
        'saregret_90': 50,
        'saregret_10': 20,
        'saregret_50': 30,
        # 'avg_risk_aa': 80,
    }
    score = calculate_score(results=results, weights=weights)
    meta = {
        "outputid": outputid,
        "config_path": str((yaml_path / f"{outputid}.yaml").resolve()),
    }
    
    # Remove the temporary config file and results file to avoid using excess storage
    yaml_file_path = yaml_path / f"{outputid}.yaml"
    if yaml_file_path.exists():
        yaml_file_path.unlink()
    print(f"Completed experiment {outputid} with score: {score}")
    return score, meta


def score_func1(sampled_params):
    score, _ = _run_experiment(sampled_params)
    return score


def get_hyperopt_params():
    param_space1 = {
        # 'B': hp.choice('B', [5, 15, 25, 50]),
        # 'gamma': hp.uniform('gamma', 0.01, 0.2),
        # 'score_window': hp.choice('score_window', [5, 10, 15, 20, 50]),
        # 'power': hp.uniform('power', 0.3, 2),
        # The line `'e_coeff_init': hp.loguniform('e_coeff_init', -5, 5),` is defining a
        # hyperparameter for the Bayesian optimization process using the `hyperopt` library in Python.
        # 'e_coeff_init': hp.loguniform('e_coeff_init', -5, 5),
        'hyperopt_manual_args':{
            # 'use_score': hp.choice('use_score', [True, False]),
            # 'score_context_size': hp.choice('score_context_size', [3, 5]),
            'pred_context_size': hp.choice('pred_context_size', [3, 5]),
        },
        'optim_arg': {
            'weight_threshold': hp.uniform('weight_threshold', 0.1, 0.8),
            'r_coeff': hp.loguniform('r_coeff', -2, 2),
            'kernel_sigma': hp.uniform('kernel_sigma', 0.1, 2.0),
        },
        'd_factor': hp.choice('d_factor', [1.0]),
    }
    return score_func1, param_space1


# convert a saved trial object to acceptable format for hyperopt
def load_trials(trials_file_name, space):
    trials = pickle.load(open(trials_file_name, "rb"))
    best_trial = {}
    for key, val in trials.best_trial['misc']['vals']:
        best_trial[key] = val[0]
    print(space_eval(space, best_trial))


def _unwrap_misc_vals(misc_vals: dict):
    # hyperopt stores vals as lists; space_eval expects scalars for each key
    return {k: (v[0] if isinstance(v, list) and len(v) > 0 else v) for k, v in misc_vals.items()}


def _truncate_text(text: str, limit: int = 2000) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"...[truncated {len(text) - limit} chars]"


def _extract_stdio_from_runtime_error(err: Exception):
    # Our RuntimeError message includes "STDOUT:" and "STDERR:" blocks.
    msg = str(err)
    if "STDOUT:" not in msg or "STDERR:" not in msg:
        return "", ""
    try:
        stdout_part = msg.split("STDOUT:", 1)[1]
        stdout, stderr = stdout_part.split("STDERR:", 1)
        return stdout.strip(), stderr.strip()
    except Exception:
        return "", ""


def write_trials_log(trials, params_space, log_file: Path):
    columns = [
        "tid",
        "loss",
        "status",
        "params",
        "outputid",
        "returncode",
        "config_path",
        "results_path",
        "stdout",
        "stderr",
        "exception",
    ]
    rows = []
    for trial in trials.trials:
        tid = trial.get("tid")
        result = trial.get("result", {}) or {}
        loss = result.get("loss")
        status = result.get("status")
        try:
            params = space_eval(params_space, _unwrap_misc_vals(trial["misc"]["vals"]))
        except Exception:
            # Fall back to raw (unwrapped) values if space_eval can't resolve
            params = _unwrap_misc_vals(trial["misc"]["vals"])
        rows.append({
            "tid": tid,
            "loss": loss,
            "status": status,
            "params": json.dumps(params, sort_keys=True),
            "outputid": result.get("outputid", ""),
            "returncode": result.get("returncode", ""),
            "config_path": result.get("config_path", ""),
            "results_path": result.get("results_path", ""),
            "stdout": _truncate_text(result.get("stdout", "")),
            "stderr": _truncate_text(result.get("stderr", "")),
            "exception": result.get("exception", ""),
        })
    if len(rows) == 0:
        pd.DataFrame(columns=columns).to_csv(log_file, index=False)
        return
    pd.DataFrame(rows, columns=columns).to_csv(log_file, index=False)


def run_hyper_opt(trials_file:str, results_file:str, rounds:int, batch_size:int = 1, log_file: Path = None, rebuild_log_only: bool = False):
    # TODO: run without using trials_save_file is fine. When use the trials_save_file, exception is throwed.
    score_func, params_space = get_hyperopt_params()
    trials_path = Path(trials_file)
    if rebuild_log_only:
        if trials_path.exists():
            try:
                trials = pickle.load(open(trials_path, "rb"))
                if log_file is not None:
                    write_trials_log(trials, params_space, log_file)
                    print(f"Wrote log to {log_file}")
            except Exception as e:
                print(f"Warning: failed to load trials from {trials_path}: {e}.")
        else:
            print(f"No trials file found at {trials_path}")
        return

    if batch_size <= 1:
        trials = Trials()
        if trials_path.exists():
            try:
                trials = pickle.load(open(trials_path, "rb"))
            except Exception as e:
                print(f"Warning: failed to load trials from {trials_path}: {e}. Starting fresh.")
                trials = Trials()
        if log_file is not None and len(trials.trials) > 0:
            write_trials_log(trials, params_space, log_file)
        best_param = fmin(
            fn=score_func,
            space=params_space,
            max_evals=rounds,
            algo=tpe.suggest,
            trials=trials,
        )
        best_params_to_save = space_eval(params_space, best_param)
        save_pickle(best_params_to_save, results_file)
        if log_file is not None:
            write_trials_log(trials, params_space, log_file)
        try:
            pickle.dump(trials, open(trials_path, "wb"))
        except Exception as e:
            print(f"Warning: failed to save trials to {trials_path}: {e}")
        return

    # Parallel, batched hyperopt with TPE
    trials = Trials()
    if trials_path.exists():
        try:
            trials = pickle.load(open(trials_path, "rb"))
        except Exception as e:
            print(f"Warning: failed to load trials from {trials_path}: {e}. Starting fresh.")
            trials = Trials()
    if log_file is not None and len(trials.trials) > 0:
        write_trials_log(trials, params_space, log_file)

    domain = base.Domain(score_func, params_space)
    rng = np.random.default_rng()

    while len(trials.trials) < rounds:
        batch = min(batch_size, rounds - len(trials.trials))
        new_ids = list(range(len(trials.trials), len(trials.trials) + batch))
        new_trials = tpe.suggest(new_ids, domain, trials, rng)

        params_list = []
        for trial in new_trials:
            params = space_eval(params_space, _unwrap_misc_vals(trial['misc']['vals']))
            params['_run_id'] = f"trial{trial['tid']}"
            params_list.append(params)

        results_by_tid = {}
        with ThreadPoolExecutor(max_workers=batch) as executor:
            future_to_tid = {
                executor.submit(_run_experiment, params): trial['tid']
                for params, trial in zip(params_list, new_trials)
            }
            for future in as_completed(future_to_tid):
                tid = future_to_tid[future]
                try:
                    loss, meta = future.result()
                    results_by_tid[tid] = {
                        "loss": loss,
                        "status": base.STATUS_OK,
                        **meta,
                    }
                except Exception as e:
                    stdout, stderr = _extract_stdio_from_runtime_error(e)
                    results_by_tid[tid] = {
                        "loss": float("inf"),
                        "status": base.STATUS_FAIL,
                        "exception": repr(e),
                        "stdout": stdout,
                        "stderr": stderr,
                    }

        now = datetime.utcnow()
        for trial in new_trials:
            trial["state"] = base.JOB_STATE_DONE
            trial["result"] = results_by_tid.get(trial["tid"], {"loss": float("inf"), "status": base.STATUS_FAIL})
            trial["refresh_time"] = now

        trials.insert_trial_docs(new_trials)
        trials.refresh()

        try:
            pickle.dump(trials, open(trials_path, "wb"))
        except Exception as e:
            print(f"Warning: failed to save trials to {trials_path}: {e}")
        if log_file is not None:
            write_trials_log(trials, params_space, log_file)

    try:
        if len(trials.trials) == 0 or trials.best_trial is None:
            print("No successful trials to save.")
            return
        if getattr(trials, "argmin", None) is not None and len(trials.argmin) > 0:
            best_params_to_save = space_eval(params_space, trials.argmin)
        else:
            best_params_to_save = space_eval(params_space, _unwrap_misc_vals(trials.best_trial["misc"]["vals"]))
        save_pickle(best_params_to_save, results_file)
    except base.AllTrialsFailed:
        print("All trials failed; no best params to save.")
        return
    
global_vars = {
    'config': '3600',
    'outputid': '3600',
    'method': 'dtaci',
    'data': 'lane',
    'pf': 0.6,
}

# example run: python opt_new.py -r -e 1 -i 5 -c 1040 -o 1040 -m cpt -d cyclone -pf 0.66
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', '-r', action='store_true')
    parser.add_argument('--exp_id', '-e') # hyperopt experiment id
    parser.add_argument('--iters', '-i', default=20) # number of hyperopt iterations
    
    parser.add_argument('--config', '-c', type=str, default='2000')
    parser.add_argument('--outputid', '-o', type=str, default='2000')
    parser.add_argument('--method', '-m', type=str, default='cpt')
    parser.add_argument('--data', '-d', type=str, default='cycloneg')
    parser.add_argument('--pretrain_fraction', '-pf', type=float, default=0.66)
    parser.add_argument('--batch_size', '-b', type=int, default=1)  # number of parallel jobs per iteration
    parser.add_argument('--rebuild_log', action='store_true')  # rebuild CSV from trials file only

    cml_args = parser.parse_args()
    exp_id = int(cml_args.exp_id)
    iters = int(cml_args.iters)
    
    global_vars['config'] = cml_args.config
    global_vars['outputid'] = cml_args.outputid
    global_vars['method'] = cml_args.method
    global_vars['data'] = cml_args.data
    global_vars['pf'] = cml_args.pretrain_fraction

    results_dir = Path('results/hyperopt')
    if results_dir.exists() == False:
        results_dir.mkdir(parents=True, exist_ok=True)
    trials_file = results_dir / f'{exp_id}.trials'
    results_file = results_dir / f'{exp_id}.pkl'
    log_file = results_dir / f'{exp_id}_runs.csv'
    if cml_args.run:
        run_hyper_opt(
            trials_file=trials_file,
            results_file=results_file,
            rounds=iters,
            batch_size=cml_args.batch_size,
            log_file=log_file,
            rebuild_log_only=cml_args.rebuild_log,
        )
    else:
        best_param = load_pickle(results_file)
        print(best_param)
