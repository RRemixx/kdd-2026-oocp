import numpy as np
from tqdm import tqdm
from scipy.optimize import bisect
from pathlib import Path

from core import data
from core.data import TimeSeriesDataTemplate
from core.method.acMCP import AcMCP, learned_scores_df_to_array
from core.method.cpid import CPID_Scale
from core.method.cp_utils import *
from core.method.score_func import *
from core.method.optim import *
from core.eval.visualization import *
from core.eval.show_traj import *
from core.method.knn import *

import shutil
import time

def run_cpid_sim(data_generator: TimeSeriesDataTemplate, score_func_args, alpha=0.3, T_obs=1000, H=15, N=20, start_t=10, method_opt='cpt', plot=False, S_max=10, save_path="results", print_max_score=False, twodim=False, learned_scores=None, expand_boundaries=0, params=None, additional_context=None, debug=False):
    """
    Run ACI simulation with given parameters.
    
    Args:
        T_obs (int): observed length
        start_t (int): start time for evaluation
        cold_start_period (int): cold start period
        alpha (float): initial alpha value for ACI
        H (int): forecast horizon
        P (np.ndarray): transition matrix (if None, generates with default params)
        k (tuple): AR coefficients
        b (tuple): bias terms
        sigma (tuple): noise standard deviations
        N (int): number of sampled trajectories
        expand_boundaries (float): boundary expansion parameter
        save_path (str): path to save results
        twodim: 2d dataset
        learned_Fs: learned Fs from pretraining
        learned_max_scores: learned max scores from pretraining
    """
    # ACI initialization
    alpha_minimum, alpha_maximum = 0.0, 1.0
    scores = np.zeros((T_obs, H))
    alphas = np.zeros((T_obs, H))
    alphas_mid = np.zeros((T_obs, H))
    alphas_radius = np.zeros((T_obs, H))
    qs = np.zeros((T_obs, H))
    baseline_alphas = np.ones((T_obs, H)) * alpha
    
    n_traj_samples = params.get('optim_arg', {}).get('n_traj_samples', 200)
    all_traj_samples = np.full((T_obs, n_traj_samples, H), np.nan)
    all_traj_weights = []
    traj_sample_acc = np.zeros((T_obs, H))
    current_traj_scores = np.zeros(H)
    
    all_qts = np.zeros((T_obs, H))
    all_boundaries = np.zeros((T_obs, H, 2))
    
    d_forecasts = np.full((T_obs, H), np.nan)
    
    score_window = params.get('score_window', 50)
    
    S_max_vector = np.ones(H) * S_max
    if learned_scores is not None:
        for h in range(H):
            score_filter_dict = get_scores_filter_dict(h, t=None)
            learned_scores_h = get_proximal_scores(learned_scores, info_dict=score_filter_dict)
            if len(learned_scores_h) > 0:
                S_max_vector[h] = np.quantile(learned_scores_h, 0.95) * 1.2
    Fs = np.ones((T_obs, H)) * alpha
    
    covered_all = np.ones((T_obs, H))
    horizon_alphats = np.zeros(T_obs)
    samples = np.zeros((T_obs, N, H))
    if method_opt == 'cpid':
        optional_args = params.get('optional_args', {})
        optional_args['T'] = T_obs
        optional_args['max_score'] = S_max
        acis = [CPID_Scale(alpha=alpha, gamma=params['gamma'], power=params['power'], d_factor=params.get('d_factor', 1.0), score_window=score_window, horizon=h,optional_args=optional_args) for h in range(H)]

    ground_truths = np.zeros((T_obs, H))
    prediction_intervals = np.zeros((T_obs, H, N, 2))  # For storing prediction intervals
    timestamps = []
    score_func = score_function
    if twodim:
        samples = np.zeros((T_obs, N, H, 2))  # For storing samples in 2D
        ground_truths = np.zeros((T_obs, H, 2))  # For storing ground truths in 2D
        score_func = score_function_2d
    
    # parameters
    optim_arg = params.get('optim_arg', {})
    if 'alpha' not in optim_arg:
        optim_arg['alpha'] = alpha
    optim_arg['e_coeff'] = np.ones(H) * optim_arg['e_coeff']

    # Score function
    score_function_type = score_func_args.get('type', 'abs-r')
    score_function_optional_args = score_func_args.get('optional_args', {})
    
    if plot:
        save_dir = Path(save_path)
        if save_dir.exists() and save_dir.is_dir():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Debug
        last_boundaries = np.zeros((H, 2))
        empirical_dist = {}
        debug_plot_save_path = Path(f"{save_path}/debug_plots")
        if not debug_plot_save_path.exists():
            debug_plot_save_path.mkdir(parents=True, exist_ok=True)
        
        debug_plot_save_path = Path(f"{save_path}/debug_plots/{data_generator.current_subset}")
        debug_plot_save_path.mkdir(parents=True, exist_ok=True)
    
    # Get S_max from scores before start_t
    for t in range(start_t):
        y_truth, samp = data_generator.get_trajectory_samples(t, random=False)
        if samp.shape[0] > N:
            cur_samp = samp[:N]
        else:
            cur_samp = samp
        for h in range(H):
            # Compute scores and betas (scores and Fs computed at this time step may not be observed yet)
            scores[t, h] = score_func(y_truth, cur_samp, h, type=score_function_type, optional_args=score_function_optional_args)
    if params.get('dynamic_S_max', False):
        assert start_t > H
        for h in range(H):
            observed_scores = scores[:start_t-h, h]
            S_max_vector[h] = np.quantile(observed_scores, 0.95) * 2
            if method_opt == 'cpid' or method_opt == 'acmcp':
                for aci in acis:
                    aci.update_max_score(S_max_vector[h])
    # init qt for cpid
    if method_opt == 'cpid' or method_opt == 'acmcp':
        # assert start_t > H
        for h in range(H):
            # collect at least 5 scores from nearby horizons
            collected_scores = collect_observed_scores(scores, h, start_t, n=10)
            if learned_scores is not None:
                score_filter_dict = get_scores_filter_dict(h, t=None)
                additional_scores = get_proximal_scores(learned_scores, info_dict=score_filter_dict)
                collected_scores = np.concatenate((collected_scores, additional_scores))
            acis[h].init_qt(collected_scores)
                    
    # Main simulation loop
    if T_obs < 120:
        t_range = range(T_obs)
    else:
        t_range = tqdm(range(T_obs))
    for t in t_range:
        current_start_time = time.time()
        current_time = data_generator.get_reference_time(t)
        y_truth, samp = data_generator.get_trajectory_samples(t, random=False)

        cur_samp = samp
        samples[t] = cur_samp
        
        weights_t = None

        boundaries = np.zeros((H, 2))
        cur_Fs = np.ones((H)) * alpha_minimum
        cur_Fs_filled = np.full((H), np.nan)
        timestamps.append(current_time)
        enhanced_scores_t = [] 
        q_nexts = np.zeros(H)

        alpha_nexts = np.zeros(H)
        for h in range(H):
            # Compute scores and betas (scores and Fs computed at this time step may not be observed yet)
            scores[t, h] = score_func(y_truth, cur_samp, h, type=score_function_type, optional_args=score_function_optional_args)
            observed_scores = []
            if t > h:
                observed_scores = scores[:t-h, h]
            score_filter_dict = get_scores_filter_dict(h, t=t)
            if learned_scores is not None:
                learned_scores_h = get_proximal_scores(learned_scores, info_dict=score_filter_dict)
            else:
                learned_scores_h = np.array([])
            enhanced_scores = concat_scores(observed_scores, learned_scores_h, score_window)
            if t > start_t and params.get('dynamic_S_max', False):
                S_max_vector[h] = np.quantile(enhanced_scores, 0.95) * 2
            enhanced_scores_t.append(enhanced_scores)
            if len(enhanced_scores) > 1:
                cur_Fs[h] = compute_beta(enhanced_scores, scores[t, h], alpha_minimum=alpha_minimum, alpha_maximum=alpha_maximum)
                cur_Fs_filled[h] = cur_Fs[h]
                Fs[t, h] = cur_Fs[h]
                
            # Update coverage and boundaries
            low, high, alpha_next = 0.0, 1.0, alpha
            if t > h:
                covered = observed_scores[-1] <= qs[t-h-1, h]
                covered_all[t-h-1, h] = covered
                beta_t = Fs[t-h-1, h]
                
                low, high, alpha_next, q_next = acis[h].update(covered, scores=enhanced_scores_t[h])
            else:
                low, high, alpha_next, q_next = acis[h].blind_update(scores=enhanced_scores_t[h])
            q_nexts[h] = q_next
            baseline_alphas[t, h] = (low + high) / 2
            boundaries[h, 0] = low
            boundaries[h, 1] = high
            alpha_nexts[h] = alpha_next
                        
        if t > start_t and expand_boundaries >= 0:
            u_star, traj_samples, weights = cpid_optim_step(
                t=t, 
                start_t=start_t,
                scores=scores,
                boundaries=boundaries, 
                alpha=alpha, 
                optim_arg=params.get('optim_arg', None),
                additional_context=additional_context,
                ground_truths=ground_truths,
                predictions=np.mean(samples, axis=1),
                betas=Fs,
            )
            weights_t = weights
            all_traj_samples[t] = traj_samples
        else:
            u_star = q_nexts    

        all_traj_weights.append(weights_t)
        
        # Compute quantiles for each horizon
        for h in range(H):
            qs[t, h] = max(u_star[h], 0.0)
            if np.isnan(qs[t, h]):
                print('NaN qs detected')
                print('t:', t, 'h:', h, 'u_star:', u_star[h])
            # Store ground truth and prediction intervals
            ground_truths[t, h] = y_truth[h]
            if not twodim:
                current_prediction_intervals = inverse_score_function(qs[t, h], samples[t, :, h], score_function_type, score_function_optional_args)
                for i in range(len(current_prediction_intervals)):
                    prediction_intervals[t, h, i, 0] = current_prediction_intervals[i][0]
                    prediction_intervals[t, h, i, 1] = current_prediction_intervals[i][1]
        
        if alpha == 0.001 and plot:
            debug_knn(u_star, boundaries[:, 0], boundaries[:, 1], cur_Fs, all_traj_samples[t], traj_sample_acc, debug_plot_save_path, t, H)
        
        # for debug cpid
        all_qts[t] = q_nexts
        for h in range(H):
            all_boundaries[t, h, 0] = boundaries[h, 0]
            all_boundaries[t, h, 1] = boundaries[h, 1]
        
    
    if print_max_score:
        print(f'max scores for each horizon:  {np.max(scores, axis=0)}')
    
    if plot and alpha == 0.1:
        # Evaluation and plotting
        debug_cpid_boundaries(all_qts=all_qts, all_boundaries=all_boundaries, qs=qs, scores=scores,save_path=debug_plot_save_path)
        evaluate_coverage(scores, qs, alpha, save_path=f"{save_path}/coverage_{'expand_boundaries'}.png")
        evaluate_horizon_coverage(scores, qs, alpha, save_path=f"{save_path}/horizon_coverage_{'expand_boundaries'}.png")
        # Plot alphas for horizon 0 and horizon_alphats

    if twodim:
        return timestamps, ground_truths, (samples, qs), scores, qs
    return timestamps, ground_truths, prediction_intervals, scores, qs