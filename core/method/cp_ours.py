import numpy as np
from tqdm import tqdm
from scipy.optimize import bisect
from pathlib import Path

from core import data
from core.data import TimeSeriesDataTemplate
from core.method.acMCP import AcMCP, learned_scores_df_to_array
from core.method.cpid import CPID
from core.method.cp_utils import *
from core.method.score_func import *
from core.method.optim import *
from core.eval.visualization import *
from core.eval.show_traj import *
from core.method.knn import *

import shutil
import time

def run_aci_simulation(data_generator: TimeSeriesDataTemplate, score_func_args, alpha=0.3, T_obs=1000, H=15, N=20, start_t=10, method_opt='cpt', plot=False, S_max=10, save_path="results", print_max_score=False, twodim=False, learned_scores=None, expand_boundaries=0, params=None, additional_context=None, debug=False):
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
    if method_opt == 'cpt':
        acis = [SetValuedACI(alpha_init=alpha, gamma=params['gamma'], alpha_min=alpha_minimum, alpha_max=alpha_maximum, power=params['power'], d_factor=params.get('d_factor', 1.0), score_window=score_window, max_delta=params.get('max_delta', 0.1)) for _ in range(H)]
    elif method_opt == 'dtaci':
        acis = [DtACI(alpha_init=alpha, alpha_min=alpha_minimum, alpha_max=alpha_maximum, gammas=0.001 * 2 ** np.arange(8), sigma=1/1000, eta=2.72, power=params['power'], d_factor=params.get('d_factor', 1.0), score_window=score_window, max_delta=params.get('max_delta', 0.1)) for _ in range(H)]
    elif method_opt == 'cpid':
        optional_args = params.get('optional_args', {})
        optional_args['T'] = T_obs
        optional_args['max_score'] = S_max
        acis = [CPID(alpha=alpha, gamma=params['gamma'], power=params['power'], d_factor=params.get('d_factor', 1.0), score_window=score_window, horizon=h,optional_args=optional_args) for h in range(H)]
    elif method_opt == 'acmcp':
        additional_dataset = None
        optional_args = params.get('optional_args', {})
        optional_args['T'] = T_obs
        optional_args['max_score'] = S_max
        if learned_scores is not None:
            additional_dataset = learned_scores_df_to_array(learned_scores, n_samples=50)
        acis = [AcMCP(alpha=alpha, gamma=params['gamma'], power=params['power'], d_factor=params.get('d_factor', 1.0), score_window=score_window, optional_args=optional_args, additional_datasets=additional_dataset) for _ in range(H)]
    elif method_opt == 'cfrnn':
        acis = [CFRNN(alpha_init=alpha, H=H, alpha_min=alpha_minimum, alpha_max=alpha_maximum, score_window=score_window) for _ in range(H)]

    ground_truths = np.zeros((T_obs, H))
    prediction_intervals = np.zeros((T_obs, H, N, 2))  # For storing prediction intervals
    timestamps = []
    score_func = score_function
    if twodim:
        samples = np.zeros((T_obs, N, H, 2))  # For storing samples in 2D
        ground_truths = np.zeros((T_obs, H, 2))  # For storing ground truths in 2D
        score_func = score_function_2d
    
    # parameters
    horizon_alphat = alpha
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
    
    
    dist2true = []
    
    duration_list = {
        'checkpoint1': [],
        'checkpoint2': [],
        'checkpoint3': [],
        'current': []
    }
    
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
                
                # update traj scores if applicable
                current_traj_scores[h] = 0.5*traj_sample_acc[t-h-1, h] + 0.5*current_traj_scores[h]
                
                if method_opt == 'cpt' or method_opt == 'dtaci' or method_opt == 'cfrnn':
                    low, high, alpha_next = acis[h].update(covered, beta_t=beta_t)
                elif method_opt == 'cpid':
                    low, high, alpha_next, q_next = acis[h].update(covered, scores=enhanced_scores_t[h]) # here cpid uses future scores but it's ok since we only use these scores to convert qt to alphat, later it will be converted back to qt using the same set of scores
                    q_nexts[h] = q_next
                elif method_opt == 'acmcp':
                    q_next, bias_correction = acis[h].update(covered, t=t, h=h, forecast_errors=scores, new_lr_feature=d_forecasts[t, :h])
                    d_forecasts[t, h] = bias_correction
                    alpha_next = q_next
            else:
                if method_opt == 'cpt' or method_opt == 'dtaci' or method_opt == 'cfrnn':
                    # coverage is not observed yet
                    low, high, alpha_next = acis[h].blind_update()
                elif method_opt == 'cpid':
                    low, high, alpha_next, q_next = acis[h].blind_update(scores=enhanced_scores_t[h])
                    q_nexts[h] = q_next
                elif method_opt == 'acmcp':
                    q_next, bias_correction = acis[h].blind_update()
                    d_forecasts[t, h] = bias_correction
                    alpha_next = q_next
                    
            baseline_alphas[t, h] = (low + high) / 2
            
            if method_opt == 'cpt' or method_opt == 'dtaci':
                low, high = symmetric_boundaries(low, high, alpha_next)
                # low, high = adaptive_interval_size(current_traj_scores[h], low, high, k=3)
            boundaries[h, 0] = low
            boundaries[h, 1] = high
            alpha_nexts[h] = alpha_next
                        
        if t > start_t and expand_boundaries >= 0:
            if params.get('oracle_dist', False):
                processed_Fs = get_ideal_dist(cur_Fs, sigma=0.2, num_samples=processed_Fs.shape[0])
            u_star, traj_samples, weights = optim_step(
                t=t, 
                start_t=start_t,
                scores=scores,
                cali_scores=enhanced_scores_t,
                boundaries=boundaries, 
                alphats=alpha_nexts, 
                alpha=alpha, 
                optim_arg=params.get('optim_arg', None),
                additional_context=additional_context,
                ground_truths=ground_truths,
                predictions=np.mean(samples, axis=1),
                betas=Fs,
            )
            weights_t = weights
            all_traj_samples[t] = traj_samples
            # Score for the samples based on pinball loss. Note that the current score is not accessible. We can only use these scores up to t-h-1.
            for h in range(H):
                if cur_Fs_filled[h] is np.nan:
                    continue
                try:
                    empirical_quantile = np.quantile(traj_samples[:, h], 1 - alpha_nexts[h])
                    traj_sample_acc[t, h] = pinball_loss(cur_Fs[h], empirical_quantile, 1 - alpha_nexts[h])
                except Exception as e:
                    pass
        else:
            u_star = alpha_nexts        

        all_traj_weights.append(weights_t)
        alphas[t] = u_star
        alphas_mid[t] = (boundaries[:, 0] + boundaries[:, 1]) / 2
        alphas_radius[t] = (boundaries[:, 1] - boundaries[:, 0]) / 2
        
        # Compute quantiles for each horizon
        for h in range(H):
            # For CPID, if alpha_next is -1, it means qt is larger than all observed scores, in this case, we simply use the qt from CPID to guarantee coverage.
            enhanced_scores = enhanced_scores_t[h]
            if method_opt == 'acmcp':
                qs[t, h] = u_star[h]
            else:
                if len(enhanced_scores) > 1:
                    qs[t, h] = quantile_function(enhanced_scores, 1 - u_star[h], S_max_vector[h])
                else:
                    qs[t, h] = S_max_vector[h]
            if method_opt == 'cpid' and alpha_nexts[h] == -1:
                qs[t, h] = q_nexts[h]
            qs[t, h] = max(qs[t, h], 0)  # Ensure non-negativity
            if np.isnan(qs[t, h]) or qs[t, h] < 0:
                print('NaN or negative qs detected')
                print('t:', t, 'h:', h, 'u_star:', u_star[h])
            # Store ground truth and prediction intervals
            ground_truths[t, h] = y_truth[h]
            if not twodim:
                current_prediction_intervals = inverse_score_function(qs[t, h], samples[t, :, h], score_function_type, score_function_optional_args)
                for i in range(len(current_prediction_intervals)):
                    prediction_intervals[t, h, i, 0] = current_prediction_intervals[i][0]
                    prediction_intervals[t, h, i, 1] = current_prediction_intervals[i][1]

        current_end_time = time.time()
        duration_list['current'].append(current_end_time - current_start_time)
        
        if alpha == 0.001 and plot:
            debug_knn(u_star, boundaries[:, 0], boundaries[:, 1], cur_Fs, all_traj_samples[t], traj_sample_acc, debug_plot_save_path, t, H)
        
        # for debug cpid
        all_qts[t] = q_nexts
        for h in range(H):
            if len(enhanced_scores_t[h]) > 1:
                all_boundaries[t, h, 0] = np.quantile(enhanced_scores_t[h], 1 - boundaries[h, 0])
                all_boundaries[t, h, 1] = np.quantile(enhanced_scores_t[h], 1 - boundaries[h, 1])
            else:
                all_boundaries[t, h, 0] = S_max_vector[h]
                all_boundaries[t, h, 1] = S_max_vector[h]
        
    
    if print_max_score:
        print(f'max scores for each horizon:  {np.max(scores, axis=0)}')
    
    if plot and alpha == 0.1:
        # Evaluation and plotting
        debug_cpid_boundaries(all_qts=all_qts, all_boundaries=all_boundaries, qs=qs, scores=scores, save_path=debug_plot_save_path)
        debug_samples(Fs=Fs, ground_truths=ground_truths, sampled_trajs=all_traj_samples, weights=all_traj_weights, save_path=save_path, samples2plot=20, alpha_selections=alphas, baseline_alpha_selections=baseline_alphas)
        evaluate_coverage(scores, qs, alpha, save_path=f"{save_path}/coverage_{'expand_boundaries'}.png")
        evaluate_horizon_coverage(scores, qs, alpha, save_path=f"{save_path}/horizon_coverage_{'expand_boundaries'}.png")
        # Plot alphas for horizon 0 and horizon_alphats
        plot_alpha_evolution(alphas, alphas_mid, alphas_radius, horizon_alphats, start_t, alpha, save_path)
        for h in range(H):
            plot_predictions_w_coverage(
                alpha=alpha,
                time_indexes=range(T_obs),
                horizon=h,
                timestamps=timestamps,
                ground_truths=ground_truths,
                prediction_intervals=prediction_intervals,
                rolling_window=15,
                save_path=Path(f"{save_path}/debug_samples/predictions_horizon_{h}.png"),
            )

    if twodim:
        return timestamps, ground_truths, (samples, qs), scores, qs
    return timestamps, ground_truths, prediction_intervals, scores, qs

class SetValuedACI:
    def __init__(self, alpha_init, gamma, alpha_min=0.0, alpha_max=1.0, power=1/2, d_factor=1.0, score_window=100, max_delta=0.1):
        """
        Args:
          alpha_init (float): initial miscoverage rate α₁
          gamma      (float): learning rate γ for ACI step
          alpha_min  (float): lower bound for α_t (e.g. 0)
          alpha_max  (float): upper bound for α_t (e.g. 1)
        """
        self.alpha = alpha_init    # Target miscoverage rate
        self.alpha_t = alpha_init  # Current alpha value
        self.gamma = gamma         # Learning rate
        self.alpha_min = alpha_min # Minimum allowed alpha
        self.alpha_max = alpha_max # Maximum allowed alpha 
        self.t = 1                # Time step
        self.power = power         # Power for delta calculation
        self.d_factor = d_factor   # Factor for delta calculation
        self.window_length = score_window # Window length for score calculation
        self.max_delta = max_delta # Maximum delta value

    def blind_update(self):
        """ Perform a blind update of the ACI interval without considering coverage."""
        alpha_next = self.alpha
        delta = get_delta(self.t, power=self.power, alphat=alpha_next, d_factor=self.d_factor, max_delta=self.max_delta)
        lower = max(self.alpha_min, alpha_next - delta)
        upper = min(self.alpha_max, alpha_next + delta)
        self.t += 1
        return lower, upper, alpha_next

    def update(self, in_interval, beta_t=None):
        """
        Perform the ACI update and return an interval of plausible alpha values.

        Args:
        in_interval (bool): True if y_t ∈ C_{α_t}(x_t), False otherwise

        Returns:
        lower (float): Lower bound of the alpha interval S_t
        upper (float): Upper bound of the alpha interval S_t
        """
        # Indicator for miscoverage: 1 if outside interval, else 0
        err_t = 0.0 if in_interval else 1.0

        # ACI subgradient step: α_{t+1} = α_t - γ (err_t - α)
        alpha_next = self.alpha_t - self.gamma * (err_t - self.alpha)

        # Clip to [alpha_min, alpha_max]
        alpha_next = min(max(alpha_next, self.alpha_min), self.alpha_max)
        delta = get_delta(self.t, power=self.power, alphat=alpha_next, d_factor=self.d_factor, max_delta=self.max_delta)
        # If alpha_next is at the boundaries, set delta to 0
        if alpha_next == self.alpha_min or alpha_next == self.alpha_max:
            delta = 0.0
        # Set S_t = [alpha_next - delta, alpha_next + delta]
        lower = max(self.alpha_min, alpha_next - delta)
        upper = min(self.alpha_max, alpha_next + delta)

        # Update state
        self.alpha_t = alpha_next
        self.t += 1

        return lower, upper, alpha_next

def vec_zero_min(x):
    return np.minimum(x, 0)

def pinball(u, alpha):
    return alpha * u - vec_zero_min(u)

class DtACI:
    def __init__(self, alpha_init, alpha_min=0.0, alpha_max=1.0, gammas=0.001 * 2 ** np.arange(8), sigma=1/1000, eta=2.72, power=1/2, d_factor=1.0, score_window=100, max_delta=0.1):
        """
        Args:
          alpha_init (float): initial miscoverage rate α₁
          gamma      (float): learning rate γ for ACI step
          alpha_min  (float): lower bound for α_t (e.g. 0)
          alpha_max  (float): upper bound for α_t (e.g. 1)
        """
        self.alpha = alpha_init    # Target miscoverage rate
        self.gammas = np.array(gammas)       # List of learning rates
        self.sigma = sigma         # Sigma value
        self.eta = eta             # Eta value
        self.power = power         # Power for delta calculation
        self.d_factor = d_factor   # Factor for delta calculation
        self.window_length = score_window # Window length for score calculation
        
        self.alpha_t = alpha_init  # Current alpha value
        self.alpha_min = alpha_min # Minimum allowed alpha
        self.alpha_max = alpha_max # Maximum allowed alpha
        self.t = 1                 # Time step
        self.max_delta = max_delta # Maximum delta value
        # init
        self.k = len(gammas)
        self.expert_alphas = np.full(self.k, alpha_init)
        self.expert_ws = np.ones(self.k)
        self.cur_expert = np.random.choice(self.k)
        self.expert_cumulative_losses = np.zeros(self.k)
        self.expert_probs = np.full(self.k, 1/self.k)
    
    def blind_update(self):
        """ Perform a blind update of the ACI interval without considering coverage."""
        alpha_next = self.alpha
        delta = get_delta(self.t, power=self.power, alphat=alpha_next, d_factor=self.d_factor, max_delta=self.max_delta)
        lower = max(self.alpha_min, alpha_next - delta)
        upper = min(self.alpha_max, alpha_next + delta)
        self.t += 1
        return lower, upper, alpha_next

    def update(self, in_interval, beta_t=None):
        """
        Perform the DtACI update and return an interval of plausible alpha values.

        Args:
        in_interval (bool): True if y_t ∈ C_{α_t}(x_t), False otherwise

        Returns:
        lower (float): Lower bound of the alpha interval S_t
        upper (float): Upper bound of the alpha interval S_t
        """
        expert_losses = pinball(beta_t - self.expert_alphas, self.alpha)
        
        # update expert alphas
        prev_expert_idx = self.cur_expert
        prev_chosen_alpha = self.expert_alphas[prev_expert_idx]
        self.expert_alphas = self.expert_alphas + self.gammas * (self.alpha - (self.expert_alphas > beta_t).astype(float))
        # update for previous expert
        err_star = 0.0 if in_interval else 1.0
        self.expert_alphas[prev_expert_idx] = prev_chosen_alpha + self.gammas[prev_expert_idx] * (self.alpha - err_star)
        
        # update expert weights
        if self.eta < np.inf:
            expert_bar_ws = self.expert_ws * np.exp(-self.eta * expert_losses)
            expert_next_ws = (1 - self.sigma) * expert_bar_ws / np.sum(expert_bar_ws) + self.sigma / self.k
            expert_probs = expert_next_ws / np.sum(expert_next_ws)
            self.cur_expert = np.random.choice(self.k, p=expert_probs)
            self.expert_ws = expert_next_ws
        else:
            self.expert_cumulative_losses += expert_losses
            self.cur_expert = np.argmin(self.expert_cumulative_losses)
        
        # get next alpha
        alpha_next = self.expert_alphas[self.cur_expert]
        
        scale_factor = 1 + 1 / (self.t - max(self.t-self.window_length,0) + 1)
        self.alpha_t = alpha_next
        alpha_next = 1 + (alpha_next-1) * scale_factor
        alpha_next = min(max(alpha_next, self.alpha_min), self.alpha_max)
        
        # get interval
        delta = get_delta(self.t, power=self.power, alphat=alpha_next, d_factor=self.d_factor, max_delta=self.max_delta)

        # If alpha_next is at the boundaries, set delta to 0
        if alpha_next == self.alpha_min or alpha_next == self.alpha_max:
            delta = 0.0
        # Set S_t = [alpha_next - delta, alpha_next + delta]
        lower = max(self.alpha_min, alpha_next - delta)
        upper = min(self.alpha_max, alpha_next + delta)

        # Update state
        self.t += 1

        return lower, upper, alpha_next

class CFRNN:
    def __init__(self, alpha_init, H, alpha_min=0.0, alpha_max=1.0, score_window=100,):
        """
        Args:
          alpha_init (float): initial miscoverage rate α₁
          alpha_min  (float): lower bound for α_t (e.g. 0)
          alpha_max  (float): upper bound for α_t (e.g. 1)
        """
        self.alpha = alpha_init    # Target miscoverage rate
        self.alpha_min = alpha_min # Minimum allowed alpha
        self.alpha_max = alpha_max # Maximum allowed alpha 
        self.window_length = score_window # Window length for score calculation
        self.horizon = H
        self.alpha_adjusted = self.alpha / H
        
    def blind_update(self):
        alpha_next = self.alpha_adjusted
        return 0, 0, alpha_next

    def update(self, in_interval, beta_t=None):
        return self.blind_update()
