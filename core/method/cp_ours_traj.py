import numpy as np
from tqdm import tqdm
from scipy.optimize import bisect
from pathlib import Path

from core import data
from core.data import TimeSeriesDataTemplate
from core.method.cpid import CPID
from core.method.cp_utils import *
from core.method.score_func import *
from core.method.optim import *
from core.eval.visualization import *
import shutil
import time

def remove_outliers(A: np.ndarray) -> np.ndarray:
    Q1 = np.quantile(A, 0.25)
    Q3 = np.quantile(A, 0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Remove outliers
    filtered_data = A[(A >= lower_bound) & (A <= upper_bound)]
    return filtered_data

def filter_and_trim_array(A: np.ndarray, quantile_level) -> np.ndarray:
    """
    Filters columns of array A based on a threshold vector T and trims them
    to the smallest resulting length.

    Args:
        A (np.ndarray): A NumPy array of shape (T_initial, H).

    Returns:
        np.ndarray: A NumPy array of shape (K, H), where K is the smallest
                    length after filtering, and K <= T_initial.
                    Returns an empty array if no values meet the criteria
                    or if the input array A is empty.
    """
    if A.size == 0:
        raise ValueError("Input array A is empty.")
    T = np.nanquantile(A, quantile_level, axis=0)

    H = A.shape[1]
    filtered_columns = []
    min_length = float('inf')

    # Step 1: Filter each column
    for h in range(H):
        # Keep values in A[:, h] that are smaller than T[h]
        filtered_values = A[A[:, h] <= T[h], h]
        filtered_columns.append(filtered_values)
        if len(filtered_values) < min_length:
            min_length = len(filtered_values)

    # Handle the case where no values met the criteria for any column
    if min_length == 0:
        raise ValueError("No values met the criteria for at least one column.")

    # Step 2: Trim all arrays to the smallest length
    # and Stack them horizontally
    result_columns = []
    for col_data in filtered_columns:
        result_columns.append(col_data[:min_length].reshape(-1, 1)) # Reshape to a column vector

    # If result_columns is empty (e.g., if H was 0), handle it
    if not result_columns:
        return np.array([])

    # Concatenate the trimmed columns horizontally
    result_array = np.hstack(result_columns)

    return result_array

def collect_scores(data_generator: TimeSeriesDataTemplate, score_func_args, T_obs=1000, H=15, N=20, start_t=10, twodim=False, subset_name=None):
    """
    Collection of information for ACI simulation. Get Fs and S_max from additional subsets.
    """ 
    if twodim:
        score_func = score_function_2d
    
    # Score function
    score_function_type = score_func_args.get('type', 'abs-r')
    score_function_optional_args = score_func_args.get('optional_args', {})
    
    # Main simulation loop
    record_list = []
    for t in range(T_obs):
        current_time = data_generator.get_reference_time(t)
        y_truth, samp = data_generator.get_trajectory_samples(t, random=False)
        samp = samp[:N]
        for h in range(H):
            record = {
                'time': current_time,
                'time_idx': t,
                'horizon': h,
                'score': score_func(y_truth, samp, h, type=score_function_type, optional_args=score_function_optional_args),
                'ground_truth': y_truth[h],
                'lat': np.nanmean(samp[:, h, 0]) if twodim else np.nanmean(samp[:, h]),
                'lon': np.nanmean(samp[:, h, 1]) if twodim else 0.0,
                'subset': subset_name,
            }
            record_list.append(record)
    return record_list

def alpha_selection(boundaries, alphats, alpha, Fs, expand_boundaries=0, optim_arg=None):
    ideal_coverage = np.ones(Fs.shape[1]) 
    rho_target = np.floor(alpha * Fs.shape[1])
    # oc_params = {'u_intvl_num': 50, 'invalid_state_penalty': 1e8}
    if expand_boundaries < 0:
        return np.mean(boundaries, axis=1), ideal_coverage
    lower_bounds = np.clip(boundaries[:, 0]-expand_boundaries, 0, 1)
    upper_bounds = np.clip(boundaries[:, 1]+expand_boundaries, 0, 1)
    if rho_target <= 0:
        print("rho is 0", end='; ')
        # return np.zeros(boundaries.shape[0])
        return np.ones(boundaries.shape[0]) * lower_bounds, ideal_coverage
    # alphas = np.mean(boundaries, axis=1)
    if optim_arg.get('monotone', False):
        enforce_monotone = optim_arg.get('enforce_monotone', False)
        lambda_inc = optim_arg.get('lambda_inc', 1.0)
        u_c = optim_arg.get('u_c', 1.0)
        alphas, _ = mcdp_ste_monotone(rho_target, optim_arg, Fs, lower_bounds, upper_bounds, enforce_monotone=enforce_monotone, lam_inc=lambda_inc, u_c=u_c)
    else:
        alphas, _, ideal_coverage = mcdp_ste(rho_target, optim_arg, Fs, lower_bounds, upper_bounds, alpha, alphats)
    # alphas = np.min(boundaries, axis=1)  # Ensure alphas are within the boundaries
    return np.array(alphas), ideal_coverage

def symmetric_boundaries(lower, upper, center):
    lower_dist = center - lower
    upper_dist = upper - center
    min_dist = min(lower_dist, upper_dist)
    new_lower = center - min_dist
    new_upper = center + min_dist
    return new_lower, new_upper

def get_ideal_dist_helper(mu, sigma=0.2, num_samples=100):
    return np.random.normal(mu, sigma, num_samples)

def get_ideal_dist(mus, sigma=0.2, num_samples=100):
    H = len(mus)
    ideal_dists = np.zeros((num_samples, H))
    for h in range(H):
        ideal_dists[:, h] = get_ideal_dist_helper(mus[h], sigma, num_samples)
    ideal_dists = np.clip(ideal_dists, 0, 1)
    return ideal_dists

def run_aci_simulation(data_generator: TimeSeriesDataTemplate, score_func_args, alpha=0.3, T_obs=1000, H=15, N=20, start_t=10, method_opt='cpt', plot=False, S_max=10, save_path="results", print_max_score=False, twodim=False, learned_scores=None, expand_boundaries=0, params=None, debug=False):
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
    
    score_window = params.get('score_window', 50)
    
    S_max_vector = np.ones(H) * S_max
    if learned_scores is not None:
        for h in range(H):
            score_filter_dict = {'horizon': (h, 0),}
            learned_scores_h = get_proximal_scores(learned_scores, info_dict=score_filter_dict)
            if len(learned_scores_h) > 0:
                S_max_vector[h] = np.quantile(learned_scores_h, 0.95) * 1.2
    Fs = np.ones((T_obs, H)) * alpha
    
    covered_all = np.ones((T_obs, H))
    horizon_alphats = np.zeros(T_obs)
    samples = np.zeros((T_obs, N, H))
    if method_opt == 'cpt':
        acis = [SetValuedACI(alpha_init=alpha, gamma=params['gamma'], alpha_min=alpha_minimum, alpha_max=alpha_maximum, power=params['power'], d_factor=params.get('d_factor', 1.0), score_window=score_window) for _ in range(H)]
    elif method_opt == 'dtaci':
        acis = [DtACI(alpha_init=alpha, alpha_min=alpha_minimum, alpha_max=alpha_maximum, gammas=0.001 * 2 ** np.arange(8), sigma=1/1000, eta=2.72, power=params['power'], d_factor=params.get('d_factor', 1.0), score_window=score_window) for _ in range(H)]
    elif method_opt == 'cpid':
        acis = [CPID(alpha=alpha, gamma=params['gamma'], power=params['power'], d_factor=params.get('d_factor', 1.0), score_window=score_window, optional_args=params.get('optional_args', {})) for _ in range(H)]

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

    # Debug
    last_boundaries = np.zeros((H, 2))
    empirical_dist = {}
    debug_plot_save_path = Path(f"{save_path}/debug_plots")
    if not debug_plot_save_path.exists():
        debug_plot_save_path.mkdir(parents=True, exist_ok=True)

    # Score function
    score_function_type = score_func_args.get('type', 'abs-r')
    score_function_optional_args = score_func_args.get('optional_args', {})
    
    if plot:
        save_dir = Path(save_path)
        if save_dir.exists() and save_dir.is_dir():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

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
    # init qt for cpid
    if method_opt == 'cpid':
        # assert start_t > H
        for h in range(H):
            # collect at least 5 scores from nearby horizons
            collected_scores = collect_observed_scores(scores, h, start_t, n=10)
            if learned_scores is not None:
                score_filter_dict = {'horizon': (h, 0),}
                additional_scores = get_proximal_scores(learned_scores, info_dict=score_filter_dict)
                collected_scores = np.concatenate((collected_scores, additional_scores))
            acis[h].init_qt(collected_scores)
                    
    # Main simulation loop
    for t in range(T_obs):
        current_start_time = time.time()
        current_time = data_generator.get_reference_time(t)
        y_truth, samp = data_generator.get_trajectory_samples(t, random=False)

        pred_betas_t = np.full((0, H), np.nan)
        additional_samp = None
        if samp.shape[0] > N:
            cur_samp = samp[:N]
            if params.get('use_additional_samples', False):
                additional_samp = samp[N:]
                pred_betas_t = np.full((len(additional_samp), H), np.nan)
        else:
            cur_samp = samp
        samples[t] = cur_samp

        boundaries = np.zeros((H, 2))
        cur_Fs = np.ones((H)) * alpha_minimum
        timestamps.append(current_time)
        enhanced_scores_t = [] 

        alpha_nexts = np.zeros(H)
        for h in range(H):
            # Compute scores and betas (scores and Fs computed at this time step may not be observed yet)
            scores[t, h] = score_func(y_truth, cur_samp, h, type=score_function_type, optional_args=score_function_optional_args)
            observed_scores = []
            if t > h:
                observed_scores = scores[:t-h, h]
            score_filter_dict = {'horizon': (h, 0),}
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
                Fs[t, h] = cur_Fs[h]
                if additional_samp is not None and len(additional_samp) > 0:
                    for k in range(additional_samp.shape[0]):
                        pred_score = score_func(additional_samp[k, h], cur_samp[:, h, :], type=score_function_type, optional_args=score_function_optional_args)
                        if np.isnan(pred_score):
                            continue
                        pred_betas_t[k, h] = compute_beta(enhanced_scores, pred_score, alpha_minimum=alpha_minimum, alpha_maximum=alpha_maximum)
            # Update coverage and boundaries
            if t > h:
                covered = observed_scores[-1] <= qs[t-h-1, h]
                covered_all[t-h-1, h] = covered
                beta_t = Fs[t-h-1, h]
                if method_opt == 'cpt' or method_opt == 'dtaci':
                    low, high, alpha_next = acis[h].update(covered, beta_t=beta_t)
                else:
                    low, high, alpha_next = acis[h].update(covered, scores=enhanced_scores_t[h]) # here cpid uses future scores but it's ok since we only use these scores to convert qt to alphat, later it will be converted back to qt using the same set of scores
            else:
                if method_opt == 'cpt' or method_opt == 'dtaci':
                    # coverage is not observed yet
                    low, high, alpha_next = acis[h].blind_update()
                else:
                    low, high, alpha_next = acis[h].blind_update(scores=enhanced_scores_t[h])
            if method_opt == 'cpt' or method_opt == 'dtaci':
                low, high = symmetric_boundaries(low, high, alpha_next)
            boundaries[h, 0] = low
            boundaries[h, 1] = high
            alpha_nexts[h] = alpha_next

            if t > h:
                horizon_cov_err = scores[t-h-1, h] > qs[t-h-1, h]
                # optim_arg['e_coeff'][h] += br_learning_rate * (horizon_cov_err - alpha)
        # prepare Fs and alpha selection
        t_end = t - H
        if t_end < 0:
            current_seq_Fs = np.full((0, H), np.nan)
        else:
            current_seq_Fs = Fs[max(t_end-params['B'], 0):t_end]
        similar_betas = np.full((0, H), np.nan)
        # merge
        processed_Fs = np.concatenate([current_seq_Fs, similar_betas, pred_betas_t], axis=0)
        u_star = alpha_nexts
        # if processed_Fs.shape[0] <= 0:
        #     u_star = alpha_nexts
        # else:
        #     if params['var_a'] < 0.99:
        #         processed_Fs = filter_and_trim_array(processed_Fs, params['var_a'])
        #     dist2true.append(np.nanmean(np.abs(processed_Fs - cur_Fs)))
        #     # ideal distribution
        #     if params.get('oracle_dist', False):
        #         processed_Fs = get_ideal_dist(cur_Fs, sigma=0.2, num_samples=processed_Fs.shape[0])
        #     u_star, ideal_coverage = alpha_selection(boundaries, alpha_nexts, horizon_alphat, processed_Fs, expand_boundaries, optim_arg)
        
        checkpoint3_time = time.time()
            
        alphas[t] = u_star
        alphas_mid[t] = (boundaries[:, 0] + boundaries[:, 1]) / 2
        alphas_radius[t] = (boundaries[:, 1] - boundaries[:, 0]) / 2
        
        # Compute quantiles for each horizon
        for h in range(H):
            enhanced_scores = enhanced_scores_t[h]
            if len(enhanced_scores) > 1:
                qs[t, h] = quantile_function(enhanced_scores, 1 - u_star[h], S_max_vector[h])
            else:
                qs[t, h] = S_max_vector[h]
            # Store ground truth and prediction intervals
            ground_truths[t, h] = y_truth[h]
            if not twodim:
                current_prediction_intervals = inverse_score_function(qs[t, h], samples[t, :, h], score_function_type, score_function_optional_args)
                for i in range(len(current_prediction_intervals)):
                    prediction_intervals[t, h, i, 0] = current_prediction_intervals[i][0]
                    prediction_intervals[t, h, i, 1] = current_prediction_intervals[i][1]

        current_end_time = time.time()
    
    
    if print_max_score:
        print(f'max scores for each horizon:  {np.max(scores, axis=0)}')
    
    if plot:
        # Evaluation and plotting
        evaluate_coverage(scores, qs, alpha, save_path=f"{save_path}/coverage_{'expand_boundaries'}.png")
        evaluate_horizon_coverage(scores, qs, alpha, save_path=f"{save_path}/horizon_coverage_{'expand_boundaries'}.png")
        # Plot alphas for horizon 0 and horizon_alphats
        plot_alpha_evolution(alphas, alphas_mid, alphas_radius, horizon_alphats, start_t, alpha, save_path)

    # print(f"Average dist to true betas: {np.nanmean(dist2true)}")

    # plot score trajectories and beta trajectories
    
    
    if twodim:
        return timestamps, ground_truths, (samples, qs)
    return timestamps, ground_truths, prediction_intervals

def get_delta(t, power, alphat, d_factor):
    mu_t = 1 / np.power(t, power) * d_factor
    delta_t = mu_t * min(1-alphat, alphat)
    return delta_t

class SetValuedACI:
    def __init__(self, alpha_init, gamma, alpha_min=0.0, alpha_max=1.0, power=1/2, d_factor=1.0, score_window=100):
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

    def blind_update(self):
        """ Perform a blind update of the ACI interval without considering coverage."""
        alpha_next = self.alpha
        delta = get_delta(self.t, power=self.power, alphat=alpha_next, d_factor=self.d_factor)
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
        delta = get_delta(self.t, power=self.power, alphat=alpha_next, d_factor=self.d_factor)
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
    def __init__(self, alpha_init, alpha_min=0.0, alpha_max=1.0, gammas=0.001 * 2 ** np.arange(8), sigma=1/1000, eta=2.72, power=1/2, d_factor=1.0, score_window=100):
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
        delta = get_delta(self.t, power=self.power, alphat=alpha_next, d_factor=self.d_factor)
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
        delta = get_delta(self.t, power=self.power, alphat=alpha_next, d_factor=self.d_factor)

        # If alpha_next is at the boundaries, set delta to 0
        if alpha_next == self.alpha_min or alpha_next == self.alpha_max:
            delta = 0.0
        # Set S_t = [alpha_next - delta, alpha_next + delta]
        lower = max(self.alpha_min, alpha_next - delta)
        upper = min(self.alpha_max, alpha_next + delta)

        # Update state
        self.t += 1

        return lower, upper, alpha_next