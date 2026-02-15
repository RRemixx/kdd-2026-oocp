import numpy as np
from tqdm import tqdm
from scipy.optimize import bisect
from pathlib import Path

from core.data import TimeSeriesDataTemplate
from core.method.cp_utils import quantile_function, concat_scores
from core.method.score_func import *
from core.method.optim import *
from core.eval.visualization import *
import shutil

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
    T = np.quantile(A, quantile_level, axis=0)

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


def information_collection(data_generator: TimeSeriesDataTemplate, score_func_args, T_obs=1000, H=15, N=20, start_t=10, twodim=False):
    """
    Collection of information for ACI simulation. Get Fs and S_max from additional subsets.
    """    
    # ACI initialization
    alpha_minimum, alpha_maximum = 0.0, 1.0
    scores = np.zeros((T_obs, H))
    alphas = np.zeros((T_obs, H))
    qs = np.zeros((T_obs, H))
    Fs = np.zeros((T_obs, H))
    covered_all = np.zeros((T_obs, H))
    samples = np.zeros((T_obs, N, H))

    ground_truths = np.zeros((T_obs, H))
    prediction_intervals = np.zeros((T_obs, H, N, 2))  # For storing prediction intervals
    timestamps = []
    score_func = score_function
    if twodim:
        samples = np.zeros((T_obs, N, H, 2))  # For storing samples in 2D
        ground_truths = np.zeros((T_obs, H, 2))  # For storing ground truths in 2D
        score_func = score_function_2d
    
    # Score function
    score_function_type = score_func_args.get('type', 'abs-r')
    score_function_optional_args = score_func_args.get('optional_args', {})
    
    # Main simulation loop
    for t in range(T_obs):
        current_time = data_generator.get_reference_time(t)
        y_truth, samp = data_generator.get_trajectory_samples(t, random=False)
        samples[t] = samp
        cur_Fs = np.ones((H)) * alpha_minimum
        for h in range(H):
            scores[t, h] = score_func(y_truth, samp, h, type=score_function_type, optional_args=score_function_optional_args)
            if t >= start_t:
                def solve(beta):
                    q = quantile_function(scores[:t, h], 1 - beta, 1e6)
                    return q - scores[t, h]
                try:
                    cur_Fs[h] = bisect(solve, alpha_minimum, alpha_maximum, xtol=1e-2)
                except ValueError:
                    print(f"Warning: bisect failed for t={t}, h={h}. Using alpha_minimum.")
        Fs[t] = cur_Fs
    # Collect Fs and scores
    Fs = Fs[start_t:]
    return Fs, scores

def information_aggregation(Fs_list, scores_list):
    # Stack Fs and scores arrays
    aggregated_Fs = np.vstack(Fs_list)
    aggregated_scores = np.vstack(scores_list)
    # get max scores for each horizon
    max_scores = np.percentile(aggregated_scores, 90, axis=0) * 1.1  # 90% quantile, scaled by 1.1
    return aggregated_Fs, aggregated_scores, max_scores

def alpha_selection(boundaries, alpha, Fs, expand_boundaries=0, optim_arg=None):    
    rho_target = np.floor(alpha * Fs.shape[1])
    oc_params = {'u_intvl_num': 50, 'invalid_state_penalty': 1e8}
    if expand_boundaries < 0:
        return np.mean(boundaries, axis=1)
    lower_bounds = np.clip(boundaries[:, 0]-expand_boundaries, 0, 1)
    upper_bounds = np.clip(boundaries[:, 1]+expand_boundaries, 0, 1)
    if rho_target < 0:
        print("rho is 0", end='; ')
        # return np.zeros(boundaries.shape[0])
        return np.ones(boundaries.shape[0]) * lower_bounds
    # alphas = np.mean(boundaries, axis=1)
    if optim_arg is not None:
        enforce_monotone = optim_arg.get('enforce_monotone', False)
        lambda_inc = optim_arg.get('lambda_inc', 1.0)
        u_c = optim_arg.get('u_c', 1.0)
        alphas, _ = mcdp_ste_monotone(rho_target, oc_params, Fs, lower_bounds, upper_bounds, enforce_monotone=enforce_monotone, lam_inc=lambda_inc, u_c=u_c)
    else:
        alphas, _ = mcdp_ste(rho_target, oc_params, Fs, lower_bounds, upper_bounds)
    # alphas = np.min(boundaries, axis=1)  # Ensure alphas are within the boundaries
    return np.array(alphas)

def run_aci_simulation(data_generator: TimeSeriesDataTemplate, score_func_args, alpha=0.3, T_obs=1000, H=15, N=20, start_t=10, expand_boundaries=0, power=0.5, B=50, var_a=0.5, br_learning_rate=0.1, gamma=0.05, d_factor=1.0, score_window=1000, plot=False, S_max=10, save_path="results", print_max_score=False, twodim=False, learned_Fs=None, learned_scores=None, learned_max_scores=None, optim_arg=None):
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
    
    S_max_vector = np.ones(H) * S_max
    if learned_max_scores is not None:
        S_max_vector = learned_max_scores
    Fs = np.ones((T_obs, H)) * alpha
    learned_Fs_length = learned_Fs.shape[0] if learned_Fs is not None else 0
    if learned_Fs is not None:
        # concatenate learned_Fs along the first axis
        Fs = np.concatenate([learned_Fs, Fs], axis=0)
    
    covered_all = np.ones((T_obs, H))
    horizon_alphats = np.zeros(T_obs)
    samples = np.zeros((T_obs, N, H))
    acis = [DtACI(alpha_init=alpha, gamma=gamma, alpha_min=alpha_minimum, alpha_max=alpha_maximum, power=power, d_factor=d_factor) for _ in range(H)]

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

    # Debug
    last_boundaries = np.zeros((H, 2))
    
    # Score function
    score_function_type = score_func_args.get('type', 'abs-r')
    score_function_optional_args = score_func_args.get('optional_args', {})
    
    if plot:
        save_dir = Path(save_path)
        if save_dir.exists() and save_dir.is_dir():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Main simulation loop
    for t in range(T_obs):
        current_time = data_generator.get_reference_time(t)
        y_truth, samp = data_generator.get_trajectory_samples(t, random=False)
        samples[t] = samp
        
        boundaries = np.zeros((H, 2))
        cur_Fs = np.ones((H)) * alpha_minimum
        
        timestamps.append(current_time)
        for h in range(H):
            # Compute part (scores and Fs computed at this time step may not be observed yet)
            scores[t, h] = score_func(y_truth, samp, h, type=score_function_type, optional_args=score_function_optional_args)
            observed_scores = []
            if t > h:
                observed_scores = scores[:t-h, h]
            enhanced_scores = concat_scores(observed_scores, learned_scores[:, h] if learned_scores is not None else None, score_window)
            if len(enhanced_scores) > 1:
                def solve(beta):
                    q = quantile_function(enhanced_scores, 1 - beta, S_max_vector[h])
                    return q - scores[t, h]
                try:
                    cur_Fs[h] = bisect(solve, alpha_minimum, alpha_maximum, xtol=1e-2)
                except ValueError:
                    cur_Fs[h] = alpha_minimum
                Fs[t+learned_Fs_length] = cur_Fs
                
            # Update coverage and boundaries
            if t > h:
                covered = observed_scores[-1] <= qs[t-h-1, h]
                covered_all[t-h-1, h] = covered
                low, high = acis[h].update(covered)
            else:
                # coverage is not observed yet
                low, high = acis[h].blind_update()
            boundaries[h, 0] = low
            boundaries[h, 1] = high

        # update horizon coverage (can only be computed and updated after t >= H)
        if t >= H:
            horizon_covered = []
            for h in range(H):
                horizon_covered.append(scores[t-h-1, h] <= qs[t-h-1, h])
            horizon_coverage = np.mean(horizon_covered)
            horizon_alphat = horizon_alphat + br_learning_rate * (horizon_coverage - 1 + alpha)
            horizon_alphat = np.clip(horizon_alphat, 0, 1.0)
        
        # prepare Fs and alpha selection
        t_end = t + learned_Fs_length - H
        if t_end <= 0:
            u_star = np.mean(boundaries, axis=1)
        else:
            processed_Fs = Fs[max(t_end-B, 0):t_end]
            processed_Fs = filter_and_trim_array(processed_Fs, var_a)
            u_star = alpha_selection(boundaries, horizon_alphat, processed_Fs, expand_boundaries, optim_arg)
        alphas[t] = u_star
        alphas_mid[t] = (boundaries[:, 0] + boundaries[:, 1]) / 2
        alphas_radius[t] = (boundaries[:, 1] - boundaries[:, 0]) / 2
        
        # Compute quantiles for each horizon
        for h in range(H):
            observed_scores = []
            if t > h:
                observed_scores = scores[:t-h, h]
            enhanced_scores = concat_scores(observed_scores, learned_scores[:, h] if learned_scores is not None else None, score_window)
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
    
    if plot:
        # prepare the additional information for the plot
        # Print relevant information for debugging
        print(f"Time step: {t}")
        print("Horizon alphas:", horizon_alphats[t-1])
        print("Quantiles (qs) this step:", qs[t])
        print("Previous alpha (t-1):", alphas[t-1] if t > 0 else "N/A")
        low_bounds = last_boundaries[:, 0]
        high_bounds = last_boundaries[:, 1]
        print("Last lower bounds:", low_bounds)
        print("Last upper bounds:", high_bounds)
        print("-" * 60)
        # plot_forecast_step(t, data_generator.get_observations(t, 5), data_generator.get_ground_truth(t), scores, qs, samp, alphas[t-1], save_path)
        last_boundaries = boundaries.copy()
    
    if print_max_score:
        print(f'max scores for each horizon:  {np.max(scores, axis=0)}')
    
    if plot:
        # Evaluation and plotting
        evaluate_coverage(scores, qs, alpha, save_path=f"{save_path}/coverage_{expand_boundaries}.png")
        evaluate_horizon_coverage(scores, qs, alpha, save_path=f"{save_path}/horizon_coverage_{expand_boundaries}.png")
        # Plot alphas for horizon 0 and horizon_alphats
        plot_alpha_evolution(alphas, alphas_mid, alphas_radius, horizon_alphats, start_t, alpha, save_path)
    if twodim:
        return timestamps, ground_truths, (samples, qs)
    return timestamps, ground_truths, prediction_intervals

def vec_zero_min(x):
    return np.minimum(x, 0)

def pinball(u, alpha):
    return alpha * u - vec_zero_min(u)

class DtACI:
    def __init__(self, alpha_init, alpha_min=0.0, alpha_max=1.0, gammas=0.001 * 2 ** np.arange(8), sigma=1/1000, eta=2.72, power=1/2, d_factor=1.0):
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
        
        self.alpha_t = alpha_init  # Current alpha value
        self.alpha_min = alpha_min # Minimum allowed alpha
        self.alpha_max = alpha_max # Maximum allowed alpha
        self.t = 1                 # Time step
        
        # init
        self.k = len(gammas)

        self.gammas = []

        self.expert_alphas = np.full(self.k, alpha_init)
        self.expert_ws = np.ones(self.k)
        self.cur_expert = np.random.choice(self.k)
        self.expert_cumulative_losses = np.zeros(self.k)
        self.expert_probs = np.full(self.k, 1/self.k)

    def update(self, beta_t):
        """
        Perform the DtACI update and return an interval of plausible alpha values.

        Args:
        in_interval (bool): True if y_t ∈ C_{α_t}(x_t), False otherwise

        Returns:
        lower (float): Lower bound of the alpha interval S_t
        upper (float): Upper bound of the alpha interval S_t
        """
        err_t = 0.0 if beta_t >= self.alpha_t else 1.0
        expert_losses = pinball(beta_t - self.expert_alphas, self.alpha)
        
        # update expert alphas
        self.expert_alphas = self.expert_alphas + self.gammas * (self.alpha - (self.expert_alphas > beta_t).astype(float))
        
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
        
        # get interval
        # Define shrinking radius δ_t = (alpha_max - alpha_min) / sqrt(t)
        D = (self.alpha_max - self.alpha_min) * self.d_factor
        delta = D / np.power(self.t, self.power)  # Default square root, but can be modified by changing 1/2

        # If alpha_next is at the boundaries, set delta to 0
        if alpha_next == self.alpha_min or alpha_next == self.alpha_max:
            delta = 0.0
        # Set S_t = [alpha_next - delta, alpha_next + delta]
        lower = max(self.alpha_min, alpha_next - delta)
        upper = min(self.alpha_max, alpha_next + delta)

        # Update state
        self.alpha_t = alpha_next
        self.t += 1

        return lower, upper