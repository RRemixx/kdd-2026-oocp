import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.data import TimeSeriesDataTemplate
from core.method.score_func import *
from core.method.optim import *

def get_delta(t, power, alphat, d_factor, max_delta=0.1):
    mu_t = 1 / np.power(t, power) * d_factor
    delta_t = mu_t * min(1-alphat, alphat)
    delta_t = min(delta_t, max_delta)
    return delta_t

def adaptive_interval_size(score, lower, upper, k):
    scale_factor = np.exp(-k * score)
    center = (lower + upper) / 2
    half_width = (upper - lower) / 2 * scale_factor
    new_lower = center - half_width
    new_upper = center + half_width
    return new_lower, new_upper

def pinball_loss(y_true, y_pred, alpha):
    """
    Compute the pinball loss for quantile regression.
    
    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted quantile values.
        alpha (float): Quantile level (between 0 and 1).
    Returns:
        float: Pinball loss value.
    """
    delta = y_true - y_pred
    loss = np.maximum(alpha * delta, (alpha - 1) * delta)
    return np.mean(loss)

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
                'ground_truth_0': y_truth[h][0] if twodim else y_truth[h],
                'ground_truth_1': y_truth[h][1] if twodim else 0.0,
                'prediction_mean_0': np.nanmean(samp[:, h, 0]) if twodim else np.nanmean(samp[:, h]),
                'prediction_mean_1': np.nanmean(samp[:, h, 1]) if twodim else 0.0,
                'lat': np.nanmean(samp[:, h, 0]) if twodim else np.nanmean(samp[:, h]),
                'lon': np.nanmean(samp[:, h, 1]) if twodim else 0.0,
                'subset': subset_name,
            }
            record_list.append(record)
    return record_list

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

def plot_nonoverlapping_batches(A_, n_trajs, n_steps, figure_path):
    A = A_[::n_steps]
    T, H = A.shape
    horizon = np.arange(H)

    for t in range(0, T - n_trajs + 1, n_trajs):
        plt.figure(figsize=(6, 4))
        for k in range(n_trajs):
            plt.plot(horizon, A[t + k], alpha=0.7)

        plt.title(f"Trajectories t = {t} to {t + n_trajs - 1}")
        plt.grid(True)
        plt.savefig(figure_path / f'{t}_{t + n_trajs - 1}.jpg')
        plt.close()

def collect_observed_scores(scores, h, start_t, n=10):
    """
    Collect observed scores for a given horizon h and time start_t.

    Parameters
    ----------
    scores : np.ndarray
        Array of shape (T, H), where T is time and H is horizon.
    h : int
        Target horizon index (0-based).
    start_t : int
        Current time index t. Observable scores at horizon h are scores[:t-h, h].
    n : int
        Threshold number of scores. If there are fewer than n at horizon h,
        we borrow from horizons h-1, h-2, ... until we reach (at least) n
        or run out of horizons.

    Returns
    -------
    np.ndarray
        1D array of collected scores (length min(total_available, n)).
    """
    T, H = scores.shape
    if not (0 <= h < H):
        raise ValueError(f"h must be in [0, {H-1}], got {h}")

    collected = []
    total = 0

    # Go from horizon h down to 0
    for cur_h in range(h, -1, -1):
        # At horizon cur_h, observable scores at time start_t:
        # scores[:start_t - cur_h, cur_h]
        end_idx = start_t - cur_h
        if end_idx <= 0:
            continue  # nothing observable yet for this horizon

        obs = scores[:end_idx, cur_h]
        if obs.size == 0:
            continue

        collected.append(obs)
        total += obs.size

        if total >= n:
            break

    if not collected:
        # No observable scores at all
        return np.array([])

    collected = np.concatenate(collected)

    # If we got more than n, keep the first n so we preserve as much as possible
    # from the target horizon h (which we added first).
    if collected.size > n:
        collected = collected[:n]

    return collected

    
def quantile_function(scores, quantile_level_, S_max):
    """
    Calculate the quantile threshold.
    
    Args:
        scores (array): Non-conformity scores
        alpha (float): Target miscoverage rate
        S_max (float): Maximum score value
    
    Returns:
        float: Quantile threshold
    """
    # finite sample correction
    n = len(scores)
    quantile_level = (n + 1) * quantile_level_ / n
    # Calculate the quantile threshold
    if quantile_level >= 1:
        return S_max
    if quantile_level <= 0:
        return 0.0
    q = np.quantile(scores, quantile_level)
    return min(q, S_max)

def get_proximal_scores(scores_df, info_dict, num_scores=10, return_scores=True):
    """
    Retrieve the most relevant non-conformity scores from a DataFrame.
    
    Args:
        scores_df (pd.DataFrame): DataFrame containing non-conformity scores with columns 'time' and 'score'.
        info_dict (dict): Dictionary of information to match against DataFrame columns, e.g., {'h': (1, 1), 't': (1, 2)}.
        num_scores (int): Number of recent scores to retrieve.

    Returns:
        np.ndarray: Array of the most relevant non-conformity scores.
    """
    filtered_df = scores_df.copy()
    for key, value in info_dict.items():
        if key in scores_df.columns:
            val, radius = value
            filtered_df = filtered_df[(filtered_df[key] >= val - radius) & (filtered_df[key] <= val + radius)]
    key_name = 'score' if return_scores else 'ground_truth'
    scores = filtered_df[key_name].values
    if len(scores) < num_scores:
        return scores
    return scores[-num_scores:]

def compute_beta(
    scores,
    obs,
    alpha_minimum=0.0,
    alpha_maximum=1.0,
):
    """
    Compute the beta value(s) for given prediction scores using the quantile function.
    Returns
    -------
    pred_betas : float or np.ndarray
        Computed beta value(s). Returns a float if `pred_scores` is scalar,
        otherwise an array of the same shape as `pred_scores`.
    """
    _scores = np.asarray(scores, dtype=float)
    _scores = _scores[np.isfinite(_scores)]
    if _scores.size == 0:
        raise ValueError("scores is empty or all non-finite.")
    _scores.sort()
    _n = _scores.size
    _smin = _scores[0]
    _smax = _scores[-1]
    def compute_single(val):
        if val > _smax:
            return float(alpha_minimum)
        if val <= _smin:
            return float(alpha_maximum)
        # First index with x[i] >= val
        i = int(np.searchsorted(_scores, val, side="left"))  # 0 <= i <= n
        # beta_max = 1 - i/n
        beta = 1.0 - (i / _n)
        return float(np.clip(beta, alpha_minimum, alpha_maximum))
    obs_arr = np.atleast_1d(obs)
    betas = np.array([compute_single(obs_arr[i]) for i in range(len(obs_arr))])
    return betas

def covered_event(score, quantile):
    return score <= quantile

def concat_scores(current_scores, previous_scores=None, score_window=1000):
    """
    Concatenate current scores with previous scores.
    If previous_scores is empty, return current_scores.
    If previous_scores is not empty, concatenate along the first axis.
    If the concatenated array exceeds score_window, trim it to the last score_window entries.
    """
    if previous_scores is None or previous_scores.size == 0:
        return current_scores
    concatenated = np.concatenate((previous_scores, current_scores), axis=0)
    if concatenated.shape[0] > score_window:
        concatenated = concatenated[-score_window:]
    return concatenated

def plot_beams(y_truth, forecast_samples, states, scores):
    """
    Plot sample trajectories, ground-truth future, and state sequence.
    
    Args:
        y_truth (np.ndarray): ground truth trajectory
        forecast_samples (np.ndarray): shape (N, H) sampled trajectories
        states (np.ndarray): state sequence
        scores (np.ndarray): shape (H), scores for each horizon
    """
    N, H = forecast_samples.shape
    times = np.arange(H)
    
    plt.figure(figsize=(10, 6))
    # sampled trajectories
    for i in range(N):
        plt.fill_between(times, forecast_samples[i] - scores, forecast_samples[i] + scores,
                         alpha=0.3, color='blue', label='_nolegend_')
    # ground truth
    # plot ground truth with color based on whether it falls within any interval
    for t in range(len(times)):
        in_any_interval = any((y_truth[t] >= forecast_samples[i][t] - scores[t]) and 
                             (y_truth[t] <= forecast_samples[i][t] + scores[t]) 
                             for i in range(N))
        color = 'green' if in_any_interval else 'red'
        if t > 0:  # Plot line segments
            plt.plot(times[t-1:t+1], y_truth[t-1:t+1], color=color, linewidth=2.5, 
                    label='Ground Truth (covered)' if color=='green' and t==1 else 
                          'Ground Truth (not covered)' if color=='red' and t==1 else "_nolegend_")
    # # state sequence
    # plt.step(times, states, where='post', color='green', label='State')
    
    plt.xlabel("Time")
    plt.ylabel("Value / State")
    plt.title("Forecast vs Ground Truth")
    plt.legend()
    plt.tight_layout()
    plt.show()
