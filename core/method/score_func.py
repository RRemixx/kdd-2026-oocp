import numpy as np


def score_function(y_true, y_pred_samples, horizon=0, type='abs-r', optional_args=None):
    """
    Calculate the non-conformity score.
    
    Args:
        y_true (array): True values
        y_pred_samples (array): Predicted samples from the model
        horizon (int): Forecast horizon - 1
        type (str): Type of score function to use. Options are : 'abs-r' for absolute residual, 'pcp' for probabilistic conformal prediction, 'pcp-u' for unions of pcp
        optional_args (dict): Additional arguments for the score function
    
    Returns:
        float: Score value
    """
    true_response = y_true[horizon]
    samples = y_pred_samples[:, horizon]
    if type == 'abs-r':
        pred = np.nanmean(samples)
        d = abs(true_response - pred)
        return d
    if type == 'pcp':
        # Find the minimum absolute distance between true response and any sample
        d = min(abs(true_response - samples))
        return d
    if type == 'pcp-u':
        quantile_d = optional_args.get('quantile_d', 0.5)
        distances = abs(true_response - samples)
        d = np.quantile(distances, quantile_d)
        return d
    print(f"Unknown score function type: {type}")
    return None


def dist(point1, point2):
    return np.linalg.norm(point1 - point2)


def score_function_2d(y_true, y_pred_samples, horizon=-1, type='abs-r', optional_args=None):
    """
    Calculate the non-conformity score.
    
    Args:
        y_true (array): True values
        y_pred_samples (array): Predicted samples from the model
        horizon (int): Forecast horizon - 1
        type (str): Type of score function to use. Options are : 'abs-r' for absolute residual, 'pcp' for probabilistic conformal prediction, 'pcp-u' for unions of pcp
        optional_args (dict): Additional arguments for the score function
    
    Returns:
        float: Score value
    """
    if horizon >= 0:
        true_response = y_true[horizon, :] # (2,)
        samples = y_pred_samples[:, horizon, :] # (N, 2)
    else:
        true_response = y_true
        samples = y_pred_samples
    if type == 'abs-r':
        pred = np.nanmean(samples, axis=0)
        d = dist(true_response, pred)
        return d
    if type == 'pcp':
        # Find the minimum absolute distance between true response and any sample
        distances_to_samples = np.array([dist(true_response, sample) for sample in samples])
        return np.min(distances_to_samples)
    if type == 'pcp-u':
        quantile_d = optional_args.get('quantile_d', 0.5)
        distances_to_samples = np.array([dist(true_response, sample) for sample in samples])
        d = np.quantile(distances_to_samples, quantile_d)
        return d
    print(f"Unknown score function type: {type}")
    return None


def covered_union(y_true, intervals):
    """
    Check whether at least one interval covers the ground truth.

    Parameters:
        y_true (float): Ground truth value.
        intervals (numpy.ndarray): Array of shape (N, 2), each row = [lower, upper].

    Returns:
        bool: True if any interval contains y_true.
    """
    # True if y_true lies in any [lower_i, upper_i]
    return np.any((y_true >= intervals[:, 0]) & (y_true <= intervals[:, 1]))


def covered_union_2d(y_true, sample, radius, score_type, optional_args=None):
    score = score_function_2d(y_true, sample, horizon=-1, type=score_type, optional_args=optional_args)
    return score <= radius


def q_coverage(intervals: np.ndarray, q: float):
    """
    Given an array of intervals shape (N,2) and fraction q,
    returns an array of sub-intervals where at least q fraction of the intervals overlap.
    """
    N = len(intervals)
    threshold = q * N
    # Build event list
    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    # Sort events by position
    events.sort(key=lambda x: x[0])
    
    coverage_intervals = []
    count = 0
    prev_pos = None
    in_region = False
    
    for pos, delta in events:
        if prev_pos is not None and in_region:
            # Add segment [prev_pos, pos] if still in region
            coverage_intervals.append((prev_pos, pos))
        
        count += delta
        in_region = (count >= threshold)
        prev_pos = pos
    
    # Merge adjacent or overlapping intervals
    merged = []
    for start, end in coverage_intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return np.array(merged)


def inverse_score_function(score, y_pred_samples, type='abs-r', optional_args=None):
    """
    Get the prediction interval for a given score.
    
    Args:
        score (float): Non-conformity score
        y_pred_samples (array): Predicted samples from the model
        type (str): Type of score function to use. Options are : 'abs-r' for absolute residual, 'pcp' for probabilistic conformal prediction, 'pcp-u' for unions of pcp
        optional_args (dict): Additional arguments for the score function
    
    Returns:
        list: Lower and upper bounds of the prediction interval
    """
    if type == 'abs-r':
        # For absolute residual, the prediction interval is centered around the sample mean
        pred = np.mean(y_pred_samples)
        prediction_intervals = np.zeros((1, 2))
        lower_bound = pred - score
        upper_bound = pred + score
        prediction_intervals[0] = [lower_bound, upper_bound]
        return prediction_intervals
    if type == 'pcp':
        N = len(y_pred_samples)
        prediction_intervals = np.zeros((N, 2))
        for i in range(N):
            sample = y_pred_samples[i]
            lower_bound = sample - score
            upper_bound = sample + score
            prediction_intervals[i] = [lower_bound, upper_bound]
        return prediction_intervals
    if type == 'pcp-u':
        quantile_d = optional_args.get('quantile_d', 0.5)
        N = len(y_pred_samples)
        all_intervals = np.zeros((N, 2))
        for i in range(N):
            sample = y_pred_samples[i]
            lower_bound = sample - score
            upper_bound = sample + score
            all_intervals[i] = [lower_bound, upper_bound]
        prediction_intervals = q_coverage(all_intervals, quantile_d)
        return np.array(prediction_intervals)