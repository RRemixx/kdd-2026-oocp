import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from tqdm import tqdm


def pinball_loss(score, q, q_level):
    """
    Pinball loss for quantile regression:
      L(q, y) = max( alpha * (y - q), (alpha - 1) * (y - q) )
    Parameters:
        score (np.ndarray): shape (T,), ground truth values y_t
        q (np.ndarray): shape (T,), predicted quantiles q_t
        q_level (float): quantile level (0 < q_level < 1)
    Returns:
        float: mean pinball loss over T time steps
    """
    diffs = score - q
    losses = np.maximum(q_level * diffs, (q_level - 1) * diffs)
    return np.mean(losses)

def min_pinball_loss(score, q_level):
    """
    Minimum pinball loss for quantile regression:
      L*(alpha) = min_q E[ L(q, y) ]
    Parameters:
        score (np.ndarray): shape (T,), ground truth values y_t
        q_level (float): quantile level (0 < q_level < 1)
    Returns:
        float: minimum pinball loss
    """
    q_star = np.quantile(score, q_level)
    return pinball_loss(score, q_star, q_level)

def strongly_adaptive_regret(scores, qs, alpha, k=20):
    """
    Strongly Adaptive Regret over all sub-intervals of length k:
      SAReg(T,k) = max_{τ=1..T-k+1} [ ∑_{t=τ}^{τ+k-1} scores[t]
                                    - ∑_{t=τ}^{τ+k-1} qs[t] ]
    """
    T = len(scores)
    if T < 30:
        k_ = T
    else:
        k_ = k
    max_regret = -np.inf
    for tau in range(T - k_ + 1):
        current_scores = scores[tau:tau + k_]
        current_qs = qs[tau:tau + k_]
        loss1 = pinball_loss(current_scores, current_qs, 1-alpha)
        loss2 = min_pinball_loss(current_scores, 1-alpha)
        regret = loss1 - loss2
        if regret > max_regret:
            max_regret = regret
    return max_regret

def union_length(intervals):
    """
    Compute the total length of the union of potentially overlapping intervals.

    Parameters:
        intervals (numpy.ndarray): Array of shape (N, 2), each row = [lower, upper].

    Returns:
        float: Length of the union of these intervals on the real line.
    """
    # Sort intervals by their lower bound
    sorted_intervals = intervals[np.argsort(intervals[:, 0])]
    total = 0.0
    current_start, current_end = sorted_intervals[0]

    for low, high in sorted_intervals[1:]:
        if low > current_end:
            # Disjoint: add previous segment length, start a new one
            total += current_end - current_start
            current_start, current_end = low, high
        else:
            # Overlap: extend the current segment if needed
            current_end = max(current_end, high)

    # Add the last segment
    total += current_end - current_start
    return total


def compute_horizon_coverage(cov, alphas, horizons):
    """
    Compute horizon coverage per time-step for each alpha. For a fixed alpha:
        1) For each time t and each horizon h, use coverage_per_time
        2) Average over horizons to get coverage_t_per_alpha[t]

    Returns:
        dict: horizon_cov[alpha] = numpy array of shape (T,), where
                horizon_cov[alpha][t] = mean coverage over all horizons for time t.
    """
    horizon_cov = {}
    for alpha in alphas:
        cov_matrix = np.stack(
            [cov[alpha][h] for h in horizons],
            axis=1
        )  # shape: (T, H)
        cov_per_t = np.mean(cov_matrix, axis=1)  # shape: (T,)
        horizon_cov[alpha] = cov_per_t
    return horizon_cov

def compute_rolling_coverage_per_horizon(cov, alphas, horizons, window=1):
    """
    Compute rolling coverage per time-step for each alpha.
    
    Returns:
        dict: rolling_horizon_cov[alpha][h] = numpy array of shape (T,), where
                rolling_horizon_cov[alpha][h][t] = rolling mean coverage for time t.
    """
    rolling_horizon_cov = {}
    for alpha in alphas:
        rolling_horizon_cov[alpha] = {}
        for h in horizons:
            series = cov[alpha][h]
            rolling_mean = np.convolve(series, np.ones(window) / window, mode='valid')
            rolling_horizon_cov[alpha][h] = rolling_mean
    return rolling_horizon_cov


def interval_width(prediction_intervals):
    """
    Calculate the average interval width (union length) across all time steps.

    Parameters:
        prediction_intervals (numpy.ndarray): Shape (T, N, 2). At each t,
                                               there are N intervals [lower, upper].

    Returns:
        float: Mean union length over all T time steps.
    """
    T, N, _ = prediction_intervals.shape
    lengths = np.zeros(T)
    for t in range(T):
        lengths[t] = union_length(prediction_intervals[t, :, :])
    return np.mean(lengths), lengths

def interval_area(sample_arr, qs_arr, score_type, optional_args=None, plot=False):
    """
    Compute an area metric based on prediction intervals.

    Parameters:
        sample_arr (numpy.ndarray): Array of shape (T, N, 2)
        qs_arr (numpy.ndarray): Array of shape (T,) representing quantiles of non-conformity scores.
    Returns:
        float: Computed area metric.
    """
    verified = not plot
    if score_type == 'abs-r':
        return np.mean(qs_arr**2 * np.pi), qs_arr**2 * np.pi

    elif score_type == 'pcp':
        T = sample_arr.shape[0]
        union_areas = []
        for t in range(T):
            q = qs_arr[t]
            samp = sample_arr[t, :, :]
            
            # Create circles (as shapely geometries) at each of the N points with radius q
            circles = [Point(x, y).buffer(q) for x, y in samp]
            # Compute the union of all circles to avoid overcounting overlaps
            union_area = unary_union(circles).area
            union_areas.append(union_area)
            
            if not verified:
                print("q:", q)
                print("samp:", samp)
                print("Union area:", union_area)
                fig, ax = plt.subplots()
                for circle in circles:
                    x, y = circle.exterior.xy
                    ax.plot(x, y, lw=2, alpha=0.7)
                ax.set_aspect('equal', adjustable='box')
                plt.title("Circles Plot")
                plt.show()
                verified = True
    
        # Return the average union area over all T time steps
        return np.mean(union_areas), union_areas

    elif score_type == 'pcp-u':
        # similar to when score_type is pcp. But here, only the area covered by at least quantile_d fraction of samples is considered
        quantile_d = optional_args.get("quantile_d", None)
        T = sample_arr.shape[0]
        union_areas = []
        # Resolution of grid for numerical approximation
        resolution = 20

        for t in tqdm(range(T)):
            q = qs_arr[t]
            samp = sample_arr[t, :, :]
            N = samp.shape[0]
            circles = [Point(x, y).buffer(q) for x, y in samp]

            # Determine the bounding box that covers all circles
            minx, miny, maxx, maxy = unary_union(circles).bounds

            xs = np.linspace(minx, maxx, resolution)
            ys = np.linspace(miny, maxy, resolution)
            X, Y = np.meshgrid(xs, ys)
            points = np.column_stack([X.ravel(), Y.ravel()])

            coverage_counts = np.zeros(points.shape[0])
            for circle in circles:
                # Check if the point lies inside the circle
                coverage_counts += np.array([circle.contains(Point(pt[0], pt[1])) for pt in points])
            # Only count points covered by at least quantile_d fraction of circles
            threshold = quantile_d * N
            valid = coverage_counts >= threshold

            # Approximate the area: fraction of valid points * total area of bounding box
            cell_area = ((maxx - minx) / (resolution - 1)) * ((maxy - miny) / (resolution - 1))
            approx_area = valid.sum() * cell_area
            union_areas.append(approx_area)
            
            if not verified:
                fig2, ax2 = plt.subplots()
                for circle in circles:
                    x, y = circle.exterior.xy
                    ax2.plot(x, y, lw=2, alpha=0.7)
                valid_points = points[valid]
                invalid_points = points[~valid]
                ax2.scatter(valid_points[:, 0], valid_points[:, 1], color='red', s=10, label='Valid Points')
                ax2.scatter(invalid_points[:, 0], invalid_points[:, 1], color='blue', s=10, label='Other Points')
                ax2.set_aspect('equal', adjustable='box')
                ax2.set_title("Circles with Grid Points")
                ax2.legend()
                plt.show()
                verified = True
                print("Quantile d:", quantile_d)
                print("Sample points:", samp)
                print("q:", q)
                print("Union area with quantile d:", approx_area)

        return np.mean(union_areas), union_areas

    else:
        raise ValueError("Unknown score_type: {}".format(score_type))


def calibration_score(coverage_vals, alphas):
    """
    Calculate the calibration score across alphas: mean absolute difference
    between observed coverage and (1 - alpha).

    Parameters:
        coverage_vals (list or numpy.ndarray): Observed coverage for each alpha.
        alphas (list or numpy.ndarray): Corresponding alpha values.

    Returns:
        float: Calibration score.
    """
    coverage_vals = np.array(coverage_vals)
    alphas = np.array(alphas)
    expected = 1 - alphas
    gaps = coverage_vals - expected
    return np.mean(np.abs(gaps))


def calibration_plot(coverage_vals, alphas, path=None):
    """
    Generate a calibration plot: observed coverage vs. target (1 - alpha).

    Parameters:
        coverage_vals (list or numpy.ndarray): Observed coverage per alpha.
        alphas (list or numpy.ndarray): Alpha values.
        path (str, optional): If provided, save figure to this path.

    Returns:
        None
    """
    alphas_arr = np.array(alphas)
    coverage_arr = np.array(coverage_vals)
    target_coverage = 1 - alphas_arr

    plt.figure(figsize=(8, 6))
    plt.plot(target_coverage, coverage_arr, marker='o', label='Observed Coverage')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.xlabel('Target Coverage (1 - Alpha)')
    plt.ylabel('Observed Coverage')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid()
    if path:
        plt.savefig(path)
    plt.close()


def horizon_coverage_overall(coverage_t):
    """
    Calculate the mean coverage over all time steps (T) and horizons (H) for one alpha.:

    Parameters:
        coverage_t (numpy.ndarray): Shape (T), where coverage_t[t] = average coverage at time t across all horizons.

    Returns:
        float: Mean coverage over all time steps and horizons.
    """
    return np.mean(coverage_t)


def horizon_coverage_calibration_score(hc_vals, alphas):
    """
    Calculate calibration score across alphas for horizon coverage:
    mean absolute difference between observed horizon coverage and (1 - alpha).

    Parameters:
        hc_vals (list or numpy.ndarray): Observed horizon coverage per alpha.
        alphas (list or numpy.ndarray): Alpha values.

    Returns:
        float: Horizon coverage calibration score.
    """
    hc_vals = np.array(hc_vals)
    alphas = np.array(alphas)
    expected = 1 - alphas
    gaps = hc_vals - expected
    return np.mean(np.abs(gaps))


def horizon_coverage_calibration_plot(hc_vals, alphas, path=None):
    """
    Generate a calibration plot for horizon coverage.

    Parameters:
        hc_vals (list or numpy.ndarray): Observed horizon coverage per alpha.
        alphas (list or numpy.ndarray): Alpha values.
        path (str, optional): If provided, save figure to this path.

    Returns:
        None
    """
    alphas_arr = np.array(alphas)
    hc_arr = np.array(hc_vals)
    target_coverage = 1 - alphas_arr

    plt.figure(figsize=(8, 6))
    plt.plot(target_coverage, hc_arr, marker='o', label='Horizon Coverage')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.xlabel('Target Coverage (1 - Alpha)')
    plt.ylabel('Observed Horizon Coverage')
    plt.title('Horizon Coverage Calibration Plot')
    plt.legend()
    plt.grid()
    if path:
        plt.savefig(path)
    plt.close()


def safety_score(coverage_t, alpha, window=1, threshold=0.1):
    """
    Calculate the safety score over all horizons:
    It checks how often the rolling mean of horizon coverage (over time t) goes outside [target ± threshold].

    Parameters:
        y_truths (numpy.ndarray): Shape (T, H).
        prediction_intervals (numpy.ndarray): Shape (T, H, N, 2).
        alpha (float): Miscoverage rate.
        window (int): Rolling window size (in time dimension).
        threshold (float): Allowed deviation from target coverage.

    Returns:
        float: Fraction of time steps where rolling coverage ∉ [target ± threshold].
    """
    # Compute rolling mean over time dimension
    rolling_mean = np.convolve(coverage_t, np.ones(window) / window, mode='valid')
    target_cov = 1 - alpha
    out_of_bounds = (rolling_mean < target_cov - threshold)
    return np.mean(out_of_bounds)

def risk_type_helper(vals, risk_type):
    if risk_type == 'absolute':
        vals = np.abs(vals)
    elif risk_type == 'relu':
        vals = np.maximum(0, vals)
    elif risk_type == 'raw':
        vals = vals
    else:
        raise ValueError("Invalid risk_type. Use 'absolute', 'relu', or 'raw'.")
    return vals

def rolling_horizon_coverage_gaps(hc_vals, alpha, window=1, risk_type='absolute'):
    if window <= 1:
        rolling_mean = hc_vals
    else:
        rolling_mean = np.convolve(hc_vals, np.ones(window) / window, mode='valid')
    target_cov = 1 - alpha
    vals = target_cov - rolling_mean
    vals = risk_type_helper(vals, risk_type)
    return vals


def quantiles_rolling_coverage(coverage_t, alpha, window=1):
    vals = rolling_horizon_coverage_gaps(coverage_t, alpha, window, risk_type='raw')
    return np.quantile(vals, q=[0.25, 0.50, 0.75])


def var_at_risk(errors, risk_level=0.9):
    """
    Value at Risk (VaR) at miscoverage rate alpha:
      smallest u such that P[|error| <= u] >= risk_level.
    Equivalently, the (risk_level)-quantile of |errors|.
    Parameters:
        errors (np.ndarray of shape (T,)): e_t
        risk_level (float): miscoverage rate (0 < risk_level < 1)
    Returns:
        float: VaR threshold
    """
    return np.quantile(errors, risk_level)


def exponential_risk(errors, theta):
    """
    Exponential Risk R_θ = θ * log E[ exp( |e_t| / θ ) ]
    Parameters:
        errors (np.ndarray of shape (T,)): e_t
        theta (float): risk-sensitivity parameter
    Returns:
        float: exponential risk
    """
    return theta * np.log(np.mean(np.exp(errors / theta)))


def worst_case_local_coverage_error(errors):
    return np.max(errors)


def max_consecutive_violations(gaps, delta):
    """
    Maximum consecutive violations:
    longest run of |coverage_gap_t| > delta.
    Parameters:
        gaps (np.ndarray of shape (T,)): coverage gaps (see coverage_gap_at_t)
        delta (float): allowed tolerance
    Returns:
        int: maximum ℓ such that ∃ t with |gaps[t:t+ℓ]| > delta
    """
    max_len = 0
    cur = 0
    for g in gaps:
        if g > delta:
            cur += 1
            if cur > max_len:
                max_len = cur
        else:
            cur = 0
    return max_len

def num_violations(gaps, delta):
    """
    Number of violations:
    count of time steps where |coverage_gap_t| > delta.
    Parameters:
        gaps (np.ndarray of shape (T,)): coverage gaps (see coverage_gap_at_t)
        delta (float): allowed tolerance
    Returns:
        int: number of time steps t with |gaps[t]| > delta
    """
    return np.sum(gaps > delta)

def monotonicity_score(data, alphas, horizons):
    sorted_alphas = sorted(alphas)
    total_score = np.zeros(len(horizons))
    T = data[sorted_alphas[0]][horizons[0]]['intervals'].shape[0]
    for h in horizons:
        prev_interval_widths = None
        for alpha in sorted_alphas:
            intervals_h_alpha = data[alpha][h]['intervals']  # shape: (T, N, 2)
            assert intervals_h_alpha.shape[1] == 1, "N should be 1 for monotonicity score"
            interval_widths = intervals_h_alpha[:, 0, 1] - intervals_h_alpha[:, 0, 0]  # shape: (T,)
            if prev_interval_widths is not None:
                diff = interval_widths - prev_interval_widths
                total_score[h] += np.sum(diff[diff > 0])
            prev_interval_widths = interval_widths
    total_score /= T
    return total_score

def distribution_consistent_ratio(data, alphas, horizons):
    sorted_alphas = sorted(alphas)
    T = data[sorted_alphas[0]][horizons[0]]['intervals'].shape[0]
    consistency_vec = np.zeros((len(horizons), T))
    for h in horizons:
        prev_interval_widths = None
        for alpha in sorted_alphas:
            intervals_h_alpha = data[alpha][h]['intervals']  # shape: (T, N, 2) note that here N = 1
            assert intervals_h_alpha.shape[1] == 1, "N should be 1 for distribution consistent ratio"
            interval_widths = intervals_h_alpha[:, 0, 1] - intervals_h_alpha[:, 0, 0]  # shape: (T,)
            if prev_interval_widths is not None:
                diff = interval_widths - prev_interval_widths
                consistency_vec[h] += (diff > 0).astype(float)
            prev_interval_widths = interval_widths
    consistent_ts = (consistency_vec == 0).astype(float)
    total_score = np.mean(consistent_ts, axis=1)
    return total_score

def get_mean(mydict, level=1):
    if level == 1:
        return np.mean(list(mydict.values()))
    elif level == 2:
        return np.mean([np.mean(list(v.values())) for v in mydict.values()])
    else:
        raise ValueError("Unsupported level")
