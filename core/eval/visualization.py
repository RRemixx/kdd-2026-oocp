import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
import pandas as pd

def covered_event(score, quantile):
    return score <= quantile

def plot_forecast_step(t, observed, forecast, scores, qs, samp, alphas, save_path, additional_info=''):
    radius = qs[t, :]

    # Use seaborn style for more scientific look
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot observed data with deeper black
    ax.plot(range(len(observed)), observed, color='#202020', 
            linewidth=2, label='Observed', alpha=0.9)

    # Plot forecast with modern color scheme
    forecast_x = range(len(observed)-1, len(observed)+len(forecast)-1)
    is_covered = scores[t, :] <= qs[t, :]

    # Use more sophisticated colors
    covered_color = '#2ecc71'  # emerald green
    uncovered_color = '#e74c3c'  # pomegranate red

    for i in range(len(forecast)):
        color = uncovered_color if not is_covered[i] else covered_color
        ax.plot(forecast_x[i], forecast[i], 'o', color=color, 
                markersize=6, alpha=0.9)

    # Add empty lines for legend
    ax.plot([], [], 'o', color=covered_color, label='Covered forecast')
    ax.plot([], [], 'o', color=uncovered_color, label='Uncovered forecast')

    # Plot prediction intervals with better transparency
    forecast_samples = samp
    for i in range(len(forecast_samples)):
        ax.plot(forecast_x, forecast_samples[i], color='#3498db', alpha=0.8, linewidth=1.5)
        ax.fill_between(forecast_x, 
                        forecast_samples[i] - radius,
                        forecast_samples[i] + radius,
                        color='#3498db', alpha=0.2)  # light blue

    # --- Added: annotate x-tick labels with alpha values for each forecast horizon ---
    # original time steps
    obs_ticks = list(range(len(observed)))
    obs_labels = [str(i) for i in obs_ticks]
    # forecast horizons
    fcst_ticks = list(forecast_x)
    # attach corresponding alpha to each horizon label
    fcst_labels = [
        f"{tick} (α={alphas[h]:.2f})" 
        for h, tick in enumerate(fcst_ticks)
    ]
    # combine and set
    all_ticks = obs_ticks + fcst_ticks
    all_labels = obs_labels + fcst_labels
    ax.set_xticks(all_ticks)
    ax.set_xticklabels(all_labels, rotation=45, fontsize=10)
    # -------------------------------------------------------------------------------

    # Improve aesthetics
    ax.set_xlabel('Time step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'Forecast at t={t}, {additional_info}', fontsize=14, pad=15)
    ax.legend(frameon=True, fancybox=True, shadow=True)

    # Adjust layout and save with higher DPI
    plt.tight_layout()
    forecast_plot_dir = Path(save_path) / 'forecast_plots'
    if not forecast_plot_dir.exists():
        forecast_plot_dir.mkdir(parents=True)
    plt.savefig(forecast_plot_dir / f"{t}.png", dpi=200, bbox_inches='tight')
    plt.close()


def plot_forecast_step_2d(samples: np.ndarray,
                          ground_truth: np.ndarray,
                          qs: np.ndarray,
                          sample_color: str = 'lightgray',
                          sample_alpha: float = 0.5,
                          gt_inside_color: str = 'green',
                          gt_outside_color: str = 'red',
                          circle_alpha: float = 0.3,
                          save_path: str = "forecast.png"):
    """
    Plot 2D forecast samples with per-step circles and color-coded ground truth.

    Args:
        samples: array of shape (N, H, 2) of predicted trajectories
        ground_truth: array of shape (H, 2) of true positions
        qs: array of length H of circle radii for each step
        sample_color: color for sample trajectories
        sample_alpha: transparency for sample lines
        gt_inside_color: color for GT points inside any circle
        gt_outside_color: color for GT points outside all circles
        circle_alpha: transparency for the circles
    """
    _, H, dim = samples.shape
    assert dim == 2, "samples must be (N, H, 2)"
    assert ground_truth.shape == (H, 2), "ground_truth must be (H, 2)"
    assert qs.shape == (H,), "qs must have length H"

    plt.figure(figsize=(8, 8))

    # plot each sample trajectory and its circles
    for traj in samples:
        plt.plot(traj[:, 0], traj[:, 1],
                 color=sample_color, alpha=sample_alpha)
        for h in range(H):
            circ = Circle(traj[h],
                          qs[h],
                          fill=True,
                          alpha=circle_alpha)
            plt.gca().add_patch(circ)

    # plot ground truth points, color by coverage
    for h in range(H):
        gt_pt = ground_truth[h]
        dists = np.linalg.norm(samples[:, h] - gt_pt, axis=1)
        inside = np.min(dists) <= qs[h]
        plt.scatter(gt_pt[0], gt_pt[1],
                    color=(gt_inside_color if inside else gt_outside_color),
                    edgecolors='k', s=60, zorder=10)

    plt.gca().set_aspect('equal', 'box')
    plt.xlabel('x')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_coverage(scores, quantiles, alpha=0.1, window=20, save_path=None):
    """
    Evaluate coverage rates per horizon.
    
    Args:
        scores (array): Shape T x H, T is total time steps, H is horizon. Non-conformity scores
        quantiles (array): Shape T x H. Quantile thresholds
        alpha (float): Target miscoverage rate (default: 0.1)
        window (int): Window size for rolling coverage calculation (default: 20)
        save_path (str): Optional path to save the plot (default: None)
    
    Returns:
        None
    """
    T, H = scores.shape
    
    # Create subplots for each horizon
    fig, axes = plt.subplots(H, 1, figsize=(12, 4*H))
    if H == 1:
        axes = [axes]
        
    for h in range(H):
        covered = np.array([covered_event(s, q) for s, q in zip(scores[:,h], quantiles[:,h])])
        
        # Calculate metrics for this horizon
        realized_coverage = covered.astype(float)
        running_coverage = np.cumsum(covered) / np.arange(1, len(covered) + 1)
        target_coverage = np.ones_like(covered) * (1-alpha)
        
        # Calculate rolling coverage over fixed window
        rolling_coverage = np.array([
            np.mean(covered[max(0,i-window):i])
            for i in range(1, T+1)
        ])
        
        # Plot on corresponding subplot
        axes[h].plot(realized_coverage, 'b.', alpha=0.3, label='Realized coverage')
        axes[h].plot(running_coverage, 'g-', label='Running coverage')
        axes[h].plot(rolling_coverage, 'r-', label=f'{window}-step coverage')
        axes[h].plot(target_coverage, 'k--', label='Target coverage')
        axes[h].set_xlabel('Time step')
        axes[h].set_ylabel('Coverage')
        axes[h].set_title(f'Coverage Analysis - Horizon {h+1}')
        axes[h].legend()
        axes[h].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def evaluate_horizon_coverage(scores, quantiles, alpha=0.1, window=20, save_path=None):
    """
    Evaluate coverage rates across horizons for each time step.
    
    Args:
        scores (array): Shape T x H, T is total time steps, H is horizon. Non-conformity scores
        quantiles (array): Shape T x H. Quantile thresholds
        alpha (float): Target miscoverage rate (default: 0.1)
        window (int): Window size for rolling coverage calculation (default: 20)
        save_path (str): Optional path to save the plot (default: None)
    
    Returns:
        None
    """
    T, H = scores.shape
    
    # Calculate coverage across horizons for each time step
    coverage_per_step = np.zeros(T)
    for t in range(T):
        covered = np.array([covered_event(scores[t,h], quantiles[t,h]) for h in range(H)])
        coverage_per_step[t] = np.mean(covered)
    
    # Calculate running average
    running_coverage = np.cumsum(coverage_per_step) / np.arange(1, T+1)
    
    # Calculate rolling coverage over fixed window
    rolling_coverage = np.array([
        np.mean(coverage_per_step[max(0,i-window):i])
        for i in range(1, T+1)
    ])
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(coverage_per_step, 'b.', alpha=0.3, label='Coverage per step')
    plt.plot(running_coverage, 'g-', label='Running average')
    plt.plot(rolling_coverage, 'r-', label=f'{window}-step average')
    plt.axhline(y=1-alpha, color='k', linestyle='--', label='Target coverage')
    plt.xlabel('Time step')
    plt.ylabel('Coverage rate across horizons')
    plt.title('Coverage Analysis Across All Horizons')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_scores(scores, quantiles, alphas, alpha=0.1, save_path=None):
    """
    Plot the scores and quantiles for each horizon, along with coverage rates and alphas
    
    Args:
        scores (array): Shape T x H, T is total time steps, H is horizon. Non-conformity scores
        quantiles (array): Shape T x H. Quantile thresholds
        alphas (array): Shape T x H. Alpha values
        alpha (float): Target miscoverage rate (default: 0.1) 
        save_path (str): Optional path to save the plot (default: None)
    
    Returns:
        None
    """
    T, H = scores.shape
    
    # Create subplots - three rows per horizon
    fig, axes = plt.subplots(H, 3, figsize=(15, 4*H))
    if H == 1:
        axes = axes.reshape(1, -1)
        
    for h in range(H):
        # Calculate coverage at each timestep
        covered = np.array([covered_event(s, q) for s, q in zip(scores[:,h], quantiles[:,h])])
        
        # Calculate running and rolling coverage
        running_coverage = np.cumsum(covered) / np.arange(1, len(covered) + 1)
        window = min(20, T)
        rolling_coverage = np.array([
            np.mean(covered[max(0,i-window):i]) 
            for i in range(1, T+1)
        ])
        
        # Plot 1: Scores and quantiles
        axes[h,0].plot(scores[:,h], 'b.', alpha=0.3, label='Scores')
        axes[h,0].plot(quantiles[:,h], 'r-', label='Quantiles')
        axes[h,0].set_title(f'Horizon {h+1} - Scores and Quantiles')
        axes[h,0].set_xlabel('Time step')
        axes[h,0].set_ylabel('Value')
        axes[h,0].legend()
        axes[h,0].grid(True)
        
        # Plot 2: Coverage rates
        axes[h,1].plot(running_coverage, 'g-', label='Running coverage')
        axes[h,1].plot(rolling_coverage, 'b-', label=f'{window}-step coverage')
        axes[h,1].axhline(y=1-alpha, color='r', linestyle='--', label='Target')
        axes[h,1].set_title(f'Horizon {h+1} - Coverage Rates')
        axes[h,1].set_xlabel('Time step')
        axes[h,1].set_ylabel('Coverage rate')
        axes[h,1].legend()
        axes[h,1].grid(True)
        
        # Plot 3: Alpha values
        axes[h,2].plot(alphas[:,h], 'k-', label='Alpha')
        axes[h,2].axhline(y=alpha, color='r', linestyle='--', label='Target')
        axes[h,2].set_title(f'Horizon {h+1} - Alpha Values')
        axes[h,2].set_xlabel('Time step')
        axes[h,2].set_ylabel('Alpha')
        axes[h,2].legend()
        axes[h,2].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_alpha_evolution(alphas, alphas_mid, alphas_radius, horizon_alphats, start_t, alpha, save_path, h=0):
    plt.figure(figsize=(10, 6))
    time_points = np.arange(len(alphas[start_t:, h]))
    
    # Plot the plausible range
    plt.fill_between(time_points, 
                        alphas_mid[start_t:, h] - alphas_radius[start_t:, h],
                        alphas_mid[start_t:, h] + alphas_radius[start_t:, h],
                        alpha=0.2, label='Plausible range')
    
    # Plot alphas and horizon_alphats
    plt.plot(alphas[start_t:, h], label=f'Alpha h={h}', alpha=0.5)
    plt.plot(alphas_mid[start_t:, h], label=f'Alpha mid h={h}', linestyle='--', alpha=0.5)
    plt.plot(horizon_alphats[start_t:], label='Horizon alpha', color='g', alpha=0.5)
    plt.axhline(y=alpha, color='r', linestyle='-', label='Target alpha')
    
    plt.xlabel('Time step')
    plt.ylabel('Alpha value')
    plt.title(f'Evolution of alphas and horizon alpha over time (h={h})')
    plt.legend()
    plt.savefig(f'{save_path}/alphas_h{h}.png')
    plt.close()


def plot_predictions_w_coverage(
    alpha, time_indexes, horizon, timestamps, ground_truths, prediction_intervals, rolling_window, save_path=None
):
    """
    Plot ground truth vs. prediction intervals and rolling coverage for a single forecast horizon.

    Coverage at time t is 1 if y_true[t] lies in *any* of the N intervals at that t.
    """
    # select the data for this horizon
    x = np.array(timestamps)[time_indexes]
    y_true = ground_truths[time_indexes, horizon]
    intervals = prediction_intervals[time_indexes, horizon]  # shape (len(time_indexes), N, 2)

    # compute coverage: 1 if y_true is in any (lower_i, upper_i)
    covered = np.any(
        (y_true[:, None] >= intervals[..., 0]) &
        (y_true[:, None] <= intervals[..., 1]),
        axis=1
    ).astype(int)

    # rolling coverage rate
    try:
        dt_index = pd.to_datetime(x, errors="raise")
        cov_series = pd.Series(covered, index=dt_index)
    except Exception:
        # Fallback for non-date timestamps (e.g., integer time codes)
        cov_series = pd.Series(covered, index=pd.RangeIndex(start=0, stop=len(x)))
    rolling_cov = cov_series.rolling(window=rolling_window, min_periods=1).mean()

    # plotting
    fig, (ax0, ax1) = plt.subplots(
        nrows=2, ncols=1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # top panel: truth + envelope
    ax0.plot(x, y_true, '-x', label="Ground truth")
    for xi, (lows, ups) in zip(x, zip(intervals[..., 0], intervals[..., 1])):
        # lows, ups each length N
        ax0.vlines([xi]*len(lows), lows, ups, color="orange", alpha=0.9, linewidth=3)
    ax0.set_ylabel("Value")
    ax0.set_title(f"Horizon = {horizon}: Predictions vs. Ground Truth")
    ax0.legend(loc="upper left")
    ax0.grid(True)

    # bottom panel: coverage and rolling rate
    ax1.plot(x, covered, marker="o", linestyle="", label="Pointwise coverage")
    ax1.plot(x, rolling_cov, linestyle="-", linewidth=2, label=f"{rolling_window}-step rolling coverage")
    ax1.axhline(1.0, color="red", linestyle="--", linewidth=1, label="100% coverage")
    ax1.axhline(1-alpha, color="orange", linestyle="--", linewidth=1, label=f"Target coverage (1 - {alpha})")
    ax1.set_ylabel("Coverage")
    ax1.set_xlabel("Time")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="upper left")
    ax1.grid(True)

    fig.autofmt_xdate()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)

def debug_knn(u_star, lbs, ubs, cur_Fs, traj_samples, traj_sample_acc, save_path, t, H):
    # plot the samples v.s. the ground truth
    horizons = np.arange(H)

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    ax1 = axes[0]
    current_ground_truth_beta = cur_Fs
    ax1.plot(horizons, current_ground_truth_beta,
            label='Ground Truth Beta',
            color='black', linewidth=2)
    ax1.plot(horizons, u_star, label='Beta Choice', color='orange', linewidth=2)
    ax1.fill_between(horizons, lbs, ubs,
            color='gray', alpha=0.3, label='Beta Interval')
    for sidx in range(min(traj_samples.shape[0], 100)):
        ax1.plot(horizons, traj_samples[sidx, :],
                color='blue', alpha=0.2)
    traj_sample_quantile = np.quantile(traj_samples, 0.1, axis=0)
    ax1.plot(horizons, traj_sample_quantile,
            label='10th Percentile of Beta Samples',
            color='purple', linewidth=2)
    ax1.set_ylabel('Beta Samples')
    ax1.set_title(f'Time {t}: Sampled Betas vs Ground Truth Beta')
    ax1.legend()
    ax2 = axes[1]
    ax2.plot(horizons, traj_sample_acc[t],
            label='Second Beta Vector',
            color='red', linewidth=2)
    ax2.set_xlabel('Horizon')
    ax2.set_ylabel('Pinball Loss Value')
    ax2.legend()
    
    # coverage plot
    ax3 = axes[2]
    covereds = np.array(u_star) < cur_Fs
    ax3.plot(horizons, covereds.astype(float),
            label='Coverage Indicator',
            color='green', linewidth=2)
    ax3.set_xlabel('Horizon')
    ax3.set_ylabel('Coverage (1=Covered, 0=Not Covered)')
    
    t_save_path = save_path / 'sample_traj_plot'
    if not t_save_path.exists():
        t_save_path.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(t_save_path / f'{t}.png')
    plt.close()

def debug_samples(Fs, ground_truths, sampled_trajs, weights, save_path, samples2plot=None, alpha_selections=None, baseline_alpha_selections=None):
    Fs = np.asarray(Fs)
    ground_truths = np.asarray(ground_truths)
    sampled_trajs = np.asarray(sampled_trajs)
    save_dir = Path(save_path) / 'debug_samples'
    save_dir.mkdir(parents=True, exist_ok=True)

    if Fs.ndim == 3 and ground_truths.ndim == 3 and sampled_trajs.ndim == 4:
        raise ValueError("Input arrays should be 2D (Fs, ground_truths) and 3D (sampled_trajs), not 3D/4D.")

    if Fs.ndim != 2 or ground_truths.ndim != 2 or sampled_trajs.ndim != 3:
        raise ValueError("Fs and ground_truths must be (T, H); sampled_trajs must be (T, N, H)")

    T, H = Fs.shape
    if ground_truths.shape != (T, H) or sampled_trajs.shape[0] != T or sampled_trajs.shape[2] != H:
        raise ValueError("Shape mismatch between Fs, ground_truths, and sampled_trajs")
    if alpha_selections is not None and np.asarray(alpha_selections).shape != (T, H):
        raise ValueError("alpha_selections must be (T, H) if provided")
    if baseline_alpha_selections is not None and np.asarray(baseline_alpha_selections).shape != (T, H):
        raise ValueError("baseline_alpha_selections must be (T, H) if provided")

    n_samples = sampled_trajs.shape[1]
    if samples2plot is None:
        samples2plot = n_samples
    samples2plot = min(samples2plot, n_samples)

    for h in range(H):
        fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)

        idxs = np.arange(h, T, H)
        colors = ['tab:blue', 'tab:orange']
        for j, idx in enumerate(idxs):
            xs = np.arange(idx + 1, idx + H + 1)
            fs_block = Fs[idx, :]
            fs_mask = np.isfinite(fs_block)
            axes[0].plot(xs[fs_mask], fs_block[fs_mask],
                         color='black', linewidth=2, alpha=1, label='gt betas' if j == 0 else None)
            ys = sampled_trajs[idx, :samples2plot, :]
            axes[0].plot(xs, ys.T, color=colors[j % 2], alpha=0.05)
            median = np.nanmedian(ys, axis=0)
            axes[0].plot(xs, median, color='red', linewidth=1.5, alpha=0.9, label='Median sample' if j == 0 else None)
            
        axes[0].set_ylabel('Value')
        axes[0].set_title(f'Fs and Samples (h={h})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        for j, idx in enumerate(idxs):
            xs = np.arange(idx + 1, idx + H + 1)
            ys_all = sampled_trajs[idx, :, :]
            q10 = []
            for h in range(H):
                q10.append(np.nanquantile(ys_all[:, h], alpha_selections[idx, h]) if alpha_selections[idx, h]>0 and alpha_selections[idx, h]<1 else np.nan)
            q10 = np.array(q10)
            fs_block = Fs[idx, :]
            fs_mask = np.isfinite(fs_block)
            axes[1].plot(xs[fs_mask], fs_block[fs_mask],
                         color='black', linewidth=1.5, alpha=0.8, label='Fs block' if j == 0 else None)
            q10_mask = np.isfinite(q10)
            axes[1].plot(xs[q10_mask], q10[q10_mask],
                         color='purple', linewidth=1.5, alpha=0.9, label='q at (1-alphat) samples' if j == 0 else None)
            if alpha_selections is not None:
                a_block = np.asarray(alpha_selections)[idx, :]
                a_mask = np.isfinite(a_block)
                axes[1].plot(xs[a_mask], a_block[a_mask],
                             color='tab:blue', linewidth=1.5, alpha=0.9, label='Alpha selection' if j == 0 else None)
            if baseline_alpha_selections is not None:
                b_block = np.asarray(baseline_alpha_selections)[idx, :]
                b_mask = np.isfinite(b_block)
                axes[1].plot(xs[b_mask], b_block[b_mask],
                             color='tab:orange', linewidth=1.5, alpha=0.9, label='Baseline alpha' if j == 0 else None)
        axes[1].set_ylabel('Alpha/Fs')
        axes[1].set_title(f'Alphas, Fs, and q0.1 (h={h})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        for j, idx in enumerate(idxs):
            xs = np.arange(idx + 1, idx + H + 1)
            fs_block = Fs[idx, :]
            if alpha_selections is not None:
                a_block = np.asarray(alpha_selections)[idx, :]
                cov = (a_block < fs_block).astype(float)
                cov_mask = np.isfinite(cov)
                axes[2].plot(xs[cov_mask], cov[cov_mask],
                             color='tab:blue', marker='o', linestyle='None', alpha=0.9, label='Coverage (alpha)' if j == 0 else None)
                if np.any(cov_mask):
                    avg_cov = np.nanmean(cov[cov_mask])
                    axes[2].plot(xs, np.full_like(xs, avg_cov, dtype=float),
                                 color='tab:blue', linewidth=2, alpha=0.6, label='Avg coverage (alpha)' if j == 0 else None)
            if baseline_alpha_selections is not None:
                b_block = np.asarray(baseline_alpha_selections)[idx, :]
                cov_b = (b_block < fs_block).astype(float)
                cov_b_mask = np.isfinite(cov_b)
                axes[2].plot(xs[cov_b_mask], cov_b[cov_b_mask],
                             color='tab:orange', marker='x', linestyle='None', alpha=0.9, label='Coverage (baseline)' if j == 0 else None)
                if np.any(cov_b_mask):
                    avg_cov_b = np.nanmean(cov_b[cov_b_mask])
                    axes[2].plot(xs, np.full_like(xs, avg_cov_b, dtype=float),
                                 color='tab:orange', linewidth=2, alpha=0.6, label='Avg coverage (baseline)' if j == 0 else None)
        axes[2].set_ylabel('Coverage')
        axes[2].set_title(f'Coverage (h={h})')
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        for j, idx in enumerate(idxs):
            xs = np.arange(idx + 1, idx + H + 1)
            gt_block = ground_truths[idx, :]
            gt_mask = np.isfinite(gt_block)
            axes[3].plot(xs[gt_mask], gt_block[gt_mask],
                         color='black', linewidth=2, alpha=0.8, label='Ground Truth block' if j == 0 else None)
        axes[3].set_xlabel('Time step')
        axes[3].set_ylabel('Value')
        axes[3].set_title(f'Ground Truth (h={h})')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        def process_weight_t(w):
            if w is None:
                return 0
            return np.max(w)
        weights2plot = [process_weight_t(weights[t]) for t in range(len(weights))]
        axes[4].plot(np.arange(len(weights2plot)), weights2plot,
                     color='brown', marker='o', linestyle='-', alpha=0.9, label='Max weight per time step')
        axes[4].set_xlabel('Time step')
        axes[4].set_ylabel('Max Weight')
        axes[4].set_title(f'Max Weights over Time (h={h})')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / f'h_{h}.png', dpi=150)
        plt.close(fig)

def debug_cpid_boundaries(all_qts, all_boundaries, qs, scores, save_path):
    T, H = all_qts.shape
    fig, axes = plt.subplots(H, 1, figsize=(10, 4*H), sharex=True)
    if H == 1:
        axes = [axes]
    
    for h in range(H):
        axes[h].plot(all_qts[:, h], label='CPID (q_t)', color='blue')
        axes[h].fill_between(np.arange(T), all_boundaries[:, h, 0], all_boundaries[:, h, 1], color='blue', alpha=0.2, label='boundaries')
        axes[h].plot(qs[:, h], label='Chosen q', color='green')
        axes[h].plot(scores[:, h], label='Scores', color='red')
        axes[h].set_title(f'Horizon {h} - CPID Boundaries vs Quantiles')
        axes[h].set_xlabel('Time step')
        axes[h].set_ylabel('Value')
        axes[h].legend()
        axes[h].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path / "cpid_boundaries.png", dpi=300)
    plt.close(fig)