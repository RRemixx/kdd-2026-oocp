import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_horizon_time_heatmap(
    A,
    *,
    t_min=0,
    t_max=None,          # exclusive
    t_step=1,            # take every t_step rows (time subsampling)
    normalize=None,      # None | "global_z" | "per_horizon_z"
    center=None,         # None | "global" | "per_horizon"
    vmin=None,
    vmax=None,
    origin="lower",
    interpolation="none",
    title=None,
    figure_path=None,    # Path or str; if not None, save to this directory
    filename="heatmap.jpg",
    dpi=200,
    show=True,
):
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        raise ValueError("A must be a 2D numpy array of shape (T, H).")

    T, H = A.shape
    if t_max is None:
        t_max = T
    if not (0 <= t_min < T) or not (0 < t_max <= T) or t_min >= t_max:
        raise ValueError(f"Invalid t_min/t_max: got t_min={t_min}, t_max={t_max}, T={T}.")
    if t_step <= 0:
        raise ValueError("t_step must be a positive integer.")

    A_sub = A[t_min:t_max:t_step].astype(float, copy=False)

    # Optional centering
    if center == "global":
        A_sub = A_sub - np.nanmean(A_sub)
    elif center == "per_horizon":
        A_sub = A_sub - np.nanmean(A_sub, axis=0, keepdims=True)
    elif center is not None:
        raise ValueError('center must be None, "global", or "per_horizon".')

    # Optional normalization
    if normalize == "global_z":
        mu = np.nanmean(A_sub)
        sd = np.nanstd(A_sub)
        if sd > 0:
            A_sub = (A_sub - mu) / sd
    elif normalize == "per_horizon_z":
        mu = np.nanmean(A_sub, axis=0, keepdims=True)
        sd = np.nanstd(A_sub, axis=0, keepdims=True)
        sd = np.where(sd > 0, sd, 1.0)
        A_sub = (A_sub - mu) / sd
    elif normalize is not None:
        raise ValueError('normalize must be None, "global_z", or "per_horizon_z".')

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(
        A_sub,
        aspect="auto",
        origin=origin,
        interpolation=interpolation,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel("Horizon (h)")
    ax.set_ylabel(f"Time index (t), slice [{t_min}:{t_max}:{t_step}]")
    ax.set_title(title or "Horizon × Time Heatmap")

    if H > 30:
        ax.set_xticks(np.linspace(0, H - 1, 6, dtype=int))
    if A_sub.shape[0] > 40:
        ax.set_yticks(np.linspace(0, A_sub.shape[0] - 1, 6, dtype=int))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("A[t, h]")

    fig.tight_layout()

    # ---- Save figure step (requested) ----
    if figure_path is not None:
        figure_path = Path(figure_path)
        figure_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(figure_path / filename, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
    return fig, ax






import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_quantile_envelope_median(
    A,
    *,
    t_min=0,
    t_max=None,          # exclusive
    t_step=1,            # subsample time: A[t_min:t_max:t_step]
    q_outer=(10, 90),    # outer band quantiles
    q_inner=(25, 75),    # inner band quantiles
    figure_path=None,    # directory to save (Path or str). None -> no save
    filename="quantile_envelope.jpg",
    dpi=200,
    show=True,
):
    """
    Quantile envelope + median trajectory over horizons.
    A: np.ndarray of shape (T, H)
    Computes quantiles across time for each horizon h.
    """
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        raise ValueError("A must be a 2D numpy array of shape (T, H).")

    T, H = A.shape
    if t_max is None:
        t_max = T
    if not (0 <= t_min < T) or not (0 < t_max <= T) or t_min >= t_max:
        raise ValueError(f"Invalid t_min/t_max: got t_min={t_min}, t_max={t_max}, T={T}.")
    if t_step <= 0:
        raise ValueError("t_step must be a positive integer.")

    A_sub = A[t_min:t_max:t_step].astype(float, copy=False)
    horizon = np.arange(H)

    # quantiles across time (axis=0), per horizon
    lo_outer, hi_outer = np.nanpercentile(A_sub, q_outer, axis=0)
    lo_inner, hi_inner = np.nanpercentile(A_sub, q_inner, axis=0)
    med = np.nanpercentile(A_sub, 50, axis=0)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.fill_between(horizon, lo_outer, hi_outer, alpha=0.2, label=f"{q_outer[0]}–{q_outer[1]}%")
    ax.fill_between(horizon, lo_inner, hi_inner, alpha=0.35, label=f"{q_inner[0]}–{q_inner[1]}%")
    ax.plot(horizon, med, linewidth=2.0, label="Median")

    ax.set_xlabel("Horizon (h)")
    ax.set_ylabel("Value")
    ax.set_title("Quantile Envelope + Median Trajectory")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    if figure_path is not None:
        figure_path = Path(figure_path)
        figure_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(figure_path / filename, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
    return fig, ax


import numpy as np

def sample_S_at_t(
    Fs: np.ndarray,
    t: int,
    *,
    K: int = None,
    num_samples: int = 1000,
    mode: str = "knn",          # "knn" or "kernel"
    knn_k: int = None,
    kernel_sigma: float = None,
    normalize: bool = True,
    eps: float = 1e-8,
    rng: np.random.Generator = None,
):
    """
    Sample from an empirical conditional distribution for S_t by resampling past full vectors S_j,
    with weights based on similarity between contexts X_j and X_t.

    Fs[u, h] = s_u^{h+1}, shape (T, H).

    Context:
        X_u = (s_{u-1}^1, s_{u-2}^2, ..., s_{u-K}^K) = (Fs[u-1,0], Fs[u-2,1], ..., Fs[u-K,K-1])

    Candidates:
        j in {K, ..., t-1}  (strictly earlier times)
    """
    if rng is None:
        rng = np.random.default_rng()

    T, H = Fs.shape
    if K is None:
        K = min(H, 5)

    if t < K:
        raise ValueError(f"t={t} must be >= K={K} to build context.")
    if t <= 0:
        raise ValueError("t must be positive (needs past context).")

    def context(u: int) -> np.ndarray:
        return np.array([Fs[u - k, k - 1] for k in range(1, K + 1)], dtype=float)

    cand_idx = np.arange(K, t)  # j = K, ..., t-1
    if cand_idx.size == 0:
        raise ValueError(f"No candidates for t={t}. Increase t or reduce K.")

    Ss = Fs[cand_idx, :]                       # (N, H)
    Xs = np.vstack([context(j) for j in cand_idx])  # (N, K)
    Xt = context(t)                            # (K,)

    if normalize:
        mu = Xs.mean(axis=0, keepdims=True)
        sd = Xs.std(axis=0, keepdims=True) + eps
        Xs = (Xs - mu) / sd
        Xt = (Xt - mu.squeeze()) / sd.squeeze()

    dists = np.linalg.norm(Xs - Xt[None, :], axis=1)  # (N,)
    N = dists.size

    if mode == "knn":
        if knn_k is None:
            knn_k = int(np.sqrt(N)) + 1
        knn_k = min(knn_k, N)
        nn = np.argsort(dists)[:knn_k]
        weights = np.zeros(N, dtype=float)
        weights[nn] = 1.0 / knn_k

    elif mode == "kernel":
        if kernel_sigma is None:
            kernel_sigma = np.median(dists) + eps
        weights = np.exp(-0.5 * (dists / (kernel_sigma + eps)) ** 2)
        weights = weights / (weights.sum() + eps)
        if len(weights) > 1:
            weights[-1] = 1- np.sum(weights[:-1])
        else:
            weights[-1] = 1 
        # print(dists)
        # print(weights)
        # print('sum weights is ', np.sum(weights))

    else:
        raise ValueError("mode must be 'knn' or 'kernel'.")

    chosen_local = rng.choice(np.arange(N), size=num_samples, replace=True, p=weights)
    samples_t = Ss[chosen_local, :]  # (num_samples, H)

    return samples_t, weights, cand_idx


def sample_S_for_all_t(
    Fs: np.ndarray,
    start_t: int,
    *,
    K: int = None,
    num_samples: int = 1000,
    mode: str = "knn",
    knn_k: int = None,
    kernel_sigma: float = None,
    normalize: bool = True,
    random_state: int = None,
):
    """
    Calls sample_S_at_t for each t in [start_t, T-H], storing:
      - samples for S_t
      - weights over candidates
      - candidate indices
      - ground-truth trajectory S_t (= Fs[t,:])

    Returns
    -------
    out : dict with keys
        times        : (N_times,) int array
        gt           : (N_times, H) array, ground truth S_t
        samples      : (N_times, num_samples, H) array
        weights_list : list of length N_times, each (N_cand_t,) array
        cand_idx_list: list of length N_times, each (N_cand_t,) int array
    """
    rng = np.random.default_rng(random_state)
    T, H = Fs.shape
    if K is None:
        K = min(H, 5)

    end_t = T - H
    if start_t > end_t:
        raise ValueError(f"start_t={start_t} must be <= T-H={end_t}.")

    start_t_eff = max(start_t, K)
    times = np.arange(start_t_eff, end_t + 1)

    samples_list = []
    weights_list = []
    cand_idx_list = []
    gt_list = []

    for t in times:
        s_t_samples, w_t, cand_t = sample_S_at_t(
            Fs,
            t,
            K=K,
            num_samples=num_samples,
            mode=mode,
            knn_k=knn_k,
            kernel_sigma=kernel_sigma,
            normalize=normalize,
            rng=rng,
        )
        samples_list.append(s_t_samples)
        weights_list.append(w_t)
        cand_idx_list.append(cand_t)
        gt_list.append(Fs[t, :].astype(float))

    out = {
        "times": times,
        "gt": np.stack(gt_list, axis=0),                 # (N_times, H)
        "samples": np.stack(samples_list, axis=0),       # (N_times, num_samples, H)
        "weights_list": weights_list,                    # ragged
        "cand_idx_list": cand_idx_list,                  # ragged
    }
    return out



import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_samples_vs_true(
    out: dict,
    *,
    horizons=None,
    q_lo: float = 0.10,
    q_hi: float = 0.90,
    use_median: bool = True,
    max_points: int = None,
    title: str = None,
    figsize=(10, 5),
    figpath: str = None,   # NEW
    dpi: int = 200,
    show: bool = True,
):
    """
    Visualize sampled trajectories against ground-truth trajectories.

    If figpath is provided, figures are saved with suffixes:
        - "_timeseries.png"
        - "_coverage.png"
        - "_scatter.png"
    """

    times = np.asarray(out["times"])
    gt = np.asarray(out["gt"])              # (N, H)
    samples = np.asarray(out["samples"])    # (N, B, H)

    N, H = gt.shape
    if horizons is None:
        horizons = list(range(H))

    # optionally downsample time axis
    if max_points is not None and N > max_points:
        idx = np.linspace(0, N - 1, max_points).round().astype(int)
        times = times[idx]
        gt = gt[idx]
        samples = samples[idx]
        N = len(times)

    # compute summary stats over samples
    qL = np.quantile(samples, q_lo, axis=1)   # (N, H)
    qU = np.quantile(samples, q_hi, axis=1)   # (N, H)
    center = np.median(samples, axis=1) if use_median else np.mean(samples, axis=1)

    # ---------- Figure 1: per-horizon time series ----------
    nplots = len(horizons)
    fig1, axes = plt.subplots(
        nplots, 1,
        figsize=(figsize[0], max(figsize[1], 2.2 * nplots)),
        sharex=True
    )
    if nplots == 1:
        axes = [axes]

    for ax, h in zip(axes, horizons):
        ax.fill_between(times, qL[:, h], qU[:, h], alpha=0.25,
                        label=f"sample q[{q_lo:.2f},{q_hi:.2f}]")
        ax.plot(times, center[:, h], linewidth=1.5,
                label="sample median" if use_median else "sample mean")
        ax.plot(times, gt[:, h], linewidth=1.5, linestyle="--", label="true")
        ax.set_ylabel(f"h={h+1}")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("time t")
    if title is None:
        title = "Samples vs True (per horizon)"
    fig1.suptitle(title)
    axes[0].legend(loc="best")
    fig1.tight_layout()

    if figpath is not None:
        fig1.savefig(f"{figpath}_timeseries.png", dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig1)

    # ---------- Figure 2: empirical coverage ----------
    cover = (gt >= qL) & (gt <= qU)          # (N, H)
    cover_rate = cover[:, horizons].mean(axis=0)

    fig2, ax2 = plt.subplots(
        1, 1,
        figsize=(max(7, 1.2 * len(horizons)), 4)
    )
    ax2.bar([h + 1 for h in horizons], cover_rate)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlabel("horizon")
    ax2.set_ylabel(f"empirical coverage in q[{q_lo:.2f},{q_hi:.2f}] band")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_title("Empirical coverage of true S_t")

    fig2.tight_layout()
    if figpath is not None:
        fig2.savefig(f"{figpath}_coverage.png", dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig2)

    # ---------- Figure 3: true vs sample center scatter ----------
    fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6))
    x = center[:, horizons].reshape(-1)
    y = gt[:, horizons].reshape(-1)

    ax3.scatter(x, y, s=8, alpha=0.35)
    mn = min(x.min(), y.min())
    mx = max(x.max(), y.max())
    ax3.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.5)
    ax3.set_xlabel("sample center")
    ax3.set_ylabel("true S_t")
    ax3.set_title("True vs Sample Center")
    ax3.grid(True, alpha=0.3)

    fig3.tight_layout()
    if figpath is not None:
        fig3.savefig(f"{figpath}_scatter.png", dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig3)


import os
import numpy as np
import matplotlib.pyplot as plt

def plot_sample_trajectories_simple(
    out: dict,
    *,
    t_idx: int = 0,
    horizons=None,
    max_samples: int = 50,
    use_median: bool = True,
    alpha: float = 0.15,
    lw_sample: float = 1.0,
    lw_true: float = 2.5,
    figsize=(6, 4),
    figpath: str = None,
    dpi: int = 200,
    show: bool = True,
):
    """
    Simple visualization of sampled trajectories vs true trajectory.

    Parameters
    ----------
    out : dict
        Output of sample_S_for_all_t
    t_idx : int
        Index into out["times"] (which time step to visualize)
    horizons : list[int] or None
        Horizons to plot (0-indexed). If None, plot all.
    max_samples : int
        Max number of sampled trajectories to draw (for clarity)
    use_median : bool
        Plot median (True) or mean (False) of samples
    figpath : str or None
        If provided, save figure to this path
    """
    if t_idx >= len(out['gt']):
        return
    
    times = out["times"]
    gt = out["gt"][t_idx]                  # (H,)
    samples = out["samples"][t_idx]        # (B, H)
    B, H = samples.shape

    if horizons is None:
        horizons = list(range(H))

    h_idx = np.array(horizons)
    x = np.arange(1, len(h_idx) + 1)

    # subsample trajectories for visibility
    if B > max_samples:
        sel = np.random.choice(B, size=max_samples, replace=False)
        samples_plot = samples[sel][:, h_idx]
    else:
        samples_plot = samples[:, h_idx]

    center = (
        np.median(samples[:, h_idx], axis=0)
        if use_median else
        np.mean(samples[:, h_idx], axis=0)
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # sample trajectories
    for s in samples_plot:
        ax.plot(x, s, color="C0", alpha=alpha, linewidth=lw_sample)

    # sample center
    ax.plot(
        x, center,
        color="C1",
        linewidth=2.0,
        label="sample median" if use_median else "sample mean",
    )

    # true trajectory
    ax.plot(
        x, gt[h_idx],
        color="black",
        linestyle="--",
        linewidth=lw_true,
        label="true",
    )

    ax.set_xlabel("horizon h")
    ax.set_ylabel("score s")
    ax.set_title(f"Sampled trajectories vs true (t = {times[t_idx]})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    if figpath is not None:
        fig.savefig(figpath, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
