import numpy as np
import matplotlib.pyplot as plt

def generate_transition_matrix(self_prob=0.95, seed=None):
    """Generate a 2×2 Markov transition matrix with given self-transition probability."""
    if seed is not None:
        np.random.seed(seed)
    off = 1 - self_prob
    P = np.array([[self_prob, off],
                  [off,        self_prob]])
    return P

def simulate_markov_ar(T, P, k, b, sigma, seed=None):
    """
    Simulate a 2-state Markov AR(1) series of length T.
    Returns full series y_full and states_full of length T.
    """
    if seed is not None:
        np.random.seed(seed)
    y_full = np.zeros(T)
    states_full = np.zeros(T, dtype=int)
    states_full[0] = np.random.choice([0, 1])
    for t in range(1, T):
        prev = states_full[t-1]
        states_full[t] = np.random.choice([0, 1], p=P[prev])
        s = states_full[t]
        y_full[t] = k[s] * y_full[t-1] + b[s] + np.random.normal(0, sigma[s])
    return y_full, states_full

def sample_markov_ar_trajectories(N, H, P, k, b, sigma, y0=0.0, seed=None):
    """
    Sample N future trajectories (length H) starting from y0 under the same Markov AR model.
    Trajectories do not include y0.
    """
    rng = np.random.default_rng(seed)
    y_samples = np.zeros((N, H))
    state_samples = np.zeros((N, H), dtype=int)
    y_prev = y0
    state_samples[:, 0] = rng.integers(0, 2, size=N)
    for i in range(N):
        s = state_samples[i, 0]
        y_samples[i, 0] = k[s] * y_prev + b[s] + rng.normal(0, sigma[s])
    
    for t in range(1, H):
        for i in range(N):
            prev_s = state_samples[i, t-1]
            state_samples[i, t] = rng.choice(2, p=P[prev_s])
            s = state_samples[i, t]
            y_samples[i, t] = k[s] * y_samples[i, t-1] + b[s] + rng.normal(0, sigma[s])
    return y_samples, state_samples

# def sample_markov_ar_trajectories(N, H, P, k, b, sigma, y0=0.0, seed=None):
#     """
#     Sample N future trajectories (length H) starting from y0 under the same Markov AR model.
#     Trajectories include y0 as their first entry.
#     """
#     rng = np.random.default_rng(seed)
#     y_samples = np.zeros((N, H))
#     state_samples = np.zeros((N, H), dtype=int)
#     y_samples[:, 0] = y0
#     state_samples[:, 0] = rng.integers(0, 2, size=N)
#     for t in range(1, H):
#         for i in range(N):
#             prev_s = state_samples[i, t-1]
#             state_samples[i, t] = rng.choice(2, p=P[prev_s])
#             s = state_samples[i, t]
#             y_samples[i, t] = k[s] * y_samples[i, t-1] + b[s] + rng.normal(0, sigma[s])
#     return y_samples, state_samples

def plot_forecast_comparison(y_truth, forecast_samples, states):
    """
    Plot sample trajectories, ground-truth future, and state sequence.
    
    Args:
        y_truth (np.ndarray): ground truth trajectory
        forecast_samples (np.ndarray): shape (N, H) sampled trajectories
        states (np.ndarray): state sequence
    """
    N, H = forecast_samples.shape
    times = np.arange(H)
    
    plt.figure(figsize=(10, 6))
    # sampled trajectories
    for i in range(N):
        plt.plot(times, forecast_samples[i], alpha=0.3, color='blue', label='_nolegend_')
    # ground truth
    plt.plot(times, y_truth, linewidth=2.5, color='red', label='Ground truth')
    # state sequence
    plt.step(times, states, where='post', color='green', label='State')
    
    plt.xlabel("Time")
    plt.ylabel("Value / State")
    plt.title("Forecast vs Ground Truth")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_y_by_state(y, states):
    """
    Plot the series y with points colored by the regime state.

    Args:
        y (np.ndarray): 1D array of observations.
        states (np.ndarray): 1D array of integer state labels (e.g., 0 or 1).
    """
    t = np.arange(len(y))
    plt.figure(figsize=(10, 4))
    # points for state 0
    plt.scatter(t[states == 0], y[states == 0], label="State 0")
    # points for state 1
    plt.scatter(t[states == 1], y[states == 1], label="State 1")
    plt.xlabel("Time")
    plt.ylabel("y")
    plt.title("Time Series Colored by State")
    plt.legend()
    plt.tight_layout()
    plt.show()