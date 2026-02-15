import numpy as np
from datetime import datetime
from core.data.data_utils import generate_transition_matrix, simulate_markov_ar, sample_markov_ar_trajectories
from core.data.data_template import TimeSeriesDataTemplate

class MarkovARData(TimeSeriesDataTemplate):
    def __init__(self, T_obs: int, H: int, N: int, data_args: dict):
        super().__init__(T_obs, H, N, data_args)
        self.seeds = data_args.get('seeds', [42, 53, 64])
        self.subsets = self.seeds
        self.data = {}
        for seed in self.seeds:
            # Pull AR parameters out of data_args
            P     = data_args.get('P', None)
            k     = data_args.get('k', (0.8, -0.5))
            b     = data_args.get('b', (1.0, 0.0))
            sigma = data_args.get('sigma', (0.1, 0.1))

            # If no transition matrix provided, generate one
            self.P = generate_transition_matrix(self_prob=0.98, seed=seed)
            self.k = k
            self.b = b
            self.sigma = sigma

            # Simulate a full trajectory of length T_obs + H
            T_total = T_obs + H
            self.y_full, self.states_full = simulate_markov_ar(
                T_total, self.P, self.k, self.b, self.sigma, seed=seed
            )

            # Split into observed vs. future
            self.y_obs     = self.y_full[:T_obs]
            self.states_obs = self.states_full[:T_obs]
            self.data[seed] = (self.y_full, self.states_full, self.y_obs, self.states_obs)

    def set_subset(self, subset):
        """Set the current subset to the specified seed index."""
        self.current_subset = subset
        self.y_full, self.states_full, self.y_obs, self.states_obs = self.data[subset]
        self.seed = subset

    def get_reference_time(self, t: int):
        # Here we simply return the integer timestep.
        if t < 0 or t >= self.T_obs:
            raise IndexError("Timestep t out of bounds for T_obs.")
        return t

    def get_observations(self, t: int, window: int):
        start = max(0, t - window + 1)
        end   = t + 1
        return self.y_obs[start:end]

    def get_ground_truth(self, t: int):
        # Return true future y[t+1 : t+H+1]
        if t < 0 or t + self.H >= len(self.y_full):
            raise IndexError("Cannot get ground truth: t+H out of range.")
        return self.y_full[t+1 : t + self.H + 1]

    def get_trajectory_samples(self, t: int, random: bool = False):
        """
        y_truth: true future for [t+1 … t+H]
        samp:    N simulated trajectories of length H, starting from y_obs[t].
        """
        y_truth = self.get_ground_truth(t)
        y0 = self.y_obs[t]
        seed = self.seed + t if random else self.seed

        samp, _ = sample_markov_ar_trajectories(
            self.N, self.H, self.P, self.k, self.b, self.sigma, y0=y0, seed=seed
        )
        return y_truth, samp
