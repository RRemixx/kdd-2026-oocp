import numpy as np
from core.data.data_template import TimeSeriesDataTemplate

class LanePredictionDataProcessor(TimeSeriesDataTemplate):
    def __init__(self, T_obs: int, H: int, N: int, data_args: dict):
        super().__init__(T_obs, H, N, data_args)

        # Read path from data_args
        data_path = data_args.get('data_path', None)
        max_cases = data_args.get('max_cases', 1000)

        # Load NPZ data
        data = np.load(data_path)
        self.trajectories = data['trajectories']  # Shape: (4773, 11, 10, 5, 2)
        self.serial_codes = data.get('serial_codes', None)
        
        # Validate dimensions
        if self.trajectories.shape[3] < H:
            raise ValueError(f"Prediction horizon in data ({self.trajectories.shape[3]}) is less than requested H={H}")
        if self.trajectories.shape[2] - 1 < N:  # -1 because first row is ground truth
            raise ValueError(f"Available trajectory samples ({self.trajectories.shape[2] - 1}) is less than requested N={N}")
        
        # Number of test cases
        self.num_cases = min(self.trajectories.shape[0], max_cases)
        # Number of time steps per case
        self.num_steps = self.trajectories.shape[1]
        
        # Initialize quantities for subset management
        self.subsets = np.arange(self.num_cases)
        self.current_subset = -1
        self._data = None
        
    def set_subset(self, case_idx: int):
        """Set the current subset to the specified case index."""
        if case_idx < 0 or case_idx >= self.num_cases:
            raise IndexError("Case index out of bounds.")
        self.current_subset = case_idx
        self._data = self.trajectories[case_idx]

    def get_reference_time(self, t: int):
        """Returns the case and step indices for a given timestep t"""
        return t

    def get_observations(self, t: int, window: int):
        """Get past trajectory observations for the specified time step"""
        # If window is larger than available steps, use all available steps
        window = min(window, t)
        # Get past ground truth trajectories (from step_idx-window to step_idx-1)
        if window > 0:
            past_obs = self.trajectories[self.current_subset, t-window:t, 0, 0, :]
        else:
            past_obs = np.empty((0, self.H, 2))
        return past_obs

    def get_ground_truth(self, t: int):
        """Get the ground truth trajectory for the specified time step"""
        return self._data[t, 0, :self.H, :]

    def get_trajectory_samples(self, t: int, random: bool = False):
        """Get sampled trajectory predictions for the specified time step"""
        # Ground truth is the first row
        ground_truth = self.get_ground_truth(t)
        
        # Get all available predictions (excluding ground truth)
        available_samples = self._data[t, 1:, :self.H, :]

        # Sample N trajectories
        num_to_sample = min(self.N, available_samples.shape[0])
        
        if random:
            # Random sampling
            indices = np.random.choice(available_samples.shape[0], num_to_sample, replace=False)
            samples = available_samples[indices]
        else:
            # Take the first N samples
            samples = available_samples[:num_to_sample]
        return ground_truth, samples


if __name__ == '__main__':
    # Example usage
    data_args = {
        'data_path': 'data/lane_data.npz'
    }
    processor = LanePredictionDataProcessor(T_obs=11, H=5, N=9, data_args=data_args)
    
    import matplotlib.pyplot as plt

    # Specify a single case index and activate it
    case_idx = 10
    processor.set_subset(case_idx)

    # Visualize the first 5 time steps (or fewer if num_steps < 5)
    for t in range(11):
        window = 5

        # Retrieve data
        obs = processor.get_observations(t, window)
        gt = processor.get_ground_truth(t)
        _, samples = processor.get_trajectory_samples(t, random=True)

        # Plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        if obs.size > 0:
            plt.plot(obs[:, 0], obs[:, 1], 'ko-', label='Observations')
        plt.plot(gt[:, 0], gt[:, 1], 'b-o', label='Ground Truth')
        for i, sample in enumerate(samples):
            plt.plot(
                sample[:, 0],
                sample[:, 1],
                '--',
                color='gray',
                alpha=0.7,
                label='Sample Trajectories' if i == 0 else None
            )
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Case {case_idx}, Step {t}')
        plt.legend()
        plt.savefig(f'case_{case_idx}_step_{t}.png')
        plt.close()