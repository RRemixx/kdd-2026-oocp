import numpy as np
from core.data.data_template import TimeSeriesDataTemplate

class LanePredictionDataOnlineProcessor(TimeSeriesDataTemplate):
    def __init__(self, T_obs: int, H: int, N: int, data_args: dict):
        super().__init__(T_obs, H, N, data_args)

        # Read path from data_args
        data_path = data_args.get('data_path', None)

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
        self.num_cases = self.trajectories.shape[0]
        # Number of time steps per case
        self.num_steps = self.trajectories.shape[1]

    def get_reference_time(self, t: int):
        """Returns the case and step indices for a given timestep t"""
        if t < 0 or t >= self.num_cases * self.num_steps:
            raise IndexError("Timestep t out of bounds.")
        
        case_idx = t // self.num_steps
        step_idx = t % self.num_steps
        return (case_idx, step_idx)

    def get_observations(self, t: int, window: int):
        """Get past trajectory observations for the specified time step"""
        (case_idx, step_idx) = self.get_reference_time(t)

        # If window is larger than available steps, use all available steps
        window = min(window, step_idx)
        
        # Get past ground truth trajectories (from step_idx-window to step_idx-1)
        if window > 0:
            past_obs = self.trajectories[case_idx, step_idx-window:step_idx, 0, 0, :]
        else:
            past_obs = np.empty((0, self.H, 2))
            
        return past_obs

    def get_ground_truth(self, t: int):
        """Get the ground truth trajectory for the specified time step"""
        case_idx, step_idx = self.get_reference_time(t)
        
        # Ground truth is the first row in the trajectories dimension
        ground_truth = self.trajectories[case_idx, step_idx, 0, :self.H, :]
        
        return ground_truth

    def get_trajectory_samples(self, t: int, random: bool = False):
        """Get sampled trajectory predictions for the specified time step"""
        case_idx, step_idx = self.get_reference_time(t)
        
        # Ground truth is the first row
        ground_truth = self.get_ground_truth(t)
        
        # Get all available predictions (excluding ground truth)
        available_samples = self.trajectories[case_idx, step_idx, 1:, :self.H, :]
        
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
        'data_path': 'data/data_online.npz'
    }
    processor = LanePredictionDataProcessor(T_obs=1000, H=5, N=9, data_args=data_args)
    
    import matplotlib.pyplot as plt

    # Specify case and step indices
    for step_idx in range(5):
        case_idx = 10
        
        t = case_idx * processor.num_steps + step_idx

        # Choose a window size for past observations (e.g., 5 time steps)
        window = 5

        # Retrieve data
        obs = processor.get_observations(t, window)
        gt = processor.get_ground_truth(t)
        _, samples = processor.get_trajectory_samples(t, random=True)

        # Visualize the trajectories
        plt.figure(figsize=(8, 6))
        if obs.shape[0] > 0:
            plt.plot(obs[:, 0], obs[:, 1], 'ko-', label='Observations')
        plt.plot(gt[:, 0], gt[:, 1], 'b-o', label='Ground Truth')
        for i, sample in enumerate(samples):
            # Label only the first sample for clarity
            plt.plot(sample[:, 0], sample[:, 1], '--', label='Sample Trajectories' if i == 0 else None)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Lane Prediction Visualization (Case 0, Step 10)')
        plt.legend()
        plt.show()