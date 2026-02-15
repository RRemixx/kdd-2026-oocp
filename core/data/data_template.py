from abc import ABC, abstractmethod

class TimeSeriesDataTemplate(ABC):
    """
    Abstract template for any time‐series data generator / processor that:
      - keeps track of T_obs (length of observation period),
        H (forecast horizon), N (number of samples), and data_args.
      - exposes four methods:
          get_reference_time(t)
          get_observations(t, window)
          get_ground_truth(t)
          get_trajectory_samples(t, random=False)
    """

    def __init__(self, T_obs: int, H: int, N: int, data_args: dict):
        """
        Args:
            T_obs (int): Number of time‐steps in the observation period.
            H (int): Forecast horizon (number of future steps to predict).
            N (int): Number of sample trajectories to draw.
            data_args (dict): Arbitrary keyword arguments specific to each subclass.
        """
        self.T_obs = T_obs
        self.H = H
        self.N = N
        self.data_args = data_args
        self.subsets = ['default']
    
    def set_subset(self, subset: str):
        """
        Select which data subset should be used for subsequent calls.
        
        Args:
            subset (str): Name of the subset to activate (must be in self.subsets).
        
        Raises:
            ValueError: If the subset name is not recognized.
        """
        if subset not in self.subsets:
            raise ValueError(f"Subset '{subset}' not recognized. Available subsets: {self.subsets}")
        self.current_subset = subset
    
    def pretrain_split(self, fraction: float, total_fraction: float = 1.0):
        """
        Split the subsets into pretraining and evaluation sets based on a fraction.
        Args:
            fraction (float): Fraction of the data to use for pretraining (0.0 to 1.0).
        Raises:
            ValueError: If the fraction is not between 0.0 and 1.0.
        """
        if not (0.0 <= fraction <= 1.0):
            raise ValueError("Fraction must be between 0.0 and 1.0")
        self.pretrain_fraction = fraction
        # split the self.subsets to create a new pretrain_subset. Move the first fraction of subsets to pretrain_subset
        total_subsets = int(len(self.subsets) * total_fraction)
        num_pretrain = int(total_subsets * fraction)
        self.pretrain_subset = self.subsets[:num_pretrain]
        self.subsets = self.subsets[num_pretrain:total_subsets]
        print(f"Pretrain subsets: {self.pretrain_subset}, Eval subsets: {self.subsets}")
        return self.pretrain_subset, self.subsets

    @abstractmethod
    def get_reference_time(self, t: int):
        """
        Return a “reference” label (e.g., a date or timestep index) for index t.

        Args:
            t (int): Current time index (0‐based).

        Returns:
            Any: Some representation of time (e.g., datetime, int index) corresponding to t.
        """
        raise NotImplementedError

    @abstractmethod
    def get_observations(self, t: int, window: int):
        """
        Return the past “window” observations ending at time t (inclusive).

        Args:
            t (int): Current time index (0‐based).
            window (int): How many past time steps to include.

        Returns:
            np.ndarray or list or pd.Series: Observed values for [t‐window+1, …, t].
        """
        raise NotImplementedError

    @abstractmethod
    def get_ground_truth(self, t: int):
        """
        Return the ground‐truth future trajectory of length H starting at t+1.

        Args:
            t (int): Current time index (0‐based).

        Returns:
            np.ndarray or list: True values for [t+1, t+2, …, t+H].
        """
        raise NotImplementedError

    @abstractmethod
    def get_trajectory_samples(self, t: int, random: bool = False):
        """
        Draw (or retrieve) N sample trajectories of length H starting from t.

        Args:
            t (int): Current time index (0‐based).
            random (bool): If True, draw samples randomly; otherwise, use a fixed order.

        Returns:
            tuple:
              - y_truth (np.ndarray or list): True future trajectory for [t+1…t+H].
              - samp   (np.ndarray of shape (N, H)): N sampled prediction trajectories.
        """
        raise NotImplementedError
