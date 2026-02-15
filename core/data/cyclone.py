import pickle
import numpy as np
import datetime
from typing import Any, Dict, List, Tuple

from core.data.data_template import TimeSeriesDataTemplate
from core.utils import load_pickle

class CycloneDataset(TimeSeriesDataTemplate):
    """
    TimeSeriesDataTemplate implementation for a preprocessed Cyclone dataset
    loaded from a pickle file.

    The pickle file should contain a dict of:
      track_id -> {
        init_time (datetime) -> {
          "times": List[datetime],
          "ground_truth": np.ndarray shape (H+1, 2),
          "samples": np.ndarray shape (H+1, N, 2)
        }
      }
    """

    def __init__(self, T_obs: int, H: int, N: int, data_args: Dict[str, Any]):
        """
        Args:
            pkl_path: Path to the preprocessed pickle file.
            T_obs: Number of past observations to return.
            H: Forecast horizon (number of future steps).
            N: Number of sample trajectories to draw.
        """
        super().__init__(T_obs, H, N, data_args)
        self.max_T_obs = T_obs
        self.pkl_path = data_args.get('pkl_path', '')
        self.samples_key = data_args.get('samples_key', 'samples')
        # load entire dataset from pickle
        self._all_data = load_pickle(self.pkl_path)
        
        
        print(f"Loaded Cyclone dataset from {self.pkl_path} with {len(self._all_data)} tracks.")
        for track_id, data in self._all_data.items():
            print(f"Track ID: {track_id}, Number of init_times: {len(data)}")

        self.track_id: str = ''
        self.init_times: List[datetime.datetime] = []
        self._track_data: Dict[datetime.datetime, Dict[str, Any]] = {}
        self.subsets = list(self._all_data.keys())

    def set_subset(self, track_id: str):
        """
        Select a specific track_id to use. Filters init_times and per-init data.
        """
        if track_id not in self._all_data:
            raise ValueError(f"Track ID '{track_id}' not found in data.")
        print(f"Setting track_id to {track_id} with {len(self._all_data[track_id])} init_times.")
        self.track_id = track_id
        self.current_subset = track_id
        # data for this track
        self._track_data = self._all_data[track_id]
        # sorted list of init_times
        self.init_times = list(self._track_data.keys())
        self.T_obs = min(self.max_T_obs, len(self.init_times))

    def get_reference_time(self, t: int) -> datetime.datetime:
        """
        Return the init_time corresponding to index t.
        """
        return self.init_times[t]

    def get_observations(self, t: int, window: int) -> np.ndarray:
        """
        No observations are returned.
        """
        return np.zeros((window, 2))

    def get_ground_truth(self, t: int) -> np.ndarray:
        """
        Return the true trajectory of length H starting at t (including lead_time=0).
        """
        ref_time = self.get_reference_time(t)
        data = self._track_data[ref_time]
        # ground_truth shape is (H+1, 2), take first H entries
        gt = data['ground_truth'][0 : self.H, :]
        return np.array(gt)

    def get_trajectory_samples(
        self, t: int, random: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve N sample trajectories of length H starting from the initial step.
        Does not skip the first lead_time=0 step.

        Returns:
            y_truth: np.ndarray of shape (H, 2)
            samples: np.ndarray of shape (N, H, 2)
        """
        ref_time = self.get_reference_time(t)
        data = self._track_data[ref_time]
        samples_all: np.ndarray = data[self.samples_key]  # shape (H+1, total_N, 2)

        # include first H steps (indices 0…H-1)
        samples = samples_all[0 : self.H, :, :]   # shape (H, total_N, 2)
        # transpose to (total_N, H, 2)
        samples = np.transpose(samples, (1, 0, 2))

        total_N = samples.shape[0]
        if total_N < self.N:
            raise ValueError(f"Not enough samples ({total_N}) for N={self.N}")

        indices = np.arange(total_N)
        if random:
            np.random.shuffle(indices)
        # indices = indices[: self.N]
        samp = samples[indices, :, :]

        # ground truth for the same H steps
        y_truth = self.get_ground_truth(t)  # shape (H, 2)
        return y_truth, samp