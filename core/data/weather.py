import random
import numpy as np
import pandas as pd
from core.data.data_template import TimeSeriesDataTemplate

def process_weather_data(data_path, start_idx, end_idx):
    try:
        df = pd.read_csv(data_path, header=0)
        df['region'] = df['region'].astype(str)
        locations = sorted(df['region'].unique())
        df['idx'] = df['idx'].astype(int)
        df.sort_values(by='idx', inplace=True)
        df = df[(df['idx'] >= int(start_idx)) & (df['idx'] <= int(end_idx))].copy()
        return df, locations
    except Exception as e:
        print(f"Error processing data file: {e}")
        return None, None

class WeatherDataProcessor(TimeSeriesDataTemplate):
    def __init__(self, T_obs: int, H: int, N: int, data_args: dict):
        super().__init__(T_obs, H, N, data_args)
        self.data_path = data_args.get('data_path', '')
        target_regions = data_args.get('target_regions', None)
        start_idx = data_args.get('start_idx', 0)
        end_idx = data_args.get('end_idx', 100)
        df, regions = process_weather_data(self.data_path, start_idx, end_idx)
        if target_regions is not None:
            target_regions = [loc for loc in target_regions if loc in regions]
        else:
            target_regions = regions
        self.seeds = data_args.get('seeds', [42])
        self.subsets = []
        for region in target_regions:
            for seed in self.seeds:
                self.subsets.append(f"{region}_{seed}")
        self.df = df
        self.current_df = None
        self.current_idx = None
    
    def set_subset(self, subset):
        super().set_subset(subset)
        region, seed = subset.rsplit('_', 1)
        # set seeds for reproducibility
        np.random.seed(int(seed))
        random.seed(int(seed))
        self.current_df = self.df[self.df['region'] == region].copy()
        self.current_idx = self.current_df['idx'].unique()

    def get_reference_time(self, t: int):
        return self.current_idx[t]

    def get_observations(self, t: int, window: int):
        idx = self.get_reference_time(t)
        window_start = idx - window
        mask = (self.current_df['idx'] >= window_start) & (self.current_df['idx'] < idx)
        obs_df = self.current_df.loc[mask].sort_values('idx')
        obs_df = obs_df[obs_df['ahead'] == 1]
        return obs_df['true'].to_numpy()

    def get_ground_truth(self, t: int):
        idx = self.get_reference_time(t)
        traj = []
        rows = self.current_df[self.current_df['idx'] == idx]
        for h in range(1, self.H + 1):
            current_row = rows[rows['ahead'] == h]
            traj.append(float(current_row[f'true']))
        return np.array(traj)

    def get_trajectory_samples(self, t: int, random: bool = False):
        idx = self.get_reference_time(t)
        ground_truth = self.get_ground_truth(t)
        rows = self.current_df[self.current_df['idx'] == idx]
        samples = np.full((self.N, self.H), np.nan)
        for h in range(1, self.H + 1):
            current_row = rows[rows['ahead'] == h]
            samples[0, h - 1] = float(current_row[f'pred'])
        return ground_truth, samples
