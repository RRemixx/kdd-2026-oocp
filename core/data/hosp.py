import numpy as np
import random
import pandas as pd
from datetime import timedelta
from epiweeks import Week
from core.data.data_template import TimeSeriesDataTemplate

def process_hosp_data(data_path, start_epiweek, end_epiweek):
    try:
        df = pd.read_csv(data_path, header=0)
        df['region'] = df['region'].astype(str)
        df['epiweek'] = df['epiweek'].astype(int)
        locations = sorted(df['region'].unique())
        df.sort_values(by='epiweek', inplace=True)
        df = df[(df['epiweek'] >= int(start_epiweek)) & (df['epiweek'] <= int(end_epiweek))].copy()
        df['epiweek'] = df['epiweek'].astype(str)
        return df, locations
    except Exception as e:
        print(f"Error processing data file: {e}")
        return None, None

class HospDataProcessor(TimeSeriesDataTemplate):
    def __init__(self, T_obs: int, H: int, N: int, data_args: dict):
        super().__init__(T_obs, H, N, data_args)
        self.data_path = data_args.get('data_path', '')
        target_regions = data_args.get('target_regions', None)
        start_epiweek = data_args.get('start_epiweek', '202001')
        end_epiweek = data_args.get('end_epiweek', '202612')
        df, regions = process_hosp_data(self.data_path, start_epiweek, end_epiweek)
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
        self.current_epiweeks = None
    
    def set_subset(self, subset):
        super().set_subset(subset)
        region, seed = subset.rsplit('_', 1)
        # set seeds for reproducibility
        np.random.seed(int(seed))
        random.seed(int(seed))
        self.current_df = self.df[self.df['region'] == region].copy()
        self.current_epiweeks = self.current_df['epiweek'].unique()

    def get_reference_time(self, t: int):
        return self.current_epiweeks[t]

    def get_observations(self, t: int, window: int):
        epiweek = self.get_reference_time(t)
        window_start = (Week.fromstring(epiweek) - timedelta(weeks=window)).cdcformat()
        mask = (self.current_df['epiweek'] >= window_start) & (self.current_df['epiweek'] < epiweek)
        obs_df = self.current_df.loc[mask].sort_values('epiweek')
        return obs_df['true_1'].to_numpy()

    def get_ground_truth(self, t: int):
        epiweek = self.get_reference_time(t)
        traj = []
        row = self.current_df[self.current_df['epiweek'] == epiweek]
        for h in range(1, self.H + 1):
           traj.append(float(row[f'true_{h}']))
        return np.array(traj)

    def get_trajectory_samples(self, t: int, random: bool = False):
        epiweek = self.get_reference_time(t)
        ground_truth = self.get_ground_truth(t)
        row = self.current_df[self.current_df['epiweek'] == epiweek]
        samples = np.full((self.N, self.H), np.nan)
        for h in range(1, self.H + 1):
            samples[0, h - 1] = float(row[f'pred_{h}'])
        return ground_truth, samples
