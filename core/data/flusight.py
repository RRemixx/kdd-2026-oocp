import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path
from core.data.data_template import TimeSeriesDataTemplate

def process_flusight_ground_truth(ground_truth_path, location='US'):
    """
    Read the ground truth data from the a csv file and filter results into a DataFrame.
    
    Args:
        ground_truth_path (Path): Path to the ground truth directory
        location (str): Location to filter
    
    Returns:
        pd.DataFrame: Processed ground truth results
    """
    try:
        df = pd.read_csv(ground_truth_path, header=0)
        df['location'] = df['location'].astype(str)
        # Print unique locations present in the DataFrame
        df_locations = sorted(df['location'].unique())
        df_filtered = df[df['location'] == location]
        df_filtered.reset_index(drop=True, inplace=True)
        df_filtered.sort_values(by='date', inplace=True)
        return df_filtered, df_locations
    except Exception as e:
        print(f"Error processing ground truth data: {e}")
        return None

class FluSightDataProcessor(TimeSeriesDataTemplate):
    def __init__(self, T_obs: int, H: int, N: int, data_args: dict):
        super().__init__(T_obs, H, N, data_args)
        _, locations = process_flusight_ground_truth(Path('data/target-hospital-admissions.csv'), location='US')
        # print(f"Locations found in ground truth data: {locations}")
        # print(f"Total locations: {len(locations)}")
        locations = [loc for loc in locations if loc != '72' and loc != '25' and loc != '27']
        # print(f"Filtered locations (excluding '72'): {locations}")
        self.subsets = locations

        # # Read paths & threshold from data_args
        # team_predictions_path = data_args.get('team_predictions_path', 'data/flusight_predictions.csv')
        # ground_truth_path     = data_args.get('ground_truth_path',     'data/flusight_ground_truth.csv')
        # threshold             = data_args.get('threshold', 10)

        # # Load CSVs
        # self.predictions_df  = pd.read_csv(team_predictions_path, parse_dates=['date'])
        # self.ground_truth_df = pd.read_csv(ground_truth_path, parse_dates=['date'])
        # self.threshold       = threshold

        # # Determine valid dates that have ≥ threshold predictions
        # self.valid_dates = self.get_valid_dates()
        # if len(self.valid_dates) < T_obs:
        #     raise ValueError(f"Not enough valid dates: found {len(self.valid_dates)}, needed ≥ {T_obs}")

        # # Make sure valid_dates are datetime objects
        # self.valid_dates = pd.to_datetime(self.valid_dates)
    
    def set_subset(self, subset):
        self.current_subset = subset
        
        data_args = self.data_args
        csvs_path = data_args.get('csvs_path', 'data/flusight_csvs')
        team_predictions_path = Path(csvs_path) / f'flusight_predictions_wo_ensemble_{subset}.csv'
        ground_truth_path = Path(csvs_path) / f'FluSight_ground_truth_{subset}.csv'
        threshold             = data_args.get('threshold', 10)
        # Load CSVs
        self.predictions_df  = pd.read_csv(team_predictions_path, parse_dates=['date'])
        self.ground_truth_df = pd.read_csv(ground_truth_path, parse_dates=['date'])
        self.threshold       = threshold
        # Determine valid dates that have ≥ threshold predictions
        self.valid_dates = self.get_valid_dates()
        if len(self.valid_dates) < self.T_obs:
            raise ValueError(f"Not enough valid dates: found {len(self.valid_dates)}, needed ≥ {self.T_obs}")
        # Make sure valid_dates are datetime objects
        self.valid_dates = pd.to_datetime(self.valid_dates)

    def get_valid_dates(self):
        dates = np.sort(self.ground_truth_df['date'].unique())
        counts = []
        for d in dates:
            counts.append(self.predictions_df[self.predictions_df['date'] == d].shape[0])
        valid = dates[np.array(counts) >= self.threshold]
        return valid

    def get_reference_time(self, t: int):
        if t < 0 or t >= len(self.valid_dates):
            raise IndexError("Timestep t out of bounds of valid dates.")
        return self.valid_dates[t]

    def get_observations(self, t: int, window: int):
        date_t = self.get_reference_time(t)
        window_start = date_t - timedelta(weeks=window)
        mask = (self.ground_truth_df['date'] >= window_start) & (self.ground_truth_df['date'] < date_t)
        obs_df = self.ground_truth_df.loc[mask].sort_values('date')
        return obs_df['value'].to_numpy()

    def get_ground_truth(self, t: int):
        date_t = self.get_reference_time(t)
        traj = []
        for h in range(1, self.H + 1):
            target_date = date_t + timedelta(weeks=h)
            row = self.ground_truth_df[self.ground_truth_df['date'] == target_date]
            if not row.empty:
                # If the row is not empty, extract the value
                traj.append(float(row['value'].iloc[0]))
            else:
                # If the row is empty, append NaN
                # print(f"No ground truth data for date {target_date} on {self.current_subset}, appending NaN.")
                traj.append(0.0)  # or np.nan, depending on your preference
        return np.array(traj)

    def get_trajectory_samples(self, t: int, random: bool = False):
        date_t = self.get_reference_time(t)
        ground_truth = self.get_ground_truth(t)

        preds_at_t = self.predictions_df[self.predictions_df['date'] == date_t]
        if preds_at_t.empty:
            return ground_truth, np.empty((0, self.H))

        available = len(preds_at_t)
        num_to_sample = min(self.N, available)

        if random:
            sampled_df = preds_at_t.sample(n=num_to_sample, random_state=None)
        else:
            sampled_df = preds_at_t.head(num_to_sample)

        traj_cols = [f'pred_{i}' for i in range(self.H)]
        samples = []
        for _, row in sampled_df.iterrows():
            traj = [float(row[col]) if col in row.index else 0 for col in traj_cols]
            samples.append(traj)

        return ground_truth, np.array(samples)
