import numpy as np
from sklearn.linear_model import LinearRegression
from core.data.data_template import TimeSeriesDataTemplate
import copy


class LinearRegressionDataGenerator(TimeSeriesDataTemplate):
    def __init__(self, T_obs: int, H: int, N: int, data_args: dict):
        """
        Initializes the LinearRegressionDataGenerator and generates the time series data.

        Args:
            T_obs (int): Number of time‐steps in the observation period.
                         This is the range [0, T_obs-1] from which 't' can be sampled.
            H (int): Forecast horizon (number of future steps to predict).
            N (int): Number of sample trajectories to draw.
            data_args (dict): Arbitrary keyword arguments specific to this generator:
                'slope' (float): Slope of the true underlying linear relationship.
                'intercept' (float): Intercept of the true underlying linear relationship.
                'noise_std' (float): Standard deviation of the Gaussian noise for observations.
                'feature_window' (int): The number of past observations to use as features
                                        for the linear regression model. This is the "dimension of features".
        """
        super().__init__(T_obs, H, N, data_args)

        self.slope = self.data_args.get('slope', 0.5)
        self.intercept = self.data_args.get('intercept', 10.0)
        self.noise_std = self.data_args.get('noise_std', 2.0)
        self.feature_window = self.data_args.get('feature_window', 10) # Default to 10 past observations
        split_ratios = self.data_args.get('split_ratios', [0.7, 0.15, 0.15])
        train_ratio, test_ratio, calibration_ratio = split_ratios

        # --- Data Generation in __init__ ---
        # Generate the full time series, including future steps needed for ground truth
        # The total length needed is T_obs (for observations) + H (for the last ground truth prediction)
        total_series_length = int((self.T_obs + self.H + 1) / test_ratio + 10)
        # Generate true underlying linear data
        self._true_series = np.array([self.slope * i + self.intercept for i in range(total_series_length)])

        # Generate observed data by adding noise to the true series
        self.observed_series = self._true_series[:total_series_length] + np.random.normal(0, self.noise_std, total_series_length)
        
        self.generate_and_split_dataset(
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            calibration_ratio=calibration_ratio,
            random_state=42,
        )
        
        print(f"Train split: {len(self.X_train)} samples")
        print(f"Test split: {len(self.X_test)} samples")
        print(f"Calibration split: {len(self.X_calib)} samples")

        # Initialize the linear regression model
        self.train_models(model_num=N)

        # For efficient training data creation in get_trajectory_samples,
        # we can pre-compute the (X, y) pairs for potential training.
        # However, the model training is part of `get_trajectory_samples`
        # as it needs to be trained on data *up to t*.
        # So, the model will be refitted or updated at each call to get_trajectory_samples.
    
    def generate_and_split_dataset(self,
                                   train_ratio: float = 0.7,
                                   test_ratio: float = 0.15,
                                   calibration_ratio: float = 0.15,
                                   random_state: int = 42):
        """
        Generates (X, y) data points from the observed time series, shuffles them,
        and splits them into training, testing, and calibration sets.

        Each X is `self.feature_window` past observed values.
        Each y is `self.H` future true values.

        Args:
            train_ratio (float): Proportion of data for the training set.
            test_ratio (float): Proportion of data for the test set.
            calibration_ratio (float): Proportion of data for the calibration set.
            random_state (int): Seed for reproducibility of shuffling.

        Returns:
            tuple: (X_train, y_train, X_test, y_test, X_calib, y_calib)
                Where X and y are numpy arrays.
        """
        if not np.isclose(train_ratio + test_ratio + calibration_ratio, 1.0):
            raise ValueError("Train, test, and calibration ratios must sum to 1.0")

        X_data, y_data = [], []

        # Iterate through the observed series to create (feature, target) pairs
        # The latest time point 't' for which we can extract an X and an H-step y
        # is T_obs - H (if y uses observed values) or T_obs (if y uses true values,
        # and _true_series extends beyond T_obs).
        # We need `feature_window` observations for X and `H` true values for y.
        # The last possible starting index for X is `T_obs - 1 - feature_window + 1`
        # and the last true value needed for y is `_true_series[T_obs + H - 1]`.
        # So, the loop should go up to `T_obs - 1 - feature_window + 1 - H`? No.
        # Let's consider the index `i` as the *start* of the feature window `X`.
        # Then `X` is `observed_series[i : i + feature_window]`
        # And `y` is `_true_series[i + feature_window : i + feature_window + H]`

        # The loop runs from the first possible starting point of `X`
        # to the last possible starting point of `X` such that `y` is still within `_true_series`
        # Smallest `i` is 0.
        # Largest `i` is `len(self.observed_series) - self.feature_window - self.H` if targets are also observed.
        # But if targets are `_true_series` and `_true_series` extends to `self.T_obs + self.H`,
        # then the last `i` is such that `i + self.feature_window + self.H - 1` is still within `_true_series`.
        # The latest point `X` can end is `self.T_obs - 1`. So, `i + self.feature_window - 1 <= self.T_obs - 1`.
        # Which means `i <= self.T_obs - self.feature_window`.
        # And the target `y` starts at `i + self.feature_window`. It ends at `i + self.feature_window + H - 1`.
        # This must be `<= len(self._true_series) - 1`.
        # So `i + self.feature_window + H - 1 <= self.T_obs + self.H - 1`.
        # `i + self.feature_window <= self.T_obs`.
        # This means `i` can go up to `self.T_obs - self.feature_window`.

        max_start_idx_for_X = len(self.observed_series) - self.feature_window
        if max_start_idx_for_X < 0:
            raise ValueError(f"Feature window ({self.feature_window}) is larger than observed series ({self.T_obs}). Cannot create data points.")

        for i in range(max_start_idx_for_X + 1):
            x_sample = self.observed_series[i : i + self.feature_window]
            y_sample = self._true_series[i + self.feature_window : i + self.feature_window + self.H]

            # Basic sanity check (should always be true with correct range)
            if len(x_sample) == self.feature_window and len(y_sample) == self.H:
                X_data.append(x_sample)
                y_data.append(y_sample)

        X_data = np.array(X_data)
        y_data = np.array(y_data)

        if len(X_data) == 0:
            raise ValueError("No data points could be created with the given feature_window and horizon. "
                             "Check T_obs, feature_window, and H parameters.")

        # Compute the number of samples for each split
        n_samples = len(X_data)
        n_train = int(np.round(train_ratio * n_samples))
        n_test = int(np.round(test_ratio * n_samples))
        n_calib = n_samples - n_train - n_test  # Ensure all samples are used

        # Shuffle indices for reproducibility
        rng = np.random.RandomState(random_state)
        indices = rng.permutation(n_samples)

        train_idx = indices[:n_train]
        test_idx = indices[n_train:n_train + n_test]
        calib_idx = indices[n_train + n_test:]

        self.X_train, self.y_train = X_data[train_idx], y_data[train_idx]
        self.X_test, self.y_test = X_data[test_idx], y_data[test_idx]
        self.X_calib, self.y_calib = X_data[calib_idx], y_data[calib_idx]
    
    def train_models(self, model_num):
        self.models = []
        n_train = len(self.X_train)
        split_size = n_train // model_num

        for i in range(model_num):
            start = i * split_size
            end = (i + 1) * split_size if i < model_num - 1 else n_train
            X_sub = self.X_train[start:end]
            y_sub = self.y_train[start:end]
            model = LinearRegression()
            model.fit(X_sub, y_sub)
            self.models.append(model)

    def get_reference_time(self, t: int):
        """
        Return the data index for test data
        """
        return t

    def get_observations(self, t: int, window: int):
        """
        Return the past 'window' observations ending at time t (inclusive).
        These observations would serve as features for a model.
        """
        if not (0 <= t < self.T_obs):
            raise ValueError(f"Index t={t} is out of test data bounds [0, {self.T_obs-1}]")
        if window > len(obs):
            raise ValueError(f"Window should be smaller than {len(obs)}")
        obs = self.X_test[t]
        return obs[-window:]

    def get_ground_truth(self, t: int):
        """
        Return the ground-truth future trajectory of length H starting at t+1.
        """
        if not (0 <= t < self.T_obs):
            raise ValueError(f"Index t={t} is out of test data bounds [0, {self.T_obs-1}]")
        return self.y_test[t]

    def get_trajectory_samples(self, t, random = False):
        y_truth = self.get_ground_truth(t)
        X_t = self.X_test[t]
        samp = np.array([model.predict(X_t.reshape(1, -1)).flatten() for model in self.models])
        return y_truth, samp
    
    def get_calib_data(self):
        ground_truths = []
        samps = []
        for i in range(len(self.X_calib)):
            current_X = self.X_calib[i]
            current_y = self.y_calib[i]
            samp = np.array([model.predict(current_X.reshape(1, -1)).flatten() for model in self.models])
            ground_truths.append(current_y)
            samps.append(samp)
        return np.array(ground_truths), np.array(samps)
        