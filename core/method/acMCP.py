import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import warnings

from core.method.cpid import find_beta, cpid_update_step, get_params

warnings.filterwarnings("ignore")

def learned_scores_df_to_array(df: pd.DataFrame, n_samples: int=50):
    all_arrays = []
    for subset_name, group in df.groupby('subset'):
        pivot_df = group.pivot(index='time_idx', columns='horizon', values='score')
        pivot_df = pivot_df.sort_index(axis=0).sort_index(axis=1)
        train_data = pivot_df.to_numpy()
        all_arrays.append(train_data)
    combined_array = np.concatenate(all_arrays, axis=0)
    rand_indices = np.random.permutation(combined_array.shape[0])
    n_samples_ = min(n_samples, combined_array.shape[0])
    selected_array = combined_array[rand_indices[:n_samples_], :]
    return selected_array

class AcMCP:
    def __init__(self, alpha, gamma, power=1/2, d_factor=1.0, score_window=100, optional_args=None, additional_datasets=None):
        """
        Args:
          alpha (float): target miscoverage rate α
          gamma (float): learning rate γ for ACI step
          power (float): power for delta calculation
          d_factor (float): factor for delta calculation
        """
        self.alpha = alpha    # Target miscoverage rate
        self.gamma = gamma         # Learning rate
        self.t = 1                # Time step
        self.power = power         # Power for delta calculation
        self.d_factor = d_factor   # Factor for delta calculation
        self.q_min = 0.0        # Minimum quantile observed
        self.alpha_min = 0.0    # Minimum alpha
        self.alpha_max = 1.0    # Maximum alpha
        
        self.ncal = optional_args.get('ncal', 10)
        self.rolling = optional_args.get('rolling', False)
        self.scorecast = optional_args.get('scorecast', True)
        self.additional_datasets = additional_datasets
        self.run_lr = optional_args.get('run_lr', True)
        
        self.S_max = optional_args.get('max_score', 1.0)
        self.T = optional_args.get('T', 200)
        self.Csat, self.KI = get_params(self.T, self.S_max)
        # self.Csat = optional_args.get('Csat', 0.1)
        # self.KI = optional_args.get('KI', 1.0)
        self.integrate = optional_args.get('integrate', True)
        self.score_window = score_window  # Window size for score history
        
        self.init_range = None
        self.qt = None
        self.covereds = []
    
    def init_qt(self, current_scores):
        self.qt = np.quantile(current_scores, 1 - self.alpha)
        # print('Initial qt is:', self.qt)
        self.init_range = np.quantile(current_scores, 0.95) * 1.5
        # print('Initial range is:', self.init_range)
    
    def update_max_score(self, max_score):
        self.S_max = max_score
        self.Csat, self.KI = get_params(self.T, max_score)
        
    def blind_update(self):
        if self.qt is None:
            return 0, 0
        self.t += 1
        return self.qt, 0.0

    def update(self, in_interval, t, h, forecast_errors=None, new_lr_feature=None):
        # Indicator for miscoverage: 1 if outside interval, else 0
        self.covereds.append(in_interval)
        lr_t = 0.01 * np.max(forecast_errors) if forecast_errors is not None else self.gamma
        # PI update
        q_next = cpid_update_step(
            self.covereds,
            self.score_window,
            self.t,
            self.qt,
            self.alpha,
            lr_t,
            self.Csat,
            self.KI,
            self.integrate,
            max_score=self.S_max
        )
        self.qt = q_next
        # Scorecast adjustment
        bias_correction = 0.0
        if self.scorecast and len(forecast_errors) > self.ncal:
            if h == 0:
                bias_correction = np.mean(forecast_errors[self.ncal:t-h, h])
                # print('Using mean bias correction for h=0:', bias_correction)
            else:
                ma_val = self._run_ma(forecast_errors, t, h)
                # B. Linear Regression using previous scorecast predictions as features
                lr_val = self._run_lr(forecast_errors, new_lr_feature, t, h) if self.run_lr else ma_val
                bias_correction = (ma_val + lr_val) / 2
        if np.isnan(bias_correction):
            bias_correction = 0.0
        q_next += bias_correction        # Update state
        self.t += 1
        return q_next, bias_correction       

    def _run_ma(self, errors, t, h):
        """
        Trained on h-step-ahead forecast errors available up to time t.
        Original Notation: e_{1+h|1}, ..., e_{t|t-h}
        Our Notation: e_0^h, ..., e_{t-h}^h
        """
        # These are errors for the SPECIFIC horizon h realized in the past.
        # In our matrix errors[time, horizon], this is the column (h-1).
        # We only take values up to time t.
        history = errors[:t-h, h]
        history = history[~np.isnan(history)] # Remove lead-in NaNs
        
        if len(history) < self.ncal:
            return 0.0
        
        try:
            # ARIMA(0,0, h-1) as specified in the paper
            model = ARIMA(history, order=(0, 0, h - 1)).fit()
            # Forecast h steps ahead from the last known error e_{t|t-h}
            arima_forecast = model.forecast(steps=h)
            assert not np.isnan(arima_forecast[-1])
            return arima_forecast[-1]
        except:
            print('ARIMA forecast failed, returning mean')
            return np.mean(history)

    def _run_lr(self, errors, x_input, t, h, max_data_size=50):
        """
        Regressing e_{t+h|t} on past steps e_{t+h-1|t}, ..., e_{t+1|t}.
        Our Notation: e_t^h based on e_t^0, ..., e_t^{h-1}
        """
        # Training: We look at past origins 'i' where all horizons 1...h have realized.
        # We need to find origins where the error for horizon h is already known.
        # Horizon h from origin i is known at time i+h. So i + h <= t.
        data = []
        max_origin = t - h
        
        train_data = errors[self.ncal : max_origin, :]
        if self.additional_datasets is not None:
            train_data = np.concatenate((self.additional_datasets, train_data), axis=0)
        if max_data_size is not None and len(train_data) > max_data_size:
            train_data = train_data[-max_data_size:, :]
        if len(train_data) < 5:
            return 0.0
        # Split into features (horizons 1 to h-1) and target (horizon h)
        X_train, y_train = train_data[:, :h], train_data[:, h]
        model = LinearRegression().fit(X_train, y_train)
        
        # Inference: To predict bias for origin t and horizon h (e_{t+h|t}),
        # we use the already calculated scorecasts for horizons 1...(h-1)
        # as the predictors, just like the R code's 'newdata' logic.
        return model.predict(x_input.reshape(1, -1))[0]