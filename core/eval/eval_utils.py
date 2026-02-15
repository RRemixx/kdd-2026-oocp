import numpy as np
import pandas as pd
import csv
from core.constants import *
from core.eval.metrics import *
from core.method.score_func import covered_union, covered_union_2d

def aggregate_results(all_results):
    """
    all_results: dict mapping subset -> results_dict
    returns: (mean_results, std_results) with same structure as a single results_dict
    """
    print("Aggregating results...")
    results_list = list(all_results.values())
    if not results_list:
        return {}, {}
    
    if len(results_list) == 1:
        # If only one result, return it directly
        return results_list[0], None

    sample = results_list[0]
    mean_results = {}
    std_results = {}
    
    # save horizon_coverage_t
    if 'horizon_coverage_t' in sample:
        mean_results['horizon_coverage_t'] = {}
        for k, v in all_results.items():
            mean_results['horizon_coverage_t'][k] = v['horizon_coverage_t']

    # Group metrics by structure
    two_level = [
        'interval_width', 'coverage_avg',
        'var', 'exponential_risk',
        'max_consecutive_violations',
        'var_h', 'exponential_risk_h',
        'max_consecutive_violations_h',
        'num_violations_nrw',
        'exponential_risk_nrw',
        'saregret',
    ]
    one_level = [
        'calibration_score_h', 'horizon_cov_overall',
        'worst_case_risk', 'worst_case_risk_h', 'avg_risk_nrw', 'monotonicity_score', 'dcr', 'avg_violations_aa', 'avg_exponential_risk_aa',
    ]
    array_level = ['horizon_coverage_t']
    scalar_level = ['calibration_score_hc', 'avg_interval_width_aa', 'avg_risk_aa']
    
    # compute average interval width here
    mean_interval_width = []
    for results in results_list:
        interval_width_vals = []
        for k1, subdict in results['interval_width'].items():
            for k2, val in subdict.items():
                interval_width_vals.append(val)
        mean_interval_width.append(float(np.mean(interval_width_vals)))
    mean_results['mean_interval_width'] = float(np.mean(mean_interval_width))
    std_results['mean_interval_width'] = float(np.std(mean_interval_width))
    
    # compute saregret here
    # average saregret for 0.1 (all horizons)
    saregret_01_vals = []
    for results in results_list:
        mean_saregret_01 = []
        for h in results['saregret'][0.1]:
            mean_saregret_01.append(results['saregret'][0.1][h])
        saregret_01_vals.append(np.mean(mean_saregret_01))
    mean_results['avg_saregret_01'] = float(np.mean(saregret_01_vals))
    std_results['avg_saregret_01'] = float(np.std(saregret_01_vals))
    # average saregret for all alphas (all horizons)
    saregret_all_vals = []
    for results in results_list:
        mean_saregret_all = []
        for alpha in results['saregret']:
            for h in results['saregret'][alpha]:
                mean_saregret_all.append(results['saregret'][alpha][h])
        saregret_all_vals.append(np.mean(mean_saregret_all))
    mean_results['avg_saregret_all'] = float(np.mean(saregret_all_vals))
    std_results['avg_saregret_all'] = float(np.std(saregret_all_vals))

    # two-level dict metrics
    for key in two_level:
        mean_results[key] = {}
        std_results[key] = {}
        for k1, subdict in sample[key].items():
            mean_results[key][k1] = {}
            std_results[key][k1] = {}
            for k2 in subdict:
                vals = [r[key][k1][k2] for r in results_list]
                mean_results[key][k1][k2] = float(np.mean(vals))
                std_results[key][k1][k2] = float(np.std(vals))

    # one-level dict metrics
    for key in one_level:
        mean_results[key] = {}
        std_results[key] = {}
        for k in sample[key]:
            vals = [r[key][k] for r in results_list]
            mean_results[key][k] = float(np.mean(vals))
            std_results[key][k] = float(np.std(vals))

    # Since T_obs for each subset can differ, we cannot aggregate horizon_coverage_t directly.
    # # array metrics
    # for key in array_level:
    #     mean_results[key] = {}
    #     std_results[key] = {}
    #     for k in sample[key]:
    #         arrs = [r[key][k] for r in results_list]
    #         stacked = np.stack(arrs, axis=0)
    #         mean_results[key][k] = np.mean(stacked, axis=0)
    #         std_results[key][k] = np.std(stacked, axis=0)

    # scalar metrics
    for key in scalar_level:
        vals = [r[key] for r in results_list]
        mean_results[key] = float(np.mean(vals))
        std_results[key] = float(np.std(vals))

    return mean_results, std_results

def save_results_flat_csv(results, csv_path):
    """
    Save the results dictionary to a single CSV file divided into three parts,
    separated by blank lines, rounding all numeric values (including alpha) to two decimal places.
    
    Parameters:
        results (dict): Output of EvalUQ.metrics(window=...)
        csv_path (str): Path where the CSV will be written.
    """
    alphas = sorted(results['interval_width'].keys())
    horizons = sorted(next(iter(results['interval_width'].values())).keys())
    cal_score_h = results['calibration_score_h']
    horizon_cov_overall = results['horizon_cov_overall']
    calibration_score_hc = results['calibration_score_hc']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        def separate_rows(metric_name):
            """
            Helper function to separate rows with a blank line.
            """
            writer.writerow([])
            writer.writerow([f"--- {metric_name} ---"])
            writer.writerow([])
        
        separate_rows("Interval Width & Coverage Average")

        # --- Part 1: interval_width & coverage_avg by (alpha, horizon) ---
        writer.writerow(["alpha", "horizon", "interval_width", "coverage_avg"])
        for alpha in alphas:
            for h in horizons:
                a = round(alpha, 3)
                iw = round(results['interval_width'][alpha][h], 3)
                cov = round(results['coverage_avg'][alpha][h], 3)
                writer.writerow([a, h, iw, cov])

        separate_rows("Calibration Score by Horizon")

        # --- Part 2: calibration_score_h by horizon ---
        writer.writerow(["horizon", "calibration_score_h"])
        for h in horizons:
            cs_h = cal_score_h.get(h, None)
            cs_h = round(cs_h, 3) if cs_h is not None else ''
            writer.writerow([h, cs_h])

        separate_rows("Horizon Coverage Overall")

        writer.writerow(["alpha", "horizon_cov_overall"])
        for alpha in alphas:
            a = round(alpha, 3)
            hc = horizon_cov_overall.get(alpha, None)
            hc = round(hc, 3) if hc is not None else ''
            writer.writerow([a, hc])

        separate_rows("Calibration Score for Horizon Coverage")

        writer.writerow(["calibration_score_hc"])
        cshc = round(calibration_score_hc, 3) if calibration_score_hc is not None else ''
        writer.writerow([cshc])
        
        def two_keys_to_rows(results_dict, key1name, key1_values, key2prefix, key2_values):
            header = [key1name] + [f"{key2prefix}_{key2}" for key2 in key2_values]
            writer.writerow(header)
            for key1 in key1_values:
                row = [round(key1, 3)]
                dict1 = results_dict.get(key1, {})
                for key2 in key2_values:
                    sc = dict1.get(key2, None)
                    sc = round(sc, 3) if sc is not None else ''
                    row.append(sc)
                writer.writerow(row)
        
        # Value at Risk (VaR), Exponential Risk and maximum Consecutive Violations
        separate_rows("Value at Risk")
        two_keys_to_rows(results['var'], 'alpha', alphas, 'risk_level', results['var'][alphas[0]].keys())
        
        separate_rows("Exponential Risk")
        two_keys_to_rows(results['exponential_risk'], 'alpha', alphas, 'theta', results['exponential_risk'][alphas[0]].keys())
        
        separate_rows("Maximum Consecutive Violations")
        two_keys_to_rows(results['max_consecutive_violations'], 'alpha', alphas, 'threshold', results['max_consecutive_violations'][alphas[0]].keys())
        
        separate_rows("Worst Case Risk")
        writer.writerow(["alpha", "worst_case_risk"])
        for alpha in alphas:
            wcr = round(results['worst_case_risk'][alpha], 3) if alpha in results['worst_case_risk'] else ''
            writer.writerow([round(alpha, 3), wcr])

        # Value at Risk (VaR), Exponential Risk and maximum Consecutive Violations per horizon
        separate_rows("Value at Risk per Horizon")
        two_keys_to_rows(results['var_h'], 'horizon', horizons, 'risk_level', results['var_h'][horizons[0]].keys())

        separate_rows("Exponential Risk per Horizon")
        two_keys_to_rows(results['exponential_risk_h'], 'horizon', horizons, 'theta', results['exponential_risk_h'][horizons[0]].keys())

        separate_rows("Maximum Consecutive Violations per Horizon")
        two_keys_to_rows(results['max_consecutive_violations_h'], 'horizon', horizons, 'threshold', results['max_consecutive_violations_h'][horizons[0]].keys())

        separate_rows("Worst Case Risk per Horizon")
        writer.writerow(["alpha", "worst_case_risk"])
        for h in horizons:
            wcr = round(results['worst_case_risk_h'][h], 3) if h in results['worst_case_risk_h'] else ''
            writer.writerow([round(h, 3), wcr])
        
        # without rolling window
        separate_rows("Average Risk without Rolling Window")
        writer.writerow(["alpha", "avg_risk_nrw"])
        for alpha in alphas:
            arv = round(results['avg_risk_nrw'][alpha], 3) if alpha in results['avg_risk_nrw'] else ''
            writer.writerow([round(alpha, 3), arv])
            
        separate_rows("Number of Violations without Rolling Window")
        two_keys_to_rows(results['num_violations_nrw'], 'alpha', alphas, 'threshold', results['num_violations_nrw'][alphas[0]].keys())

        separate_rows("Exponential Risk without Rolling Window")
        two_keys_to_rows(results['exponential_risk_nrw'], 'alpha', alphas, 'theta', results['exponential_risk_nrw'][alphas[0]].keys())
        
        separate_rows("Monotonicity Score per Horizon")
        writer.writerow(["horizon", "monotonicity_score"])
        for h in horizons:
            ms = round(results['monotonicity_score'][h], 3) if h in results['monotonicity_score'] else ''
            writer.writerow([round(h, 3), ms])

        separate_rows("Distribution Consistent Ratio per Horizon")
        writer.writerow(["horizon", "dcr"])
        for h in horizons:
            dcr = round(results['dcr'][h], 3) if h in results['dcr'] else ''
            writer.writerow([round(h, 3), dcr])
        
        separate_rows("Strongly Adaptive Regret per Horizon")
        writer.writerow(["alpha", "horizon", "saregret"])
        two_keys_to_rows(results['saregret'], 'alpha', alphas, 'horizon', horizons)
        

class EvalUQ:
    """
    Evaluation class that reads a JSON with columns:
      - time:                timestamp or index
      - alpha:               miscoverage rate
      - horizon:             integer horizon index
      - ground_truths:       a scalar (ground truth at that time & horizon)
      - prediction_intervals: array of shape (N, 2)

    After loading, it computes:
      1. coverage per time-step for each (alpha, horizon)
      2. horizon coverage for each alpha (averaged over horizons & time)
      3. various metrics via the metrics() method, including quantiles of rolling coverage
    """
    def __init__(self, json_path=None, window=10, risk_type='absolute', results_df=None, twodim=False, score_params=None, start_t=None):
        # 1) Load JSON
        if results_df is not None:
            self.df = results_df
        elif json_path is not None:
            print("Loading results from JSON:", json_path)
            self.df = pd.read_json(json_path, orient='records')
        else:
            raise ValueError("Either json_path or results_df must be provided.")
        self.df["alpha"] = self.df["alpha"].astype(float)
        self.score_params = score_params

        # 2) Identify unique alphas and horizons
        self.alphas = sorted(self.df['alpha'].unique(), key=lambda x: float(x))  # Ensure alphas are sorted numerically
        # self.alphas = [round(alpha, 2) for alpha in self.alphas]  # Round to two decimal places
        self.horizons = sorted(self.df['h'].unique())
        self.data = {alpha: {} for alpha in self.alphas}
        
        self.risk_type = risk_type
        self.risk_levels = risk_levels_c
        self.thetas = thetas_c
        self.thresholds = thresholds_c
        self.twodim = twodim
        self.start_t = start_t
        if twodim:
            self.init2d()
        else:
            self.init1d()
        self.remove_start_t(self.data, self.start_t)
        
        # compute T (number of time‐steps) for each (alpha, horizon)
        self.T = self.data[self.alphas[0]][self.horizons[0]]['y_truths'].shape[0]
        
        self.window = window
        if self.T < window*2:
            self.window = self.T // 2  # Ensure window is not larger than T
            print(f"Warning: T={self.T} is too small for window={window}. Using window={self.window} instead.")
        
        # Precompute coverage per time-step and horizon-coverage per alpha
        self.cov_per_time = self.compute_coverage_per_time()
        self.rolling_cov_per_horizon = compute_rolling_coverage_per_horizon(self.cov_per_time, self.alphas, self.horizons, window=self.window)
        self.horizon_cov_per_time = compute_horizon_coverage(self.cov_per_time, self.alphas, self.horizons)
        
        # only take alpha = 0.1
        target_alpha = 0.1
        self.rolling_cov_gaps_per_horizon = {}
        for h in self.horizons:
            self.rolling_cov_gaps_per_horizon[h] = rolling_horizon_coverage_gaps(self.rolling_cov_per_horizon[target_alpha][h], target_alpha, window=1, risk_type=self.risk_type)
        
        self.rolling_cov_gaps = {}
        for alpha in self.alphas:
            hc_vals_alpha = self.horizon_cov_per_time[alpha]
            self.rolling_cov_gaps[alpha] = rolling_horizon_coverage_gaps(hc_vals_alpha, alpha, window=self.window, risk_type=self.risk_type)
    
    def remove_start_t(self, dict, start_t):
        if start_t is None or start_t <= 0:
            return
        for alpha in self.alphas:
            for h in self.horizons:
                for key in dict[alpha][h]:
                    dict[alpha][h][key] = dict[alpha][h][key][start_t:]

    def init1d(self):
        for alpha in self.alphas:
            df_a = self.df[self.df['alpha'] == alpha]
            for h in self.horizons:
                df_ah = df_a[df_a['h'] == h].sort_values('time')
                times = df_ah['time'].to_list()
                y_arr = df_ah['ground_truth'].to_numpy(dtype=float)
                intervals_list = df_ah['prediction_interval'].to_list()
                scores_arr = df_ah['scores'].to_numpy()
                qs_arr = df_ah['qs'].to_numpy()
                if len(intervals_list) > 0:
                    intervals_arr = np.stack([np.asarray(x, dtype=float) for x in intervals_list], axis=0)
                else:
                    intervals_arr = np.zeros((0, 0, 2))
                self.data[alpha][h] = {
                    'times': times,
                    'y_truths': y_arr,
                    'intervals': intervals_arr,
                    'scores': scores_arr,
                    'qs': qs_arr,
                }

    def init2d(self):
        for alpha in self.alphas:
            df_a = self.df[self.df['alpha'] == alpha]
            for h in self.horizons:
                df_ah = df_a[df_a['h'] == h].sort_values('time')
                times = df_ah['time'].to_list()
                y_arr = df_ah['ground_truth'].to_numpy()
                sample_arr = df_ah['samples'].to_list()
                qs_arr = df_ah['qs'].to_numpy()
                scores_arr = df_ah['scores'].to_numpy()
                if len(sample_arr) > 0:
                    sample_arr = np.stack([np.asarray(x, dtype=float) for x in sample_arr], axis=0)
                else:
                    raise ValueError(f"No samples found for alpha={alpha}, horizon={h}")
                self.data[alpha][h] = {
                    'times': times,
                    'y_truths': y_arr,
                    'samples': sample_arr,
                    'qs': qs_arr,
                    'scores': scores_arr,
                }

    def compute_coverage_per_time(self):
        """
        Compute coverage per time-step for each (alpha, horizon).

        Returns:
            dict: cov[alpha][h] = numpy array of shape (T,), where
                  cov[alpha][h][t] = 1.0 if any interval at (t) covers y_truths[t], else 0.0
        """
        cov = {alpha: {} for alpha in self.alphas}

        for alpha in self.alphas:
            for h in self.horizons:
                entry = self.data[alpha][h]
                y_arr = entry['y_truths']
                T = y_arr.shape[0]
                coverage_t = np.zeros(T, dtype=float)
                
                if self.twodim:
                    sample_arr = entry['samples']             # shape: (T, N, 2)
                    qs_arr = entry['qs']                   # shape: (T,)
                    for t in range(T):
                        sample = sample_arr[t, :, :]       # shape: (N, 2)
                        qs = qs_arr[t]                   # scalar
                        ground_truth_t = y_arr[t]
                        coverage_t[t] = covered_union_2d(ground_truth_t, sample, qs, score_type=self.score_params['type'], optional_args=self.score_params.get('optional_args', None))
                else:
                    pi_arr = entry['intervals']             # shape: (T, N, 2)
                    for t in range(T):
                        intervals_t = pi_arr[t, :, :]       # shape: (N, 2)
                        ground_truth_t = y_arr[t]
                        coverage_t[t] = covered_union(ground_truth_t, intervals_t)
                cov[alpha][h] = coverage_t
        return cov

    def compute_time_varying_metrics(self):
        results_t = {
            'avg_risk': {alpha: [] for alpha in self.alphas},
            'exponential_risk': {theta: {alpha: [] for alpha in self.alphas} for theta in self.thetas},
            'coverage_per_time': self.cov_per_time,
            'pil_per_time': self.pil_per_time,
        }
        for alpha in self.alphas:
            hc_vals_alpha = self.horizon_cov_per_time[alpha]
            target_cov = 1 - alpha
            gaps = target_cov - hc_vals_alpha
            gaps = risk_type_helper(gaps, self.risk_type)
            results_t['avg_risk'][alpha] = gaps.copy()
            # R_θ = θ * log E[ exp( |e_t| / θ ) ]
            for theta in self.thetas:
                results_t['exponential_risk'][theta][alpha] = theta * np.log(np.exp(np.abs(gaps) / theta))
        return results_t

    def metrics(self):
        """
        Compute all metrics.

        Returns:
            dict with keys:
              'interval_width':         {alpha: {h: float}}
              'coverage_avg':           {alpha: {h: float}}
              'calibration_score_h':    {h: float}
              'horizon_coverage_t':     {alpha: numpy.ndarray shape (T,)}
              'horizon_cov_overall':    {alpha: float}
              'calibration_score_hc':   float
              'var':                    {alpha: {risk_level: float}}
              'exponential_risk':       {alpha: {theta: float}}
              'worst_case_risk':        {alpha: float}
              'max_consecutive_violations': {alpha: {threshold: int}}
              'var_h':                    {horizon: {risk_level: float}}
              'exponential_risk_h':       {horizon: {theta: float}}
              'worst_case_risk_h':        {horizon: float}
              'max_consecutive_violations_h': {horizon: {threshold: int}}
              
              # no rolling window here
              'exponential_risk_nrw': {alpha: {theta: float}}  # exponential risk over all horizons
              'avg_risk_nrw': {alpha: float}  # average risk over all horizons
              'num_violations_nrw': {alpha: {threshold: int}}  # total number of violations over all horizons
              'exponential_risk_nrw': {alpha: {theta: float}}  # exponential risk over all horizons
              'monotonicity_score': {horizon: float}
              'dcr': {horizon: float}
              
              # averaged metrics
              'avg_interval_width_aa': float,
              'avg_cs_aa': float,
              'avg_risk_aa': float,
              'avg_violations_aa': {threshold: float},
              'avg_exponential_risk_aa': {theta: float},
        """
        results = {
            'interval_width': {},
            'coverage_avg': {},
            'calibration_score_h': {},
            'horizon_coverage_t': {},
            'horizon_cov_overall': {},
            'calibration_score_hc': None,
            'var': {},
            'exponential_risk': {},
            'worst_case_risk': {},
            'max_consecutive_violations': {},
            'var_h': {},
            'exponential_risk_h': {},
            'worst_case_risk_h': {},
            'max_consecutive_violations_h': {},
            'avg_risk_nrw': {alpha: {} for alpha in self.alphas},  # average risk over all horizons
            'num_violations_nrw': {alpha: {} for alpha in self.alphas},  # total number of violations over all horizons
            'exponential_risk_nrw': {alpha: {} for alpha in self.alphas},
            'monotonicity_score': {},
            'dcr': {},
            'avg_interval_width_aa': None,
            'avg_risk_aa': None,
            'avg_violations_aa': {thr: None for thr in self.thresholds},
            'avg_exponential_risk_aa': {theta: None for theta in self.thetas},
            'saregret': {},
        }

        # 1) interval_width and coverage_avg per (alpha, horizon)
        coverage_by_h = {h: [] for h in self.horizons}
        self.pil_per_time = {alpha: {} for alpha in self.alphas}
        for alpha in self.alphas:
            results['interval_width'][alpha] = {}
            results['coverage_avg'][alpha] = {}
            for h in self.horizons:
                entry = self.data[alpha][h]
                cov_t = self.cov_per_time[alpha][h]     # shape: (T,)

                # average interval width at horizon h
                if self.twodim:
                    sample_arr = entry['samples']             # shape: (T, N, 2)
                    qs_arr = entry['qs']              # shape: (T,)
                    iw, pils = interval_area(sample_arr, qs_arr, score_type=self.score_params['type'], optional_args=self.score_params.get('optional_args', None), plot=False)
                else:
                    pi_arr = entry['intervals']             # shape: (T, N, 2)
                    iw, pils = interval_width(pi_arr)
                
                self.pil_per_time[alpha][h] = pils
                results['interval_width'][alpha][h] = iw

                # average coverage at horizon h
                cov_avg = float(np.mean(cov_t))
                results['coverage_avg'][alpha][h] = cov_avg

                coverage_by_h[h].append(cov_avg)

        # 2) calibration_score per horizon (across alphas)
        for h in self.horizons:
            cov_vals = coverage_by_h[h]
            results['calibration_score_h'][h] = calibration_score(cov_vals, self.alphas)

        # 3) horizon coverage time series and overall per alpha
        hc_overall_list = []
        for alpha in self.alphas:
            cov_t_alpha = self.horizon_cov_per_time[alpha]  # shape: (T,)
            results['horizon_coverage_t'][alpha] = cov_t_alpha

            hc_overall = horizon_coverage_overall(cov_t_alpha)
            results['horizon_cov_overall'][alpha] = float(hc_overall)
            hc_overall_list.append(hc_overall)

        # 5) calibration_score for horizon coverage across alphas
        results['calibration_score_hc'] = calibration_score(hc_overall_list, self.alphas)
        
        # 6) other horizon coverage related metrics per alpha
        results['var'] = {alpha: {} for alpha in self.alphas}
        results['exponential_risk'] = {alpha: {} for alpha in self.alphas}
        results['max_consecutive_violations'] = {alpha: {} for alpha in self.alphas}
        
        for alpha in self.alphas:
            gaps = self.rolling_cov_gaps[alpha]
            for risk_level in self.risk_levels:
                results['var'][alpha][risk_level] = var_at_risk(gaps, risk_level=risk_level)
            for theta in self.thetas:
                results['exponential_risk'][alpha][theta] = exponential_risk(gaps, theta=theta)
            results['worst_case_risk'][alpha] = worst_case_local_coverage_error(gaps)
            # max consecutive violations for thresholds
            results['max_consecutive_violations'][alpha] = {
                thr: max_consecutive_violations(gaps, thr) for thr in self.thresholds
            }
        
        # 7) horizon-wise metrics
        results['var_h'] = {h: {} for h in self.horizons}
        results['exponential_risk_h'] = {h: {} for h in self.horizons}
        results['worst_case_risk_h'] = {}
        results['max_consecutive_violations_h'] = {h: {} for h in self.horizons}

        for h in self.horizons:
            gaps = self.rolling_cov_gaps_per_horizon[h]
            for risk_level in self.risk_levels:
                results['var_h'][h][risk_level] = var_at_risk(gaps, risk_level=risk_level)
            for theta in self.thetas:
                results['exponential_risk_h'][h][theta] = exponential_risk(gaps, theta=theta)
            results['worst_case_risk_h'][h] = worst_case_local_coverage_error(gaps)
            # max consecutive violations for thresholds
            results['max_consecutive_violations_h'][h] = {
                thr: max_consecutive_violations(gaps, thr) for thr in self.thresholds
            }
    
        # 8) without rolling window
        for alpha in self.alphas:
            hc_vals_alpha = self.horizon_cov_per_time[alpha]
            target_cov = 1 - alpha
            gaps = target_cov - hc_vals_alpha
            gaps = risk_type_helper(gaps, self.risk_type)
            results['avg_risk_nrw'][alpha] = float(np.mean(gaps))
            results['num_violations_nrw'][alpha] = {
                thr: num_violations(gaps, thr)
                for thr in self.thresholds
            }
            results['exponential_risk_nrw'][alpha] = {}
            for theta in self.thetas:
                results['exponential_risk_nrw'][alpha][theta] = exponential_risk(gaps, theta=theta)
        
        # 9) monotonicity score per horizon
        try:
            monotonicity_scores = monotonicity_score(self.data, self.alphas, self.horizons)
            dcr = distribution_consistent_ratio(self.data, self.alphas, self.horizons)
            for h in self.horizons:
                results['monotonicity_score'][h] = monotonicity_scores[h]
                results['dcr'][h] = dcr[h]
        except Exception as e:
            print(f"Error computing monotonicity scores: {e}")
            results['monotonicity_score'] = {h: 0 for h in self.horizons}
            results['dcr'] = {h: 0 for h in self.horizons}
        
        
        # 10) averaged metrics across alphas
        results['avg_interval_width_aa'] = get_mean(results['interval_width'], level=2)
        results['avg_risk_aa'] = get_mean(results['avg_risk_nrw'], level=1)
        results['avg_violations_aa'] = {thr: get_mean({alpha: results['num_violations_nrw'][alpha][thr] for alpha in self.alphas}, level=1) for thr in self.thresholds}
        results['avg_exponential_risk_aa'] = {theta: get_mean({alpha: results['exponential_risk_nrw'][alpha][theta] for alpha in self.alphas}, level=1) for theta in self.thetas}
        results['avg_cs_aa'] = results['calibration_score_hc']
        
        # 11) strongly adaptive metrics can be added here
        for alpha in self.alphas:
            results['saregret'][alpha] = {h: -1 for h in self.horizons}  # Placeholder for strongly adaptive regret metrics
            for h in self.horizons:
                try:
                    current_qs = self.data[alpha][h]['qs']
                    current_scores = self.data[alpha][h]['scores']
                    saregret_val = strongly_adaptive_regret(scores=current_scores, qs=current_qs, alpha=alpha, k=20)
                    results['saregret'][alpha][h] = saregret_val
                except Exception as e:
                    print(f"Error computing strongly adaptive regret for alpha={alpha}, horizon={h}: {e}")
        return results
