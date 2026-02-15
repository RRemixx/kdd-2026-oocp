import numpy as np
import pandas as pd
from tqdm import tqdm
from core.data import TimeSeriesDataTemplate
from core.method.score_func import *
# from core.method.copula import copulaCPTS
from core.method.copula_cpt import copulaCPTS
from core.method.score_func import inverse_score_function, score_function


def run_iid_simulation(method:str, params:dict, data_generator: TimeSeriesDataTemplate, alpha=0.3, T_obs=1000, H=15, N=20, start_t=10, twodim=False): 
    print("#" * 50)
    print(f"Running iid simulation with method: {method}, alpha: {alpha}, T_obs: {T_obs}, H: {H}, N: {N}, start_t: {start_t}")
    # Initialization
    scores = np.zeros((T_obs, H))
    qs = np.zeros((T_obs, H))
    samples = np.zeros((T_obs, N, H))
    S_max = params['data_params']['S_max'][params['score_func_args']['type']]
    
    score_func_args = params.get('score_func_args', {})
    score_function_type = score_func_args.get('type', 'abs-r')
    score_function_optional_args = score_func_args.get('optional_args', {})
    
    ground_truths = np.zeros((T_obs, H))
    prediction_intervals = np.zeros((T_obs, H, N, 2))  # For storing prediction intervals
    timestamps = []
    
    score_func = score_function
    if twodim:
        samples = np.zeros((T_obs, N, H, 2))  # For storing samples in 2D
        ground_truths = np.zeros((T_obs, H, 2))  # For storing ground truths in 2D
        score_func = score_function_2d
    
    # Generate scores for calibration dataset
    calib_y_truths, calib_samps = data_generator.get_calib_data()
    calib_scores = np.zeros((len(calib_y_truths), H))
    for t in range(len(calib_y_truths)):
        y_truth = calib_y_truths[t]
        samp = calib_samps[t]
        for h in range(H):
            calib_scores[t, h] = score_func(y_truth, samp, h, score_function_type, score_function_optional_args)
    
    # Generate qs prediction
    optional_cp_args = params.get('optional_args', {})
    if method == 'copulacp':
        cp_method = copulaCPTS
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods are 'copulacp'.")
    mh_method = cp_method(optional_cp_args)
    qs_val = mh_method.predict(cali_scores=calib_scores, epsilon=alpha)
    for t in range(T_obs):
        qs[t] = qs_val
    
    # Generate scores for test dataset
    for t in range(T_obs):
        current_time = data_generator.get_reference_time(t)
        timestamps.append(current_time)
        y_truth, samp = data_generator.get_trajectory_samples(t, random=False)
        samples[t] = samp
        for h in range(H):
            ground_truths[t, h] = y_truth[h]
            scores[t, h] = score_func(y_truth, samp, h, score_function_type, score_function_optional_args)
    
    # Main simulation loop
    if not twodim:
        for t in tqdm(range(T_obs)):
            # Store ground truth and prediction intervals
            for h in range(H):
                current_prediction_intervals = inverse_score_function(qs[t, h], samples[t, :, h], score_function_type, score_function_optional_args)
                for i in range(len(current_prediction_intervals)):
                    prediction_intervals[t, h, i, 0] = current_prediction_intervals[i][0]
                    prediction_intervals[t, h, i, 1] = current_prediction_intervals[i][1]
    
    if twodim:
        return timestamps, ground_truths, (samples, qs)
    return timestamps, ground_truths, prediction_intervals


def run_multi_horizon_simulation(method:str, params:dict, data_generator: TimeSeriesDataTemplate, alpha=0.3, T_obs=1000, H=15, N=20, start_t=10, twodim=False, learned_scores_df=None): 
    print(f"Running multi horizon simulation with method: {method}, alpha: {alpha}, T_obs: {T_obs}, H: {H}, N: {N}, start_t: {start_t}")
    # Initialization
    scores = np.zeros((T_obs, H)) #(11,5)
    qs = np.zeros((T_obs, H)) #(11,5)

    covered_all = np.zeros((T_obs, H))
    samples = np.zeros((T_obs, N, H))

    S_max = params['data_params']['S_max'][params['score_func_args']['type']] # s_max = 2
    
    score_func_args = params.get('score_func_args', {}) 
    #  type: pcp
    #       optional_args:
    #           quantile_d: 0.2

    score_function_type = score_func_args.get('type', 'abs-r') # score_function_type =  pcp
    score_function_optional_args = score_func_args.get('optional_args', {}) #quantile_d: 0.2
    score_func = score_function # mini pc score function
    
    ground_truths = np.zeros((T_obs, H)) #(11,5)
    prediction_intervals = np.zeros((T_obs, H, N, 2))  # For storing prediction intervals
    timestamps = []

    if twodim:
        samples = np.zeros((T_obs, N, H, 2))  # For storing samples in 2D
        ground_truths = np.zeros((T_obs, H, 2))  # For storing ground truths in 2D
        score_func = score_function_2d
    
    # T_obs: 11, H: 5, N: 9, start_t: 2
    for t in range(start_t):
        y_truth, samp = data_generator.get_trajectory_samples(t, random=False)
        # print(y_truth.shape)
        # print(samp.shape)
        samples[t] = samp
        if samp.shape[0] > N:
            cur_samp = samp[:N]
        else:
            cur_samp = samp
        for h in range(H):
            # Compute scores (scores computed at this time step are not observed yet)
            scores[t, h] = score_func(y_truth, cur_samp, h, type=score_function_type, optional_args=score_function_optional_args)
    # print(scores.shape)
    if learned_scores_df is not None:
        df = learned_scores_df[learned_scores_df['time_idx']>= start_t].copy()
        filter_df = (df.pivot_table(index=["subset", "time_idx"], columns="horizon", values="score", aggfunc="first").reindex(columns=range(0, H)))
        filter_df.columns.name=None
        filter_df = filter_df.reset_index()    
    # print(learned_scores_df)
    # print(filter_df)

    # TODO: initialize the Copula method here

    for t in tqdm(range(T_obs)):
        if t < start_t: # Before the cold start time, we do not make predictions
            qs[t] = np.ones(H) * S_max
        else:
            calibration_scores = filter_df[filter_df['time_idx']==t].copy()
            cp_method = copulaCPTS(calibration_scores, H)
            qs[t] = cp_method.predict(epsilon=alpha)
        #     # TODO: predict the qs at time t
        #     pass
        #     # qs[t] = mh_method.predict(cali_scores=scores[:t, :], epsilon=alpha)
        #     # Store ground truth and prediction intervals
        # # TODO: after implementing qs prediction, uncomment the following
        # # save prediction intervals
        # print(qs)
        for h in range(H):
            ground_truths[t, h] = y_truth[h]
            if not twodim:
                current_prediction_intervals = inverse_score_function(qs[t, h], samples[t, :, h], score_function_type, score_function_optional_args)
                for i in range(len(current_prediction_intervals)):
                    prediction_intervals[t, h, i, 0] = current_prediction_intervals[i][0]
                    prediction_intervals[t, h, i, 1] = current_prediction_intervals[i][1]

    if twodim:
        return timestamps, ground_truths, (samples, qs), scores, qs
    return timestamps, ground_truths, prediction_intervals, scores, qs
