from ast import If
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.data import TimeSeriesDataTemplate
from core.method.score_func import *
from core.method.optim import *
from core.method.cp_utils import *

def get_scores_filter_dict(h, t):
    return {
        'horizon': (h, 0),
    }

def get_additional_Fs(learned_scores, score_window, start_idx=0):
    subsets = learned_scores['subset'].unique()
    horizons = learned_scores['horizon'].unique()
    Fs_records = []
    for subset in subsets:
        other_scores = learned_scores[learned_scores['subset'] != subset]
        subset_scores = learned_scores[learned_scores['subset'] == subset]
        subset_scores = subset_scores.sort_values(by='time')
        for horizon in horizons:
            score_filter_dict = get_scores_filter_dict(horizon, t=None)
            additional_scores = get_proximal_scores(other_scores, info_dict=score_filter_dict)
            scores_h_ = subset_scores[subset_scores['horizon'] == horizon]['score'].values.reshape(-1)
            for i in range(start_idx, len(scores_h_)):
                scores_h = scores_h_.copy()
                if i > horizon:
                    past_scores = scores_h[:i-horizon]
                    combined_scores = concat_scores(past_scores, additional_scores, score_window)
                else:
                    combined_scores = additional_scores[-score_window:]
                beta_i = compute_beta(combined_scores, scores_h[i], alpha_minimum=0.0, alpha_maximum=1.0)
                current_df_entry = subset_scores[(subset_scores['horizon'] == horizon) & (subset_scores['time_idx'] == i)]
                # print('Number of records in current_df_entry:', len(current_df_entry), 'Number of records in subset_scores:', len(subset_scores))
                Fs_records.append({
                    'time': current_df_entry['time'].values[0],
                    'time_idx': i,
                    'horizon': horizon,
                    'subset': subset,
                    'beta': beta_i,
                    'score': scores_h[i],
                    'prediction_mean_0': current_df_entry['prediction_mean_0'].values[0],
                    'prediction_mean_1': current_df_entry['prediction_mean_1'].values[0],
                    'ground_truth_0': current_df_entry['ground_truth_0'].values[0],
                    'ground_truth_1': current_df_entry['ground_truth_1'].values[0],
                })
    Fs_df = pd.DataFrame(Fs_records)
    return Fs_df

#---- Get Context Functions ----#
def robust_min_max_normalize(arr):
    min_val = np.percentile(arr, 5)
    max_val = np.percentile(arr, 95)
    normalized = (arr - min_val) / (max_val - min_val + 1e-8)
    normalized = np.clip(normalized, 0.0, 1.0)
    return normalized

def get_context_from_other_subsets(df, optim_arg, start_t=0):
    # Context parameters
    traj_type = optim_arg.get('traj_type', 'betas') # 'betas' or 'scores'
    traj_data = None
    data_options = optim_arg.get('data_options', {})
    
    data2use = [k for k in data_options if data_options[k].get('use', False)]
    dat2col = {'scores': ['score'], 'betas': ['beta'], 'ground_truth': ['ground_truth_0', 'ground_truth_1'], 'preds': ['prediction_mean_0', 'prediction_mean_1']}
    
    def get_data_from_subset_df(subset_df, data_type):
        value_cols = dat2col[data_type]
        data_arrays = []
        for col in value_cols:
            pivot = subset_df.pivot(index='time_idx', columns='horizon', values=col).sort_index()
            pivot = pivot.sort_index(axis=1)
            data_array = pivot.values  # (T_subset, H)
            data_array = data_array.astype(float)
            data_arrays.append(data_array)
        if len(data_arrays) == 1:
            return data_arrays[0]
        elif len(data_arrays) == 2:
            data_stacked = np.stack(data_arrays, axis=-1)  # (T_subset, H, 2)
            data_stacked = data_stacked.mean(axis=-1)  # (T_subset, H)
            return data_stacked
        return None
    
    additional_Ss_list = []
    additional_Xs_list = []
    data_types_needed = list(data2use)
    if traj_type not in data_types_needed:
        data_types_needed.append(traj_type)
    for _, subset_df in df.groupby('subset'):
        if subset_df.empty:
            continue
        data_dict = {}
        for data_type in data_types_needed:
            data_array = get_data_from_subset_df(subset_df, data_type)  # (T_subset, H)
            data_dict[data_type] = data_array
        traj_data = data_dict.get(traj_type)
        if traj_data is None:
            continue
        raw_input = []
        max_context_size = 0
        for data_type in data_types_needed:
            data = data_dict.get(data_type)
            if data_type not in data_options or not data_options[data_type].get('use', False) or data is None:
                continue
            max_context_size = max(max_context_size, data_options[data_type].get('context_size', 2))
            raw_input.append({
                'data': data,
                'data_type': data_type,
                'context_size': max_context_size,
                'context_option': data_options[data_type].get('context_option', 'triangle')
            })
        if not raw_input:
            continue
        new_start_t = max(start_t, raw_input[0]['data'].shape[1], max_context_size)
        if traj_data.shape[0] <= new_start_t:
            continue
        candidate_index = np.arange(new_start_t, traj_data.shape[0])
        Ss = traj_data[candidate_index, :]
        Xs = np.array([get_context_t(raw_input, j) for j in candidate_index])
        additional_Ss_list.append(Ss)
        additional_Xs_list.append(Xs)

    if not additional_Ss_list:
        return None
    additional_Ss = np.vstack(additional_Ss_list)
    additional_Xs = np.vstack(additional_Xs_list)
    print(f"Additional context from other subsets: {additional_Ss.shape[0]} samples.")
    return additional_Ss, additional_Xs

def get_scores_like_context(scores, t, context_option='triangle', context_size=2):
    """Extract context from scores-like array (Ground truths, scores or betas)
    Args:
        scores (np.ndarray): Array of scores with shape (T, H).
        t (int): Current time index.
        beta_context_option (str, optional): Option for context extraction. Defaults to 'triangle'. Options are 'triangle' and 'line'.

    Returns:
        np.ndarray: Extracted context at time t based on the specified option.
    """
    T, H = scores.shape
    square = scores[t-H-1:t, :]
    if square.shape[0] == 0:
        raise ValueError(f"No context available for time t={t} with horizon H={H}.")
    if context_option == 'triangle':
        mask = np.fliplr(np.triu(np.ones_like(square, dtype=bool)))
        return square[mask][:context_size]
    if context_option == 'line':
        n_rows, n_cols = square.shape
        n = min(n_rows, n_cols)
        return square[np.arange(n), n_cols - 1 - np.arange(n)][:context_size]
    raise ValueError("context_option must be 'triangle' or 'line'.")

def get_pred_like_context(predictions, t, context_size=2):
    """Extract context from prediction-like array (Base model predictions)
    Args:
        predictions (np.ndarray): Array of predictions with shape (T, H).
        t (int): Current time index.

    Returns:
        np.ndarray: Extracted context at time t.
    """
    return predictions[t, :][:context_size]

def get_context_t(raw_input, t):
    """Get context at time t.

    Args:
        raw_input (list[dict]): List of dictionaries containing raw input data and parameters. Each dictionary should have the following keys:
            - 'data' (np.ndarray): Array of data with shape (T, H) or (T, H, 2).
            - 'data_type' (str): Type of data ('scores', 'ground_truth', 'betas', 'preds').
            - 'context_size' (int): Size of the context to extract.
            - 'context_option' (str): Option for context extraction ('triangle' or 'line').
        t (int): Current time index.
    Returns:
        np.ndarray (#inputs x #context_size): Extracted contexts from all raw inputs at time t.
    """
    # Get contexts from each raw input
    contexts = []
    for input_dict in raw_input:
        data = input_dict['data'].copy()
        context_size = input_dict.get('context_size', 2)
        context_option = input_dict.get('context_option', 'triangle')
        data_type = input_dict['data_type']
        if data.ndim == 3:
            data = data.mean(axis=-1)
        if data_type in ('scores', 'ground_truth', 'betas'):
            context = get_scores_like_context(data, t, context_option=context_option, context_size=context_size)
        elif data_type == 'preds':
            context = get_pred_like_context(data, t, context_size=context_size)
        else:
            raise ValueError("data_type must be 'scores', 'ground_truth', 'betas', or 'preds'.")
        contexts.append(context)
    return np.array(contexts)

# ----- End of Get Context Functions ----- #

def get_samples_t(data_dict, cali_scores, t, start_t, optim_arg=None, additional_context=None, return_true_scale_samples=False):
    optim_arg = optim_arg or {}
    # Sampler parameters
    num_samples = optim_arg.get('n_traj_samples', 200)
    mode = optim_arg.get('mode', 'kernel')
    knn_k = optim_arg.get('knn_k', None)
    kernel_sigma = optim_arg.get('kernel_sigma', None)
    eps = optim_arg.get('eps', 1e-8)
    weight_threshold = optim_arg.get('weight_threshold', 0.5)
    # Context parameters
    traj_type = optim_arg.get('traj_type', 'betas') # 'betas' or 'scores'
    traj_data = None
    data_options = optim_arg.get('data_options', {})
    raw_input = []
    max_context_size = max([data_options[data_type].get('context_size', 2) for data_type in data_dict if data_type in data_options and data_options[data_type].get('use', False)] or [2])
    for data_type, data in data_dict.items():
        if data_type == traj_type:
            traj_data = data
        if data_type not in data_options or not data_options[data_type].get('use', False) or data is None:
            continue
        # TODO: now the code can only support same context_size for all data types, need to generalize later. Now we use the minimum context_size among all data types.)
        raw_input.append({
            'data': data,
            'data_type': data_type,
            'context_size': max_context_size,
            'context_option': data_options[data_type].get('context_option', 'triangle')
        })
    max_additional_context = optim_arg.get('max_additional_context', 1000)
        
    new_start_t = max(start_t, raw_input[0]['data'].shape[1], max_context_size)
    Ss, Xs = None, None
    a_Ss, a_Xs = None, None
    if t > new_start_t:
        candidate_index = range(new_start_t, t)
        Ss = traj_data[candidate_index, :] #(N, H)
        Xs = np.array([get_context_t(raw_input, j) for j in candidate_index]) # (N, #inputs, #context_size)
    if additional_context is not None:
        a_Ss, a_Xs = additional_context
        a_Ss = a_Ss[:max_additional_context] #(M, H)
        a_Xs = a_Xs[:max_additional_context] #(M, #inputs, #context_size)
    # Concat Ss and s_Ss
    if Ss is not None and a_Ss is not None:
        Ss = np.vstack([Ss, a_Ss])
        Xs = np.vstack([Xs, a_Xs])
    elif Ss is None and a_Ss is None:
        raise ValueError(f"No candidate indices available for time t={t}.")
    elif Ss is None and a_Ss is not None:
        Ss = a_Ss
        Xs = a_Xs
    
    # Get Xt (#inputs x #context_size)
    Xt = get_context_t(raw_input, t)[None, :, :]
        
    # Normalize Xs and Xt, calculate min / max along the second axis
    min_val = np.quantile(Xs, 0.05, axis=(0, 2), keepdims=True)
    max_val = np.quantile(Xs, 0.95, axis=(0, 2), keepdims=True)    
    Xs = (Xs - min_val) / (max_val - min_val + eps)
    Xt = (Xt - min_val) / (max_val - min_val + eps)    
    Xs = Xs.reshape(Xs.shape[0], -1)
    Xt = Xt.reshape(-1)
    H = Ss.shape[1]
    
    # Compute distances and weights
    dists = np.linalg.norm(Xs - Xt[None, :], axis=1)  # (N,)
    N = dists.size

    if mode == "knn":
        if knn_k is None:
            knn_k = int(np.sqrt(N)) + 1
        knn_k = min(knn_k, N)
        nn = np.argsort(dists)[:knn_k]
        weights = np.zeros(N, dtype=float)
        weights[nn] = 1.0 / knn_k

    elif mode == "kernel":
        if kernel_sigma is None:
            kernel_sigma = np.median(dists) + eps
        weights_raw = np.exp(-0.5 * (dists / (kernel_sigma + eps)) ** 2)
        weights_raw[weights_raw < weight_threshold] = 0.0
        if np.sum(weights_raw) < eps:
            raise ArithmeticError(f"All weights are zero at time t={t}.")
        weights = weights_raw / (weights_raw.sum() + eps)
        if len(weights) > 1:
            weights[-1] = 1- np.sum(weights[:-1])
        else:
            weights[-1] = 1 
    else:
        raise ValueError("mode must be 'knn' or 'kernel'.")

    rng = np.random.default_rng()
    chosen_local = rng.choice(np.arange(N), size=num_samples, replace=True, p=weights)
    samples_t = Ss[chosen_local, :]  # (num_samples, H)
    
    if return_true_scale_samples:
        return samples_t, weights_raw

    beta_samples_t = []
    beta_samples_buffer = {}
    for i in range(num_samples):
        idx = chosen_local[i]
        if idx in beta_samples_buffer:
            beta_sample = beta_samples_buffer[idx]
        else:
            beta_sample = np.array([compute_beta(cali_scores[h], samples_t[i, h]) for h in range(H)]).flatten()
            beta_samples_buffer[idx] = beta_sample
        beta_samples_t.append(beta_sample)
    beta_samples_t = np.array(beta_samples_t)  # (num_samples, H)
    
    return beta_samples_t, weights_raw

def optim_step(t, start_t, scores, cali_scores, boundaries, alphats, alpha, optim_arg=None, additional_context=None, ground_truths=None, predictions=None, betas=None, debug=False):
    traj_samples = None
    weights = None
    
    if debug:
        traj_samples, weights = get_samples_t(
            data_dict={
                'scores': scores,
                'ground_truth': ground_truths,
                'betas': betas,
                'preds': predictions,
            },
            cali_scores=cali_scores,  
            t=t,
            start_t=start_t,
            optim_arg=optim_arg,
            additional_context=additional_context,
        )
        u_hat = alpha_selection(
            boundaries,
            alphats,
            alpha,
            traj_samples,
            optim_arg=optim_arg,
        )
    else:
        try:
            traj_samples, weights = get_samples_t(
                data_dict={
                    'scores': scores,
                    'ground_truth': ground_truths,
                    'betas': betas,
                    'preds': predictions,
                },
                cali_scores=cali_scores,  
                t=t,
                start_t=start_t,
                optim_arg=optim_arg,
                additional_context=additional_context,
            )
        # Optimization based on the sampled trajectories
        # u_hat = np.array([np.quantile(traj_samples[:, h], q=alphats[h], axis=0) for h in range(len(alphats))])        
            u_hat = alpha_selection(
                boundaries,
                alphats,
                alpha,
                traj_samples,
                optim_arg=optim_arg,
            )
        except ValueError as e:
            print(f"Warning: {e} Returning mean of boundaries for time t={t}.")
            u_hat = np.mean(boundaries, axis=1)
        except ArithmeticError as e:
            # print(f"Warning: {e} Returning mean of boundaries for time t={t}.")
            u_hat = np.mean(boundaries, axis=1)
    u_hat = np.clip(u_hat, boundaries[:, 0], boundaries[:, 1])
    return u_hat, traj_samples, weights

def alpha_selection(boundaries, alphats, alpha, Fs, optim_arg=None):
    optim_arg = optim_arg or {}
    H = Fs.shape[1]
    rho_target = alpha * H

    alphas, _ = mcdp_traj_open_loop(
        rho_target=rho_target,
        oc_params=optim_arg,
        traj_samples=Fs,
        lower_bounds=boundaries[:, 0],
        upper_bounds=boundaries[:, 1],
        alpha=alpha,
        alphats=alphats,
    )
    return np.array(alphas)

# ---- CPID-specific optim step ---- #
def cpid_optim_step(t, start_t, scores, boundaries, alpha, optim_arg=None, additional_context=None, ground_truths=None, predictions=None, betas=None, debug=False):
    traj_samples = None
    weights = None
    try:
        traj_samples, weights = get_samples_t(
            data_dict={
                'scores': scores,
                'ground_truth': ground_truths,
                'betas': betas,
                'preds': predictions,
            },
            cali_scores=None,  
            t=t,
            start_t=start_t,
            optim_arg=optim_arg,
            additional_context=additional_context,
            return_true_scale_samples=True,
        )
        u_hat = cpid_q_selection(
            boundaries,
            alpha,
            traj_samples,
            optim_arg=optim_arg,
        )
    except ValueError as e:
        print(f"Warning: {e} Returning mean of boundaries for time t={t}.")
        u_hat = np.mean(boundaries, axis=1)
    except ArithmeticError as e:
        # print(f"Warning: {e} Returning mean of boundaries for time t={t}.")
        u_hat = np.mean(boundaries, axis=1)
    u_hat = np.clip(u_hat, boundaries[:, 0], boundaries[:, 1])
    return u_hat, traj_samples, weights

def cpid_q_selection(boundaries, alpha, traj_samples, optim_arg=None):
    optim_arg = optim_arg or {}
    H = traj_samples.shape[1]
    rho_target = alpha * H

    u_hat, _ = mcdp_traj_open_loop_cpid(
        rho_target=rho_target,
        oc_params=optim_arg,
        traj_samples=traj_samples,
        lower_bounds=boundaries[:, 0],
        upper_bounds=boundaries[:, 1],
        alpha=alpha,    
    )
    return np.array(u_hat)