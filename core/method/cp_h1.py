from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from tqdm import tqdm

from online_conformal.saocp import SAOCP

from core.data import TimeSeriesDataTemplate
from core.method.cp_utils import concat_scores, get_proximal_scores
from core.method.score_func import *

class SingleHorizonCP(ABC):
    """
    Single Horizon CP (conformal prediction) class for handling single horizon conformal prediction tasks.
    """
    def __init__(self, init_alpha:float, optional_cp_args:dict={}):
        self.alpha = init_alpha
        self.args = optional_cp_args
    
    @abstractmethod
    def update(self, t, observed_scores, observed_coverage=None, before_start: bool = False, optional_args: dict = {}):
        """
        TODO: 
        """
        raise NotImplementedError

def run_single_horizon_simulation(method:str, params:dict, data_generator: TimeSeriesDataTemplate, alpha=0.3, T_obs=1000, H=15, N=20, start_t=10, twodim=False, score_window=100, learned_scores=None):
    """
    TODO: write docstring
    """    
    # Initialization
    scores = np.zeros((T_obs, H))
    qs = np.zeros((T_obs, H))
    covered_all = np.ones((T_obs, H))
    samples = np.zeros((T_obs, N, H))
    
    # debug
    alphas = np.zeros((T_obs, H))
    scaled_alphas = np.zeros((T_obs, H))
    
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
    
    # Generate scores first
    for t in range(T_obs):
        current_time = data_generator.get_reference_time(t)
        timestamps.append(current_time)
        y_truth, samp = data_generator.get_trajectory_samples(t, random=False)
        samples[t] = samp
        for h in range(H):
            ground_truths[t, h] = y_truth[h]
            scores[t, h] = score_func(y_truth, samp, h, score_function_type, score_function_optional_args)
    
    # handle S_max
    S_max_vector = np.ones(H) * params['S_max']
    if learned_scores is not None:
        for h in range(H):
            score_filter_dict = {'horizon': (h, 0),}
            learned_scores_h = get_proximal_scores(learned_scores, info_dict=score_filter_dict)
            if len(learned_scores_h) > 0:
                S_max_vector[h] = np.quantile(learned_scores_h, 0.95) * 1.2
    if params.get('dynamic_S_max', False):
        assert start_t > H
        for h in range(H):
            observed_scores = scores[:start_t-h, h]
            S_max_vector[h] = np.quantile(observed_scores, 0.95) * 2

    optional_cp_args = params.get('optional_args', {})
    if method == 'nexcp':
        cp_method = NEXCP
    elif method == 'faci':
        cp_method = FACI
    elif method == 'saocp':
        cp_method = SAOCPWrapper
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods are 'nexcp', 'faci' and 'saocp'.")
    sh_method = []
    optional_cp_args['start_t'] = start_t
    optional_cp_args['S_max'] = S_max_vector
    for h in range(H):
        if method == 'faci':
            optional_cp_args['scores'] = scores[:, h]
        optional_cp_args['h'] = h
        sh_method.append(cp_method(alpha, optional_cp_args))
    
    # Main simulation loop
    for t in range(T_obs):
        for h in range(H):
            observed_scores = []
            # first ground truth is observed after t > h, before this point, we do not have any scores
            if t > h:
                observed_scores = scores[:t-h, h]
                covered = observed_scores[-1] <= qs[t-h-1, h]
                covered_all[t-h-1, h] = covered
            score_filter_dict = {'horizon': (h, 0),}
            if learned_scores is not None:
                learned_scores_h = get_proximal_scores(learned_scores, info_dict=score_filter_dict, num_scores=100)
            else:
                learned_scores_h = None
            enhanced_scores = concat_scores(observed_scores, learned_scores_h, score_window)
            
            if t > start_t and params.get('dynamic_S_max', False):
                S_max_vector[h] = np.quantile(enhanced_scores, 0.95) * 2
            
            before_start = t < start_t
            debug = True
            if debug:
                qs[t, h], scaled_alphat, alphat = sh_method[h].update(
                    t=t, 
                    observed_scores=enhanced_scores,
                    observed_coverage=covered_all[:t-h, h] if t > h else None,
                    before_start=before_start,
                    S_max=S_max_vector[h],
                    debug=debug,
                )
                alphas[t, h] = alphat
                scaled_alphas[t, h] = scaled_alphat

            else:
                qs[t, h] = sh_method[h].update(
                    t=t, 
                    observed_scores=enhanced_scores,
                    observed_coverage=covered_all[:t-h, h] if t > h else None,
                    before_start=before_start,
                    S_max=S_max_vector[h],
                    debug=debug,
                )
            
            if not twodim:
                current_prediction_intervals = inverse_score_function(qs[t, h], samples[t, :, h], score_function_type, score_function_optional_args)
                for i in range(len(current_prediction_intervals)):
                    prediction_intervals[t, h, i, 0] = current_prediction_intervals[i][0]
                    prediction_intervals[t, h, i, 1] = current_prediction_intervals[i][1]
        if False:
            print(f"Time step: {t}")
            print(f'alpha is ', alpha)
            print(f'scaled_alpha is ', scaled_alphas[t, 0])
            print('alpha_t: ', alphas[t, 0])
            print('score: ', scores[t, 0])
            print('q_t: ', qs[t, 0])
            print("-" * 60)

    if twodim:
        return timestamps, ground_truths, (samples, qs), scores, qs
    return timestamps, ground_truths, prediction_intervals, scores, qs


##############################################
# ---------- SAOCP Implementation ---------- #
##############################################

class SAOCPWrapper(SingleHorizonCP):
    def __init__(self, init_alpha, optional_cp_args = {}):
        super().__init__(init_alpha, optional_cp_args)
        self.cp = SAOCP(model=None, train_data=None, max_scale=optional_cp_args.get('S_max', None)[optional_cp_args.get('h', 0)], coverage=1-init_alpha)
    
    def update(self, t, observed_scores, observed_coverage=None, before_start=False, S_max=None, optional_args={}, debug=False):
        if len(observed_scores) > 0:
            self.cp.update(ground_truth=pd.Series([observed_scores[-1]]), forecast=pd.Series([0]), horizon=1)
        if before_start:
            if debug:
                return S_max, 0, 0
            return S_max
        if debug:
            return self.cp.predict(horizon=1)[1], 0, 0
        return self.cp.predict(horizon=1)[1]

##############################################
# ---------- NEXCP Implementation ---------- #
##############################################

def nex_quantile(arr, q, gamma):
    q = np.clip(q, 0, 1)
    if len(arr) == 0:
        return np.zeros(len(q)) if hasattr(q, "__len__") else 0
    weights = np.exp(np.log(gamma) * np.arange(len(arr) - 1, -1, -1))
    assert len(weights) == len(arr)
    idx = np.argsort(arr)
    weights = np.cumsum(weights[idx])
    q_idx = np.searchsorted(weights / weights[-1], q)
    return np.asarray(arr)[idx[q_idx]]

class NEXCP(SingleHorizonCP):
    def __init__(self, init_alpha, optional_cp_args = {}):
        super().__init__(init_alpha, optional_cp_args)
        self.window_length = optional_cp_args.get('window_length', 10)
        self.gamma = optional_cp_args.get('gamma', 0.9)
    
    def update(self, t, observed_scores, observed_coverage=None, before_start=False, S_max=None, optional_args={}, debug=False):
        if observed_scores is None or len(observed_scores) == 0:
            if debug:
                return S_max, 0, 0
            return S_max
        scale_factor = 1 + 1 / (t - max(t-self.window_length,0) + 1)
        quantile_level_t = np.clip(scale_factor*(1-self.alpha), 0, 1)
        
        if quantile_level_t == 1:
            qt = S_max
        else:
            qt = nex_quantile(observed_scores, quantile_level_t, gamma=self.gamma)
        
        if debug:
            return qt, quantile_level_t, self.alpha,
        return qt
    
#############################################
# ---------- FACI Implementation ---------- #
#############################################

def find_beta(recent_scores, cur_score, epsilon=0.001):
    top = 1
    bot = 0
    mid = (top + bot) / 2
    
    while top - bot > epsilon:
        if np.quantile(recent_scores, 1 - mid) > cur_score:
            bot = mid
        else:
            top = mid
        mid = (top + bot) / 2
    return mid

def vec_zero_min(x):
    return np.minimum(x, 0)

def pinball(u, alpha):
    return alpha * u - vec_zero_min(u)

def FACI_preprocess(scores,
    window_length,
    T_burnin,):
    betas = []
    for t in range(len(scores)):
        cur_beta = 0.01
        if t > T_burnin:  
            cur_score = scores[t]
            recent_scores = scores[max(0, t-window_length):t]
            cur_beta = find_beta(recent_scores, cur_score)
        betas.append(cur_beta)
    return betas

# FACI method implementation
def conformal_adapt_stable(betas, alpha, gammas, sigma=1/1000, eta=2.72):
    T = len(betas)
    k = len(gammas)
    
    alpha_seq = np.full(T, alpha)
    err_seq_adapt = np.zeros(T)
    err_seq_fixed = np.zeros(T)
    gamma_seq = np.zeros(T)
    mean_alpha_seq = np.zeros(T)
    mean_err_seq = np.zeros(T)
    mean_gammas = np.zeros(T)
    
    expert_alphas = np.full(k, alpha)
    expert_ws = np.ones(k)
    cur_expert = np.random.choice(k)
    expert_cumulative_losses = np.zeros(k)
    expert_probs = np.full(k, 1/k)
    
    for t in range(T):
        alphat = expert_alphas[cur_expert]
        alpha_seq[t] = alphat
        err_seq_adapt[t] = float(alphat > betas[t])
        err_seq_fixed[t] = float(alpha > betas[t])
        gamma_seq[t] = gammas[cur_expert]
        mean_alpha_seq[t] = np.dot(expert_probs, expert_alphas)
        mean_err_seq[t] = float(mean_alpha_seq[t] > betas[t])
        mean_gammas[t] = np.dot(expert_probs, gammas)
        
        expert_losses = pinball(betas[t] - expert_alphas, alpha)
        
        # Update expert alphas
        expert_alphas = expert_alphas + gammas * (alpha - (expert_alphas > betas[t]).astype(float))
        
        # Update expert weights
        if eta < np.inf:
            expert_bar_ws = expert_ws * np.exp(-eta * expert_losses)
            expert_next_ws = (1 - sigma) * expert_bar_ws / np.sum(expert_bar_ws) + sigma / k
            expert_probs = expert_next_ws / np.sum(expert_next_ws)
            cur_expert = np.random.choice(k, p=expert_probs)
            expert_ws = expert_next_ws
        else:
            expert_cumulative_losses += expert_losses
            cur_expert = np.argmin(expert_cumulative_losses)
    
    return {
        "alpha_seq": alpha_seq,
        "err_seq_adapt": err_seq_adapt,
        "err_seq_fixed": err_seq_fixed,
        "gamma_seq": gamma_seq,
        "mean_alpha_seq": mean_alpha_seq,
        "mean_err_seq": mean_err_seq,
        "mean_gammas": mean_gammas
    }

class FACI(SingleHorizonCP):
    def __init__(self, init_alpha, optional_cp_args = {}):
        super().__init__(init_alpha, optional_cp_args)
        self.window_length = optional_cp_args.get('window_length', 10)
        self.T_burnin = optional_cp_args.get('start_t') # start_t
        self.alpha = init_alpha
        self.gammas = optional_cp_args.get('gammas', [0.9])
        self.sigma = optional_cp_args.get('sigma', 1/1000)
        self.eta = optional_cp_args.get('eta', 2.72)
        self.h = optional_cp_args.get('h', 0)
        
        scores = optional_cp_args.get('scores', None)
        if scores is None:
            raise ValueError("Scores must be provided in optional_args.")
        betas = FACI_preprocess(scores, self.window_length, self.T_burnin)
        expert_results = conformal_adapt_stable(betas, self.alpha, self.gammas, self.sigma, self.eta)
        self.expert_alphas = expert_results['alpha_seq']
    
    def update(self, t, observed_scores, observed_coverage=None, before_start=False, S_max=None, optional_args={}, debug=False):
        if observed_scores is None or len(observed_scores) == 0:
            if debug:
                return S_max, 0, 0
            return S_max
        if t < self.h:
            cur_alpha = self.alpha
        else:
            cur_alpha = self.expert_alphas[t-self.h]
        scale_factor = 1 + 1 / (t - max(t-self.window_length,0) + 1)
        quantile = np.clip(scale_factor*(1-cur_alpha), 0, 1)
        if quantile == 0:
            q = 0
        elif quantile == 1:
            q = S_max
        else:
            q = np.quantile(observed_scores, quantile)
        if debug:
            return q, 1-quantile, cur_alpha
        return q