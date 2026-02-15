from core.method.cp_ours import *

def run_algorithm_with_multiple_alphas(alphas, data_generator: TimeSeriesDataTemplate, score_func_args, T_obs=1000, H=15, N=20, start_t=10, method_opt='cpt', plot=False, S_max=10, save_path="results", print_max_score=False, twodim=False, learned_scores=None, expand_boundaries=0, params=None):
    A = len(alphas)
    score_window = params.get('score_window', 50)
    alpha_minimum, alpha_maximum = 0.0, 1.0
    
    # One for all alphas
    scores = np.zeros((T_obs, H))
    timestamps = []
    ground_truths = np.zeros((T_obs, H))
    samples = np.zeros((T_obs, N, H))
    score_func = score_function
    score_function_type = score_func_args.get('type', 'abs-r')
    score_function_optional_args = score_func_args.get('optional_args', {})
    if twodim:
        samples = np.zeros((T_obs, N, H, 2))  # For storing samples in 2D
        ground_truths = np.zeros((T_obs, H, 2))  # For storing ground truths in 2D
        score_func = score_function_2d
    S_max_vector = np.ones(H) * S_max
    if learned_scores is not None:
        for h in range(H):
            score_filter_dict = {'horizon': (h, 0),}
            learned_scores_h = get_proximal_scores(learned_scores, info_dict=score_filter_dict)
            if len(learned_scores_h) > 0:
                S_max_vector[h] = np.quantile(learned_scores_h, 0.95) * 1.2
    
    # One for each alpha
    alphats = np.zeros((A, T_obs, H))
    alphas_mid = np.zeros((A, T_obs, H))
    alphas_radius = np.zeros((A, T_obs, H))
    qs = np.zeros((A, T_obs, H))
    Fs = np.ones((T_obs, H)) * alpha_minimum
    covered_all = np.ones((A, T_obs, H))
    prediction_intervals = np.zeros((A, T_obs, H, N, 2))  # For storing prediction intervals
    
    all_acis = []
    for i, alpha in enumerate(alphas):
        if method_opt == 'cpt':
            acis = [SetValuedACI(alpha_init=alpha, gamma=params['gamma'], alpha_min=alpha_minimum, alpha_max=alpha_maximum, power=params['power'], d_factor=params.get('d_factor', 1.0), score_window=score_window) for _ in range(H)]
        elif method_opt == 'dtaci':
            acis = [DtACI(alpha_init=alpha, alpha_min=alpha_minimum, alpha_max=alpha_maximum, gammas=0.001 * 2 ** np.arange(8), sigma=1/1000, eta=2.72, power=params['power'], d_factor=params.get('d_factor', 1.0), score_window=score_window) for _ in range(H)]
        all_acis.append(acis)
    
    # parameters
    # optim_arg = params.get('optim_arg', {})
    # if 'alpha' not in optim_arg:
    #     optim_arg['alpha'] = alpha
    # optim_arg['e_coeff'] = np.ones(H) * optim_arg.get('e_coeff_init', 0.01)
    
    dist2true = []
    
    # Main simulation loop
    for t in range(T_obs):
        current_time = data_generator.get_reference_time(t)
        y_truth, samp = data_generator.get_trajectory_samples(t, random=False)

        if samp.shape[0] > N:
            cur_samp = samp[:N]
        else:
            cur_samp = samp
        samples[t] = cur_samp

        boundaries = np.zeros((A, H, 2))
        timestamps.append(current_time)
        enhanced_scores_t = [] 
        alpha_nexts = np.zeros((A, H))
        cur_betas = np.ones((H)) * alpha_minimum
        for h in range(H):
            scores[t, h] = score_func(y_truth, cur_samp, h, type=score_function_type, optional_args=score_function_optional_args)
            observed_scores = []
            if t > h:
                observed_scores = scores[:t-h, h]
            score_filter_dict = {'horizon': (h, 0),}
            if learned_scores is not None:
                learned_scores_h = get_proximal_scores(learned_scores, info_dict=score_filter_dict)
            else:
                learned_scores_h = np.array([])
            enhanced_scores = concat_scores(observed_scores, learned_scores_h, score_window)
            enhanced_scores_t.append(enhanced_scores)
            if len(enhanced_scores) > 1:
                cur_betas[h] = compute_beta(enhanced_scores, scores[t, h], alpha_minimum=alpha_minimum, alpha_maximum=alpha_maximum)
                Fs[t, h] = cur_betas[h]
            # Update coverage and boundaries
            for i, alpha in enumerate(alphas):
                if t > h:
                    covered = observed_scores[-1] <= qs[i, t-h-1, h]
                    covered_all[i, t-h-1, h] = covered
                    beta_t = Fs[t-h-1, h]
                    low, high, alpha_next = all_acis[i][h].update(covered, beta_t=beta_t)
                else:
                    # coverage is not observed yet
                    low, high, alpha_next = all_acis[i][h].blind_update()
                low, high = symmetric_boundaries(low, high, alpha_next)
                boundaries[i, h, 0] = low
                boundaries[i, h, 1] = high
                alpha_nexts[i, h] = alpha_next
            
            # select optimal alphas for next time step
            if expand_boundaries < 0:
                u_star = alpha_nexts[:, h]  # no isotonic constraint
            else:
                u_star, _ = soft_isotonic_decreasing(alpha=alpha_nexts[:, h], low=boundaries[:, h, 0], high=boundaries[:, h, 1])
            alphats[:, t, h] = u_star
        
        # Compute quantiles for each horizon
        for i, alpha in enumerate(alphas):
            for h in range(H):
                enhanced_scores = enhanced_scores_t[h]
                if len(enhanced_scores) > 1:
                    qs[i, t, h] = quantile_function(enhanced_scores, 1 - alphats[i, t, h], S_max_vector[h])
                else:
                    qs[i, t, h] = S_max_vector[h]
                # Store ground truth and prediction intervals
                ground_truths[t, h] = y_truth[h]
                if not twodim:
                    current_prediction_intervals = inverse_score_function(qs[i, t, h], samples[t, :, h], score_function_type, score_function_optional_args)
                    for k in range(len(current_prediction_intervals)):
                        prediction_intervals[i, t, h, k, 0] = current_prediction_intervals[k][0]
                        prediction_intervals[i, t, h, k, 1] = current_prediction_intervals[k][1]
    
    if print_max_score:
        print(f'max scores for each horizon:  {np.max(scores, axis=0)}')

    if twodim:
        return timestamps, ground_truths, (samples, qs)
    return timestamps, ground_truths, prediction_intervals