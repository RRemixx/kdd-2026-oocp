import numpy as np

def mcdp_traj(rho_target, oc_params, traj_samples, lower_bounds, upper_bounds, alpha, alphats):
    """
    Markov-style control using full trajectory samples.
    ---------------------------------------------------
    Each row in traj_samples is one sampled trajectory of beta_t over t=0..H-1.
    The expected cost is computed by averaging over trajectories.

    Parameters
    ----------
    rho_target : int
        Desired number of ones at horizon H.
    oc_params : dict
        {'u_intvl_num': int,
         'invalid_state_penalty': float}
    traj_samples : ndarray, shape (n_samples, H)
        Monte-Carlo samples of beta_t trajectories.
    lower_bounds, upper_bounds : ndarray, shape (H,)
        Bounds for u_t at each t.

    Returns
    -------
    u_star : list[float]
        Optimal controls for t = 0 … H-1.
    policy  : ndarray, shape (H, H+1)
        policy[t, p] = optimal u when running count is p at time t.
    ideal_coverage : list[int]
        Greedy coverage indicator based on median transition.
    """
    n_samples, H = traj_samples.shape
    u_intvl_num = oc_params.get('u_intvl_num', 50)
    invalid_state_cost = oc_params.get('invalid_state_penalty', 1e8)
    e_coeff = oc_params.get('e_coeff', np.ones(H) * 0.01) / alpha
    r_coeff = oc_params.get('r_coeff')
    zero_penalty = 1

    p_grid = np.arange(H + 1)
    policy = np.full((H, H + 1), np.nan)
    V_next = np.maximum((p_grid - rho_target), 0)

    for t in range(H - 1, -1, -1):
        u_grid = np.linspace(lower_bounds[t], upper_bounds[t], u_intvl_num)
        beta_t = traj_samples[:, t]
        inc_mat = beta_t[:, None] > u_grid[None, :]
        V_t = np.empty_like(p_grid, dtype=float)

        for p in p_grid:
            if p > t:
                V_t[p] = invalid_state_cost
                continue

            future_costs = V_next[p + inc_mat].mean(axis=0)
            running_cost = (
                e_coeff[t] * (1 - u_grid) +
                r_coeff * np.abs(u_grid - alphats[t]) +
                zero_penalty * (u_grid == 0)
            )
            total_costs = running_cost + future_costs

            best_idx = np.argmin(total_costs)
            V_t[p] = total_costs[best_idx]
            policy[t, p] = u_grid[best_idx]

        V_next = V_t

    p = 0
    u_star = []
    ideal_coverage = []
    for t in range(H):
        u_opt = float(policy[t, p])
        u_star.append(u_opt)
        if (traj_samples[:, t] < u_opt).mean() >= 0.5:
            p = p + 1
            ideal_coverage.append(0)
        else:
            ideal_coverage.append(1)

    return u_star, policy, ideal_coverage


def mcdp_traj_open_loop(rho_target, oc_params, traj_samples, lower_bounds, upper_bounds, alpha, alphats):
    """
    Open-loop optimization over a single action sequence shared by all scenarios.
    ---------------------------------------------------------------------------
    Chooses u_0..u_{H-1} to minimize the average cost across trajectory samples.
    This ignores state feedback and solves a deterministic sequence.

    Parameters
    ----------
    rho_target : int
        Desired number of ones at horizon H.
    oc_params : dict
        {'u_intvl_num': int,
         'invalid_state_penalty': float,
         'open_loop_max_iter': int,
         'open_loop_tol': float}
    traj_samples : ndarray, shape (n_samples, H)
        Monte-Carlo samples of beta_t trajectories.
    lower_bounds, upper_bounds : ndarray, shape (H,)
        Bounds for u_t at each t.

    Returns
    -------
    u_star : list[float]
        Open-loop control sequence for t = 0 … H-1.
    cost_history : list[float]
        Average cost per iteration.
    """
    n_samples, H = traj_samples.shape
    u_intvl_num = oc_params.get('u_intvl_num', 50)
    invalid_state_cost = oc_params.get('invalid_state_penalty', 1e8)
    e_coeff = oc_params.get('e_coeff', np.ones(H) * 0.01) / alpha
    r_coeff = oc_params.get('r_coeff')
    zero_penalty = 1
    max_iter = oc_params.get('open_loop_max_iter', 10)
    tol = oc_params.get('open_loop_tol', 1e-6)

    u_star = np.clip(np.array(alphats, dtype=float), lower_bounds, upper_bounds)

    def running_cost(u_vec):
        diff = traj_samples - u_vec[None, :]
        pinball = np.maximum(alphats[None, :] * diff, (alphats[None, :] - 1) * diff).mean(axis=0)
        return float(
            np.sum(
                e_coeff * pinball +
                zero_penalty * (u_vec == 0)
            )
        ) / H

    def expected_terminal_cost(u_vec):
        counts = (traj_samples < u_vec[None, :]).sum(axis=1)
        return np.maximum(counts - rho_target, 0).mean()

    def total_cost(u_vec):
        term = expected_terminal_cost(u_vec)
        if term >= invalid_state_cost:
            return invalid_state_cost + running_cost(u_vec)
        return running_cost(u_vec) + term

    cost_history = [float(total_cost(u_star))]

    for _ in range(max_iter):
        prev_cost = cost_history[-1]
        current_running_cost = running_cost(u_star)
        for t in range(H):
            u_grid = np.linspace(lower_bounds[t], upper_bounds[t], u_intvl_num)
            # counts excluding time t for each trajectory
            base_counts = (traj_samples < u_star[None, :]).sum(axis=1) - (traj_samples[:, t] < u_star[t])
            # vectorized terminal cost for each candidate u
            inc = (traj_samples[:, t][:, None] < u_grid[None, :]).astype(int)
            counts = base_counts[:, None] + inc
            term_costs = np.maximum(counts - rho_target, 0).mean(axis=0)
            diff = traj_samples[:, t][:, None] - u_grid[None, :]
            pinball = np.maximum(alphats[t] * diff, (alphats[t] - 1) * diff).mean(axis=0)
            running = (
                e_coeff[t] * pinball +
                r_coeff * np.abs(u_grid - alphats[t]) +
                zero_penalty * (u_grid == 0)
            )
            current_diff = traj_samples[:, t] - u_star[t]
            current_pinball = np.maximum(
                alphats[t] * current_diff,
                (alphats[t] - 1) * current_diff
            ).mean()
            current_component = (
                e_coeff[t] * current_pinball +
                r_coeff * np.abs(u_star[t] - alphats[t]) +
                zero_penalty * (u_star[t] == 0)
            )
            total = running + term_costs + (current_running_cost - current_component)
            best_idx = np.argmin(total)
            u_star[t] = u_grid[best_idx]
            current_running_cost = current_running_cost - current_component + running[best_idx]

        new_cost = float(total_cost(u_star))
        cost_history.append(new_cost)
        if abs(prev_cost - new_cost) <= tol * max(1.0, prev_cost):
            break

    return u_star.tolist(), cost_history


def mcdp_traj_open_loop_cpid(rho_target, oc_params, traj_samples, lower_bounds, upper_bounds, alpha):
    """
    Open-loop optimization over a single action sequence shared by all scenarios.
    ---------------------------------------------------------------------------
    Chooses u_0..u_{H-1} to minimize the average cost across trajectory samples.
    This ignores state feedback and solves a deterministic sequence.

    Parameters
    ----------
    rho_target : int
        Desired number of ones at horizon H.
    oc_params : dict
        {'u_intvl_num': int,
         'invalid_state_penalty': float,
         'open_loop_max_iter': int,
         'open_loop_tol': float}
    traj_samples : ndarray, shape (n_samples, H)
        Monte-Carlo samples of beta_t trajectories.
    lower_bounds, upper_bounds : ndarray, shape (H,)
        Bounds for u_t at each t.

    Returns
    -------
    u_star : list[float]
        Open-loop control sequence for t = 0 … H-1.
    cost_history : list[float]
        Average cost per iteration.
    """
    n_samples, H = traj_samples.shape
    u_intvl_num = oc_params.get('u_intvl_num', 50)
    invalid_state_cost = oc_params.get('invalid_state_penalty', 1e8)
    e_coeff = oc_params.get('e_coeff', np.ones(H) * 0.01) / alpha
    r_coeff = oc_params.get('r_coeff')
    zero_penalty = 1
    max_iter = oc_params.get('open_loop_max_iter', 10)
    tol = oc_params.get('open_loop_tol', 1e-6)
    
    alphats = np.full(H, alpha)

    u_star = np.clip(np.array(alphats, dtype=float), lower_bounds, upper_bounds)

    def running_cost(u_vec):
        diff = traj_samples - u_vec[None, :]
        pinball = np.maximum(alphats[None, :] * diff, (alphats[None, :] - 1) * diff).mean(axis=0)
        return float(
            np.sum(
                e_coeff * pinball +
                zero_penalty * (u_vec == 0)
            )
        ) / H

    def expected_terminal_cost(u_vec):
        counts = (traj_samples > u_vec[None, :]).sum(axis=1)
        return np.maximum(counts - rho_target, 0).mean()

    def total_cost(u_vec):
        term = expected_terminal_cost(u_vec)
        if term >= invalid_state_cost:
            return invalid_state_cost + running_cost(u_vec)
        return running_cost(u_vec) + term

    cost_history = [float(total_cost(u_star))]

    for _ in range(max_iter):
        prev_cost = cost_history[-1]
        current_running_cost = running_cost(u_star)
        for t in range(H):
            u_grid = np.linspace(lower_bounds[t], upper_bounds[t], u_intvl_num)
            # counts excluding time t for each trajectory
            base_counts = (traj_samples > u_star[None, :]).sum(axis=1) - (traj_samples[:, t] > u_star[t])
            # vectorized terminal cost for each candidate u
            inc = (traj_samples[:, t][:, None] > u_grid[None, :]).astype(int)
            counts = base_counts[:, None] + inc
            term_costs = np.maximum(counts - rho_target, 0).mean(axis=0)
            diff = traj_samples[:, t][:, None] - u_grid[None, :]
            pinball = np.maximum(alphats[t] * diff, (alphats[t] - 1) * diff).mean(axis=0)
            running = (
                e_coeff[t] * pinball +
                r_coeff * np.abs(u_grid) +
                zero_penalty * (u_grid == 0)
            )
            current_diff = traj_samples[:, t] - u_star[t]
            current_pinball = np.maximum(
                alphats[t] * current_diff,
                (alphats[t] - 1) * current_diff
            ).mean()
            current_component = (
                e_coeff[t] * current_pinball +
                r_coeff * np.abs(u_star[t]) +
                zero_penalty * (u_star[t] == 0)
            )
            total = running + term_costs + (current_running_cost - current_component)
            best_idx = np.argmin(total)
            u_star[t] = u_grid[best_idx]
            current_running_cost = current_running_cost - current_component + running[best_idx]

        new_cost = float(total_cost(u_star))
        cost_history.append(new_cost)
        if abs(prev_cost - new_cost) <= tol * max(1.0, prev_cost):
            break

    return u_star.tolist(), cost_history