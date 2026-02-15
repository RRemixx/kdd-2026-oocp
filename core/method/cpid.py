import numpy as np
from core.method.cp_utils import get_delta

def _fmt_num(x):
    if x is None:
        return "None"
    if isinstance(x, (np.floating, float, np.integer, int)):
        return f"{x:.6g}"
    return str(x)

def find_beta(val, scores):
    return 1 - np.mean(np.array(scores) <= val)

def mytan(x, max_score):
    if x >= np.pi/2:
        # print('aaa')
        return max_score
    elif x <= -np.pi/2:
        # print('bbb')
        return -max_score
    else:
        return np.tan(x)

def saturation_fn_log(x, t, Csat, KI, max_score):
    tan_out = mytan(x * np.log(t+1)/(Csat * (t+1)), max_score)
    # if np.abs(tan_out) > 1e10:
    #     print(f'tan out is: {tan_out}, x is {x}')
    out = KI * tan_out
    return out

def cpid_update_step(covereds, score_window, t_pred, qt_prev, alpha, lr_t, Csat, KI, integrate, do_print=False, max_score=1.0):
    init_idx = max(0, len(covereds) - score_window)
    grad = alpha if covereds[-1] else -(1-alpha)
    integrator_arg = (1-np.array(covereds[init_idx:])).sum() - (len(covereds[init_idx:]))*alpha
    integrator = saturation_fn_log(integrator_arg, t_pred, Csat, KI, max_score=max_score)
    if do_print:
        print(
            f"  step  w={init_idx}:{len(covereds)} cov={int(covereds[-1])} "
            f"used covs={covereds[init_idx:]}\n"
            f"num_covered = {np.sum(covereds[init_idx:])} "
            f"alpha*len_covs = {alpha * len(covereds[init_idx:])} "
            f"len covs = {len(covereds[init_idx:])} "
            f"a={_fmt_num(alpha)} lr={_fmt_num(lr_t)} grad={_fmt_num(grad)} "
            f"iarg={_fmt_num(integrator_arg)} iraw={_fmt_num(integrator)} "
            f"int={int(integrate)} qt_prev={_fmt_num(qt_prev)}"
        )
    # Update the next quantile
    qt_next = qt_prev - lr_t * grad
    integrator = integrator if integrate else 0
    qt_next += integrator
    if do_print:
        print(
            f"  step  qt_grad={_fmt_num(qt_prev - lr_t * grad)} "
            f"iused={_fmt_num(integrator)} qt_next={_fmt_num(qt_next)}"
        )
    return qt_next

def get_params(T, max_score=1.0):
    C_sat = 150 / np.pi * (np.ceil(np.log(T)*0.05) - 1/np.log(T))
    K = max_score
    # print(f'Assign parameters for CPID: Csat is: {C_sat}, K is: {K}')
    return C_sat, K

class CPID:
    def __init__(self, alpha, gamma, power=1/2, d_factor=1.0, horizon=100, score_window=100, optional_args=None):
        """
        Args:
          alpha (float): target miscoverage rate α
          gamma (float): learning rate γ for ACI step
          power (float): power for delta calculation
          d_factor (float): factor for delta calculation
        """
        self.horizon = horizon
        self.alpha = alpha    # Target miscoverage rate
        self.gamma = gamma         # Learning rate
        self.t = 0                # Time step
        self.power = power         # Power for delta calculation
        self.d_factor = d_factor   # Factor for delta calculation
        self.q_min = 0.0        # Minimum quantile observed
        self.max_delta = optional_args.get('max_delta', 0.1)   # Maximum delta for shrinking radius
        self.alpha_min = 0.0    # Minimum alpha
        self.alpha_max = 1.0    # Maximum alpha
        self.S_max = None
        
        max_score = optional_args.get('max_score', 1.0)
        self.T = optional_args.get('T', 200)
        self.Csat, self.KI = get_params(self.T, max_score)
        # self.Csat = optional_args.get('Csat', 1)
        # self.KI = optional_args.get('KI', 1.0)
        self.integrate = optional_args.get('integrate', True)
        self.score_window = score_window  # Window size for score history
        
        self.init_range = None
        self.qt = None
        self.covereds = []

        self.verbose = optional_args.get('verbose', False)
    
    def update_max_score(self, max_score):
        self.S_max = max_score
        self.Csat, self.KI = get_params(self.T, max_score)
        # if self.verbose:
        #     print(f"Updated max score to {max_score}. New Csat: {self.Csat}, KI: {self.KI}")
        
    def init_qt(self, current_scores):
        self.qt = np.quantile(current_scores, 1 - self.alpha)
        # print('Initial qt is:', self.qt)
        self.init_range = np.quantile(current_scores, 0.95) * 1.5
        # print('Initial range is:', self.init_range)

    def blind_update(self, scores=None):
        self.t += 1
        if self.qt is None:
            return 0, 0, 0, self.S_max
        """ Perform a blind update of the ACI interval without considering coverage."""
        if len(scores) > 0 and self.qt > np.max(scores):
            return 0, 0, -1, self.qt
        alpha_next = find_beta(self.qt, scores)
        delta = get_delta(self.t, power=self.power, alphat=alpha_next, d_factor=self.d_factor, max_delta=self.max_delta)
        lower_alpha = max(alpha_next - delta, self.alpha_min)
        upper_alpha = min(alpha_next + delta, self.alpha_max)
        return lower_alpha, upper_alpha, alpha_next, self.qt

    def update(self, in_interval, scores=None):
        # Update state
        self.t += 1
        # Indicator for miscoverage: 1 if outside interval, else 0
        do_print = self.verbose and self.alpha == 0.1 and self.horizon == 1
        scores_len = len(scores) if scores is not None else 0
        scores_max = np.max(scores) if scores_len > 0 else None
        if do_print:
            print(f"\n{'='*22} CPID t={self.t} {'='*22}")
            print(
                f"  in    cov={int(in_interval)} qt={_fmt_num(self.qt)} "
                f"a={_fmt_num(self.alpha)} g={_fmt_num(self.gamma)} "
                f"n={scores_len} smax={_fmt_num(scores_max)} "
                f"sw={self.score_window} C={_fmt_num(self.Csat)} K={_fmt_num(self.KI)} "
                f"int={int(self.integrate)}"
            )
        self.covereds.append(in_interval)
        lr_t = 0.01 * np.max(scores) if scores is not None else self.gamma
        if self.S_max > 50:
            print(f"Warning: S_max is {self.S_max}, which seems very high. Check if it's set correctly.")
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
            do_print=do_print,
            max_score=self.S_max
        )
        self.qt = q_next
        
        if len(scores) > 0 and self.qt > np.max(scores):
            if do_print:
                print(
                    f"  out   EARLY qt={_fmt_num(self.qt)} > smax={_fmt_num(np.max(scores))} "
                    f"a_next=-1"
                )
                print(f"{'='*54}")
            return 0, 0, -1, self.qt
        
        # Clip to [alpha_min, alpha_max]
        alpha_next = find_beta(self.qt, scores)

        # Define shrinking radius δ_t = (alpha_max - alpha_min) / sqrt(t)
        delta = get_delta(self.t, power=self.power, alphat=alpha_next, d_factor=self.d_factor, max_delta=self.max_delta)
        delta_raw = delta

        # If alpha_next is at the boundaries, set delta to 0
        if alpha_next == self.alpha_min or alpha_next == self.alpha_max:
            delta = 0.0
        lower_alpha = max(alpha_next - delta, self.alpha_min)
        upper_alpha = min(alpha_next + delta, self.alpha_max)
        if do_print:
            print(
                f"  out   qt={_fmt_num(self.qt)} a_next={_fmt_num(alpha_next)} "
                f"d_raw={_fmt_num(delta_raw)} d={_fmt_num(delta)} "
                f"lo={_fmt_num(lower_alpha)} hi={_fmt_num(upper_alpha)}"
            )
            print(f"{'='*54}")

        return lower_alpha, upper_alpha, alpha_next, self.qt

class CPID_Scale:
    def __init__(self, alpha, gamma, power=1/2, d_factor=1.0, horizon=100, score_window=100, optional_args=None, max_delta=0.1):
        """
        Args:
          alpha (float): target miscoverage rate α
          gamma (float): learning rate γ for ACI step
          power (float): power for delta calculation
          d_factor (float): factor for delta calculation
        """
        self.horizon = horizon
        self.alpha = alpha    # Target miscoverage rate
        self.gamma = gamma         # Learning rate
        self.t = 0                # Time step
        self.power = power         # Power for delta calculation
        self.d_factor = d_factor   # Factor for delta calculation
        self.q_min = 0.0        # Minimum quantile observed
        self.max_delta = max_delta
        self.alpha_min = 0.0    # Minimum alpha
        self.alpha_max = 1.0    # Maximum alpha
        
        max_score = optional_args.get('max_score', 1.0)
        
        self.S_max = max_score
        self.max_range = self.S_max * self.max_delta
        
        self.T = optional_args.get('T', 200)
        self.Csat, self.KI = get_params(self.T, max_score)
        # self.Csat = optional_args.get('Csat', 1)
        # self.KI = optional_args.get('KI', 1.0)
        self.integrate = optional_args.get('integrate', True)
        self.score_window = score_window  # Window size for score history
        
        self.init_range = None
        self.qt = None
        self.covereds = []

        self.verbose = optional_args.get('verbose', False)
    
    def update_max_score(self, max_score):
        self.S_max = max_score
        self.Csat, self.KI = get_params(self.T, max_score)
        self.max_range = self.S_max * self.max_delta
        # if self.verbose:
        #     print(f"Updated max score to {max_score}. New Csat: {self.Csat}, KI: {self.KI}")
        
    def init_qt(self, current_scores):
        self.qt = np.quantile(current_scores, 1 - self.alpha)
        # print('Initial qt is:', self.qt)
        self.init_range = np.quantile(current_scores, 0.95) * 1.5
        # print('Initial range is:', self.init_range)

    def blind_update(self, scores=None):
        self.t += 1
        if self.qt is None:
            return 0, 0, 0, self.S_max
        """ Perform a blind update of the ACI interval without considering coverage."""
        if len(scores) > 0 and self.qt > np.max(scores):
            return 0, 0, -1, self.qt
        delta = (1 / np.power(self.t, self.power) * self.d_factor) * abs(self.qt)
        delta = min(delta, self.max_range)
        lower = max(self.qt - delta, 0.0)
        upper = self.qt + delta
        return lower, upper, self.alpha, self.qt

    def update(self, in_interval, scores=None):
        # Update state
        self.t += 1
        # Indicator for miscoverage: 1 if outside interval, else 0
        do_print = self.verbose and self.alpha == 0.1 and self.horizon == 1
        self.covereds.append(in_interval)
        lr_t = 0.01 * np.max(scores) if scores is not None else self.gamma
        if self.S_max > 50:
            print(f"Warning: S_max is {self.S_max}, which seems very high. Check if it's set correctly.")
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
            do_print=do_print,
            max_score=self.S_max
        )
        self.qt = q_next
        
        delta = (1 / np.power(self.t, self.power) * self.d_factor) * abs(self.qt)
        delta = min(delta, self.max_range)
        lower = max(self.qt - delta, 0.0)
        upper = self.qt + delta

        return lower, upper, self.alpha, self.qt