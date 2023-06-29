from rddc.tools.matrix_normal_distribution import MN
from rddc.tools import control_utils
import numpy as np
from scipy.stats import wishart

def create_sys_dist(seed, A, B, sigma):
    n = B.shape[0]
    m = B.shape[1]
    assert(A.shape == (n,n))
    wishart_n_parameters = n*n + n*m
    # wishart_scale = (np.eye(wishart_n_parameters) + np.ones((wishart_n_parameters, wishart_n_parameters))) / 2
    # E = sigma * wishart.rvs(df=wishart_n_parameters, scale=wishart_scale, size=1, random_state=seed)
    E = sigma * (np.eye(wishart_n_parameters) + np.ones((wishart_n_parameters, wishart_n_parameters))) / 2
    AB = np.hstack((A,B))
    sys_dist = MN.from_MND(m=AB, E=E, dim=(n, n+m))
    return sys_dist

def sample_MN(N_sys, trunc_threshold, controllability_tol, sys_dist : MN, rnd):
    n, p = sys_dist.dim
    m = p - n
    ## Sampling of each system and trajectory generation from it
    systems = []
    while len(systems)<N_sys:
        _M = sys_dist.sample_truncated(n=1, c=trunc_threshold, rnd=rnd)[0]
        A = _M[:, :n]
        B = _M[:, -m:]
        # Discard uncontrollable systems
        if control_utils.check_controllability(A, B, controllability_tol):
            systems.append((A,B))
        else:
            print('Uncontrollable system sampled, resampling')
    return systems