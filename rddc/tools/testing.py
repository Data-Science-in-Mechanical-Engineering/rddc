import numpy as np
import matplotlib.pyplot as plt
from rddc.tools import control_utils, trajectory
import picos as pc

def test_stability(systems, K, eps=1e-6):

    N_test = len(systems)
    #print('\nChecking spectral radii of tested systems:')
    N_stable = 0
    for sysId in range(N_test):
        A = systems[sysId][0]
        B = systems[sysId][1]
        if control_utils.check_stability(A + B@K, eps=eps):
            N_stable = N_stable + 1
    #print('{0}/{1} systems have stable spectral radius, ratio = {2}\n'.format(N_stable, N_test, N_stable/N_test))

    return N_stable

def test_performance_empiric(systems, K, T, settings, N_traj_per_sys, stab_eps=1e-6):
    #based on lqr_cost_empiric
    costs = list()
    for system in systems:
        cost, _ = lqr_cost_empiric( #TODO: account for std?
            system          = system,
            T               = T,
            K               = K,
            settings        = settings,
            stab_eps        = stab_eps,
            N_traj_per_sys  = N_traj_per_sys
        )
        costs.append(cost)
    return np.array(costs)
    # TODO: Idk what is n and self.V

def lqr_cost_empiric(system, T, K, settings, stab_eps, N_traj_per_sys=100):
    if not control_utils.check_stability(system[0] + system[1] @ K, stab_eps):
        return np.Inf, 0.
    trajectories = trajectory.generate_trajectories_test(
        systems         = [(system[0],system[1])],
        T               = T,
        N_traj_per_sys  = N_traj_per_sys,
        K               = K,
        start           = 'rand',
        noiseInfo       = settings
    )
    return lqr_cost_trajectories(trajectories, settings)

def lqr_cost_trajectories(trajectories, metric):
    cost_sum = list()
    for traj in trajectories:

        x = traj['state']
        u = traj['input']
        cost = list()
        for i in range(u.shape[1]):

            x_i = x[:,[i]]
            u_i = u[:, [i]]

            c_i = x_i.T @ metric['Q'] @ x_i + u_i.T @ metric['R'] @ u_i
            cost.append(c_i)

        cost = np.array(cost)
        cost_sum.append(cost.mean())

    cost_sum = np.array(cost_sum)
    return cost_sum.mean(), cost_sum.std()