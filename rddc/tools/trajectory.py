import numpy as np
from rddc.tools import control_utils

def generate_trajectories_synth(systems, T, rnd=np.random.default_rng(),
                            noiseInfo=dict(), N_traj_per_sys=1, start=None,
                            K=None, inputNoiseAmplitude=1.0):
    """
    Take given systems (list of tuples (A,B)), generate N_traj for each system
    each trajectory contains T_traj+1 points

    @param: systems: list of tuples (A,B)
    @param: N_traj: number of trajectories per system to generate
    @param: T: number of samples to generate per trajectory.
        The output trajectory contains T+1 states and T inputs
    @param: noiseInfo: a dictionary with m_w, bound, B_w
    @param: rnd: random number generator instance
    @param: start: vector of initial states. Default is [0...0]^T
    @param: K: preliminary stabilizing controller. If None, then random inputs are used
    @param: inputNoiseAmplitude: input noise amplitude

    @output: a flat list of dictionaries, where j-th trajectory of the i-th system is the [j + N_traj * i] element
    """
    trajectories = list()
    B_w = noiseInfo['B_w']
    m_w = noiseInfo['m_w']
    for sysId in range(len(systems)):
        A = systems[sysId][0]
        B = systems[sysId][1]
        n = B.shape[0]
        max_trajectory_norm = noiseInfo['max_trajectory_norm_factor'] * np.sqrt(n)

        while len(trajectories)<(sysId+1)*N_traj_per_sys:
            assumedBound = noiseInfo['assumedBound']
            slaterFails = 0
            conditionNumberFails = 0
            Ts = list()
            U0_merged = np.empty((n,0))
            X0_merged = np.empty((n,0))
            X1_merged = np.empty((n,0))
            while X0_merged.shape[1]<T:
                trajectory = generate_one_trajectory(
                    system      = systems[sysId],
                    T           = min(T-X0_merged.shape[1], noiseInfo['max_trajectory_length']),
                    rnd         = rnd,
                    noiseInfo   = noiseInfo,
                    start       = start,
                    K           = K,
                    inputNoiseAmplitude = inputNoiseAmplitude
                )
                U0 = trajectory['U0']
                X0 = trajectory['X0']
                X1 = trajectory['X1']

                noisy_system = {'A':A, 'B':B, 'B_w':B_w}
                T_cut = U0.shape[1]
                Phi_11 = assumedBound**2 * np.eye(m_w) * T_cut
                Phi_12 = np.zeros((m_w, T_cut))
                Phi_22 = -np.eye(T_cut)
                Phi = np.block([[Phi_11  , Phi_12],
                                [Phi_12.T, Phi_22]])
                slater_ok = control_utils.check_gen_slater_condition(U0, X0, X1, Phi)
                # willems_ok = control_utils.check_persistent_excitation_willems(X0, U0)
                # condition_number_ok = np.linalg.cond(np.vstack([U0, X0])) < 1000
                condition_number_ok = np.linalg.norm(X0[:,-1]) < max_trajectory_norm
                trajectory_ok = slater_ok and condition_number_ok
                while not trajectory_ok:
                    if not slater_ok:
                        slaterFails += 1
                        T_cut = int(np.floor(0.9 * T_cut))
                        assert T_cut >= 1, "Trajectory length of 1 was not enough to make it ok, WTF?"
                        # assumedBound = 1.1 * assumedBound
                    if not condition_number_ok: #not condition_number_ok
                        conditionNumberFails += 1
                        # print('Condition number was not ok')
                        T_cut = int(np.floor(0.7 * T_cut))
                        assert T_cut >= 1, "Trajectory length of 1 was not enough to make it ok, WTF?"
                    U0 = U0[:,:T_cut]
                    X0 = X0[:,:T_cut]
                    X1 = X1[:,:T_cut]
                    Phi_11 = assumedBound**2 * np.eye(m_w) * T_cut
                    Phi_12 = np.zeros((m_w, T_cut))
                    Phi_22 = -np.eye(T_cut)
                    Phi = np.block([[Phi_11  , Phi_12],
                                    [Phi_12.T, Phi_22]])
                    slater_ok = control_utils.check_gen_slater_condition(U0, X0, X1, Phi)
                    # willems_ok = control_utils.check_persistent_excitation_willems(X0, U0)
                    # condition_number_ok = np.linalg.cond(np.vstack([U0, X0])) < 1000
                    condition_number_ok = np.linalg.norm(X0[:,-1]) < max_trajectory_norm
                    trajectory_ok = slater_ok and condition_number_ok
                Ts.append(T_cut)
                U0_merged = np.hstack([U0_merged, U0])
                X0_merged = np.hstack([X0_merged, X0])
                X1_merged = np.hstack([X1_merged, X1])
            assert U0_merged.shape[1]==X0_merged.shape[1]
            assert X1_merged.shape[1]==X0_merged.shape[1]
            last_T_cut_overshoot = sum(Ts) - T
            Ts[-1] -= last_T_cut_overshoot
            U0_merged = U0_merged[:,:T]
            X0_merged = X0_merged[:,:T]
            X1_merged = X1_merged[:,:T]
            # print(f"Trajectories have been generated and merged: \n {Ts}")
            # print(f"Number of slater fails: {slaterFails}, Number of condition number fails: {conditionNumberFails}\n")

            trajectories.append({'U0': U0_merged, 'X0':X0_merged, 'X1':X1_merged, 'assumedBound':assumedBound})

    return trajectories

def generate_trajectories_test(systems, T, rnd=np.random.default_rng(),
                            noiseInfo=dict(), N_traj_per_sys=1, start=None,
                            K=None, inputNoiseAmplitude=0.0):
    """
    Take given systems (list of tuples (A,B)), generate N_traj for each system
    each trajectory contains T_traj+1 points

    @param: systems: list of tuples (A,B)
    @param: N_traj: number of trajectories per system to generate
    @param: T: number of samples to generate per trajectory.
        The output trajectory contains T+1 states and T inputs
    @param: noiseInfo: a dictionary with m_w, bound, B_w
    @param: rnd: random number generator instance
    @param: start: vector of initial states. Default is [0...0]^T
    @param: K: preliminary stabilizing controller. If None, then random inputs are used
    @param: inputNoiseAmplitude: input noise amplitude

    @output: a flat list of dictionaries, where j-th trajectory of the i-th system is the [j + N_traj * i] element
    """
    trajectories = list()
    for sysId in range(len(systems)):

        while len(trajectories)<(sysId+1)*N_traj_per_sys:
            trajectory = generate_one_trajectory(
                system      = systems[sysId],
                T           = T,
                rnd         = rnd,
                noiseInfo   = noiseInfo,
                start       = start,
                K           = K,
                inputNoiseAmplitude = inputNoiseAmplitude
            )
        trajectories.append(trajectory)

    return trajectories

def generate_one_trajectory(system, T, rnd, noiseInfo, start, K, inputNoiseAmplitude):
    A = system[0]
    B = system[1]
    n = B.shape[0]
    m = B.shape[1]
    m_w = noiseInfo['m_w']
    wBar = noiseInfo['bound']
    B_w = noiseInfo['B_w']
    W = noise_from_a_ball(m_w, T, wBar, rnd)

    # Initialize the input signals
    U_all = (2*rnd.random((m, T))-np.ones((m, T)))*inputNoiseAmplitude

    # Initialize system states
    X_all = np.zeros((n, T+1))
    if start is None:
        pass # keep it 0
    elif start=='rand':
        X_all[:, 0] = 2*rnd.random(n)-1
    else:
        X_all[:, 0] = start

    for i in range(T):
        if K is not None:
            #TODO: why do we add the random input here? shouldn't it be just overwritten?
            # -> No. If we overwrite it, we break persistency of excitation
            # However, we need to overwrite it for the final system testing,
            # bool inputNoise takes care of that
            U_all[:, i] = U_all[:, i] + K @ X_all[:, i]
        X_all[:, i+1] = A @ X_all[:,i] + B @ U_all[:,i] + B_w @ W[:,i]

    U0 = control_utils.hankel_matrix(U_all, 0, 1, T)
    X0 = control_utils.hankel_matrix(X_all, 0, 1, T)
    X1 = control_utils.hankel_matrix(X_all, 1, 1, T)

    return {'U0':U0, 'X0':X0, 'X1':X1}

def noise_from_a_ball(m_w, T, radius, rnd):
    """
    uniform distibution from a ball
    acquired by clipping a uniform distribution from a box

    @output: m_w x T matrix, containing T samples of dimension m_w
    """

    W = np.zeros((m_w, T))
    k = 0
    while k < T:
        w_k = radius*(2*rnd.random(m_w) - np.ones(m_w))
        if np.linalg.norm(w_k)<=radius:
            W[:,k] = w_k
            k = k + 1

    return W

def test_trajectory(X0, X1, U0, noisy_system, settings):

    # if np.linalg.norm(X_all[:,-1])>10:
    #     return False

    # if not control_utils.check_persistent_excitation_berberich(noisy_system, U_all, W):
    #     # print('failed persistency of excitation')
    #     return False
    T = U0.shape[1]
    m_w = settings['m_w']

    Phi_11 = settings['assumedBound']**2 * np.eye(m_w) * T
    Phi_12 = np.zeros((m_w, T))
    Phi_22 = -np.eye(T)
    Phi = np.block([[Phi_11  , Phi_12],
                    [Phi_12.T, Phi_22]])
    if not control_utils.check_gen_slater_condition(U0, X0, X1, Phi):
        # print('failed slater condition')
        return False
    return True