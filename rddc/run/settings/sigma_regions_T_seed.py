import numpy as np

def get_settings():
    name = 'dean_1d_sigma_regions_T_seed'
    suffix = ''
    eps = 1e-8
    controllability_tol = 1e-3
    seed = 100
    trunc_threshold = 0.95
    A = np.array([[0.9]])
    B = np.array([[1.4]])
    n = A.shape[0]         # number of state variables x_k
    m = B.shape[1]          # number of control variables u_k
    # sigma = 2e-2            # variability of system matrices
    # K_prelim = np.array([[-0.62560034]])
    K_prelim = None
    init_state = 'rand'
    input_noise_amplitude = 0.1
    # algorithm = 'robust_hinf_scenario_slemma'
    #algorithm = 'robust_stabilization_scenario_slemma'
    #check_slater = False
    #check_willems = False
    #output_verbosity = 0
    max_trajectory_norm_factor = 2
    max_trajectory_length = 50
    #num_cores = 3

    #N_test = 1              # number of systems to generate and test
    N_synth = 1            # number of systems used for controller synthesis
    T = 50                 # number of samples per trajectory

    # noise
    m_w = n                 # number of disturbance variables w_k
    B_w = np.eye(n, m_w)
    bound = 0.01             # noise bound
    assumedBound = 0.015      # noise bound assumed for robust controller synthesis

    # performance metric
    Q = np.eye(n, n)
    S = np.zeros((n, m))
    R = np.eye(m, m)
    C = np.array([[1]])
    D = np.array([[0]])

    return locals()

def get_variations():
    """
    Parameters from 'get_settings' to vary
    If a parameter is used to calculate other parameters, it should not be varied,
        since the calculation won't be adjusted
    """
    T = [5, 10, 50, 100]

    seed = [10,20,30,40]

    return locals()