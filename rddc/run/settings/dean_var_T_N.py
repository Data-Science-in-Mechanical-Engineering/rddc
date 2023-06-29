import numpy as np

def get_settings():
    name = 'dean_var_T_N'
    suffix = ''
    eps = 1e-8
    controllability_tol = 1e-3
    seed = 42
    trunc_threshold = 0.95
    A = np.array([[1.01,  0.01,  0.0], 
                    [0.01,  1.01,  0.01],
                    [0.0,   0.01,  1.01]])
    B = np.array([[1.0,   0.0,   0.0], 
                    [0.0,   1.0,   0.0],
                    [0.0,   0.0,   1.0]])
    n = A.shape[0]         # number of state variables x_k
    m = B.shape[1]          # number of control variables u_k
    sigma = 2e-2            # variability of system matrices
    # K_prelim = np.array([ [-0.62560034, -0.00824407,  0.00091957],
    #                         [-0.00767528, -0.62593008, -0.00634649],
    #                         [ 0.00121963, -0.00788259, -0.62513108]])
    K_prelim = None
    init_state = 'rand'
    input_noise_amplitude = 0.1
    # algorithm = 'robust_hinf_scenario_slemma'
    algorithm = 'robust_stabilization_scenario_slemma'
    check_slater = False
    check_willems = False
    output_verbosity = 0
    max_trajectory_norm_factor = 2
    max_trajectory_length = 50
    num_cores = 10

    N_test = 1000          # number of systems to generate and test
    N_synth = 1            # number of systems used for controller synthesis
    T = 100                 # number of samples per trajectory

    # noise
    m_w = n                 # number of disturbance variables w_k
    B_w = np.eye(n, m_w)
    bound = 0.01             # noise bound
    assumedBound = 0.15      # noise bound assumed for robust controller synthesis

    # performance metric
    Q = np.eye(n, n)
    S = np.zeros((n, m))
    R = np.eye(m, m)
    C = np.array([[1, 1, 1]])
    D = np.array([[0, 0, 0]])

    return locals()

def get_variations():
    """
    Parameters from 'get_settings' to vary
    If a parameter is used to calculate other parameters, it should not be varied,
        since the calculation won't be adjusted
    """
    # N_synth = [int(x) for x in np.ceil(np.logspace(0, np.log(300) / np.log(10), 10))],
    N_synth = [1, 2, 4, 8, 16, 32, 63, 125, 248, 494, 984]

    sigma = [0.1]

    T = [10, 18, 29, 50, 84, 143, 243, 413, 702, 1194, 2031, 3456, 5879, 10000]

    bound = [0.0005]

    assumedBound = [0.001]

    seed = [168, 158, 24, 722, 612, 644, 763, 585, 688, 555, 697, 334, 634, 340, 548, 4, 302, 135, 884, 166, 114, 913, 261, 951, 436, 104, 620, 56, 741, 948, 959, 931, 341, 672, 412, 497, 623, 107, 396, 948, 451, 581, 870, 269, 378, 184, 963, 564, 468, 974]

    return locals()