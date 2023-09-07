import numpy as np

def get_settings():
    name = 'simulation'
    suffix = 'ctrl_from_experiment'
    seed = 20
    eps = 1e-8
    controllability_tol = 1e-3
    state_idx = [0,1,3,4,6,7]
    input_idx = [9,10]
    n = len(state_idx)                  # number of state variables x_k
    m = len(input_idx)                   # number of control variables u_k

    check_slater = False
    check_willems = True
    # algorithm = 'robust_hinf_scenario_slemma'
    algorithm = 'robust_stabilization_scenario_slemma'
    # algorithm = 'robust_lqr_scenario_slemma'
    # algorithm = 'robust_h2_scenario_slemma'
    output_verbosity = 0

   
    filenames = [
        # 'hover_000000_10Hz_1.0',
        # 'hover_000003_10Hz_1.0',
        # 'hover_001002_10Hz_1.0',
        # 'hover_010001_10Hz_1.0',
        # 'hover_010201_10Hz_1.0',
        # 'hover_011002_10Hz_1.0',
        # 'hover_100000_10Hz_1.0',
        # 'hover_100012_10Hz_1.0',
        # 'hover_103010_10Hz_1.0',
        # 'hover_110002_10Hz_1.0',
        # 'hover_110010_10Hz_1.0',
        # 'hover_120001_10Hz_1.0',
        # 'hover_120002_10Hz_1.0',
        # 'hover_221000_10Hz_1.0',
        # 'hover_300002_10Hz_1.0',
        ##### Second session #####
        'hover_000000_10Hz_1.0_',
        'hover_000001_10Hz_0.9',
        'hover_000020_10Hz_0.9',
        'hover_000400_10Hz_1.0',
        'hover_000512_10Hz_0.9_fewPoints',
        # 'hover_000512_10Hz_1.0_fewPoints',
        'hover_012000_10Hz_0.9',
        # 'hover_020031_10Hz_1.0_fewPoints',
        'hover_100003_10Hz_1.0',
        'hover_101002_10Hz_0.9',
        # 'hover_101003_10Hz_0.9_fewPoints',
        'hover_120000_10Hz_1.0',
        'hover_120002_10Hz_0.9',
        'hover_201011_10Hz_1.0',
        'hover_220001_10Hz_1.0',
        # 'hover_220011_10Hz_1.0_fewPoints',
        'hover_221000_10Hz_0.9',
        'hover_300002_10Hz_0.9_fewPoints',
        # ## 50Hz #####################
        # 'hover_000000_50Hz_1.5',
        # 'hover_000000_50Hz_2.5_vicon300Hz1sample',
        # 'hover_020000_50Hz_1.1',
        # ## 20 Hz ####################
        # 'hover_000000_20Hz_0.5',
        # ## 5 Hz #####################
        # 'hover_000000_5Hz_1.0',
        # ## dx 0.5 ###################
        # 'hover_000000_10Hz_1.0_dx0.5',
        # 'hover_100000_10Hz_1.0_dx0.5',
        # 'hover_010000_10Hz_1.0_dx0.5',
    ]

    # N_synth = 10
    # N_test = 100
    # start = 0                              # time step to start sampling the trajectory with

    # noise
    m_w = n                 # number of disturbance variables w_k
    B_w = np.eye(n, m_w)
    assumedBound = 0.001     # noise bound assumed for robust controller synthesis

    # performance metric
    Q = np.eye(n, n)
    S = np.zeros((n, m))
    R = np.eye(m, m)
    C = np.array([[1,1,1,1,1,1]])
    D = np.array([[1, 1]])

    return locals()

if __name__=='__main__':
    pass