import numpy as np

def get_settings():
    name = 'simulation'
    suffix = 'like_experiment'
    seed = 20
    eps = 1e-8
    controllability_tol = 1e-3
    state_idx = [0,1,3,4,6,7]
    input_idx = [9,10]
    # We're prescribing "inputs" to our system while observing "states"
    # We're trying to control "states" through "inputs"
    n = len(state_idx)                  # number of state variables x_k
    m = len(input_idx)                   # number of control variables u_k

    A = np.zeros((n,n))
    B = np.zeros((n,m))
    # algorithm = 'robust_hinf_scenario_slemma'
    check_slater = False
    check_willems = False
    algorithm = 'robust_stabilization_scenario_slemma'
    # algorithm = 'robust_lqr_scenario_slemma'
    # algorithm = 'robust_h2_scenario_slemma'
    output_verbosity = 0

    # Extra weight distribution
    extra_loads = list() # do not touch this one, only adjust extra_loads_synth or extra_loads_test
    extra_loads_synth = list() # leave empty ("list()") if you want to pick them randomly
    # extra_loads_synth = [
        # {'mass': 0.000 ,'position':np.array([ 0.000,  0.000, -0.000]), 'form':'ball', 'size':[0.0]},
    #     {'mass': 0.0013,'position':np.array([-0.002,  0.001,  0.001]), 'form':'ball', 'size':[0.0]},
    #     {'mass': 0.0013,'position':np.array([ 0.000, -0.004,  0.002]), 'form':'ball', 'size':[0.0]},
    #     {'mass': 0.0007,'position':np.array([ 0.006, -0.001,  0.000]), 'form':'ball', 'size':[0.0]},
    #     {'mass': 0.0018,'position':np.array([-0.001, -0.001, -0.001]), 'form':'ball', 'size':[0.0]},
    # ]
    extra_loads_test = list() # leave empty ("list()") if you want to pick them randomly
    mass_range = [0.007, 0.015]
    pos_size = [0.015, 0.015, 0.01]
    # displacement_planar = 0.01
    # displacement_vert = 0.0
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
        # 'hover_000001_10Hz_0.9',
        # 'hover_000020_10Hz_0.9',
        # 'hover_000400_10Hz_1.0',
        # 'hover_000512_10Hz_0.9_fewPoints',
        # # 'hover_000512_10Hz_1.0_fewPoints',
        # 'hover_012000_10Hz_0.9',
        # # 'hover_020031_10Hz_1.0_fewPoints',
        # 'hover_100003_10Hz_1.0',
        # 'hover_101002_10Hz_0.9',
        # # 'hover_101003_10Hz_0.9_fewPoints',
        # 'hover_120000_10Hz_1.0',
        # 'hover_120002_10Hz_0.9',
        # 'hover_201011_10Hz_1.0',
        # 'hover_220001_10Hz_1.0',
        # # 'hover_220011_10Hz_1.0_fewPoints',
        # 'hover_221000_10Hz_0.9',
        # 'hover_300002_10Hz_0.9_fewPoints',
        ### 50Hz #####################
        # 'hover_000000_50Hz_1.5',
        # 'hover_000000_50Hz_2.5_vicon300Hz1sample',
        # 'hover_020000_50Hz_1.1',
        ### 20 Hz ####################
        # 'hover_000000_20Hz_0.5',
        ### 5 Hz #####################
        # 'hover_000000_5Hz_1.0',
    ]

    N_synth = 10
    N_test = 100
    start = 0                              # time step to start sampling the trajectory with
    T = 1000                               # number of samples per trajectory for controller synthesis
    T_test = 60                            # number of samples per trajectory for performance evaluation

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

    trainSettings = {
        'num_drones':N_synth,
        'sfb':None,
        'sfb_freq_hz':20,
        'num_samples':T,
        'ctrl_noise':2.0,
        'proc_noise':0.005,
        'traj':'hover',
        'part_pid_off':False,
        'traj_filename':None,
        'plot':False,
        'cut_traj':True,
        'init_rpys_spread':0.05,
        # 'init_xyzs_spread':0.01,
        'gui':True,
        'pid_type':'mellinger',
        'control_freq_hz':500,
        'simulated_delay_ms':10
    }
    testSettings = {
        'num_drones':N_test,
        'sfb':'rddc',
        'sfb_freq_hz':10,
        'num_samples':T_test,
        'ctrl_noise':0.0,
        'proc_noise':0.001,
        'traj':'8',
        'part_pid_off':True,
        'traj_filename':None,
        'plot':False,
        'cut_traj':False,
        'wrap_wp':False,
        'wind_on':False,
        'gui':True,
        'pid_type':'mellinger',
        'control_freq_hz':500,
        'simulated_delay_ms':8
    }

    return locals()

if __name__=='__main__':
    pass