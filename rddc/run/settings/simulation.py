import numpy as np

def get_settings():
    name = 'simulation'
    suffix = 'paper'
    seed = 550
    train_trajectory_path = 'traj_train_'
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
    output_verbosity = 0

    # Extra weight distribution
    mass_range = [0, 0.006]
    pos_size = 0.01
    # displacement_planar = 0.01
    # displacement_vert = 0.0

    N_synth = 15
    N_test = 100
    start = 0                              # time step to start sampling the trajectory with
    T = 300                                  # number of samples per trajectory for controller synthesis
    T_test = 60                            # number of samples per trajectory for performance evaluation

    # noise
    m_w = n                 # number of disturbance variables w_k
    B_w = np.eye(n, m_w)
    assumedBound = 0.002     # noise bound assumed for robust controller synthesis

    # performance metric
    Q = np.eye(n, n)
    S = np.zeros((n, m))
    R = np.eye(m, m)
    C = np.array([[1,1,1,1,1,1]])
    D = np.array([[1, 1]])

    trainSettings = {
        'sfb': None,
        'sfb_freq_hz':10,
        'num_samples':T,
        'ctrl_noise':1.0,
        'proc_noise':0.001,
        'traj':'hover',
        'part_pid_off':True,
        'traj_filename':None,
        'plot':False,
        'cut_traj':True,
        'init_rpys_spread':0.2,
        # 'init_xyzs_spread':0.01,
        'gui':False,
        'pid_type':'dsl'
    }
    testSettings = {
        'sfb': 'direct',
        'sfb_freq_hz':10,
        'num_samples':T_test,
        'ctrl_noise':0.0,
        'proc_noise':0.001,
        'traj':'line',
        'part_pid_off':True,
        'traj_filename':None,
        'plot':False,
        'cut_traj':False,
        'wrap_wp':False,
        'wind_on':True,
        'gui':False,
        'pid_type':'dsl'
    }

    return locals()

if __name__=='__main__':
    pass