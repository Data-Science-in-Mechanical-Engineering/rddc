import numpy as np

def get_settings():
    name = 'simulation'
    suffix = 'paper'
    seed = 42
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
    # algorithm = 'robust_lqr_scenario_slemma'
    # algorithm = 'robust_h2_scenario_slemma'
    output_verbosity = 0

    ## Extra load distribution
    extra_loads = list() # do not touch this one, only adjust extra_loads_synth or extra_loads_test
    extra_loads_synth = list() # leave empty ("list()") if you want to pick them randomly
    # extra_loads_synth = [
    #     {'mass': 0.001 ,'position':np.array([ 0.003,  0.002, -0.001]), 'form':'ball', 'size':[0.0]},
    #     {'mass': 0.0013,'position':np.array([-0.002,  0.001,  0.001]), 'form':'ball', 'size':[0.0]},
    #     {'mass': 0.0013,'position':np.array([ 0.000, -0.004,  0.002]), 'form':'ball', 'size':[0.0]},
    #     {'mass': 0.0007,'position':np.array([ 0.006, -0.001,  0.000]), 'form':'ball', 'size':[0.0]},
    #     {'mass': 0.0018,'position':np.array([-0.001, -0.001, -0.001]), 'form':'ball', 'size':[0.0]},
    # ]
    extra_loads_test = list() # leave empty ("list()") if you want to pick them randomly
    mass_range = [0, 0.005]
    pos_size = 0.015
    # displacement_planar = 0.01
    # displacement_vert = 0.0

    N_synth = 15
    N_test = 50
    start = 0                              # time step to start sampling the trajectory with
    T = 300                                  # number of samples per trajectory for controller synthesis
    T_test = 120                            # number of samples per trajectory for performance evaluation

    # noise
    m_w = n                 # number of disturbance variables w_k
    B_w = np.eye(n, m_w)
    assumedBound = 0.0002     # noise bound assumed for robust controller synthesis

    # performance metric
    Q = np.eye(n, n)*1
    S = np.zeros((n, m))
    R = np.eye(m, m)*1
    C = np.array([[1,1,1,1,1,1]])
    D = np.array([[1, 1]])

    trainSettings = {
        'num_drones':N_synth,
        'sfb':None,
        'sfb_freq_hz':50,
        'num_samples':T,
        'ctrl_noise':1.0,
        'proc_noise':0.0001,
        'traj':'hover',
        'part_pid_off':True,
        'traj_filename':None,
        'plot':False,
        'cut_traj':True,
        'init_rpys_spread':0.2,
        # 'init_xyzs_spread':0.01,
        'gui':True,
        'pid_type':'mellinger'
    }
    testSettings = {
        'num_drones':N_test,
        'sfb':'direct',
        'sfb_freq_hz':50,
        'num_samples':T_test,
        'ctrl_noise':0.0,
        'proc_noise':0.0001,
        'traj':'8',
        'part_pid_off':True,
        'traj_filename':None,
        'plot':False,
        'cut_traj':False,
        'wrap_wp':False,
        'wind_on':True,
        'gui':True,
        'user_debug_gui':False,
        'pid_type':'mellinger'
    }

    return locals()

if __name__=='__main__':
    pass