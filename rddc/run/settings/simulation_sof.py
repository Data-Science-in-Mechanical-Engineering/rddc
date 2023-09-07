import numpy as np
import os

def get_settings():
    name = 'simulation'
    suffix = 'thesis'
    seed = 273
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
    check_willems = True
    algorithm = 'robust_stabilization_scenario_slemma'
    # algorithm = 'robust_lqr_scenario_slemma'
    # algorithm = 'robust_h2_scenario_slemma'
    output_verbosity = 1

    ## Extra load distribution
    extra_loads = list() # do not touch this one, only adjust extra_loads_synth or extra_loads_test
    extra_loads_synth = list() # leave empty ("list()") if you want to pick them randomly
    # extra_loads_synth = [
        # {'mass': 0.000 ,'position':np.array([ 0.000,  0.000,  0.000]), 'form':'ball', 'size':[0.0]},
    #     {'mass': 0.0013,'position':np.array([-0.002,  0.001,  0.001]), 'form':'ball', 'size':[0.0]},
    #     {'mass': 0.0013,'position':np.array([ 0.000, -0.004,  0.002]), 'form':'ball', 'size':[0.0]},
    #     {'mass': 0.0007,'position':np.array([ 0.006, -0.001,  0.000]), 'form':'ball', 'size':[0.0]},
    #     {'mass': 0.0018,'position':np.array([-0.001, -0.001, -0.001]), 'form':'ball', 'size':[0.0]},
    # ]
    extra_loads_test = list() # leave empty ("list()") if you want to pick them randomly
    mass_range = [0.007, 0.012]
    pos_size = [0.01, 0.01, 0.002]
    # displacement_planar = 0.01
    # displacement_vert = 0.0

    N_synth = 15
    N_test = 100
    start = 0                              # time step to start sampling the trajectory with
    T = 300                                  # number of samples per trajectory for controller synthesis
    T_test = 80                            # number of samples per trajectory for performance evaluation

    # noise
    m_w = n                 # number of disturbance variables w_k
    B_w = np.eye(n, m_w)
    assumedBound = 0.001     # noise bound assumed for robust controller synthesis

    # performance metric
    Q = np.eye(n, n)*np.diag([100,100,0.1,0.1,0.01,0.01])
    S = np.zeros((n, m))
    R = np.eye(m, m)*0
    C = np.array([[1,1,1,1,1,1]])*np.array([[10,10,0.5,0.5,2,2]])*1
    D = np.array([[1, 1]])*0.01

    trainSettings = {
        'num_drones':N_synth,
        'sfb':None,
        'sfb_freq_hz':10,
        'num_samples':T,
        'ctrl_noise':1.0,
        'proc_noise':0.00001,
        'traj':'hover',
        'part_pid_off':True,
        'traj_filename':None,
        'plot':False,
        'cut_traj':True,
        'init_rpys_spread':0.1,
        'init_xyzs_spread':0.5,
        'gui':False,
        'pid_type':'mellinger'
    }
    trainSettings['traj_filename'] = os.path.join(
        'data',
        name,
        suffix,
        'train_sof',
    )
    testSettings = {
        'num_drones':N_test,
        'sfb':'rddc',
        'sfb_freq_hz':10,
        'num_samples':T_test,
        'ctrl_noise':0.0,
        'proc_noise':0.0,
        'traj':'line30',
        'part_pid_off':True,
        'traj_filename':None,
        'plot':False,
        'cut_traj':False,
        'wrap_wp':False,
        'wind_force':0.3*np.array([-np.sin(np.radians(30)), np.cos(np.radians(30)), 0]),
        'gui':False,
        'user_debug_gui':False,
        'record_reference':True,
        'pid_type':'mellinger'
    }
    testSettings['traj_filename'] = os.path.join(
        'data',
        name,
        suffix,
        'test_sof',
    )

    safe_state_lims = [
        [-1.5, 1.5],            #x
        [-1.5, 1.5],            #y
        [-0.5, 0.5],            #z

        [-1.5, 1.5],            #vx
        [-1.5, 1.5],            #vy
        [-0.5, 0.5],            #vz

        [-0.8, 0.8],            #roll
        [-0.8, 0.8],            #pitch
        [-0.1, 0.1],            #yaw

        [-10, 10],            #roll rate
        [-10, 10],            #pitch rate
        [-5, 5],            #yaw rate
    ]
    return locals()

if __name__=='__main__':
    pass