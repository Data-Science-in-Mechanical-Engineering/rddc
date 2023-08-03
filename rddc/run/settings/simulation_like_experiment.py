import numpy as np
import os

def get_settings():
    name = 'simulation'
    suffix = 'like_experiment'
    seed = 642
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
    # algorithm = 'robust_stabilization_scenario_slemma'
    algorithm = 'robust_lqr_scenario_slemma'
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
    pos_size = [0.01, 0.01, 0.003]

    N_synth = 10
    N_test = 15
    start = 0                              # time step to start sampling the trajectory with
    T = 500                               # number of samples per trajectory for controller synthesis
    T_test = 120                            # number of samples per trajectory for performance evaluation

    # noise
    m_w = n                 # number of disturbance variables w_k
    B_w = np.eye(n, m_w)
    assumedBound = 0.001     # noise bound assumed for robust controller synthesis

    # performance metric
    Q = np.eye(n, n)*np.diag([1000,1000,0.001,0.001,1,1])
    S = np.zeros((n, m))
    R = np.eye(m, m)*0
    C = np.array([[1,1,1,1,1,1]])
    D = np.array([[1, 1]])

    vicon_freq = 300

    trainSettings = {
        'num_drones':N_synth,
        'sfb':None,
        'sfb_freq_hz':20,
        'num_samples':T,
        'ctrl_noise':1.0,
        'proc_noise':0.0001,
        # 'meas_noise':[  2e-4,
        #                 2e-4,
        #                 2e-4,
        #                 2e-4*vicon_freq*2,
        #                 2e-4*vicon_freq*2,
        #                 2e-4*vicon_freq*2,
        #                 np.radians(0.01),
        #                 np.radians(0.01),
        #                 np.radians(0.01),],
        'traj':'hover',
        'part_pid_off':True,
        'traj_filename':None,
        'plot':False,
        'cut_traj':True,
        'init_rpys_spread':0.05,
        # 'init_xyzs_spread':0.01,
        'gui':True,
        'pid_type':'mellinger',
        'control_freq_hz': 500,
        'simulated_delay_ms':50
    }
    trainSettings['traj_filename'] = os.path.join(
        'data',
        name,
        suffix,
        'train' + '_' + trainSettings['traj'] + \
            '_T' + str(T) + \
            '_' + str(trainSettings['sfb_freq_hz']) + 'Hz' + \
            '_pn' + str(trainSettings['proc_noise']) + \
            '_delay' + str(trainSettings['simulated_delay_ms']) + \
            '_mass' + str(mass_range) + \
            '_pos' + str(pos_size)
    )
    testSettings = {
        'num_drones':N_test,
        'sfb':'rddc',
        'sfb_freq_hz':20,
        'num_samples':T_test,
        'ctrl_noise':0.0,
        'proc_noise':0.0001,
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
        'simulated_delay_ms':50
    }

    safe_state_lims = [
        [-100, 100],            #x
        [-100, 100],            #y
        [0.1, 2],               #z
        [-2.5, 2.5],            #vx
        [-2.5, 2.5],            #vy
        [-2.5, 2.5],            #vz
        [-0.5, 0.5],            #roll
        [-0.5, 0.5],            #pitch
        [-0.1, 0.1],            #yaw
        [-50, 50],            #roll rate
        [-50, 50],            #pitch rate
        [-50, 50],            #yaw rate
    ]
    return locals()

if __name__=='__main__':
    pass