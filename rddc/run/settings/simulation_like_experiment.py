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
    algorithm = 'robust_stabilization_scenario_slemma'
    # algorithm = 'robust_lqr_scenario_slemma'
    # algorithm = 'robust_h2_scenario_slemma'
    output_verbosity = 1

    # Extra weight distribution
    extra_loads = list() # do not touch this one, only adjust extra_loads_synth or extra_loads_test
    extra_loads_synth = list() # leave empty ("list()") if you want to pick them randomly
    # extra_loads_synth = [
        # {'mass': 0.000 ,'position':np.array([ 0.000,  0.000, -0.000]), 'form':'ball', 'size':[0.0]},
    #     {'mass': 0.011,'position':np.array([0.000,  0.008,  0.001]), 'form':'ball', 'size':[0.0]},
    #     {'mass': 0.013,'position':np.array([ 0.008, -0.004,  0.002]), 'form':'ball', 'size':[0.0]},
    #     {'mass': 0.008,'position':np.array([ -0.006, 0.001,  0.000]), 'form':'ball', 'size':[0.0]},
    #     {'mass': 0.010,'position':np.array([-0.003, -0.007, -0.001]), 'form':'ball', 'size':[0.0]},
    # ]
    extra_loads_test = list() # leave empty ("list()") if you want to pick them randomly
    mass_range = [0.007, 0.012]
    pos_size = [0.01, 0.01, 0.002]

    N_synth = 3
    N_test = 5
    start = 0.0                              # time to start sampling the trajectory with
    T = 300                               # number of samples per trajectory for controller synthesis
    T_test = 80                            # number of samples per trajectory for performance evaluation
    f = 10 #Hz

    # noise
    proc_noise = 0.0001
    m_w = n                 # number of disturbance variables w_k
    B_w = np.eye(n, m_w)
    assumedBound = 0.001     # noise bound assumed for robust controller synthesis

    # performance metric
    Q = np.eye(n, n)
    S = np.zeros((n, m))
    R = np.eye(m, m)
    C = np.array([[1,1,1,1,1,1]])
    D = np.array([[1, 1]])

    vicon_freq = 300
    vicon_error_x = 1e-3
    # vicon_error_v = vicon_error_x * vicon_freq * 2 /10
    vicon_error_v = 0.2
    vicon_error_rpy = np.radians(2)
    vicon_error_rpy_rate = vicon_error_rpy * vicon_freq * 2

    # post_meas_noise = [
    #     vicon_error_x*0,
    #     vicon_error_x*0,
    #     vicon_error_x*0,
    #     vicon_error_v,
    #     vicon_error_v,
    #     vicon_error_v,
    #     vicon_error_rpy,
    #     vicon_error_rpy,
    #     vicon_error_rpy,
    #     vicon_error_rpy_rate*0,
    #     vicon_error_rpy_rate*0,
    #     vicon_error_rpy_rate*0,
    # ]

    trainSettings = {
        'num_drones':N_synth,
        'sfb':None,
        'sfb_freq_hz':f,
        'num_samples':T,
        'ctrl_noise':1.0,
        'proc_noise':proc_noise,
        # 'meas_noise_vicon':[vicon_error_x,
        #                     vicon_error_x,
        #                     vicon_error_x,
        #                     vicon_error_v,
        #                     vicon_error_v,
        #                     vicon_error_v,
        #                     vicon_error_rpy,
        #                     vicon_error_rpy,
        #                     vicon_error_rpy,
        #                     vicon_error_rpy_rate,
        #                     vicon_error_rpy_rate,
        #                     vicon_error_rpy_rate,],
        'traj':'hover',
        'part_pid_off':True,
        'traj_filename':None,
        'plot':True,
        'cut_traj':True,
        'gui':False,
        'record_reference':True,
        'init_rpys_spread':0.0,
        'init_xyzs_spread':0.5,
        'pid_type':'mellinger',
        'control_freq_hz': 500,
        'record_reference':True,
        'simulated_delay_ms':0
    }
    trainSettings['traj_filename'] = os.path.join(
        'data',
        name,
        suffix,
        'train' + '_' + trainSettings['traj'] + \
            'N' + str(N_synth) + \
            '_mass' + str(mass_range) + \
            '_pos' + str(pos_size) + \
            '_T' + str(T) + \
            '_start' + str(start) + \
            '_' + str(trainSettings['sfb_freq_hz']) + 'Hz' + \
            '_cn' + str(trainSettings['ctrl_noise']) + \
            '_pn' + str(trainSettings['proc_noise']) + \
            '_xyzs' + str(trainSettings['init_xyzs_spread']) + \
            '_rpys' + str(trainSettings['init_rpys_spread']) + \
            '_seed' + str(seed) + \
            '_delay' + str(trainSettings['simulated_delay_ms'])
    )
    testSettings = {
        'num_drones':N_test,
        'sfb':'rddc',
        'sfb_freq_hz':f,
        'num_samples':T_test,
        'ctrl_noise':0.0,
        'proc_noise':proc_noise,
        'traj':'line30',
        'part_pid_off':True,
        'traj_filename':None,
        'plot':False,
        'cut_traj':False,
        'wrap_wp':False,
        'wind_force':0.3*np.array([-np.sin(np.radians(30)), np.cos(np.radians(30)), 0]),
        'gui':True,
        'record_reference':True,
        'pid_type':'mellinger',
        'control_freq_hz':500,
        'simulated_delay_ms':0
    }

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