import numpy as np

def get_settings():
    name = 'experiment'
    suffix = 'test'
    # seed = 550
    # train_trajectory_path = 'traj_train_'
    # eps = 1e-8
    # controllability_tol = 1e-3
    state_idx = [0,1,3,4,6,7]
    input_idx = [9,10]
    # # We're prescribing "inputs" to our system while observing "states"
    # # We're trying to control "states" through "inputs"
    # n = len(state_idx)                  # number of state variables x_k
    # m = len(input_idx)                   # number of control variables u_k

    # A = np.zeros((n,n))
    # B = np.zeros((n,m))
    # # algorithm = 'robust_hinf_scenario_slemma'
    # check_slater = False
    # check_willems = False
    # algorithm = 'robust_stabilization_scenario_slemma'
    # output_verbosity = 0

    # # Extra weight distribution
    # mass_range = [0, 0.006]
    # pos_size = 0.01
    # # displacement_planar = 0.01
    # # displacement_vert = 0.0

    # N_synth = 15
    # N_test = 100
    # start = 0                              # time step to start sampling the trajectory with
    # T_test = 60                            # number of samples per trajectory for performance evaluation

    # # noise
    # m_w = n                 # number of disturbance variables w_k
    # B_w = np.eye(n, m_w)
    # assumedBound = 0.002     # noise bound assumed for robust controller synthesis

    # # performance metric
    # Q = np.eye(n, n)
    # S = np.zeros((n, m))
    # R = np.eye(m, m)
    # C = np.array([[1,1,1,1,1,1]])
    # D = np.array([[1, 1]])

    kp_xy       = 0.4
    ki_xy       = 0.05
    kd_xy       = 0.2
    kp_z        = 1.25
    ki_z        = 0.05
    kd_z        = 0.4

    kR_xy       = 70000
    kw_xy       = 20000
    kd_omega_rp = 200
    ki_m_xy     = 0.0
    kR_z        = 60000
    kw_z        = 12000
    ki_m_z      = 500

    disable_pid = True
    # sfb_name = 'controller_exp_20Hz_1sys' # comment out to not use it
    trajectory = 'hover'
    trajectory_height = 1.0
    trajectory_period = 5.0
    trajectory_resolution = 360
    weight_combination = '000000'
    ctrl_noise = 1.0
    rddc_rate   = 10
    main_rate   = 100
    T = 100                                  # number of samples per trajectory for controller synthesis
    # total_laps = 2
    safe_state_lims = [
        [-1.0, 1.0],                #x
        [-1.0, 1.0],                #y
        [0.8, 1.2],             #z
        [-0.7, 0.7],            #vx
        [-0.7, 0.7],            #vy
        [-0.2, 0.2],            #vz
        [-10, 10],              #roll
        [-10, 10],              #pitch
        [-5, 5],              #yaw
        [-30, 30],            #roll rate
        [-30, 30],            #pitch rate
        [-10, 10],            #yaw rate
    ]

    return locals()

if __name__=='__main__':
    pass