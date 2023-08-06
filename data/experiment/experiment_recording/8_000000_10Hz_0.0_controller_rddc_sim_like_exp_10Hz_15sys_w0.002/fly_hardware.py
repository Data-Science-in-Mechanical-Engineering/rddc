#!/usr/bin/env python3

import os
import sys
import numpy as np
from datetime import datetime
import threading
import shutil
# from crazyswarm.ros_ws.src.crazyswarm.scripts.pycrazyswarm import *
# from simulation.controller import SimpleStateFeedbackController
# from rddc.run.settings.experiment import get_settings
# from rddc.tools.trajectory import get_trajectory_gerono
# from rddc.experiment.logging import bufferStateLogger

# import sys
# sys.path.insert(0, "../")
# sys.path.insert(0, ".")
from pycrazyswarm import *
from dmitrii_drones.controller import SimpleStateFeedbackController
# from dmitrii_drones.settings import get_settings
from dmitrii_drones.trajectory import *
from dmitrii_drones.cf_loggers import bufferStateLogger, viconStateLogger
import importlib
import argparse

basepath = '.'
statenames = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'roll', 'pitch', 'yaw', 'roll rate', 'pitch rate', 'yaw rate']

def threaded_log_run(logger: viconStateLogger):
    logger.log_run()

def init_controller(settings):
    state_idx = settings['state_idx']
    input_idx = settings['input_idx']
    K = np.zeros((12,12))
    path = os.path.join(basepath, 'dmitrii_drones', 'controllers')
    controller = np.load(os.path.join(path, settings['sfb_name']+'.npy'), allow_pickle=True).item()
    for row, r_idx in enumerate(input_idx):
        for col, c_idx in enumerate(state_idx):
            K[r_idx, c_idx] = controller['controller'][row, col]
    # x = 1
    # v = 1
    # a = 1
    # r = 1
    # K = K * np.array([
    #     [x]*12,[x]*12,[x]*12,
    #     [v]*12,[v]*12,[v]*12,
    #     [a]*12,[a]*12,[a]*12,
    #     [r]*12,[r]*12,[r]*12,
    # ]).T
    # K = np.array([
    #     [0]*12, [0]*12, [0]*12,
    #     [0]*12, [0]*12, [0]*12,
    #     [0]*12, [0]*12, [0]*12,
    #     [-0.45, 3.979, 0.0, 0.118, 4.873, 0.0, -10.168, -0.145, 0.0, 0.0, 0.0, 0.0],
    #     [-4.00, 0.1,   0.0, -4.3, -0.4,   0.0,  -0.06, -10.76,  0.0, 0.0, 0.0, 0.0],
    #     [0]*12,
    # ])
    
    print(f"Using controller: \n{np.array_str(K, precision=2)}")
    return SimpleStateFeedbackController(K=K)

def check_unsafe_states(settings, state_vector):
    """
    returns a list of indices of the states that don't fit into safe_state_lims in the settings
    """
    # print(f"Analyzing state for safety: {format_state(state_vector)}")
    safe_states = list()
    for idx in range(12):
        state = state_vector[idx]
        # print(f"state: {state}")
        # print(f"lims: {settings['safe_state_lims'][idx]}\n")
        # print(state > settings['safe_state_lims'][idx][0])
        # print(state < settings['safe_state_lims'][idx][1])
        # print(state > settings['safe_state_lims'][idx][0] and state < settings['safe_state_lims'][idx][1])
        safe_states.append(
            (state > settings['safe_state_lims'][idx][0]) and
            (state < settings['safe_state_lims'][idx][1])
        )
    return [idx for idx, safe in enumerate(safe_states) if not safe]

def manual_land(cf, time_helper, targetHeight, duration, N_steps=10):
    """
    only works for one crazyflie
    """
    latency = duration/N_steps
    # for cf in allcfs.crazyflies:
    start_pos = cf.position()
    for i in range(N_steps):
        z = start_pos[2] + (targetHeight-start_pos[2]) * (i+1)/N_steps
        # cf.cmdFullState(
        #     pos     = [0., 0., z],
        #     vel     = [0., 0., 0.,],
        #     acc     = [0., 0., 0.],
        #     yaw     = 0.,
        #     omega   = [0., 0., 0.,],
        # )
        cf.cmdPosition([start_pos[0],start_pos[1],z], 0,)
        time_helper.sleep(latency)


def manual_goto(cf, time_helper, target, vel=0.3, N_steps=3):
    """
    only works for one crazyflie
    """
    # for cf in allcfs.crazyflies:
    start = cf.position()
    duration = np.linalg.norm(target - start)/vel
    latency = duration/N_steps
    print(f"going from {start} to {target}")
    for i in range(N_steps):
        pos = start + (target-start) * (i+1)/N_steps
        cf.cmdPosition(pos, 0,)
        print(".")
        time_helper.sleep(latency)

def manual_stabilize(cf, time_helper, logger, last_pos, duration = 1):
    """
    only works for one crazyflie
    """
    # for cf in allcfs.crazyflies:
    rate = 20
    N = int(duration*rate)
    state = logger.retrieve_state()
    dt = logger.retrieve_timestep()
    # pos = state[0:3]
    pos = last_pos
    vel = state[3:6]
    print(f"recovering from {last_pos}, {pos} @ {vel}")
    vel[2] = 0
    for i in range(N):
        pos = [0., 0., 1.,]
        vel = vel*(1-(i+1)/N)
        cf.cmdFullState(
            pos     = pos,
            vel     = vel,
            acc     = [0., 0., 0.],
            yaw     = 0.,
            omega   = [0., 0., 0.,],
        )
        print(f". : going to: {pos} @ {vel}")
        time_helper.sleepForRate(rate)

def interp_trajectory(trajectory, period, time_target):
    """
    linear interpolation of the trajectory between two neighboring nodes
    """
    print(f"time_target: {time_target}")
    time_target = min(time_target, period*0.999999)
    print(f"time_target: {time_target}")
    s = np.mod(time_target, period) / period
    print(f"s: {s}")
    id_0 = np.mod(int(np.floor(s * trajectory.shape[0])), trajectory.shape[0])
    print(f"id0: {id_0}")
    id_1 = np.mod(int(np.ceil(s * trajectory.shape[0])), trajectory.shape[0])
    print(f"id1: {id_1}")
    if id_0<id_1:
        dist = s*trajectory.shape[0] - id_0
    else:
        dist = 0.
    print(f"dist: {dist}")
    res = list()
    for dim in range(trajectory.shape[1]):
        res_0 = trajectory[id_0, dim]
        res_1 = trajectory[id_1, dim]
        res.append(res_0 + (res_1 - res_0)*dist)
    return res

def format_state(state_vector):
    """
    Input state vector is in m, m/s, rad and rad/s
    """
    strings = ['']*12
    strings[0]  = "{:>7.3f}".format(state_vector[0])
    strings[1]  = "{:>7.3f}".format(state_vector[1])
    strings[2]  = "{:>7.3f}".format(state_vector[2])
    strings[3]  = "{:>7.4f}".format(state_vector[3])
    strings[4]  = "{:>7.4f}".format(state_vector[4])
    strings[5]  = "{:>7.4f}".format(state_vector[5])
    strings[6]  = "{:>7.0f}".format(np.degrees(state_vector[6]))
    strings[7]  = "{:>7.0f}".format(np.degrees(state_vector[7]))
    strings[8]  = "{:>7.0f}".format(np.degrees(state_vector[8]))
    strings[9]  = "{:>7.0f}".format(np.degrees(state_vector[9]))
    strings[10] = "{:>7.0f}".format(np.degrees(state_vector[10]))
    strings[11] = "{:>7.0f}".format(np.degrees(state_vector[11]))
    return strings

def print_states(ist, soll, error, ctrl):
    ist_str = format_state(ist)
    soll_str = format_state(soll)
    error_str = format_state(error)
    ctrl_str = format_state(ctrl)
    units = ['m', 'm', 'm', 'm/s', 'm/s', 'm/s', '°', '°', '°', '°/s', '°/s', '°/s']
    print('state'.ljust(16, ' ') + ' | ' + 'ist'.rjust(7, ' ') + ' | ' + 'soll'.rjust(7, ' ') + ' | ' + 'error'.rjust(7, ' ') + ' | ' + 'ctrl'.rjust(7, ' '))
    print(''*57)
    # for idx in [0,1,2,3,4,5,6,7,8,9,10,11]:
    for idx in [0,3,6]:
        print((statenames[idx]+', ').ljust(13, ' '),units[idx].ljust(3, ' ') + ' | ' + ist_str[idx] + ' | ' + soll_str[idx] + ' | ' + error_str[idx] + ' | ' + ctrl_str[idx])

def enable_pid(settings, cf):
    try:
        cf.setParam("ctrlMel/kp_xy", settings['kp_xy'])
        cf.setParam("ctrlMel/ki_xy", settings['ki_xy'])
        cf.setParam("ctrlMel/kd_xy", settings['kd_xy'])
        cf.setParam("ctrlMel/kp_z", settings['kp_z'])
        cf.setParam("ctrlMel/ki_z", settings['ki_z'])
        cf.setParam("ctrlMel/kd_z", settings['kd_z'])

        cf.setParam("ctrlMel/kR_xy", settings['kR_xy'])
        cf.setParam("ctrlMel/kw_xy", settings['kw_xy'])
        cf.setParam("ctrlMel/kd_omega_rp", settings['kd_omega_rp']) 
        cf.setParam("ctrlMel/ki_m_xy", settings['ki_m_xy'])
        cf.setParam("ctrlMel/kR_z", settings['kR_z'])
        cf.setParam("ctrlMel/kw_z", settings['kw_z'])
        cf.setParam("ctrlMel/ki_m_z", settings['ki_m_z'])
    except Exception:
        return False
    return True

def enable_slow_pid(settings, cf):
    try:
        cf.setParam("ctrlMel/kp_xy", settings['kp_xy']*0.3)
        cf.setParam("ctrlMel/ki_xy", settings['ki_xy']*0.3)
        cf.setParam("ctrlMel/kd_xy", settings['kd_xy']*0.3)
        cf.setParam("ctrlMel/kp_z", settings['kp_z']*0.8)
        cf.setParam("ctrlMel/ki_z", settings['ki_z']*0.2)
        cf.setParam("ctrlMel/kd_z", settings['kd_z']*0.8)

        cf.setParam("ctrlMel/kR_xy", settings['kR_xy'])
        cf.setParam("ctrlMel/kw_xy", settings['kw_xy'])
        cf.setParam("ctrlMel/kd_omega_rp", settings['kd_omega_rp']) 
        cf.setParam("ctrlMel/ki_m_xy", settings['ki_m_xy'])
        cf.setParam("ctrlMel/kR_z", settings['kR_z'])
        cf.setParam("ctrlMel/kw_z", settings['kw_z'])
        cf.setParam("ctrlMel/ki_m_z", settings['ki_m_z'])
    except Exception:
        return False
    return True

def enable_recovery_pid(settings, cf):
    try:
        cf.setParam("ctrlMel/kp_xy", settings['kp_xy']*0.0)
        cf.setParam("ctrlMel/ki_xy", settings['ki_xy']*0.0)
        cf.setParam("ctrlMel/kd_xy", settings['kd_xy']*0.0)
        cf.setParam("ctrlMel/kp_z", settings['kp_z']*1)
        cf.setParam("ctrlMel/ki_z", settings['ki_z']*1)
        cf.setParam("ctrlMel/kd_z", settings['kd_z']*1)

        cf.setParam("ctrlMel/kR_xy", settings['kR_xy']*0.7)
        cf.setParam("ctrlMel/kw_xy", settings['kw_xy']*0.7)
        cf.setParam("ctrlMel/kd_omega_rp", settings['kd_omega_rp']*0.7) 
        cf.setParam("ctrlMel/ki_m_xy", settings['ki_m_xy']*1)
        cf.setParam("ctrlMel/kR_z", settings['kR_z']*1)
        cf.setParam("ctrlMel/kw_z", settings['kw_z']*1)
        cf.setParam("ctrlMel/ki_m_z", settings['ki_m_z']*1)
    except Exception:
        return False
    return True

def partially_disable_pid(settings, cf):
    try:
        cf.setParam("ctrlMel/kp_xy", 0.0)
        cf.setParam("ctrlMel/ki_xy", 0.0)
        cf.setParam("ctrlMel/kd_xy", 0.0)
        cf.setParam("ctrlMel/kp_z", settings['kp_z'])
        cf.setParam("ctrlMel/ki_z", settings['ki_z'])
        cf.setParam("ctrlMel/kd_z", settings['kd_z'])

        cf.setParam("ctrlMel/kR_xy", 0.0)
        cf.setParam("ctrlMel/kw_xy", settings['kw_xy'])
        cf.setParam("ctrlMel/kd_omega_rp", settings['kd_omega_rp']) 
        cf.setParam("ctrlMel/ki_m_xy",settings['ki_m_xy'])
        cf.setParam("ctrlMel/kR_z", settings['kR_z'])
        cf.setParam("ctrlMel/kw_z", settings['kw_z'])
        cf.setParam("ctrlMel/ki_m_z", settings['ki_m_z'])
    except Exception:
        return False
    return True

def run(args):
    
    parser = argparse.ArgumentParser(description="Module to run hardware experiments on craziflies")
    parser.add_argument('--train',  action='store_true',    default=False)
    parser.add_argument('--test',   action='store_true',    default=False)
    ARGS = parser.parse_args(args)

    if not ARGS.train:
        assert ARGS.test
        test_or_train = 'test'
        settings_module = importlib.import_module('dmitrii_drones.settings_test')
    else:
        assert ARGS.train
        test_or_train = 'train'
        settings_module = importlib.import_module('dmitrii_drones.settings_train')

    #### Read the settings and create the path ###########
    settings = settings_module.get_settings()
    
    #### Init swarm ######################################
    swarm           = Crazyswarm()
    time_helper     = swarm.timeHelper
    allcfs          = swarm.allcfs
    cf              = allcfs.crazyflies[0] #This script works for just one crazyflie
    TIMESCALE       = 1.0
    
    #### Set start parameters ############################
    # for cf in allcfs.crazyflies:
    enable_pid(settings, cf)
        # #### Sanity check ################
        # P_xy=cf.getParam("ctrlMel/kp_xy")
        # P_z=cf.getParam("ctrlMel/kp_z")
        # I_xy=cf.getParam("ctrlMel/ki_xy")
        # I_z=cf.getParam("ctrlMel/ki_z")

    #### Init controller #################################
    if 'sfb_name' in settings.keys():
        rddc            = init_controller(settings)
    rddc_rate       = settings['rddc_rate'] #Hz
    main_rate       = settings['main_rate'] #Hz
    main_per_rddc   = main_rate // rddc_rate
    assert main_rate % rddc_rate == 0
    if 'total_laps' in settings.keys() and 'T' in settings.keys():
        print("Only one of 'total_laps' and 'T' should be specified in settings")
        raise ValueError

    #### Saving paths #################################
    codepath = os.getcwd()
    # print(settings['weight_combination'])
    sfb_suffix = ('_'+settings['sfb_name']) if 'sfb_name' in settings.keys() else ''
    testcase_name = settings['trajectory'] + '_' + settings['weight_combination'] + '_' + str(rddc_rate) + 'Hz_' + str(settings['ctrl_noise']) + sfb_suffix
    savepath = os.path.join('/home', 'franka_panda', 'dmitrii_drones', testcase_name)
    if not(os.path.exists(savepath)):
        os.makedirs(savepath)
    if not(os.path.exists(os.path.join(savepath,'dmitrii_drones'))):
        os.makedirs(os.path.join(savepath,'dmitrii_drones'))
    trajectory_path = os.path.join(savepath, 'trajectory.npy')
    absolute_trajectory_path = os.path.join(savepath, 'absolute_trajectory.npy')
    ref_trajectory_path = os.path.join(savepath, 'reference_trajectory.npy')
    script_path_src = os.path.join(codepath,os.path.basename(__file__))
    script_path_dst = os.path.join(savepath,os.path.basename(__file__))
    settings_path_src = os.path.join(codepath,'dmitrii_drones','settings_'+test_or_train+'.py')
    settings_path_dst = os.path.join(savepath,'dmitrii_drones','settings_'+test_or_train+'.py')
    loggers_code_path_src = os.path.join(codepath,'dmitrii_drones','cf_loggers.py')
    loggers_code_path_dst = os.path.join(savepath,'dmitrii_drones','cf_loggers.py')
    trajectory_code_path_src = os.path.join(codepath,'dmitrii_drones','trajectory.py')
    trajectory_code_path_dst = os.path.join(savepath,'dmitrii_drones','trajectory.py')
    controller_code_path_src = os.path.join(codepath,'dmitrii_drones','controller.py')
    controller_code_path_dst = os.path.join(savepath,'dmitrii_drones','controller.py')
    absolute_trajectory_default_duration = 10 #s
    absolute_trajectory_default_length = absolute_trajectory_default_duration * main_rate
    absolute_trajectory = {
        'time':np.zeros(absolute_trajectory_default_length),
        'state':np.zeros((12,absolute_trajectory_default_length)),
        'rate': main_rate,
    }
    # print(f"codepath: {codepath}")
    # print(os.path.exists(codepath))
    # print(f"savepath: {savepath}")
    # print(os.path.exists(savepath))
    # print(os.path.exists(os.path.join(savepath,'dmitrii_drones')))
    # print()
    # print(script_path_src)
    # print(script_path_dst)
    # print(settings_path_src)
    # print(settings_path_dst)
    # print()

    # raise NotImplementedError
    #### Init soll trajectory ############################
    radius = 1.0
    height = settings['trajectory_height']
    period = settings['trajectory_period'] # time (s) for one lap
    trajectory_resolution   = settings['trajectory_resolution']
    if settings['trajectory'] in ['hover']:
        pos_traj, vel_traj = get_trajectory_hover(height, trajectory_resolution)
    elif settings['trajectory'] in ['8']:
        pos_traj, vel_traj = get_trajectory_gerono(height, radius, trajectory_resolution, period)
    elif settings['trajectory'] in ['line']:
        pos_traj, vel_traj = get_trajectory_line(start=np.array([-1.,-1.,height]), finish=np.array([1.,1.,height]), num_points=trajectory_resolution, duration=period)
    else:
        print("Wrong trajectory name")
        raise ValueError

    #### Init state logging ##############################
    # logger = bufferStateLogger(buffer_size=2)
    vicon_frequency = 300 #Hz
    vicon_buffer_size = 5
    filtering_latency = vicon_buffer_size/vicon_frequency/2
    print(f"Vicon system is assumed to publish @ {vicon_frequency}Hz")
    print(f"\t We're keeping {vicon_buffer_size} samples in the buffer")
    print(f"\t We'll be averaging over the whole buffer, \n\t\tso it will result in {filtering_latency*1000:3.0f} ms filtering latency")
    logger = viconStateLogger(vicon_buffer_size=vicon_buffer_size)
    threading.Thread(target=threaded_log_run, args=(logger,)).start()

    time_helper.sleep(vicon_buffer_size/vicon_frequency*2)    # make sure to collect enough data in the buffer
    pos_cf = cf.position()
    pos_vicon = logger.retrieve_state()[0:3]
    logger.vicon2cf_translation = pos_cf - pos_vicon
    print(f"Calculated translational shift between vicon's and cf's coordinate systems:\n\t {np.array_str(logger.vicon2cf_translation, precision=4)}")
 
    #### Flight start ####################################
    print("taking off\n")
    time_helper.sleep(1.0)
    allcfs.takeoff(targetHeight=height, duration=2.5)
    time_helper.sleep(3.0)
    print(f"\tGoing to trajectory start point:")
    manual_goto(cf, time_helper=time_helper, target=pos_traj[0], vel=0.5, N_steps=40)

    print("pause\n")
    # TODO: Anto: why? Pause to learn?
    for i in range(25):
        # for cf in allcfs.crazyflies:
        cf.cmdPosition(pos_traj[0], 0,)
        time_helper.sleep(.1)

    ctrl_trajectory = {'time':[],'U0':[], 'X0':[], 'X1':[]}
    lap_counter = 0
    #### Lemniscate trajectory ###################
    # for rounds in range(total_laps):
    keep_going = True
    main_counter = 0
    while keep_going:

        communication_ok = enable_pid(settings, cf)
        if not communication_ok:
            break
        for i in range(20):
            # for cf in allcfs.crazyflies:
            cf.cmdPosition(pos_traj[0], 0,)
            time_helper.sleep(.1)

        if settings['disable_pid']:
            partially_disable_pid(settings, cf)
        start_time = time_helper.time()
        start_time_log = logger.retrieve_time()
        lap_time = 0.0
        ctrl_counter = 0
        last_pos_cf = np.zeros(3)
        # for cf in allcfs.crazyflies:
        cf.cmdPosition(pos_traj[0], 0,)
        print("Starting a new lap\n")
        lap_successful = True
        while lap_time < period:
            # print(f"cf.position: {cf.position()}")
            lap_time = time_helper.time() - start_time
            lap_time_log = logger.retrieve_time('oldest') - start_time_log
            delay = lap_time - lap_time_log
            ist = logger.retrieve_state()
            unsafe_states = check_unsafe_states(settings, ist)
            if len(unsafe_states)>0:
                # for cf in allcfs.crazyflies:
                last_pos_cf = cf.position()
                last_pos_vicon = ist[0:3]
                print(f"last_pos_cf:    {last_pos_cf}")
                print(f"last_pos_vicon: {last_pos_vicon}")
                lap_successful = False
                print(f"\t\t\tAbout to crash!")
                break
            if main_counter%main_per_rddc == 0:
                # for cf in allcfs.crazyflies:
                soll = np.zeros(12)
                soll[0:3] = interp_trajectory(pos_traj, period, lap_time)
                soll[3:6] = interp_trajectory(vel_traj*min(lap_time/5,1), period, lap_time)
                error = ist - soll
                if 'sfb_name' in settings.keys():
                    ctrl = rddc.computeControl(error[0:3], error[3:6], error[6:9], error[9:12])
                else:
                    ctrl = np.zeros(12)
                ctrl[settings['input_idx']] = ctrl[settings['input_idx']] + settings['ctrl_noise'] * (2.0*np.random.rand(len(settings['input_idx']))-1.0)
                # ctrl = ctrl/3
                cf.cmdFullState(    #giving nonzero pos and vel is okay, since PID is partially disabled
                    pos     = soll[0:3] + ctrl[0:3],            # m
                    vel     = soll[3:6] + ctrl[3:6],            # m/s
                    acc     = [0.0, 0.0, 0.0],                  # m/s^2
                    yaw     = soll[8] + ctrl[8],                # rad
                    omega   = soll[9:12] + ctrl[9:12],          # rad/s
                )
                print(f"\nControl cycle start at: {time_helper.time() - start_time}")
                print(f"\tlatency: {delay*1000:3.1f}ms")
                print_states(ist, soll, error, ctrl)
                # print(f"\tist: {np.array_str(ist, precision=2)}")
                # print(f"\tsoll: {np.array_str(soll, precision=2)}")
                # print(f"\terror: {np.array_str(error, precision=2)}")
                # print(f"\tctrl: {np.array_str(ctrl, precision=2)}")
                if ctrl_counter>0:
                    ctrl_trajectory['X1'].append(error)
                ctrl_trajectory['X0'].append(error)
                ctrl_trajectory['U0'].append(ctrl)
                ctrl_trajectory['time'].append(lap_time)
                #TODO: Hmmm, the trajectory is not perfectly equidistant. Is that a problem?
                ctrl_counter = ctrl_counter + 1
            if main_counter>(absolute_trajectory['time'].shape[0]-1):
                absolute_trajectory['time'] = np.hstack([absolute_trajectory['time'], np.zeros(absolute_trajectory_default_length)])
                absolute_trajectory['state'] = np.hstack([absolute_trajectory['state'], np.zeros((12,absolute_trajectory_default_length))])
            absolute_trajectory['time'][main_counter] = lap_time
            absolute_trajectory['state'][:, main_counter] = ist
            time_helper.sleepForRate(main_rate)
            main_counter = main_counter + 1
        ctrl_trajectory['X1'].append(None)
        lap_counter = lap_counter + 1
        if not lap_successful:
            communication_ok = True
            print(f"\tUnsafe states: {[statenames[state_id] for state_id in unsafe_states]}")
            # for cf in allcfs.crazyflies:
            communication_ok = enable_recovery_pid(settings, cf)
            if not communication_ok:
                break
            manual_stabilize(cf, time_helper=time_helper, logger=logger, last_pos=last_pos_cf, duration=0.4)
        # for cf in allcfs.crazyflies:
        communication_ok = enable_slow_pid(settings, cf)
        if not communication_ok:
            break
        print(f"\tGoing to origin:")
        manual_goto(cf, time_helper=time_helper, target=pos_traj[0], vel=0.5, N_steps=40)

        # for cf in allcfs.crazyflies:
        communication_ok = enable_slow_pid(settings, cf)
        if not communication_ok:
            break

        # for i in range(1):
        #     for cf in allcfs.crazyflies:
        #         cf.cmdPosition(pos_traj[0], 0,)
        if 'total_laps' in settings.keys():
            keep_going = lap_counter < settings['total_laps']
        else:
            keep_going = len(ctrl_trajectory['X0']) < settings['T']+lap_counter


        #### Lap report ###########################
        print(f"Lap {lap_counter} | finished: {lap_successful} | #data points: {len(ctrl_trajectory['X0'])}")

    #### Return to start ##################################
    print(f"\tGoing to origin:")
    manual_goto(cf, time_helper=time_helper, target=[0,0,height], vel=0.5, N_steps=40)

    #### Land #############################################
    if communication_ok:
        manual_land(cf, time_helper=time_helper, targetHeight=0.05, duration=2.0)
    # allcfs.land(targetHeight=0.05, duration=2.0) #it doesn't work once cmdFullState has been issued
    del(swarm)

    #### Save Everything ##################################
    print("Saving the data")
    np.save(trajectory_path, ctrl_trajectory, allow_pickle=True)
    np.save(absolute_trajectory_path, absolute_trajectory, allow_pickle=True)
    np.save(ref_trajectory_path, pos_traj, allow_pickle=True)
    shutil.copy(script_path_src, script_path_dst)
    shutil.copy(settings_path_src, settings_path_dst)
    shutil.copy(trajectory_code_path_src, trajectory_code_path_dst)
    shutil.copy(controller_code_path_src, controller_code_path_dst)
    print("Done")


    for i in range(len(ctrl_trajectory['X0'])):
        print(f"U0: {ctrl_trajectory['U0'][i][settings['input_idx']]}, {'X' if ctrl_trajectory['X1'][i] is None else ' '}")

    sys.exit('Quiting the program')

if __name__=='__main__':
    run(sys.argv[1:])