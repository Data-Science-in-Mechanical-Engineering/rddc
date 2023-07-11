#!/usr/bin/env python3

import os
import numpy as np
from datetime import datetime
import threading
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
from dmitrii_drones.settings import get_settings
from dmitrii_drones.trajectory import *
from dmitrii_drones.cf_loggers import bufferStateLogger, viconStateLogger

statenames = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'roll', 'pitch', 'yaw', 'roll rate', 'pitch rate', 'yaw rate']

def threaded_log_run(logger: viconStateLogger):
    logger.log_run()

def init_controller(settings):
    state_idx = settings['state_idx']
    input_idx = settings['input_idx']
    K = np.zeros((12,12))
    path = os.path.join('dmitrii_drones', 'controllers')
    controller = np.load(os.path.join(path, 'controller.npy'), allow_pickle=True).item()
    for row, r_idx in enumerate(input_idx):
        for col, c_idx in enumerate(state_idx):
            K[r_idx, c_idx] = controller['controller'][row, col]
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

def manual_land(allcfs, time_helper, targetHeight, duration, N_steps=10):
    """
    only works for one crazyflie
    """
    latency = duration/N_steps
    for cf in allcfs.crazyflies:
        startHeight = cf.position()[2]
        for i in range(N_steps):
            z = startHeight + (targetHeight-startHeight) * (i+1)/N_steps
            # cf.cmdFullState(
            #     pos     = [0., 0., z],
            #     vel     = [0., 0., 0.,],
            #     acc     = [0., 0., 0.],
            #     yaw     = 0.,
            #     omega   = [0., 0., 0.,],
            # )
            cf.cmdPosition([0,0,z], 0,)
            time_helper.sleep(latency)


def manual_goto(allcfs, time_helper, target, vel=0.3, N_steps=3):
    """
    only works for one crazyflie
    """
    for cf in allcfs.crazyflies:
        start = cf.position()
        duration = np.linalg.norm(target - start)/vel
        latency = duration/N_steps
        print(f"going from {start} to {target}")
        for i in range(N_steps):
            pos = start + (target-start) * (i+1)/N_steps
            cf.cmdPosition(pos, 0,)
            print(".")
            time_helper.sleep(latency)

def manual_stabilize(allcfs, time_helper, logger, last_pos, duration = 1):
    """
    only works for one crazyflie
    """
    for cf in allcfs.crazyflies:
        rate = 10
        N = int(duration*rate)
        print(f"recovering from {last_pos}")
        for i in range(N):
            state = logger.retrieve_state()
            dt = logger.retrieve_timestep()
            vel = state[3:6]*(1-(i+1)/N)
            vel[2] = 0
            pos = state[0:3] + vel*dt
            cf.cmdFullState(
                pos     = pos,
                vel     = vel,
                acc     = [0., 0., 0.],
                yaw     = 0.,
                omega   = [0., 0., 0.,],
            )
            print(".")
            time_helper.sleepForRate(rate)

def interp_trajectory(trajectory, period, time_target):
    """
    linear interpolation of the trajectory between two neighboring nodes
    """
    s = np.mod(time_target, period) / period
    id_0 = np.mod(int(np.floor(s * trajectory.shape[0])), trajectory.shape[0])
    id_1 = np.mod(int(np.ceil(s * trajectory.shape[0])), trajectory.shape[0])
    dist = s*trajectory.shape[0] - id_0
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
    strings[3]  = "{:>7.2f}".format(state_vector[3])
    strings[4]  = "{:>7.2f}".format(state_vector[4])
    strings[5]  = "{:>7.2f}".format(state_vector[5])
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
    for idx in range(12):
        print((statenames[idx]+', ').ljust(13, ' '),units[idx].ljust(3, ' ') + ' | ' + ist_str[idx] + ' | ' + soll_str[idx] + ' | ' + error_str[idx] + ' | ' + ctrl_str[idx])

def enable_pid(settings, cf):
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

def partially_disable_pid(settings, cf):
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

if __name__ == "__main__":

    #### Read the settings and create the path ###########
    settings = get_settings()
    
    datapath = os.path.join('dmitrii_drones', 'data', settings['name'], settings['suffix'])
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    
    #### Init swarm ######################################
    swarm           = Crazyswarm()
    time_helper     = swarm.timeHelper
    allcfs          = swarm.allcfs
    TIMESCALE       = 1.0
    
    #### Set start parameters ############################
    for cf in allcfs.crazyflies:
        enable_pid(settings, cf)
        # #### Sanity check ################
        # P_xy=cf.getParam("ctrlMel/kp_xy")
        # P_z=cf.getParam("ctrlMel/kp_z")
        # I_xy=cf.getParam("ctrlMel/ki_xy")
        # I_z=cf.getParam("ctrlMel/ki_z")

    #### Init controller #################################
    rddc            = init_controller(settings)
    rddc_rate       = settings['rddc_rate'] #Hz
    main_rate       = settings['main_rate'] #Hz
    main_per_rddc   = main_rate // rddc_rate
    assert main_rate % rddc_rate == 0
    #General
    # learn=True
    # learn_counter=0
    # max_learn_rounds=100
    total_rounds = settings['total_rounds']
    # continue_l=False
    # chg_at=[0,8]

    #### Init save files #################################
    #TODO: create a file naming scheme like Anto
    # if learn and (not trigger):
    #     dir="/home/franka_panda/Holz_drones/SOmel_{}_beta_{}_{}chges_{}_{}_".format(experiment_type, beta_type, len(chg_at)-1, max_learn_rounds, total_rounds)+datetime.now().strftime("%m.%d_%H.%M")
    # elif(trigger and learn):
    #     dir="/home/franka_panda/Holz_drones/ETLmel_{}_beta_{}_{}chges_{}_{}_".format(experiment_type, beta_type, len(chg_at)-1, max_learn_rounds, total_rounds)+datetime.now().strftime("%m.%d_%H.%M")
    # else:
        # dir="/home/franka_panda/Holz_drones/Baseline_{}_{}chges_{}_{}_".format(experiment_type, len(chg_at)-1, max_learn_rounds, total_rounds)+datetime.now().strftime("%m.%d_%H.%M")
    savepath = "/home/franka_panda/dmitrii_drones/noname_"+datetime.now().strftime("%m.%d_%H.%M")
    trajectory_path = savepath+"/trajectory.npy"
    if not(os.path.exists(savepath)):
        os.makedirs(savepath)
    ctrl_trajectory = {'time':[],'U0':[], 'X0':[], 'X1':[]}

    #### Init soll trajectory ############################
    radius = 1.0
    height = 1.0
    period = 10.0 # time (s) for one lap
    trajectory_resolution   = 360
    pos_traj, vel_traj = get_trajectory_gerono(height, radius, trajectory_resolution, period)
    # pos_traj, vel_traj = get_trajectory_hover(height, trajectory_resolution)

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

    # time_helper.sleep(vicon_buffer_size/vicon_frequency*2)    # make sure to collect enough data in the buffer
    # pos_cf = cf.position()
    # pos_vicon = logger.retrieve_state()[0:3]
    # logger.vicon2cf_translation = pos_cf - pos_vicon
    # print(f"Calculated translational shift between vicon's and cf's coordinate systems:\n\t {np.array_str(logger.vicon2cf_translation, precision=4)}")
 
    #### Flight start ####################################
    print("taking off\n")
    allcfs.takeoff(targetHeight=height, duration=2.0)
    time_helper.sleep(2.5)

    #TODO: Anto: why?
    # time_helper.sleep(2.5)
    # for i in range(trajectory_resolution):
    #        for cf in allcfs.crazyflies:
    #             cf.cmdPosition(pos_traj[i], 0,)           
    #        time_helper.sleep(0.017) #TODO: latency?

    print("pause\n")
    # TODO: Anto: why? Pause to learn?
    for i in range(25):
        for cf in allcfs.crazyflies:
            cf.cmdPosition([0,0,height], 0,)           
        time_helper.sleep(.1)

    #### Lemniscate trajectory ###################
    for rounds in range(total_rounds):

        for i in range(1):
            for cf in allcfs.crazyflies:
                cf.cmdPosition([0,0,height], 0,)           
            time_helper.sleep(.1)

        partially_disable_pid(settings, cf)
        start_time = time_helper.time()
        start_time_log = logger.retrieve_time()
        lap_time = 0.0
        ctrl_counter = 0
        main_counter = 0
        last_pos = np.zeros(3)
        for cf in allcfs.crazyflies:
            cf.cmdPosition([0,0,height], 0,)  
        print("Starting a new lap\n")
        lap_successful = True
        while lap_time < period:
            # print(f"cf.position: {cf.position()}")
            unsafe_states = check_unsafe_states(settings, logger.retrieve_state())
            if len(unsafe_states)>0:
                for cf in allcfs.crazyflies:
                    last_pos = cf.position()
                lap_successful = False
                print(f"\t\t\tAbout to crash!")
                break
            if main_counter==main_per_rddc:
                for cf in allcfs.crazyflies:
                    lap_time = time_helper.time() - start_time
                    lap_time_log = logger.retrieve_time('oldest') - start_time_log
                    delay = lap_time - lap_time_log
                    # print("[Script] Retrieving state from buffers:")
                    # print(f"\t pos: {logger.pos_buffer}")
                    # print(f"\t vel: {logger.vel_buffer}")
                    # print(f"\t angle: {logger.angle_buffer}")
                    ist = logger.retrieve_state()
                    # ist[7] = -ist[7] #cf -> normal, not need for vicon logger
                    # ist[10] = -ist[10] #cf -> normal, not need for vicon logger
                    # ist[6:12] = np.radians(ist[6:12]) # not need for vicon logger
                    soll = np.zeros(12)
                    soll[0:3] = interp_trajectory(pos_traj, period, lap_time)
                    soll[3:6] = interp_trajectory(vel_traj*min(lap_time/5,1), period, lap_time)
                    error = ist - soll
                    ctrl = np.zeros(12)
                    ctrl = rddc.computeControl(error[0:3], error[3:6], error[6:9], error[9:12])
                    # ctrl[settings['input_idx']] = ctrl[settings['input_idx']] + 0.2 * np.random.rand(len(settings['input_idx']))
                    # ctrl = ctrl/3
                    # ctrl[7] = -ctrl[7] #normal -> cf #unnecessary, since cmdFullState operates in normal coordinates
                    # ctrl[10] = -ctrl[10]
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
                    ctrl_trajectory['time'].append(time_helper.time())
                    #TODO: Hmmm, the trajectory is not perfectly equidistant. Is that a problem?
                    ctrl_counter = ctrl_counter + 1
                    main_counter = 0
            time_helper.sleepForRate(main_rate)
            main_counter = main_counter + 1
        ctrl_trajectory['X1'].append(None)
        enable_pid(settings, cf)
        if not lap_successful:
            print(f"\tUnsafe states: {unsafe_states}")
            manual_stabilize(allcfs=allcfs, time_helper=time_helper, logger=logger, last_pos=last_pos)
            print(f"\tGoing to origin:")
            manual_goto(allcfs=allcfs, time_helper=time_helper, target=[0,0,height])
        for i in range(1):
            for cf in allcfs.crazyflies:
                cf.cmdPosition([0,0,height], 0,)


        #### Calculate performance ###########################
        print(f"Lap {rounds+1}/{total_rounds} | finished: {lap_successful}")

        # TODO: Anto: why? Pause to learn?
        # TODO: Anto: Why command the same position 25 times? Don't they remember it?
        # -> Yep, they don't. Without any commands for 0.5s, cfs fall
        #### Return to start ##################################
        for i in range(25):
            for cf in allcfs.crazyflies:
                cf.cmdPosition([0,0,height], 0,)
            time_helper.sleep(.1)

        #### Save Trajectory ##################################
        np.save(trajectory_path, ctrl_trajectory, allow_pickle=True)

    manual_land(allcfs=allcfs, time_helper=time_helper, targetHeight=0.05, duration=2.0)
    # allcfs.land(targetHeight=0.05, duration=2.0) #Alex said it doesn't work once cmdFullState has been issued
