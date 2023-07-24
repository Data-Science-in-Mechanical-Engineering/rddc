"""
The main script for performing flight simulation
"""
import os
import time
import argparse
from datetime import datetime
import numpy as np
import pybullet as p
from rddc.tools import files

import sys
sys.path.append("gym-pybullet-drones")
from rddc.simulation import utils
# from rddc.run.settings.simulation import get_settings
from rddc.run.settings.simulation_like_experiment import get_settings
from rddc.tools.files import get_simulation_trajectory_path

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from rddc.simulation.VariationAviary import VariationAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from rddc.simulation.mellingerController import MellingerControl
from rddc.simulation.emulatorController import EmulatorControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from rddc.simulation.controller import SimpleStateFeedbackController

def starting_conditions_fulfilled(env, obs, lastObs, targets):
    """
    targets - 12 target states (pos, vel, rpy, rpy_rates)
    only checks it for the first drone
    """
    states = obs[str(0)]["state"]
    cur_pos     = states[0:3]
    cur_rpy     = states[7:10]
    cur_vel     = states[10:13]
    last_rpy    = lastObs[str(0)]["state"][7:10]
    cur_rpy_rate= (cur_rpy - last_rpy)*env.SIM_FREQ
    # if np.abs(cur_pos[2]-targets[2])>1e-3:
    #     return False
    if np.linalg.norm(cur_pos-targets[0:3])>1e-3:
        return False
    if np.linalg.norm(cur_vel-targets[3:6])>1e-3:
        return False
    if np.linalg.norm(cur_rpy-targets[6:9])>1e-3:
        return False
    if np.linalg.norm(cur_rpy_rate-targets[9:12])>1e-3:
        return False
    return True

def partially_disable_pid(ARGS, ctrl):
    """
    original PID coefficients, multiplied with a mask for disabling some of them
    """
    for i in range(ARGS.num_drones):
        ctrl[i].setPIDCoefficients(
            p_coeff_pos = np.array([.4, .4, 1.25])              * np.array([0, 0, 1]), # x, y, z
            i_coeff_pos = np.array([.05, .05, .05])             * np.array([0, 0, 1]), # x, y,  z integrated
            d_coeff_pos = np.array([.2, .2, .4])                * np.array([0, 0, 1]), # vx, vy, vz
            p_coeff_att = np.array([70000., 70000., 60000.])    * np.array([0, 0, 1]), # r, p, y
            i_coeff_att = np.array([.0, .0, 500.])              * np.array([1, 1, 1]), # r, p, y integrated
            d_coeff_att = np.array([20000., 20000., 12000.])    * np.array([1, 1, 1])  # rr, pr, yr
        )

def enable_pid(ARGS, ctrl):
    """
    original PID coefficients
    """
    for i in range(ARGS.num_drones):
        ctrl[i].setPIDCoefficients(
            p_coeff_pos = np.array([.4, .4, 1.25])              , # x, y, z
            i_coeff_pos = np.array([.05, .05, .05])             , # x, y,  z integrated
            d_coeff_pos = np.array([.2, .2, .4])                , # vx, vy, vz
            p_coeff_att = np.array([70000., 70000., 60000.])    , # r, p, y
            i_coeff_att = np.array([.0, .0, 500.])              , # r, p, y integrated
            d_coeff_att = np.array([20000., 20000., 12000.])      # rr, pr, yr
        )

def get_gerono_lemniscate_point_at_phase(R, H, phi):
    return [
        R   * np.cos(phi),
        R/2 * np.sin(2*phi),
        H
    ]

def get_gerono_lemniscate_vel_at_phase(R, H, T, phi):
    return [
        -2*np.pi/T * R * np.sin(phi),
         2*np.pi/T * R * np.cos(2*phi),
        0.
    ]

def get_circle_point_at_phase(R, H, phi):
    return [
        R * np.cos(phi),
        R * np.sin(phi),
        H
    ]

def get_circle_vel_at_phase(R, H, T, phi):
    return [
        -2*np.pi/T * R * np.sin(phi),
         2*np.pi/T * R * np.cos(phi),
        H
    ]

def get_init_xyzs(rnd, ARGS, R, H):
    spread = ARGS.init_xyzs_spread
    num_drones = ARGS.num_drones
    if ARGS.traj in ['8']:
        init_phi = np.pi/2
        INIT_XYZS = np.array([
            get_gerono_lemniscate_point_at_phase(R, H, init_phi)
            for _ in range(num_drones)
        ])
    elif ARGS.traj in ['circle']:
        init_phi = 0
        INIT_XYZS = np.array([
            get_circle_point_at_phase(R, H, init_phi)
            for _ in range(num_drones)
        ])
    elif ARGS.traj in ['hover', 'line', 'linex']:
        INIT_XYZS = np.array([
            [0.,0.,H] + (2*rnd.random(3)-1)*np.array([1,1,0])*spread #don't randomize z since we're not interested in it
            for _ in range(num_drones)
        ])
    else:
        print("Error: specified an unknown trajectory")
        raise NotImplementedError
    return INIT_XYZS

def get_init_rpys(rnd, ARGS):
    spread = ARGS.init_xyzs_spread
    num_drones = ARGS.num_drones
    INIT_RPYS = np.array([
        [0, 0, 0] + (2*rnd.random(3)-1)*np.array([1,1,0])*spread
        for _ in range(num_drones)
    ])
    return INIT_RPYS

def reset_sim(env, rnd, ARGS, R, H):
    env.INIT_XYZS = get_init_xyzs(rnd, ARGS, R, H)
    env.INIT_RPYS = get_init_rpys(rnd, ARGS)
    env.reset()
    for j in range(ARGS.num_drones):
        # have to overwrite since for some reason pybullet calculates non-zero yaw in
        # _updateAndStoreKinematicInformation() even if INIT_RPYS is 0
        env.pos[j] = env.INIT_XYZS[j, :]
        env.rpy[j] = env.INIT_RPYS[j, :]
        env.quat[j] = p.getQuaternionFromEuler(env.INIT_RPYS[j, :])
        env.vel[j] = np.array([0.,0.,0.])
        env.ang_v[j] = np.array([0.,0.,0.])

def parse_arguments(override_args=None, ignore_cli=False):
    """
    Argument parsing implemented so that the script can be called from
    both CLI or another python function with arguments passed in a dictionary
    """
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Trajectory script using PID or state-feedback control')
    parser.add_argument('--drone',              default="cf2x",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=1,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default="dyn",      type=Physics,       help='Physics updates (default: DYN)', metavar='', choices=Physics)
    parser.add_argument('--variation',          default=True,       type=str2bool,      help='Whether to use VariationAviary. For this, DYN physics are required (default: True)', metavar='')
    parser.add_argument('--gui',                default=True,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=False,      type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=False,      type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=2000,       type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=400,        type=int,           help='Control frequency in Hz (default: 100)', metavar='')
    parser.add_argument('--sfb_freq_hz',        default=10,         type=int,           help='State feedback frequency in Hz (default: 10)', metavar='')
    parser.add_argument('--duration_sec',       default=10,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--num_samples',        default=-1,         type=int,           help='Number of samples required, will recalculate and overwrite the duration', metavar='')
    parser.add_argument('--sfb',                default=None,       type=str,           help='Enable robust state feedback', metavar='', choices=[None, 'direct', 'indirect', 'prelim'])
    parser.add_argument('--ctrl_noise',         default=0.0,        type=float,         help='Reference trajectory input signal noise', metavar='')
    parser.add_argument('--proc_noise',         default=0.0,        type=float,         help='Process noise', metavar='')
    parser.add_argument('--traj',               default="hover",    type=str,           help='Trajectory to fly', metavar='', choices=['8', 'hover', 'circle', 'linex', 'line'])
    parser.add_argument('--calc_cost',          default=False,      type=str2bool,      help='Whether to calculate and save LQR cost', metavar='')
    parser.add_argument('--part_pid_off',       default=True,       type=str2bool,      help='Tell PID to not control some states', metavar='')
    parser.add_argument('--traj_filename',      default=None,       type=str,           help='Filename to save the resulting trajectory in', metavar='')
    parser.add_argument('--draw_trajectory',    default=False,      type=str2bool,      help='Draw the trajectory that the drones performs', metavar='')
    parser.add_argument('--init_xyzs_spread',   default=0.0,        type=float,         help='Spread in initial coordinates', metavar='')
    parser.add_argument('--init_rpys_spread',   default=0.0,        type=float,         help='Spread in initial orientations', metavar='')
    parser.add_argument('--cut_traj',           default=False,      type=str2bool,      help='Reset the simulation when the trajectory grows big', metavar='')
    parser.add_argument('--wind_on',            default=False,      type=str2bool,      help='Turn on wind force', metavar='')
    parser.add_argument('--wrap_wp',            default=True,       type=str2bool,      help='Wrap the trajectory. After reaching the last WP, start from the beginning', metavar='')
    parser.add_argument('--simulated_delay_ms', default=0.0,        type=float,         help='Simulation of sfb delay. Specify the latency in ms. Keep 0 to turn it off.', metavar='')
    parser.add_argument('--pid_type',           default='dsl',      type=str,           help='Type of the pid controller', metavar='', choices=['dsl', 'mellinger', 'emulator', 'simple'])

    if ignore_cli:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    if override_args is not None:
        for key, value in override_args.items():
            setattr(args, key, value)

    return args


def apply_process_noise(env, ARGS, rnd, processNoise):
    if ARGS.proc_noise>0:
        for j in range(ARGS.num_drones):
            proc_noise = np.multiply((2*rnd.random(12)-1), processNoise)
            env.pos[j] += proc_noise[0:3]
            env.vel[j] += proc_noise[3:6]
            env.rpy[j] += proc_noise[6:9]
            env.quat[j] = p.getQuaternionFromEuler(env.rpy[j])
            env.ang_v[j] += proc_noise[9:12]


def run(settings, override_args=None):
    
    ignore_cli = override_args is not None

    ARGS = parse_arguments(override_args, ignore_cli)

    rnd = np.random.default_rng(seed=settings['seed'])
    #### Initialize the simulation #############################
    H = 0.8
    R = .5

    INIT_XYZS = get_init_xyzs(rnd, ARGS, R, H)
    INIT_RPYS = get_init_rpys(rnd, ARGS)
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    PERIOD = 10
    NUM_WP = ARGS.control_freq_hz * PERIOD
    TARGET_POS = np.zeros((NUM_WP, 3, ARGS.num_drones))
    TARGET_VEL = np.zeros((NUM_WP, 3, ARGS.num_drones))
    TARGET_RPY = np.zeros((NUM_WP, 3, ARGS.num_drones))
    TARGET_RPY_RATE = np.zeros((NUM_WP, 3, ARGS.num_drones))

    if ARGS.traj in ['8']:
        for drone in range(ARGS.num_drones):
            for i in range(NUM_WP):
                t = i/NUM_WP * PERIOD
                phi = t/PERIOD * 2*np.pi + np.pi/2
                TARGET_POS[i, :, drone] =  get_gerono_lemniscate_point_at_phase(R, H, phi)
                TARGET_VEL[i, :, drone] =  get_gerono_lemniscate_vel_at_phase(R, H, PERIOD, phi)
    elif ARGS.traj in ['circle']:
        for drone in range(ARGS.num_drones):
            for i in range(NUM_WP):
                t = i/NUM_WP * PERIOD
                phi = t/PERIOD * 2*np.pi
                TARGET_POS[i, :, drone] =  get_circle_point_at_phase(R, H, phi)
                TARGET_VEL[i, :, drone] =  get_circle_vel_at_phase(R, H, PERIOD, phi)
    elif ARGS.traj in ['hover']:
        for drone in range(ARGS.num_drones):
            for i in range(NUM_WP):
                TARGET_POS[i, :, drone] =  INIT_XYZS[drone, 0], \
                                    INIT_XYZS[drone, 1], \
                                    INIT_XYZS[drone, 2]
    elif ARGS.traj in ['linex']:
        for drone in range(ARGS.num_drones):
            for i in range(NUM_WP):
                TARGET_POS[i, :, drone] =  R * np.cos((i / NUM_WP) * (2 * np.pi) + np.pi / 2) + INIT_XYZS[drone, 0], \
                                    INIT_XYZS[drone, 1], \
                                    INIT_XYZS[drone, 2]
    elif ARGS.traj in ['line']:
        dist = 2*np.sqrt(2)*R*np.sqrt(2)*2
        v_max = dist/PERIOD * 1.1
        t_acc = PERIOD - dist/v_max
        i_acc = int(t_acc * ARGS.control_freq_hz)
        for drone in range(ARGS.num_drones):
            for i in range(NUM_WP):
                TARGET_POS[i, :, drone] =  dist/np.sqrt(2) * (i / NUM_WP) + INIT_XYZS[drone, 0], \
                                    dist/np.sqrt(2) * (i / NUM_WP) + INIT_XYZS[drone, 1], \
                                    INIT_XYZS[drone, 2]
            if i < i_acc:
                TARGET_VEL[i, :, drone] =  v_max/np.sqrt(2) * (i/i_acc), \
                                    v_max/np.sqrt(2) * (i/i_acc), \
                                    0.0
            elif i > (NUM_WP-i_acc-1):
                TARGET_VEL[i, :, drone] =  v_max/np.sqrt(2) * ((NUM_WP - i-1)/i_acc), \
                                    v_max/np.sqrt(2) * ((NUM_WP - i-1)/i_acc), \
                                    0.0
            else:
                TARGET_VEL[i, :, drone] =  v_max/np.sqrt(2),\
                                    v_max/np.sqrt(2),\
                                    0.0
    else:
        print("Error: specified an unknown trajectory")
        raise NotImplementedError

    # wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(ARGS.num_drones)])
    wp_counters = np.array([0 for i in range(ARGS.num_drones)])

    TARGET_POS_COR = TARGET_POS.copy()
    TARGET_VEL_COR = TARGET_VEL.copy()
    TARGET_RPY_COR = TARGET_RPY.copy()
    TARGET_RPY_RATE_COR = TARGET_RPY_RATE.copy()

    controlNoise = np.array([1. if k in settings['input_idx'] else 0. for k in range(12)]) * ARGS.ctrl_noise
    processNoise = np.array([1. if k in settings['state_idx'] else 0. for k in range(12)]) * ARGS.proc_noise
    # if 9 in settings['state_idx'] or 10 in settings['state_idx'] or 11 in settings['state_idx']:
    #     print("Angular rates cannot be given process noise. Adjust 'state_idx' in settings")
    #     raise ValueError

    if ARGS.num_samples>0:
        ARGS.duration_sec = ARGS.num_samples / ARGS.sfb_freq_hz

    #### Create the environment with or without video capture ##
    extra_loads = list() if settings['use_urdf'] else settings['extra_loads']
    if ARGS.variation:
        env = VariationAviary(  drone_model=ARGS.drone,
                                num_drones=ARGS.num_drones,
                                initial_xyzs=INIT_XYZS,
                                initial_rpys=INIT_RPYS,
                                physics=ARGS.physics,
                                neighbourhood_radius=10,
                                freq=ARGS.simulation_freq_hz,
                                aggregate_phy_steps=AGGR_PHY_STEPS,
                                gui=ARGS.gui,
                                record=ARGS.record_video,
                                obstacles=ARGS.obstacles,
                                user_debug_gui=ARGS.user_debug_gui,
                                extra_loads=extra_loads,
                                )
    else:
        env = CtrlAviary(   drone_model=ARGS.drone,
                            num_drones=ARGS.num_drones,
                            initial_xyzs=INIT_XYZS,
                            initial_rpys=INIT_RPYS,
                            physics=ARGS.physics,
                            neighbourhood_radius=10,
                            freq=ARGS.simulation_freq_hz,
                            aggregate_phy_steps=AGGR_PHY_STEPS,
                            gui=ARGS.gui,
                            record=ARGS.record_video,
                            obstacles=ARGS.obstacles,
                            user_debug_gui=ARGS.user_debug_gui
                            )
    if settings['use_urdf']:
        os.replace(settings['urdfBackupPath'], settings['urdfOriginalPath']) #restore the urdf now, so that the controller doesn't notice it

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    datapath = os.path.join('data', settings['name'], settings['suffix'])
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    logger = Logger(logging_freq_hz=int(ARGS.simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=ARGS.num_drones,
                    output_folder=datapath
                    )

    #### Initialize the controllers ############################
    # if ARGS.ctrl in ["pid"]:
    # ctrl = [DSLPIDControl(drone_model=ARGS.drone) for i in range(ARGS.num_drones)]
    if ARGS.pid_type in ['mellinger']:
        ctrl = [MellingerControl(drone_model=ARGS.drone) for i in range(ARGS.num_drones)]
    if ARGS.pid_type in ['emulator']:
        if ARGS.variation:
            print("ERROR: can't have variationAviary and emulatorController at the same time")
            raise NotImplementedError
        ctrl = [EmulatorControl(
            drone_model=ARGS.drone,
            load_mass=extra_loads[i]['mass'],
            load_J=extra_loads[i]['J'],
            load_pos=extra_loads[i]['position']
        )
        for i in range(ARGS.num_drones)]
    elif ARGS.pid_type in ['simple']:
        ctrl = [SimplePIDControl(drone_model=ARGS.drone) for i in range(ARGS.num_drones)]
    elif ARGS.pid_type in ['dsl']:
        ctrl = [DSLPIDControl(drone_model=ARGS.drone) for i in range(ARGS.num_drones)]

    state_idx = settings['state_idx']
    input_idx = settings['input_idx']
    #n = settings['n']
    #m = settings['m']
    #K = np.eye(12,12)
    K = np.zeros((12,12))
    if  ARGS.sfb is not None:
        path = datapath
        if ARGS.sfb in ['direct']:
            controller_filename = 'controller.npy'
        elif ARGS.sfb in ['indirect']:
            controller_filename = 'controller_sysId_LQR.npy'
        elif ARGS.sfb in ['prelim']:
            controller_filename = 'controller_prelim.npy'
        controller = np.load(os.path.join(path, controller_filename), allow_pickle=True).item()
        for row, r_idx in enumerate(input_idx):
            for col, c_idx in enumerate(state_idx):
                K[r_idx, c_idx] = controller['controller'][row, col]
            #K[input_idx][:, state_idx] = controller['controller']
    lqr = [SimpleStateFeedbackController(K=K) for i in range(ARGS.num_drones)]

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    SFB_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.sfb_freq_hz))
    CTRL_PER_SFB = SFB_EVERY_N_STEPS // CTRL_EVERY_N_STEPS
    action = {str(i): np.array([0,0,0,0]) for i in range(ARGS.num_drones)}
    START = time.time()

    TRAJ_LENGTH = int(np.floor(ARGS.sfb_freq_hz * ARGS.duration_sec))
    traj_counters = np.array([0 for _ in range(ARGS.num_drones)])
    trajectories = [{   'U0':np.zeros((12,TRAJ_LENGTH)),
                        'X0':np.zeros((12,TRAJ_LENGTH)),
                        'X1':np.zeros((12,TRAJ_LENGTH))
                    }   for _ in range(ARGS.num_drones)]
    
    reference = {   'timestamps': [list() for j in range(ARGS.num_drones)],
                    'cur_states': [list() for j in range(ARGS.num_drones)],
                    'targets': [list() for j in range(ARGS.num_drones)],
                    'orig_targets': [list() for j in range(ARGS.num_drones)],
                    'frequency': int(ARGS.simulation_freq_hz/AGGR_PHY_STEPS/CTRL_EVERY_N_STEPS)
    }

    # For drawing the drone's trajectory
    drone_ids = env.getDroneIds()
    prev_positions = {i: p.getBasePositionAndOrientation(drone_ids[i])[0] for i in range(ARGS.num_drones)}
    # Trajectory color (R, G, B)
    line_color = [  [196/255, 78/255, 82/255],
                    [85/255, 168/255, 104/255],
                    [76/255, 114/255, 1767/255]]
    skip_time = 1.0
    skip_steps_before_drawing = skip_time * env.SIM_FREQ
    draw_time = 3.0
    draw_steps = draw_time * env.SIM_FREQ
    firstLine = True

    resetRequired = False
    sfb_on = False
    first_sfb_happened = False
    i = 0
    iLastReset = 0
    # for i in range(0, int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):
    # waitWithSampling = settings['start']
    waitForSfb = 0.01
    obs, reward, done, info = env.step(action)
    lastObs = obs
    while True:

        if ARGS.wind_on:
            if i/env.SIM_FREQ > 1.5:
                env.WIND_FORCE = np.array([-1., 1., 0.])/20
            if i/env.SIM_FREQ > 2.0:
                env.WIND_FORCE = [0., 0., 0.]

        if (not sfb_on) and waitForSfb*env.SIM_FREQ<(i-iLastReset):
            sfb_on = True

        if sfb_on and ARGS.part_pid_off:
            partially_disable_pid(ARGS, ctrl)

        if i%SFB_EVERY_N_STEPS == 0 and sfb_on and first_sfb_happened:
            for j in range(ARGS.num_drones):
                states      = obs[str(j)]["state"]
                cur_pos     = states[0:3]
                cur_rpy     = states[7:10]
                cur_vel     = states[10:13]
                # cur_rpy_rate= states[13:16]
                last_rpy    = lastObs[str(j)]["state"][7:10]
                cur_rpy_rate= (cur_rpy - last_rpy)*env.SIM_FREQ
                d_pos = cur_pos - TARGET_POS[wp_counters[j], :, j]
                d_vel = cur_vel - TARGET_VEL[wp_counters[j], :, j]
                d_rpy = cur_rpy - TARGET_RPY[wp_counters[j], :, j]
                d_rpy_rate = cur_rpy_rate - TARGET_RPY_RATE[wp_counters[j], :, j]
                trajectories[j]['X1'][0:3, traj_counters[j]] = d_pos
                trajectories[j]['X1'][3:6, traj_counters[j]] = d_vel
                trajectories[j]['X1'][6:9, traj_counters[j]] = d_rpy
                trajectories[j]['X1'][9:12, traj_counters[j]] = d_rpy_rate
                traj_counters[j] = traj_counters[j] + 1
            if traj_counters[0] >= TRAJ_LENGTH:
                # break out now to not oversample X0 and U0
                break

            if resetRequired:
                reset_sim(env, rnd, ARGS, R, H)
                resetRequired = False
                for j in range(ARGS.num_drones):
                    ctrl[j].reset()
                #     wp_counters[j] = 0
                # waitWithSampling = settings['start']
                sfb_on = False
                first_sfb_happened = False
                action = {str(j): np.array([0,0,0,0]) for j in range(ARGS.num_drones)}
                obs, reward, done, info = env.step(action)
                lastObs = obs
                enable_pid(ARGS, ctrl)
                TARGET_POS_COR = TARGET_POS.copy()
                TARGET_VEL_COR = TARGET_VEL.copy()
                TARGET_RPY_COR = TARGET_RPY.copy()
                TARGET_RPY_RATE_COR = TARGET_RPY_RATE.copy()
                iLastReset = i

        #### Step the simulation ###################################
        if i%SFB_EVERY_N_STEPS == 0 and sfb_on:
            d = 0   # latency simulation
            if ARGS.simulated_delay_ms>0:
                max_delay_ms = ARGS.simulated_delay_ms
                sfb_delay_ms = max_delay_ms*rnd.random()
                # sfb_delay_ms = max_delay_ms*np.random.random()
                ms_per_wp = 1000/ARGS.control_freq_hz
                sfb_delay_wp = sfb_delay_ms / ms_per_wp
                d = int(sfb_delay_wp)
            first_sfb_happened = True
            # print(f"Check: {waitWithSampling}")

            for j in range(ARGS.num_drones):
                input_correction = np.zeros(12)
                states = obs[str(j)]["state"]
                cur_pos     = states[0:3]
                cur_rpy     = states[7:10]
                cur_vel     = states[10:13]
                last_rpy    = lastObs[str(j)]["state"][7:10]
                cur_rpy_rate= (cur_rpy - last_rpy)*env.SIM_FREQ
                d_pos = cur_pos - TARGET_POS[wp_counters[j], :, j]
                d_vel = cur_vel - TARGET_VEL[wp_counters[j], :, j]
                d_rpy = cur_rpy - TARGET_RPY[wp_counters[j], :, j]
                d_rpy_rate = cur_rpy_rate - TARGET_RPY_RATE[wp_counters[j], :, j]

                if ARGS.sfb is not None:
                    input_correction = lqr[j].computeControl(d_pos, d_vel, d_rpy, d_rpy_rate)

                if ARGS.ctrl_noise>0:
                    ctrl_noise = np.multiply((2*rnd.random(12)-1), controlNoise)
                    # d_state = np.concatenate([d_pos, d_vel, d_rpy])
                    # input_correction += ctrl_noise*(0.2+np.linalg.norm(d_state, np.Inf))
                    input_correction += ctrl_noise

                TARGET_POS_COR[wp_counters[j]+d:wp_counters[j]+CTRL_PER_SFB+d, :, j]       = TARGET_POS[wp_counters[j]+d:wp_counters[j]+CTRL_PER_SFB+d, :, j] + input_correction[0:3]
                TARGET_VEL_COR[wp_counters[j]+d:wp_counters[j]+CTRL_PER_SFB+d, :, j]       = TARGET_VEL[wp_counters[j]+d:wp_counters[j]+CTRL_PER_SFB+d, :, j] + input_correction[3:6]
                TARGET_RPY_COR[wp_counters[j]+d:wp_counters[j]+CTRL_PER_SFB+d, :, j]       = TARGET_RPY[wp_counters[j]+d:wp_counters[j]+CTRL_PER_SFB+d, :, j] + input_correction[6:9]
                TARGET_RPY_RATE_COR[wp_counters[j]+d:wp_counters[j]+CTRL_PER_SFB+d, :, j]  = TARGET_RPY_RATE[wp_counters[j]+d:wp_counters[j]+CTRL_PER_SFB+d, :, j] + input_correction[9:12]

                trajectories[j]['X0'][0:3, traj_counters[j]] = d_pos
                trajectories[j]['X0'][3:6, traj_counters[j]] = d_vel
                trajectories[j]['X0'][6:9, traj_counters[j]] = d_rpy
                trajectories[j]['X0'][9:12, traj_counters[j]] = d_rpy_rate
                trajectories[j]['U0'][0:3, traj_counters[j]] = input_correction[0:3]
                trajectories[j]['U0'][3:6, traj_counters[j]] = input_correction[3:6]
                trajectories[j]['U0'][6:9, traj_counters[j]] = input_correction[6:9]
                trajectories[j]['U0'][9:12, traj_counters[j]] = input_correction[9:12]
                print(f"Finished sample #{traj_counters[j]}")

                # check the state vector for signs of instability
                d_state = np.concatenate([d_pos, d_vel, d_rpy])
                if ARGS.cut_traj:
                    if (np.linalg.norm(d_vel[:2])>1 or np.linalg.norm(d_rpy[:2], np.Inf)>1):
                        print(f"Reset required!, {traj_counters[0]}")
                        resetRequired = True
            # else:
                # print(f"Need to wait {waitWithSampling} steps until I start sampling")
                # waitWithSampling -= 1

        #### Compute control at the desired frequency ##############
        if i%CTRL_EVERY_N_STEPS == 0:

            #### Compute control for the current way point #############
            for j in range(ARGS.num_drones):
                try:
                    action[str(j)], _, _ = ctrl[j].computeControlFromState(
                        control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                        state=obs[str(j)]["state"],
                        # target_pos=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2]]),
                        target_pos=TARGET_POS_COR[wp_counters[j], :, j],
                        target_vel=TARGET_VEL_COR[wp_counters[j], :, j],
                        target_rpy=TARGET_RPY_COR[wp_counters[j], :, j],
                        target_rpy_rates=TARGET_RPY_RATE_COR[wp_counters[j], :, j],
                    )
                except ValueError:
                    print('Value Error detected')

            #### Save the previous waypoint and go to the next way point and loop #####################
            for j in range(ARGS.num_drones):
                states = obs[str(j)]["state"]
                cur_pos     = states[0:3]
                cur_rpy     = states[7:10]
                cur_vel     = states[10:13]
                # cur_rpy_rate= states[13:16]
                if 'lastObs' in locals():
                    last_rpy = lastObs[str(j)]["state"][7:10]
                else:
                    last_rpy = cur_rpy
                cur_rpy_rate= (cur_rpy - last_rpy)*env.SIM_FREQ
                reference['timestamps'][j].append(i/env.SIM_FREQ)
                reference['cur_states'][j].append(np.hstack([
                    cur_pos,
                    cur_vel,
                    cur_rpy,
                    cur_rpy_rate]))
                reference['targets'][j].append(np.hstack([
                    TARGET_POS_COR[wp_counters[j], :, j],
                    TARGET_VEL_COR[wp_counters[j], :, j],
                    TARGET_RPY_COR[wp_counters[j], :, j],
                    TARGET_RPY_RATE_COR[wp_counters[j], :, j]]))
                reference['orig_targets'][j].append(np.hstack([
                    TARGET_POS[wp_counters[j], :, j],
                    TARGET_VEL[wp_counters[j], :, j],
                    TARGET_RPY[wp_counters[j], :, j],
                    TARGET_RPY_RATE[wp_counters[j], :, j]]))
                if sfb_on:
                    if ARGS.wrap_wp:
                        wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0
                    else:
                        wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else NUM_WP-1


            for j in range(ARGS.num_drones):
                logger.log(drone=j,
                    timestamp=i/env.SIM_FREQ,
                    state= obs[str(j)]["state"],
                    #control=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
                    control=np.hstack([TARGET_POS[wp_counters[j], :, j], INIT_RPYS[j, :], np.zeros(6)])
                    )

        # Draw the drone trajectories
        if i%(CTRL_EVERY_N_STEPS*10) == 0 and ARGS.draw_trajectory and i>skip_steps_before_drawing :
            progress = (i - skip_steps_before_drawing)/draw_steps

            num_parallel_lines = 50 # 50 for final shot
            radius = 0.001
            for j in range(len(drone_ids)):
                current_position = p.getBasePositionAndOrientation(drone_ids[j])[0]
                direction = np.array(current_position) - np.array(prev_positions[j])
                direction_norm = np.linalg.norm(direction)

                if direction_norm > 0:
                    # Compute the normal and binormal vectors
                    normal = np.cross(direction, np.array([0, 0, 1])) if np.linalg.norm(np.cross(direction, np.array([0, 0, 1]))) > 0 else np.array([1, 0, 0])
                    normal = normal / np.linalg.norm(normal)
                    binormal = np.cross(direction, normal)
                    binormal = binormal / np.linalg.norm(binormal)

                    # Create parallel lines around the central trajectory line
                    for k in range(num_parallel_lines):
                        angle = 2 * np.pi * k / num_parallel_lines
                        displacement = radius * (np.cos(angle) * normal + np.sin(angle) * binormal)
                        start = prev_positions[j] + displacement
                        end = current_position + displacement
                        if firstLine:
                            pass
                        else:
                            # color = [*line_color[j], progress]
                            color = line_color[j]
                            p.addUserDebugLine(start, end, color, lineWidth=1, lifeTime=0)

                    prev_positions[j] = current_position
            if firstLine:
                firstLine = False

            if progress>1:
                camera_distance = 2.3
                camera_yaw = 46
                camera_pitch = -40 # -43
                camera_target_pos = [-0.5, 1.0, -0.6]
                p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_pos)
                input("Say cheeeeese")

        #### Sync the simulation ###################################
        if ARGS.gui:
            sync(i, START, env.TIMESTEP)

        lastObs = obs
        obs, reward, done, info = env.step(action)
        if i%SFB_EVERY_N_STEPS==0:
            apply_process_noise(env, ARGS, rnd, processNoise)
            obs = env._computeObs()
        i += 1

    env.close()

    #### Save the simulation results ###########################

    if ARGS.calc_cost:
        from rddc.tools.testing import lqr_cost_trajectories

        costs = np.zeros(ARGS.num_drones)
        for j in range(ARGS.num_drones):
            trajectory = {'state':None, 'input':None}
            trajectory['state'] = trajectories[j]['state'][state_idx, :]
            trajectory['input'] = trajectories[j]['input'][input_idx, :]
            costs[j], _ = lqr_cost_trajectories([trajectory], metric=settings)
        print('Average LQR cost: {0:4.3g}'.format(np.mean(costs)))
        files.save_dict_npy(os.path.join(logger.OUTPUT_FOLDER, "performance-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+".npy"), {'costs':costs})

    if ARGS.traj_filename is None:
        # traj_filename = "trajectory-sfb-rate-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
        train_or_test = 'train' if ARGS.sfb is None else 'test'
        traj_filename = get_simulation_trajectory_path(settings, train_or_test)
    else:
        traj_filename = ARGS.traj_filename
    np.save(traj_filename+".npy", trajectories, allow_pickle=True)

    reference['timestamps'] = np.array(reference['timestamps'])
    # So far, all states in 'reference' have been recorded in a matrix with dim0=drone, dim1=time, dim2=state
    # However, our usual format is dim1=state, dim2=time
    # That's why we need transposition here.
    reference['cur_states'] = np.transpose(np.array(reference['cur_states']), axes=(0,2,1))
    reference['targets'] = np.transpose(np.array(reference['targets']), axes=(0,2,1))
    reference['orig_targets'] = np.transpose(np.array(reference['orig_targets']), axes=(0,2,1))
    ### Save the states and the reference trajectory
    np.save(traj_filename+"_reference.npy", reference, allow_pickle=True)

    #### Plot the simulation results ###########################
    if ARGS.plot:
        #logger.plot()
        utils.plot_states_and_targets(logger, reference)

if __name__ == "__main__":

    settings = get_settings()

    run(settings)
