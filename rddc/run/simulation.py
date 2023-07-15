"""
Here, multiple drone flight simulation are started and the trajectory data is gathered
Then, the trajectories are used to generate a robust controller
the created controller is saved in corresponding folder
it can be tested on a specified number of varying drones
the resulting trajectory can be plotted
"""
from rddc.tools import control_utils, controller_synthesis, files
from rddc.simulation import fly, utils
# from rddc.run.settings.simulation_like_experiment import get_settings
from rddc.run.settings.simulation import get_settings
import numpy as np
import os
import argparse
from rddc.simulation import fly
import matplotlib.pyplot as plt
import seaborn as sns
from rddc.tools.files import get_simulation_trajectory_path

# def get_test_trajectory_filename(settings, seed=None):
#     if seed is None:
#         seed = settings['seed']
#     return 'test_' + settings['testSettings']['traj'] +'_' + settings['testSettings']['sfb']  + '_seed' + str(seed)

# def get_train_trajectory_filename(settings):
#     return 'train_' + settings['trainSettings']['traj'] + '_seed' + str(settings['seed'])

def get_trajectories(settings, paths):
    """
    this trajectory is in format suitable for training the controller:
    each state is not an absolute value, but a deviation from the target state
    """
    trajectories = list()
    state_idx = settings['state_idx']
    input_idx = settings['input_idx']
    for path in paths:
        trajectories_origin = np.load(path, allow_pickle=True)
        for single_trajectory in trajectories_origin:
            trajectory = {
                'U0': single_trajectory['U0'][input_idx, :],
                'X0': single_trajectory['X0'][state_idx, :],
                'X1': single_trajectory['X1'][state_idx, :],
                'assumedBound':settings['assumedBound']
            }
            trajectories.append(trajectory)

    return trajectories

def get_reference(settings, path):
    """
    this is a target trajectory given to a quadcopter, (in an absolute format)
    """
    state_idx = settings['state_idx']
    reference = np.load(path, allow_pickle=True).item()['orig_targets'][0, state_idx, :]
    return reference

def get_absolute_trajectories(settings, paths):
    """
    this trajectory is in absolute format. (see get_trajectories())
    """
    trajectories = list()
    state_idx = settings['state_idx']
    for path in paths:
        trajectories_all = np.load(path, allow_pickle=True).item()['cur_states']
        # print(trajectories_all)
        # print(trajectories_all.shape[0])
        for trajectory_drone in trajectories_all:
            trajectory = trajectory_drone[state_idx, :]
            trajectories.append(trajectory)
    return trajectories

def check_trajectories(settings, trajectories, test_type='willems'):

    N = len(trajectories) # number of trajectories
    # T = trajectories[0]['U0'].shape[1] # length of one trajectory

    for trajId in range(N):
        U0 = trajectories[trajId]['U0']
        X0 = trajectories[trajId]['X0']
        X1 = trajectories[trajId]['X1']
        if test_type in ['willems']:
            assert control_utils.check_persistent_excitation_willems(states=X0, inputs=U0, tol=1e-3)
        elif test_type in ['slater']:
            assumedBound = settings['assumedBound']
            m_w = X0.shape[0]
            T = U0.shape[1] # length of one trajectory
            Phi_11 = assumedBound**2 * np.eye(m_w) * T
            Phi_12 = np.zeros((m_w, T))
            Phi_22 = -np.eye(T)
            Phi = np.block([[Phi_11  , Phi_12],
                            [Phi_12.T, Phi_22]])
            assert control_utils.check_gen_slater_condition(U0, X0, X1, Phi)
        else:
            print('Unknown trajectory test type')
            raise ValueError

def synthesize_controller(settings, trajectories4synth):
    path = os.path.join('data', settings['name'], settings['suffix'])
    func = getattr(controller_synthesis, settings['algorithm'])
    K = func(
        trajectories    = trajectories4synth,
        noiseInfo       = settings,
        perfInfo        = settings,
        verbosity       = settings['output_verbosity']
    )

    # Sanity check: Least squares solution
    m = settings['m']
    n = settings['n']
    Q = settings['Q']
    R = settings['R']
    U0 = np.hstack([trajectories4synth[sysId]['U0'] for sysId in range(len(trajectories4synth))])
    X0 = np.hstack([trajectories4synth[sysId]['X0'] for sysId in range(len(trajectories4synth))])
    X1 = np.hstack([trajectories4synth[sysId]['X1'] for sysId in range(len(trajectories4synth))])
    BA = np.linalg.lstsq(np.block([[U0],[X0]]).T, X1.T, rcond=None)[0].T
    B = BA[:, :m]
    A = BA[:, -n:]
    print("Least squares A is: \n{0}".format(np.array_str(A, precision=3, suppress_small=True)))
    print("Least squares B is: \n{0}".format(np.array_str(B, precision=3, suppress_small=True)))
    print('\nSpectral radius of the identified open loop: \n{}\n'.format(control_utils.spectral_radius(A)))
    if control_utils.check_controllability(A, B, tol=None):
        print('Identified system is controllable')
    else:
        print('Identified system is not controllable')
    from scipy.linalg import solve_discrete_are
    X_nom = np.array(np.array(solve_discrete_are(A, B, Q, R)))
    K_nom = - np.linalg.inv(R + B.T @ X_nom @ B) @ (B.T @ X_nom @ A)
    print('\nOptimal LQR controller: \n{}\n'.format(np.array_str(K_nom, precision=3)))
    files.save_dict_npy(os.path.join(path, 'controller_sysId_LQR.npy'), {'controller': K_nom})

    if K is None:
        K_str = None
    elif isinstance(K, float):
        K_str = 'nan'
    else:
        K_str = np.array_str(K, precision=3)
    print('\nOptimal DDC controller: \n{}\n'.format(K_str))

    files.save_dict_npy(os.path.join(path, 'controller.npy'), {'controller': K})

    return K

def plotTrajectories(settings):
    # seeds = [settings['seed']+i+settings['N_synth'] for i in range(settings['N_test'])]
    # paths = [os.path.join('data', settings['name'], settings['suffix'], get_test_trajectory_filename(settings, seed) +'_reference.npy')
                # for seed in seeds]
    paths = [get_simulation_trajectory_path(settings, 'test', reference=True)+'.npy']
    trajectories4test = get_absolute_trajectories(settings, paths)
    reference = get_reference(settings, paths[0])
    colors = sns.color_palette("deep", len(trajectories4test))
    for trajId in range(len(trajectories4test)):
        x = trajectories4test[trajId][0,:]
        y = trajectories4test[trajId][1,:]
        plt.plot(x,y,'--',color=colors[trajId])
    ref_x = reference[0, :]
    ref_y = reference[1, :]
    plt.plot(ref_x,ref_y,'-k')
    plt.show()

def training_parallel(settings_base):
    """
    do not change urdf
    either take load samples from extra_loads_synth or sample it from the given distribution
    calculate the moment of inertia for each extra load
    save the extra loads in settings, so they can be accessed in the simulation
    """
    settings = settings_base.copy()
    if len(settings['extra_loads_synth'])>0:
        settings['N_synth'] = len(settings['extra_loads_synth'])
        sample_loads = False
    else:
        sample_loads = True
    settings['trainSettings']['num_drones'] = settings['N_synth']
    settings['trainSettings']['traj_filename'] = get_simulation_trajectory_path(settings, 'train')
    rnd = np.random.default_rng(settings['seed'])
    for sysId in range(settings['N_synth']):
        if sample_loads:
            extra_load = utils.get_load_sample_box(
                rnd         = rnd,
                mass_range  = settings['mass_range'],
                pos_size    = settings['pos_size'],
            )
        else:
            extra_load = settings['extra_loads_synth'][sysId]
        print(f"Using the following extra load:\n {extra_load}")
        J = utils.J_from_extra_mass(extra_load['mass'], extra_load['position'], extra_load['form'], extra_load['size'])
        print(f"J calculated:\n {np.array_str(J, precision=6)}")
        extra_load.update({'J':J})
        settings['extra_loads'].append(extra_load)
    fly.run(settings, settings['trainSettings'])

def training_serial(settings_base):
    """
    either take load samples from extra_loads_synth or sample it from the given distribution
    calculate the moment of inertia for each extra load
    apply each of the extra loads to the urdf file, so it can be accessed in the simulation
    """
    droneUrdfPath = os.path.join('gym-pybullet-drones', 'gym_pybullet_drones', 'assets')
    originalPath = os.path.join(droneUrdfPath, 'cf2x.urdf')
    backupPath = os.path.join(droneUrdfPath, 'cf2x_backup.urdf')
    os.replace(originalPath, backupPath) #backup
    if len(settings_base['extra_loads'])>0:
        sample_loads = False
    else:
        sample_loads = True
    for sysId in range(settings_base['N_synth']):
        settings = settings_base.copy()
        settings['seed'] += sysId
        settings['trainSettings']['traj_filename'] = get_simulation_trajectory_path(settings, 'train', seed=settings['seed'])
        settings['trainSettings']['num_drones'] = 1
        rnd = np.random.default_rng(settings['seed'])
        if sample_loads:
            # extra_load = utils.get_load_sample_realistic(
            #     rnd,
            #     mass_range = settings['mass_range'],
            #     displacement_planar = settings['displacement_planar'],
            #     displacement_vert = settings['displacement_vert']
            # )
            extra_load = utils.get_load_sample_box(
                rnd         = rnd,
                mass_range  = settings['mass_range'],
                pos_size    = settings['pos_size'],
            )
        else:
            extra_load = settings['extra_loads'][sysId]
        print(f"Using the following extra load:\n {extra_load}")
        J = utils.J_from_extra_mass(extra_load['mass'], extra_load['position'], extra_load['form'], extra_load['size'])
        print(f"J calculated:\n {np.array_str(J, precision=6)}")
        extra_load.update({'J':J})
        utils.update_urdf_mass_and_inertia(backupPath, originalPath, extra_load)
        settings.update({'urdfBackupPath':backupPath, 'urdfOriginalPath':originalPath})
        fly.run(settings, settings['trainSettings'])
    # os.replace(backupPath, originalPath) #restore (commented out, since it's done in fly.py)

def testing_parallel(settings_base):
    """
    do not change urdf
    either take load samples from extra_loads_test or sample it from the given distribution
    calculate the moment of inertia for each extra load
    save the extra loads in settings, so they can be accessed in the simulation
    """
    settings = settings_base.copy()
    if len(settings['extra_loads_test'])>0:
        settings['N_test'] = len(settings['extra_loads_test'])
        sample_loads = False
    else:
        sample_loads = True
    settings['testSettings']['num_drones'] = settings['N_test']
    settings['testSettings']['traj_filename'] = get_simulation_trajectory_path(settings, 'test')
    rnd = np.random.default_rng(settings['seed'])
    for sysId in range(settings['N_test']):
        if sample_loads:
            extra_load = utils.get_load_sample_box(
                rnd         = rnd,
                mass_range  = settings['mass_range'],
                pos_size    = settings['pos_size'],
            )
        else:
            extra_load = settings['extra_loads_test'][sysId]
        print(f"Using the following extra load:\n {extra_load}")
        J = utils.J_from_extra_mass(extra_load['mass'], extra_load['position'], extra_load['form'], extra_load['size'])
        print(f"J calculated:\n {np.array_str(J, precision=6)}")
        extra_load.update({'J':J})
        settings['extra_loads'].append(extra_load)
    fly.run(settings, settings['testSettings'])

def testing_serial(settings_base):
    """
    either take load samples from extra_loads_test or sample it from the given distribution
    calculate the moment of inertia for each extra load
    apply each of the extra loads to the urdf file, so it can be accessed in the simulation
    """
    droneUrdfPath = os.path.join('gym-pybullet-drones', 'gym_pybullet_drones', 'assets')
    originalPath = os.path.join(droneUrdfPath, 'cf2x.urdf')
    backupPath = os.path.join(droneUrdfPath, 'cf2x_backup.urdf')
    if len(settings_base['extra_loads_test'])>0:
        sample_loads = False
    else:
        sample_loads = True
    for sysId in range(settings_base['N_test']):
        os.replace(originalPath, backupPath) #backup
        settings = settings_base.copy()
        settings['seed'] += sysId
        settings['testSettings']['traj_filename'] = get_simulation_trajectory_path(settings, 'test', seed=settings['seed'])
        settings['testSettings']['num_drones'] = 1
        rnd = np.random.default_rng(settings['seed'])
        if sample_loads:
            # extra_load = utils.get_load_sample_realistic(
            #     rnd,
            #     mass_range = settings['mass_range'],
            #     displacement_planar = settings['displacement_planar'],
            #     displacement_vert = settings['displacement_vert']
            # )
            extra_load = utils.get_load_sample_box(
                rnd         = rnd,
                mass_range  = settings['mass_range'],
                pos_size    = settings['pos_size'],
            )
        else:
            extra_load = settings['extra_loads_test'][sysId]
        print(f"Using the following extra load:\n {extra_load}")
        J = utils.J_from_extra_mass(extra_load['mass'], extra_load['position'], extra_load['form'], extra_load['size'])
        print(f"J calculated:\n {np.array_str(J, precision=6)}")
        extra_load.update({'J':J})
        utils.update_urdf_mass_and_inertia(backupPath, originalPath, extra_load)
        settings.update({'urdfBackupPath':backupPath, 'urdfOriginalPath':originalPath})
        fly.run(settings, settings['testSettings'])
    # os.replace(backupPath, originalPath) #restore (commented out, since it's done in fly.py)

def run_modular(settings_base):

    if ARGS.train:
        if settings_base['use_urdf']:
            training_serial(settings_base)
        else:
            training_parallel(settings_base)

    if ARGS.K:
        settings = settings_base.copy()
        if settings_base['use_urdf']:
            seeds = [settings['seed']+i for i in range(settings['N_synth'])]
            paths = [get_simulation_trajectory_path(settings, 'train', seed=seed)+'.npy' for seed in seeds]
        else:
            paths = [get_simulation_trajectory_path(settings, 'train')+'.npy']
        trajectories4synth = get_trajectories(settings, paths)
        # check_trajectories(settings, trajectories4synth, test_type='willems')
        K = synthesize_controller(settings, trajectories4synth=trajectories4synth)
        # print('\nDDC controller: \n{}\n'.format(K))

    if ARGS.test:
        if settings_base['use_urdf']:
            testing_serial(settings_base)
        else:
            testing_parallel(settings_base)
        # # droneUrdfPath = os.path.join('gym-pybullet-drones', 'gym_pybullet_drones', 'assets')
        # # originalPath = os.path.join(droneUrdfPath, 'cf2x.urdf')
        # # backupPath = os.path.join(droneUrdfPath, 'cf2x_backup.urdf')
        # # os.replace(originalPath, backupPath) #backup
        # settings = settings_base.copy()
        # rnd = np.random.default_rng(settings['seed'])
        # settings.update({'extra_loads':list()})
        # for sysId in range(settings_base['N_test']):
        #     # settings['seed'] += sysId + settings['N_synth']
        #     # settings['testSettings']['traj_filename'] = get_test_trajectory_filename(settings)
        #     # extraLoad = settings['testWeights'][sysId]
        #     # extraLoad = utils.get_load_sample_realistic(
        #     #     rnd,
        #     #     mass_range = settings['mass_range'],
        #     #     displacement_planar = settings['displacement_planar'],
        #     #     displacement_vert = settings['displacement_vert']
        #     # )
        #     extra_load = utils.get_load_sample_box(
        #         rnd         = rnd,
        #         mass_range  = settings['mass_range'],
        #         pos_size    = settings['pos_size'],
        #     )
        #     print(f"Using the following extra load:\n {extra_load}")
        #     J = utils.J_from_extra_mass(extra_load['mass'], extra_load['position'], extra_load['form'], extra_load['size'])
        #     print(f"J calculated:\n {np.array_str(J, precision=6)}")
        #     extra_load.update({'J':J})
        #     settings['extra_loads'].append(extra_load)
        # fly.run(settings, settings['testSettings'])
        # # os.replace(backupPath, originalPath) #restore

    if ARGS.eval:
        settings = settings_base.copy()
        settings['testSettings']['sfb'] = ARGS.eval
        plotTrajectories(settings)


if __name__=='__main__':

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Modular script that can simulate drone flights and synthesize FSB controllers based on them')
    parser.add_argument('--train',      action='store_true',       default=False,     help='Run training flight simulations and save their trajectories')
    parser.add_argument('--K',          action='store_true',       default=False,     help='Run controller synthesis')
    parser.add_argument('--test',       type=str,                  default=None,      help='Run test flight simulations and save their trajectories')
    parser.add_argument('--eval',       type=str,                  default=None,      help='Plot the test flight trajectory')
    parser.add_argument('--use_urdf',   action='store_true',       default=False,     help='Whether to change quadcopter through the urdf file or through direct adjustments in simulation')
    ARGS = parser.parse_args()

    settings = get_settings()
    settings.update({'use_urdf':ARGS.use_urdf})
    if ARGS.test is not None:
        settings['testSettings']['sfb'] = ARGS.test
    run_modular(settings)