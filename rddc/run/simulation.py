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
# from rddc.run.settings.simulation import get_settings
import importlib
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
def print_ctrl_summary(K):
    if K is None:
        return
    if isinstance(K, float):
        return
    string = "{0:5.2f}\t{1:5.2f}\t{2:5.2f}\t{3:5.2f}\t{4:5.2f}\t{5:5.2f}\t".format(
        min(K[0,1], -K[1,0]),
        max( abs(K[1,1]), abs(K[0,0])),
        min(K[0,3], -K[1,2]),
        max( abs(K[1,3]), abs(K[0,2])),
        min(-K[0,4], -K[1,5]),
        max( abs(K[1,4]), abs(K[0,5])),
    )
    print(string)


def plot_extra_loads(settings):
    xs = [1000*load['position'][0] for load in settings['extra_loads']]
    ys = [1000*load['position'][1] for load in settings['extra_loads']]
    # zs = [1000*load['position'][2] for load in settings['extra_loads']]
    ms = [10000*load['mass'] for load in settings['extra_loads']]
    dx =1000*np.max(settings['pos_size'])*1.1
    # plt.plot([-dx, dx], [0,0], 'k-')
    # plt.plot([0, 0], [-dx,dx], 'k-')
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.scatter(xs, ys, ms)
    plt.xlim((-dx, dx))
    plt.ylim((-dx, dx))
    plt.show()

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

def apply_measurement_noise(settings, rnd, trajectory):
    T = trajectory['X0'].shape[1]
    n = len(settings['state_idx'])
    meas_noise = np.array([x for idx, x in enumerate(settings['post_meas_noise']) if idx in settings['state_idx']])
    for i in range(T):
        trajectory['X0'][:,i] += (2*rnd.random(n)-1)*meas_noise
        trajectory['X1'][:,i] += (2*rnd.random(n)-1)*meas_noise

def concatenate_trajectories_as_one(trajectories):
    """
    this trajectory is in format suitable for training the controller:
    each state is not an absolute value, but a deviation from the target state
    """
    n = trajectories[0]['X0'].shape[0]
    m = trajectories[0]['U0'].shape[0]
    big_trajectory = {
                'U0': np.zeros((m,0)),
                'X0': np.zeros((n,0)),
                'X1': np.zeros((n,0)),
                'assumedBound':trajectories[0]['assumedBound']
    }
    for trajectory in trajectories:
        assert trajectory['U0'].shape[1] == trajectory['X1'].shape[1]
        assert trajectory['X0'].shape[1] == trajectory['X1'].shape[1]
        big_trajectory['U0'] = np.hstack([big_trajectory['U0'], trajectory['U0']])
        big_trajectory['X0'] = np.hstack([big_trajectory['X0'], trajectory['X0']])
        big_trajectory['X1'] = np.hstack([big_trajectory['X1'], trajectory['X1']])
    return big_trajectory

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

def synthesize_controller(settings, trajectories4synth, which_controllers):
    path = os.path.join('data', settings['name'], settings['suffix'])

    if 'rddc' in which_controllers:
        func = getattr(controller_synthesis, settings['algorithm'])
        K = func(
            trajectories    = trajectories4synth,
            noiseInfo       = settings,
            perfInfo        = settings,
            verbosity       = settings['output_verbosity']
        )
        if K is None:
            K_str = None
        elif isinstance(K, float):
            K_str = 'nan'
        else:
            K_str = np.array_str(K, precision=3)
        print('\nOptimal RDDC controller: \n{}\n'.format(K_str))
        print_ctrl_summary(K)
        files.save_dict_npy(os.path.join(path, 'controller_rddc.npy'), {'controller': K})

    if 'lslqr' in which_controllers:
        func = getattr(controller_synthesis, 'sysId_ls_lqr')
        K = func(
            trajectories    = trajectories4synth,
            sysInfo         = settings,
            verbosity       = settings['output_verbosity']
        )
        if K is None:
            K_str = None
        elif isinstance(K, float):
            K_str = 'nan'
        else:
            K_str = np.array_str(K, precision=3)
        print('\nOptimal LS-LQR controller: \n{}\n'.format(K_str))
        print_ctrl_summary(K)
        files.save_dict_npy(os.path.join(path, 'controller_lslqr.npy'), {'controller': K})

    if 'ceddc' in which_controllers:
        func = getattr(controller_synthesis, settings['algorithm'])
        K = func(
            trajectories    = [trajectories4synth[0]],
            noiseInfo       = settings,
            perfInfo        = settings,
            verbosity       = settings['output_verbosity']
        )
        if K is None:
            K_str = None
        elif isinstance(K, float):
            K_str = 'nan'
        else:
            K_str = np.array_str(K, precision=3)
        print('\nOptimal CE-DDC controller: \n{}\n'.format(K_str))
        print_ctrl_summary(K)
        files.save_dict_npy(os.path.join(path, 'controller_ceddc.npy'), {'controller': K})
    
    if 'ddc' in which_controllers:
        func = getattr(controller_synthesis, settings['algorithm'])
        K = func(
            trajectories    = [concatenate_trajectories_as_one(trajectories4synth)],
            noiseInfo       = settings,
            perfInfo        = settings,
            verbosity       = settings['output_verbosity']
        )
        if K is None:
            K_str = None
        elif isinstance(K, float):
            K_str = 'nan'
        else:
            K_str = np.array_str(K, precision=3)
        print('\nOptimal DDC controller: \n{}\n'.format(K_str))
        files.save_dict_npy(os.path.join(path, 'controller_ddc.npy'), {'controller': K})

def plotTrajectories(settings):
    # seeds = [settings['seed']+i+settings['N_synth'] for i in range(settings['N_test'])]
    # paths = [os.path.join('data', settings['name'], settings['suffix'], get_test_trajectory_filename(settings, seed) +'_reference.npy')
                # for seed in seeds]
    controller_suffix = settings['testSettings']['sfb']
    paths = [get_simulation_trajectory_path(settings, 'test', controller_suffix, reference=True)+'.npy']
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
    controller_suffix = settings['trainSettings']['sfb']
    settings['trainSettings']['num_drones'] = settings['N_synth']
    if settings['trainSettings']['traj_filename'] is None:
        settings['trainSettings']['traj_filename'] = get_simulation_trajectory_path(settings, 'train', controller_suffix)
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
    plot_extra_loads(settings)
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
        controller_suffix = settings['trainSettings']['sfb']
        if settings['trainSettings']['traj_filename'] is None:
            settings['trainSettings']['traj_filename'] = get_simulation_trajectory_path(settings, 'train', controller_suffix, seed=settings['seed'])
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
    controller_suffix = settings['testSettings']['sfb']
    settings['testSettings']['num_drones'] = settings['N_test']
    if settings['testSettings']['traj_filename'] is None:
        settings['testSettings']['traj_filename'] = get_simulation_trajectory_path(settings, 'test', controller_suffix)
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
        controller_suffix = settings['testSettings']['sfb']
        settings['seed'] += sysId
        if settings['testSettings']['traj_filename'] is None:
            settings['testSettings']['traj_filename'] = get_simulation_trajectory_path(settings, 'test', controller_suffix, seed=settings['seed'])
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

    if ARGS.K is not None:
        settings = settings_base.copy()
        controller_suffix = settings['trainSettings']['sfb']
        if len(ARGS.paths) > 0: # if paths are given from CLI
            paths = ARGS.paths
        elif settings_base['trainSettings']['traj_filename'] is not None: #if paths are specified in settings
            paths = [settings_base['trainSettings']['traj_filename']+'.npy']
        else:
            if settings_base['use_urdf']:
                seeds = [settings['seed']+i for i in range(settings['N_synth'])]
                paths = [get_simulation_trajectory_path(settings, 'train', controller_suffix,  seed=seed)+'.npy' for seed in seeds]
            else:
                paths = [get_simulation_trajectory_path(settings, 'train', controller_suffix)+'.npy']
        trajectories4synth = get_trajectories(settings, paths)
        rnd = np.random.default_rng(settings['seed'])
        if 'post_meas_noise' in settings_base.keys():
            for trajectory in trajectories4synth:
                # print(f"before:\n {trajectory['X0'][:,:5]}")
                apply_measurement_noise(settings_base, rnd, trajectory)
                # print(f"after:\n {trajectory['X0'][:,:5]}")
        # check_trajectories(settings, trajectories4synth, test_type='willems')
        if len(ARGS.K)>0:
            which_controllers = ARGS.K
        else:
            which_controllers = [settings_base['testSettings']['sfb']]
        synthesize_controller(settings, trajectories4synth=trajectories4synth, which_controllers=which_controllers)
        # print('\nDDC controller: \n{}\n'.format(K))

    if ARGS.test:
        if settings_base['use_urdf']:
            testing_serial(settings_base)
        else:
            testing_parallel(settings_base)

    if ARGS.eval:
        settings = settings_base.copy()
        plotTrajectories(settings)


if __name__=='__main__':

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Modular script that can simulate drone flights and synthesize FSB controllers based on them')
    parser.add_argument('--train',      action='store_true',        default=False,     help='Run training flight simulations and save their trajectories')
    parser.add_argument('--K',          nargs='*',                  default=None,      help='Type of the controller to synthesize. Use just --K to run controller synthesis. Controller type identified from --mode, if none is specified.')
    parser.add_argument('--test',       action='store_true',        default=None,      help='Run test flight simulations and save their trajectories')
    parser.add_argument('--eval',       action='store_true',        default=None,      help='Plot the test flight trajectory')
    parser.add_argument('--use_urdf',   action='store_true',        default=False,     help='Whether to change quadcopter through the urdf file or through direct adjustments in simulation')
    parser.add_argument('--paths',      nargs='*',                  default=[],        help='Use only particular paths to synthesize the controller with')
    parser.add_argument('--mode',       type=str,                   default='test',    help='Which mode to run simulation in', choices=['test', 'rddc', 'ceddc', 'like_experiment', 'ddc'] )
    ARGS = parser.parse_args()

    module = importlib.import_module('rddc.run.settings.simulation' + '_' + ARGS.mode)

    settings = module.get_settings()
    settings.update({'use_urdf':ARGS.use_urdf})
    run_modular(settings)