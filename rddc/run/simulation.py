"""
Here, multiple drone flight simulation are started and the trajectory data is gathered
Then, the trajectories are used to generate a robust controller
the created controller is saved in corresponding folder
it can be tested on a specified number of varying drones
the resulting trajectory can be plotted
"""
from rddc.tools import control_utils, controller_synthesis, files
from rddc.simulation import fly, utils
from rddc.run.settings.simulation import get_settings
import numpy as np
import os
import argparse
from rddc.simulation import fly
import matplotlib.pyplot as plt
import seaborn as sns

def get_test_trajectory_filename(settings, seed=None):
    if seed is None:
        seed = settings['seed']
    return 'test_' + settings['testSettings']['traj'] +'_' + settings['testSettings']['sfb']  + '_seed' + str(seed)

def get_train_trajectory_filename(settings):
    return 'train_' + settings['trainSettings']['traj'] + '_seed' + str(settings['seed'])

def get_trajectories(settings, paths):
    trajectories = list()
    state_idx = settings['state_idx']
    input_idx = settings['input_idx']
    for path in paths:
        trajectories_origin = np.load(path, allow_pickle=True)
        trajectory = {
            'U0': trajectories_origin[0]['U0'][input_idx, :],
            'X0': trajectories_origin[0]['X0'][state_idx, :],
            'X1': trajectories_origin[0]['X1'][state_idx, :],
            'assumedBound':settings['assumedBound']
        }
        trajectories.append(trajectory)

    return trajectories

def get_reference(settings, path):
    state_idx = settings['state_idx']
    reference = np.load(path, allow_pickle=True).item()['orig_targets'][0, state_idx, :]
    return reference

def get_absolute_trajectories(settings, paths):
    trajectories = list()
    state_idx = settings['state_idx']
    for path in paths:
        trajectory = np.load(path, allow_pickle=True).item()['cur_states'][0, :, state_idx]
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
    seeds = [settings['seed']+i+settings['N_synth'] for i in range(settings['N_test'])]
    paths = [os.path.join('data', settings['name'], settings['suffix'], get_test_trajectory_filename(settings, seed) +'_reference.npy')
                for seed in seeds]
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

def run_modular(settings_base):

    if ARGS.train:
        droneUrdfPath = os.path.join('gym-pybullet-drones', 'gym_pybullet_drones', 'assets')
        originalPath = os.path.join(droneUrdfPath, 'cf2x.urdf')
        backupPath = os.path.join(droneUrdfPath, 'cf2x_backup.urdf')
        os.replace(originalPath, backupPath) #backup
        for sysId in range(settings_base['N_synth']):
            settings = settings_base.copy()
            settings['seed'] += sysId
            settings['trainSettings']['traj_filename'] = get_train_trajectory_filename(settings)
            # extraLoad = settings['trainWeights'][sysId]
            rnd = np.random.default_rng(settings['seed'])
            # extraLoad = utils.get_load_sample_realistic(
            #     rnd,
            #     mass_range = settings['mass_range'],
            #     displacement_planar = settings['displacement_planar'],
            #     displacement_vert = settings['displacement_vert']
            # )
            extraLoad = utils.get_load_sample_box(
                rnd         = rnd,
                mass_range  = settings['mass_range'],
                pos_size    = settings['pos_size'],
            )
            print(f"Using the following extra load:\n {extraLoad}")
            utils.update_urdf_mass_and_inertia(
                URDFPATH = backupPath,
                NEW_URDFPATH = originalPath,
                extra_load = extraLoad
            )
            fly.run(settings, settings['trainSettings'])
        os.replace(backupPath, originalPath) #restore

    if ARGS.K:
        settings = settings_base.copy()
        seeds = [settings['seed']+i for i in range(settings['N_synth'])]
        paths = [os.path.join('data', settings['name'], settings['suffix'], 'train_' + settings['trainSettings']['traj'] + '_seed' + str(seed) + '.npy')
                    for seed in seeds]
        trajectories4synth = get_trajectories(settings, paths)
        # check_trajectories(settings, trajectories4synth, test_type='willems')
        K = synthesize_controller(settings, trajectories4synth=trajectories4synth)
        # print('\nDDC controller: \n{}\n'.format(K))

    if ARGS.test:
        droneUrdfPath = os.path.join('gym-pybullet-drones', 'gym_pybullet_drones', 'assets')
        originalPath = os.path.join(droneUrdfPath, 'cf2x.urdf')
        backupPath = os.path.join(droneUrdfPath, 'cf2x_backup.urdf')
        os.replace(originalPath, backupPath) #backup
        for sysId in range(settings_base['N_test']):
            settings = settings_base.copy()
            settings['testSettings']['sfb'] = ARGS.test
            settings['seed'] += sysId + settings['N_synth']
            settings['testSettings']['traj_filename'] = get_test_trajectory_filename(settings)
            # extraLoad = settings['testWeights'][sysId]
            rnd = np.random.default_rng(settings['seed'])
            # extraLoad = utils.get_load_sample_realistic(
            #     rnd,
            #     mass_range = settings['mass_range'],
            #     displacement_planar = settings['displacement_planar'],
            #     displacement_vert = settings['displacement_vert']
            # )
            extraLoad = utils.get_load_sample_box(
                rnd         = rnd,
                mass_range  = settings['mass_range'],
                pos_size    = settings['pos_size'],
            )
            print(f"Using the following extra load:\n {extraLoad}")
            utils.update_urdf_mass_and_inertia(
                URDFPATH = backupPath,
                NEW_URDFPATH = originalPath,
                extra_load = extraLoad
            )

            fly.run(settings, settings['testSettings'])
        os.replace(backupPath, originalPath) #restore

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
    ARGS = parser.parse_args()

    settings = get_settings()
    # settings['suffix'] = ''
    run_modular(settings)