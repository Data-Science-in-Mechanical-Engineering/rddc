"""
Here, multiple drone flight simulation are started and the trajectory data is gathered
Then, the trajectories are used to generate a robust controller
the created controller is saved in corresponding folder
it can be tested on a specified number of varying drones
the resulting trajectory can be plotted
"""
from rddc.tools import control_utils, controller_synthesis, files
from rddc.run.settings.simulation_like_experiment import get_settings
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def get_trajectories(settings, paths):
    trajectories = list()
    state_idx = settings['state_idx']
    n = len(state_idx)
    input_idx = settings['input_idx']
    m = len(input_idx)
    for path in paths:
        # trajectories_origin = np.load(path, allow_pickle=True)
        # trajectory = {
        #     'U0': trajectories_origin[0]['U0'][input_idx, :],
        #     'X0': trajectories_origin[0]['X0'][state_idx, :],
        #     'X1': trajectories_origin[0]['X1'][state_idx, :],
        #     'assumedBound':settings['assumedBound']
        # }

        trajectories_origin = np.load(path, allow_pickle=True).item()
        # num_breaks = sum([x is None for x in trajectories_origin['X1']])
        # num_total = len(trajectories_origin['U0'])
        # num_points = num_total - num_breaks
        indices = [i for i, x in enumerate(trajectories_origin['X1']) if x is not None]
        num_points = len(indices)
        trajectory = {'U0': np.zeros((m, num_points)), 'X0': np.zeros((n, num_points)), 'X1': np.zeros((n, num_points)), 'assumedBound':settings['assumedBound']}
        for idx in range(num_points):
            idx_origin = indices[idx]
            trajectory['U0'][:, idx] = trajectories_origin['U0'][idx_origin][input_idx]
            trajectory['X0'][:, idx] = trajectories_origin['X0'][idx_origin][state_idx]
            trajectory['X1'][:, idx] = trajectories_origin['X1'][idx_origin][state_idx]

        trajectories.append(trajectory)

    return trajectories

def synthesize_controller(settings, trajectories4synth):
    path = os.path.join('data', 'experiment', ARGS.filename)
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

def run(settings, ARGS):

    paths = [os.path.join('data', 'experiment', ARGS.filename, 'trajectory.npy')]
    trajectories4synth = get_trajectories(settings, paths)
    # check_trajectories(settings, trajectories4synth, test_type='willems')
    K = synthesize_controller(settings, trajectories4synth=trajectories4synth)
    # print('\nDDC controller: \n{}\n'.format(K))

if __name__=='__main__':

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Modular script to synthesize an RDDC controller from experiment data')
    parser.add_argument('--filename',           default='?',      type=str,           help='Experiment file name', metavar='')
    # parser.add_argument('--train',      action='store_true',       default=False,     help='Run training flight simulations and save their trajectories')
    # parser.add_argument('--K',          action='store_true',       default=False,     help='Run controller synthesis')
    # parser.add_argument('--test',       action='store_true',       default=False,     help='Run test flight simulations and save their trajectories')
    # parser.add_argument('--eval',       action='store_true',       default=False,     help='Plot the test flight trajectory')
    ARGS = parser.parse_args()

    settings = get_settings()
    # settings['suffix'] = ''
    run(settings, ARGS)