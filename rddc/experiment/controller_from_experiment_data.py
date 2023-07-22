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

        print(f"Using trajectory at: \t{path}\nnumber of points:\t{num_points}")
        trajectories.append(trajectory)

    return trajectories

def synthesize_controller(settings, trajectories4synth):
    path = os.path.join('data', 'experiment')
    func = getattr(controller_synthesis, settings['algorithm'])
    K = func(
        trajectories    = trajectories4synth,
        noiseInfo       = settings,
        perfInfo        = settings,
        verbosity       = settings['output_verbosity']
    )

    func = getattr(controller_synthesis, 'sysId_ls_lqr')
    K_nom = func(
        trajectories    = trajectories4synth,
        sysInfo         = settings,
        verbosity       = settings['output_verbosity']
    )
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

def run(settings):

    paths = [os.path.join('data', 'experiment', filename, 'trajectory.npy') for filename in settings['filenames']]
    trajectories4synth = get_trajectories(settings, paths)
    # check_trajectories(settings, trajectories4synth, test_type='willems')
    K = synthesize_controller(settings, trajectories4synth=trajectories4synth)
    # print('\nDDC controller: \n{}\n'.format(K))

if __name__=='__main__':
    settings = get_settings()

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Modular script to synthesize an RDDC controller from experiment data')
    if not 'filenames' in settings:
        settings.update({'filenames':[]})
        parser.add_argument('--filenames', nargs='+', default=[], help='Experiment file names. It has not been found in settings, so it must be specified at input', metavar='')
    else:
        parser.add_argument('--filenames', nargs='*', default=[], help='Experiment file names. It has been found in settings, however, the input will replace them', metavar='')
    ARGS = parser.parse_args()
    # print(ARGS.filenames)
    if 'only' in ARGS.filenames: # only use CLI filenames for controller synthesis -> remove filenames from settings
        settings.update({'filenames':[]})
        ARGS.filenames.remove('only')
    settings['filenames'] = settings['filenames'] + ARGS.filenames
    if len(settings['filenames'])==0:
        print("No training trajectories were given. Please specify the folder names in settings or/and in the input arguments")

    # settings['suffix'] = ''
    run(settings)