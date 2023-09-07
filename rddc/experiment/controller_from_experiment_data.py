"""
Here, multiple drone flight simulation are started and the trajectory data is gathered
Then, the trajectories are used to generate a robust controller
the created controller is saved in corresponding folder
it can be tested on a specified number of varying drones
the resulting trajectory can be plotted
"""
from rddc.tools import control_utils, controller_synthesis, files
from rddc.run.settings.controller_from_exp_data import get_settings
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def get_trajectories(settings, paths, ignore_first_point=False):
    trajectories = list()
    state_idx = settings['state_idx']
    n = len(state_idx)
    input_idx = settings['input_idx']
    m = len(input_idx)
    for path in paths:
        trajectories_origin = np.load(path, allow_pickle=True).item()
        if ignore_first_point:
            indices = []
            for i in range(1,len(trajectories_origin['X1'])):
                if trajectories_origin['X1'][i] is None:
                    continue
                if trajectories_origin['X1'][i-1] is None:
                    continue
                indices.append(i)
        else:
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

def perform_postprocessing(trajectories, rng : np.random.RandomState):
    pp_trajectories = list()
    for trajectory in trajectories:
        pp_trajectory = trajectory.copy()
        state_shift = (2*rng.random((6,1))-1) * np.array([[5,5,  5,5,  5,5]]).T
        pp_trajectory['X0'] = trajectory['X0'] + state_shift
        pp_trajectory['X1'] = trajectory['X1'] + state_shift
        pp_trajectories.append(pp_trajectory)
    return pp_trajectories

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

    rng = np.random.default_rng(settings['seed'])
    paths = [os.path.join('data', 'experiment', filename, 'trajectory.npy') for filename in settings['filenames']]
    trajectories4synth = get_trajectories(settings, paths, ignore_first_point=False)
    if settings['postprocessing']:
        trajectories4synth = perform_postprocessing(trajectories4synth, rng)
    # check_trajectories(settings, trajectories4synth, test_type='willems')
    K = synthesize_controller(settings, trajectories4synth=trajectories4synth)
    # print('\nDDC controller: \n{}\n'.format(K))

if __name__=='__main__':
    settings = get_settings()

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Modular script to synthesize an RDDC controller from experiment data')
    parser.add_argument('--pp', action='store_true', default=False, help='Perform postprocessing. Cut-off first point or two of each trajectory, and add shifts to position')
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
    settings['postprocessing'] = ARGS.pp
    if len(settings['filenames'])==0:
        print("No training trajectories were given. Please specify the folder names in settings or/and in the input arguments")

    # settings['suffix'] = ''
    run(settings)