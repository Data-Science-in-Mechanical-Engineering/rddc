import os
import numpy as np

def get_variable_location(name, settings):
    if name in settings:
        return 'settings.npy'
    elif name in ['N_stable', 'cost_distribution', 'costs']:
        return 'results.npy'
    elif name in ['systems4test']:
        return 'systems4test.npy'
    elif name in ['systems4synth']:
        return 'systems4synth.npy'
    elif name in ['controller']:
        return 'controller.npy'
    elif name in ['trajectories4synth']:
        return 'trajectories4synth.npy'
    else:
        raise NotImplementedError

def get_simulation_trajectory_path(settings, train_or_test, controller, reference=False, seed=None):
    if seed is not None:
        seed_suffix = '_seed' + str(seed)
    else:
        seed_suffix = ''
    if reference:
        reference_suffix = '_reference'
    else:
        reference_suffix = ''
    if controller is None:
        controller_suffix = '_no_sfb'
    else:
        controller_suffix = '_'+controller
    filepath = os.path.join(
        'data', 
        settings['name'], 
        settings['suffix'], 
        train_or_test + '_' + settings[train_or_test + 'Settings']['traj'] + controller_suffix + seed_suffix + reference_suffix
    )
    return filepath

def get_suffix_part(name, value):
    if isinstance(value, int):
        value_str = str(value)
    else:
        value_str = '{:.4g}'.format(value)
    return name + '=' + value_str

def save_dict_npy(path, dictionary, mode='wb'):
    #assert(isinstance(dictionary, dict), f"Saved variable must be a dictionary, got {type(dictionary)} instead.")
    numpy_data = {key: np.array(value) for key, value in dictionary.items()}
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, mode) as file:
        np.save(file, dictionary, allow_pickle=True)
