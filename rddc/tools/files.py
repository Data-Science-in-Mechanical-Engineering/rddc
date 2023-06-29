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
