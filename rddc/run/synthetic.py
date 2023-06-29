import numpy as np
from rddc.tools import control_utils as cu
from rddc.tools import istarmap
import importlib
import argparse
import multiprocessing as mp
from tqdm import tqdm
from rddc.run import main

def parse_arguments(override_args=None, ignore_cli=False):
    """
    Argument parsing implemented so that the script can be called from
    both CLI or another python function with arguments passed in a dictionary
    """
    parser = argparse.ArgumentParser(description='Trajectory script using PID or state-feedback control')
    parser.add_argument('--testcase',   required=True,  type=str,   help='Test case to run',        metavar='', choices=['dean', 'dean_1d', 'custom_3d'])
    parser.add_argument('--mode',       required=True,  type=str,   help='Mode to run the code in', metavar='', choices=['test', 'var_T_N', 'var_sigma_N'])

    if ignore_cli:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    if override_args is not None:
        for key, value in override_args.items():
            setattr(args, key, value)

    return args

def test_settings(settings):
    print('Spectral radius of the specified system (open loop): \n{}'.format(cu.spectral_radius(settings['A'])))
    if settings['K_prelim'] is not None:
        print(f" Spectral radius with a given preliminary controller (closed loop): {cu.spectral_radius(settings['A'] + settings['B'] @ settings['K_prelim'])}")
    assert cu.check_controllability(settings['A'], settings['B'], settings['controllability_tol'])
    print('Specified system is controllable')

def run(override_args=None):
    ignore_cli = override_args is not None

    args = parse_arguments(override_args, ignore_cli)

    module = importlib.import_module('rddc.run.settings.' + args.testcase + '_' + args.mode)

    settings = module.get_settings()
    test_settings(settings)

    variations = module.get_variations()
    variations_sizes = [len(_) for _ in variations.values()]
    variations_total = np.prod(variations_sizes)

    params = [(run_id, settings, variations) for run_id in range(variations_total)]
    # for p in tqdm(params):        #Serial run, for debugging
    #     main.run_variation(*p)
    with mp.Pool(settings['num_cores']) as pool:
        for _ in tqdm(pool.istarmap(main.run_variation, params), total=variations_total):
            pass

if __name__=='__main__':
    run()