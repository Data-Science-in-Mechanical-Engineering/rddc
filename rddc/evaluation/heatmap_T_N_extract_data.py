import numpy as np
import argparse
import importlib
import os
import rddc.evaluation.tools as evaltools

import pandas as pd
# import rddc.run.settings.dean_test as module

basepath = '.'

def parse_arguments():
    parser = argparse.ArgumentParser(description='Heatmap plot for different testcases')
    parser.add_argument('--testcase',   required=True,  type=str,   help='Test case to run',        metavar='', choices=['dean', 'dean_1d', 'custom_3d'])
    parser.add_argument('--mode',       required=True,  type=str,   help='Mode to run the code in', metavar='', choices=['test', 'var_sigma_N'])
    args = parser.parse_args()
    return args

args = parse_arguments()
module = importlib.import_module('rddc.run.settings.' + args.testcase + '_' + args.mode)

settings = module.get_settings()
variations = module.get_variations()

# variations['sigma'] = np.sort(np.append(variations['sigma'], [0.3, 0.03]))
variations_fixed = dict()
variations_to_fold = list()

## EXTRACTING THE DATA
variations_to_fold = ['seed']
# automatic base on the settings file
variations_fixed = {key : value[0] for key, value in variations.items()
                        if len(value)==1 and key not in variations_to_fold}
arrays, var, nonnans = evaltools.get_scalars(
    settings=settings,
    variables=['N_stable', 'N_test'],
    variations_all=variations,
    variations_fixed=variations_fixed,
    variations_to_fold=variations_to_fold,
    # basepath='/home/alex/tmp_mnt/home/alex/robust-data-driven-control/data'
    basepath = 'data'
)

# GENERATING RANDOM DATA FOR DEBUGGING
# var = {'N_synth':[int(x) for x in np.ceil(np.logspace(0, np.log(300) / np.log(10), 10))],
#         'T':[x for x in np.logspace(1, 4, 7)]}
# arrays = {'N_stable':np.random.randint(0, 1000, (10,7)), 'N_test':np.ones((10,7))*1000}
# nonnans = {'N_stable': np.random.randint(0, 5, (10,7))}

## POSTPROCESSING AND CREATING A DATA FRAME
num_Nsynths = len(var['N_synth'])
num_Ts = len(var['T'])
ratios_stable = 100*np.divide(arrays['N_stable'], arrays['N_test'])
ratios_with_controller = nonnans['N_stable']
Ts, N_synths = np.meshgrid(var['T'], var['N_synth'])
nonnan_mask = ratios_with_controller > 0
data = pd.DataFrame({
    'T':Ts.ravel(),
    'N_synth':N_synths.ravel(),
    'ratio_stable':ratios_stable.ravel(),
    'ratio_with_controller':ratios_with_controller.ravel(),
    'controller_found':nonnan_mask.ravel()
})

data.to_csv(os.path.join(basepath, 'data', 'dean_var_T_N', 'compressed_data.csv'))

