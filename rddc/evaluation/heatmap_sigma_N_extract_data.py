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
# variations_fixed = {'bound':0.0005, 'T':500, 'assumedBound':0.001} # manual
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
#         'sigma':[x for x in np.logspace(-3/2, 0, 14)]}
# arrays = {'N_stable':np.random.randint(0, 1000, (10,14)), 'N_test':np.ones((10,14))*1000}
# nonnans = {'N_stable': np.random.randint(0, 5, (10,14))}

## POSTPROCESSING AND CREATING A DATA FRAME
num_Nsynths = len(var['N_synth'])
num_sigmas = len(var['sigma'])
ratios_stable = 100*np.divide(arrays['N_stable'], arrays['N_test'])
ratios_with_controller = nonnans['N_stable']
sigmas, N_synths = np.meshgrid(var['sigma'], var['N_synth'])
nonnan_mask = ratios_with_controller > 0
data = pd.DataFrame({
    'sigma':sigmas.ravel(),
    'N_synth':N_synths.ravel(),
    'ratio_stable':ratios_stable.ravel(),
    'ratio_with_controller':ratios_with_controller.ravel(),
    'controller_found':nonnan_mask.ravel()
})

data.to_csv(os.path.join(basepath, 'data', 'dean_var_sigma_N', 'compressed_data.csv'))

