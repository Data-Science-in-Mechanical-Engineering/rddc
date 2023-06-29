import numpy as np
import matplotlib.pyplot as plt
from run.dean_single import get_settings
from run.dean_variation import  get_variations
# from run.dean_1d_single import get_settings
# from run.dean_1d_variation import get_variations
import evaluation.tools as evaltools

settings = get_settings()
variations = get_variations()
variations_fixed = dict()
variations_to_fold = list()

variations_fixed = {'bound':0.01, 'T':20, 'sigma':0.03, 'assumedBound':0.03}

arrays, var = evaltools.get_scalars(
    settings=settings,
    variables=['N_stable', 'N_test'],
    variations_all=variations,
    variations_fixed=variations_fixed,
    variations_to_fold=variations_to_fold
)
ratios = np.divide(arrays['N_stable'], arrays['N_test'])

fig, ax = plt.subplots()
c = ax.plot(var['N_synth'], ratios)
# set the limits of the plot to the limits of the data
ax.set_xlabel('N_synth')
ax.set_ylabel('N_stable / N_test')

plt.show()