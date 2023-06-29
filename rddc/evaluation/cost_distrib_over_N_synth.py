import numpy as np
import matplotlib.pyplot as plt
from run.dean_single import get_settings
from run.dean_variation import get_variations
# from run.dean_1d_single import get_settings
# from run.dean_1d_variation import get_variations
import evaluation.tools as evaltools
import seaborn as sns
import pandas as pd

settings = get_settings()
variations = get_variations()
variations_fixed = {'sigma':0.01, 'bound':0.01, 'T':20, 'assumedBound': 0.03}
arrays, var = evaltools.get_arrays(
    settings=settings,
    variables=['costs'],
    variations_all=variations,
    variations_fixed=variations_fixed
)
data = pd.DataFrame()
for i in range(len(var['N_synth'])):
    _data = pd.DataFrame()
    _data['costs'] = arrays['costs'][i]
    _data['N_synth'] = var['N_synth'][i]
    data = pd.concat([data, _data])

fig, ax = plt.subplots()
sns.boxenplot(data=data, x='N_synth', y='costs')
ax.set_title(f'Cost over N_synth for: {variations_fixed}')
ax.set_yscale('log')
# set the limits of the plot to the limits of the data
ax.set_xlabel('N_synth')
ax.set_ylabel('cost')

plt.show()