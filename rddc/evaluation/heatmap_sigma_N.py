import numpy as np
import argparse
import importlib
import matplotlib as mpl
# Use the pgf backend (must be set before pyplot imported)
mpl.use('pgf')
import rddc.evaluation.tools as evaltools
c, params = evaltools.get_colors_and_plot_params('CDC_paper')  # specify font size etc.,
mpl.rcParams.update(params)

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import seaborn as sns
# import rddc.run.settings.dean_settings_var_sigma_N as module #for production
import rddc.run.settings.dean_test as module #for testing and development
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
# from run.dean_1d_single import get_settings
# from run.dean_1d_variation import get_variations

## SETTING UP COLORS AND APPEARENCE
# with open(os.path.join('rddc','tools','RWTHcolors.json')) as json_file:
#         RWTHcolors = json.load(json_file)
# hexLow = evaltools.rgb01_to_hex(RWTHcolors['rot100'])
# hexHigh = evaltools.rgb01_to_hex(RWTHcolors['gruen100'])
# cmap = sns.color_palette("blend:"+hexLow+','+hexHigh, as_cmap=True)
# cmap = sns.color_palette("YlGnBu", as_cmap=True) #sequential
# cmap = sns.color_palette("PuBuGn", as_cmap=True) #sequential
# cmap = sns.color_palette("RdYlGn", as_cmap=True) #divergent, intuitive
cmap = ListedColormap(sns.color_palette("RdYlGn", n_colors=10)) #divergent, intuitive
# cmap = ListedColormap(sns.diverging_palette(12, 130, s=100, l=50, sep=50, n=10, center='light'))
# cmap = sns.light_palette("seagreen", as_cmap=True) #sequentialm one-color
# cmap = sns.cubehelix_palette(start=1.0, rot=0.4, hue=1.0, reverse=True, as_cmap=True) #sequential
# cmap = sns.color_palette("light:"+hexHigh, as_cmap=True) #sequential, one-color
# #rgb( [153, 0, 0], [255, 213, 0], [162, 255, 153]) #dark red to light green
# #rgb( [255, 128, 128], [255, 213, 0], [13, 77, 0]) #dark red to light green
# cmap = LinearSegmentedColormap.from_list("mycmap",[
#     evaltools.rgb_to_hex([255, 156, 156]),
#     evaltools.rgb_to_hex([255, 213, 0]),
#     evaltools.rgb_to_hex([13, 90, 0]), 
# ])
norm = colors.Normalize(vmin=0, vmax=100)
# sns.set_theme()

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
# nonnans = {'N_stable': np.random.randint(0, 3, (10,14))}

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
data_with_controller = data[data['controller_found']==True].copy()
data_with_controller['circle_size_log'] = np.log10(data_with_controller['ratio_with_controller'])
data_without_controller = data[data['controller_found']==False]

## PLOTTING
(fig_width_in, fig_height_in) = evaltools.get_size(245, subplots=(1,1), fraction=1)
fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in*1.5))
fig.set_dpi(300)
fig.subplots_adjust(bottom=0.12, top=0.95, left=0.18, right=0.95)
ax_bbox_in_width = ax.get_window_extent().width / fig.get_dpi()
ax_bbox_in_height = ax.get_window_extent().height / fig.get_dpi()
# point size is to be given in (typographical dot)^2, typographical dot is 1/72 inch
circle_max_size = min((ax_bbox_in_width/num_Nsynths*72)**2, (ax_bbox_in_height/num_sigmas*72)**2)*0.61
ax.set_xscale('log')
ax.set_yscale('log')
sns.scatterplot(
    data=data_with_controller, x='N_synth', y='sigma',
    hue='ratio_stable', size='ratio_with_controller',
    sizes=(circle_max_size/10, circle_max_size), palette=cmap, ax=ax, legend=False,
    linewidth=0, zorder=2
)

sns.scatterplot(
    data=data_without_controller, x="N_synth", y="sigma",
    marker="x", color="black", s=20, linewidth=1, ax=ax, zorder=2)
# ax.set_title(f'Stability heatmap for: {variations_fixed}, averaged over {variations_to_fold}')
# set the limits of the plot to the limits of the data
# ax.axis([N_synths.min(), N_synths.max(), sigmas.min(), sigmas.max()])
cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, drawedges=False,
                    location='top',
                    label='Percentage of stable systems')
ax.set_xlabel(r'Number of observed systems $N$')
ax.set_ylabel(r'Indicator for variation size  $\sigma^2$')
ax.set_xticks(var['N_synth'], labels=[str(i) for i in var['N_synth']], minor=False) #
# ax.set_yticks(var['sigma'], labels=["{:3.2g}".format(i) for i in var['sigma']], minor=False) #
yticks = [  *list(np.array([3,4,5,6,7,8,9])*1e-2),
            *list(np.array([1,2,3,4,5,6,7,8,9])*1e-1),
            # *list(np.array([1,2,3,4,5,6,7,8,9])*1e-1),
            1
]
ax.set_yticks(yticks, minor=False) #
ax.minorticks_off()
ax.grid(True, linestyle=':', linewidth=0.5, zorder=1, which='major')
# plt.show()

# plt.show()

plt.savefig('figures/heatmap_sigma_N.pgf', format='pgf')
plt.savefig('figures/heatmap_sigma_N.pdf', format='pdf', dpi=300)

plt.close()