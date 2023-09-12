import numpy as np
import os
import matplotlib as mpl
# Use the pgf backend (must be set before pyplot imported)
mpl.use('pgf')
import rddc.evaluation.tools as evaltools
c, params = evaltools.get_colors_and_plot_params('CDC_paper')  # specify font size etc.,
mpl.rcParams.update(params)

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea
from matplotlib.offsetbox import AnchoredText
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

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
basepath = '.'

data = pd.read_csv(os.path.join(basepath, 'data', 'dean_var_T_N', 'compressed_data.csv'))
N_synths = data.N_synth.unique()
num_Nsynths = N_synths.shape[0]
Ts = data['T'].unique()
num_Ts = Ts.shape[0]
data_with_controller = data[data['controller_found']==True]
data_with_controller['circle_size_log'] = np.log10(data_with_controller['ratio_with_controller'])
data_without_controller = data[data['controller_found']==False]

## PLOTTING
(fig_width_in, fig_height_in) = evaltools.get_size(245, subplots=(1,1), fraction=1)
fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in*1.5))
fig.set_dpi(300)
# fig.subplots_adjust(bottom=0.18, top=0.9, left=0.18, right=0.95)
fig.subplots_adjust(bottom=0.12, top=0.95, left=0.18, right=0.95)
ax_bbox_in_width = ax.get_window_extent().width / fig.get_dpi()
ax_bbox_in_height = ax.get_window_extent().height / fig.get_dpi()
# point size is to be given in (typographical dot)^2, typographical dot is 1/72 inch
circle_max_size = min((ax_bbox_in_width/num_Nsynths*72)**2, (ax_bbox_in_height/num_Ts*72)**2)*0.61
ax.set_xscale('log')
ax.set_yscale('log')
sns.scatterplot(
    data=data_with_controller, x='N_synth', y='T',
    hue='ratio_stable', size='ratio_with_controller',
    sizes=(circle_max_size/10, circle_max_size), palette=cmap, ax=ax, legend=False,
    linewidth=0, zorder=2
)

sns.scatterplot(
    data=data_without_controller, x="N_synth", y="T",
    marker="x", color="black", s=20, linewidth=1, ax=ax, zorder=2)
# ax.set_title(f'Stability heatmap for: {variations_fixed}, averaged over {variations_to_fold}')
# set the limits of the plot to the limits of the data
# ax.axis([N_synths.min(), N_synths.max(), Ts.min(), Ts.max()])
cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, drawedges=False,
                    location='top',
                    anchor = (1.0, 0.0),
                    shrink = 0.67,
                    ticks = [0, 20, 40, 60, 80, 100],
                    label='% stable systems')

#### Second legend: circle sizes #####
max_circle_size_legend = 5.2
ada = AnchoredDrawingArea(  90, 35, 0, 0,
                            loc=('upper left'), frameon=False,
                            bbox_to_anchor=(-0.25, 1.35),
                            bbox_transform=ax.transAxes)
ada.drawing_area.add_artist(Circle((10, 5), max_circle_size_legend, fc="k"))
ada.drawing_area.add_artist(Circle((30, 5), max_circle_size_legend/np.sqrt(3), fc="k"))
ada.drawing_area.add_artist(Circle((47.5, 5), max_circle_size_legend/np.sqrt(10), fc="k"))
ax.set_ylim()
ax.scatter([2.0], [2.9e4], marker="x", color="black", s=20, linewidth=1, clip_on=False)
ax.add_artist(ada)
text_height = 1.2465
ax.add_artist(AnchoredText("100",
                    loc='upper left', frameon=False,
                    bbox_to_anchor=(-0.24, text_height),
                    bbox_transform=ax.transAxes))
ax.add_artist(AnchoredText("50",
                    loc='upper left', frameon=False,
                    bbox_to_anchor=(-0.12, text_height),
                    bbox_transform=ax.transAxes))
ax.add_artist(AnchoredText("2",
                    loc='upper left', frameon=False,
                    bbox_to_anchor=(-0.01, text_height),
                    bbox_transform=ax.transAxes))
ax.add_artist(AnchoredText("0",
                    loc='upper left', frameon=False,
                    bbox_to_anchor=(0.08, text_height),
                    bbox_transform=ax.transAxes))

ax.add_artist(AnchoredText("% informative data",
                    loc='upper left', frameon=False,
                    bbox_to_anchor=(-0.25, 1.325),
                    bbox_transform=ax.transAxes))


ax.set_xlabel(r'Number of observed systems $N$')
ax.set_ylabel(r'Trajectory length $T$')
ax.set_xticks(N_synths, labels=[str(i) for i in N_synths], minor=False) #
# ax.set_yticks(var['T'], labels=["{:3.2g}".format(i) for i in var['T']], minor=False) #
yticks = [  *list(np.array([1,2,3,4,5,6,7,8,9])*1e1),
            *list(np.array([1,2,3,4,5,6,7,8,9])*1e2),
            *list(np.array([1,2,3,4,5,6,7,8,9])*1e3),
            # *list(np.array([1,2,3,4,5,6,7,8,9])*1e-1),
            10000
]
ax.set_yticks(yticks, minor=False) #
ax.minorticks_off()
ax.grid(True, linestyle=':', linewidth=0.5, zorder=1, which='major')
# plt.show()

# plt.show()

plt.savefig('figures/heatmap_T_N.pgf', format='pgf')
plt.savefig('figures/heatmap_T_N.pdf', format='pdf', dpi=300)

plt.close()