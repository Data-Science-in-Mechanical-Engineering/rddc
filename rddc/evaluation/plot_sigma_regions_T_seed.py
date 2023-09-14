"""
Uses a manually picked set of 1d systems to synthesize a controller
to use it later in the sigma region plot
"""
import matplotlib as mpl
mpl.use('pgf')

import rddc.evaluation.tools as evaltools
c, params = evaltools.get_colors_and_plot_params('CDC_paper')  # specify font size etc.,
mpl.rcParams.update(params)
from rddc.run import main
import numpy as np
import os
import matplotlib.pyplot as plt
from rddc.run.settings.sigma_regions_T_seed import get_settings, get_variations
from rddc.tools.control_utils import draw_sigma_XU, plot_systems4synth


basepath = '.'
resolution = 150
settings = get_settings()
variations = get_variations()
settings['suffix'] = 'thesis'
systems = [
    [settings['A'], settings['B']],
]
Ts = variations['T']
seeds = variations['seed']
all_Ts = list()
all_seeds = list()
for T in Ts:
    for seed in seeds:
        all_Ts.append(T)
        all_seeds.append(seed)

(fig_width_in, fig_height_in) = evaltools.get_size(245, subplots=(1,1), fraction=1)
fig, axes = plt.subplots(figsize=(fig_width_in, fig_height_in), nrows=len(Ts), ncols=len(seeds))
fig.tight_layout(h_pad=-1.8, w_pad=-1.25)

for row in range(len(Ts)):
    settings['T'] = Ts[row]
    for col in range(len(seeds)):
        settings['seed'] = seeds[col]
        ax = axes[row,col]
        trajectories4synth = main.generate_trajectories4synth(settings, systems4synth=systems)
        # if row ==0 and col==0:
        for trajectory in trajectories4synth:
            sigma_marker = draw_sigma_XU(trajectory, settings, noise_criterion=1, limA=[0.6, 1.2], limB=[1.1, 1.7], resolution=resolution, colormap="Blues", hatch=[''], ax=ax)
        system_marker = plot_systems4synth(systems, color=evaltools.rgb_to_hex([0,0,50]), ax=ax)
        # ax.set_xlabel(r'$A$')
        ax.set_xticks([0.7, 1.1])
        ax.set_yticks([1.2, 1.6])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        for tick in ax.yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
        for tick in ax.xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
        if col > 0:
            ax.set_yticklabels([])
            for tick in ax.yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
        else:
            ax.set_ylabel(r'$M = {}$'.format(Ts[row]), rotation=0, ha='left')
            ax.yaxis.set_label_coords(-1.12, 0.35)
        if row < len(Ts)-1:
            ax.set_xticklabels([])
            for tick in ax.xaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
        ax.set_xlim([0.6, 1.2])
        ax.set_ylim([1.1, 1.7])
        # ax.axis('equal')
        ax.grid(True, linestyle=":", linewidth=0.5)

fig.legend(
    [sigma_marker, system_marker],
    [r'Uncertainty set $\Sigma_{\tau_i}$', r'True system $\theta_i$',],
    loc=[0.1,0.88],
    borderaxespad=0., ncol=2, columnspacing=2.0
)
fig.text(0.15, 0.47, r'A')
fig.text(0.555, 0.03, r'B')
# fig.align_ylabels(axes[:,0])
# ax.annotate("",
#             xy=(0.16, 0.9), xycoords='figure fraction',
#             xytext=(0.16, 0.54), textcoords='figure fraction',
#             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
# ax.annotate("",
#             xy=(0.16, 0.1), xycoords='figure fraction',
#             xytext=(0.16, 0.47), textcoords='figure fraction',
#             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

fig.subplots_adjust(bottom=0.1, top=0.85, left=0.2, right=.93)
plt.savefig(os.path.join(basepath, 'figures','sigma_regions_T_seed.pdf'))
plt.savefig(os.path.join(basepath, 'figures','sigma_regions_T_seed.pgf'), format='pgf', dpi=300)
plt.close()