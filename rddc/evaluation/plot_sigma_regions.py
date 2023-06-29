"""
Creates a plot of 1-D systems in parameter space,
including their uncertainties due to process noise
showing the stable region of a controller, used stabilizable systems
and a possible system which would the problem infeasible

run
```
python -m rddc.run.1d_sigma_regions
```
before using this script
"""

import matplotlib as mpl
mpl.use('pgf')

import rddc.evaluation.tools as evaltools
c, params = evaltools.get_colors_and_plot_params('CDC_paper')  # specify font size etc.,
mpl.rcParams.update(params)
from rddc.tools import control_utils
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

resolution = 300
# basepath = '/home/alex/projects/rddc/code_new/robust-data-driven-control'
basepath = '.'

path = os.path.join(basepath, 'data', 'dean_1d_sigma_regions', 'working')
systems = np.load(os.path.join(path, 'systems4synth.npy'), allow_pickle=True).item()['systems4synth']
trajectories = np.load(os.path.join(path, 'trajectories4synth.npy'), allow_pickle=True).item()['trajectories4synth']
K = np.load(os.path.join(path, 'controller.npy'), allow_pickle=True).item()['controller']
settings = np.load(os.path.join(path, 'settings.npy'), allow_pickle=True).item()
num_good_systems = len(systems)

path = os.path.join(basepath,'data', 'dean_1d_sigma_regions', 'extra')
systems_extra = np.load(os.path.join(path, 'systems4synth.npy'), allow_pickle=True).item()['systems4synth']
trajectories_extra = np.load(os.path.join(path, 'trajectories4synth.npy'), allow_pickle=True).item()['trajectories4synth']
K_extra = np.load(os.path.join(path, 'controller.npy'), allow_pickle=True).item()['controller']
# assert np.isnan(K_extra)
settings = np.load(os.path.join(path, 'settings.npy'), allow_pickle=True).item()


(fig_width_in, fig_height_in) = evaltools.get_size(245, subplots=(1,1), fraction=1)
fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in*1.2))

stable_region = control_utils.plot_stable_region(K, hatch='////')
i = 0
sigma_good = []
for trajectory in trajectories:
    sigma_good.append(control_utils.draw_sigma_XU(trajectory, settings, noise_criterion=1, limA=[0.5, 1.5], limB=[0, 2], resolution=resolution, colormap="Blues", hatch=['']))
    i += 1

# sigma_bad = control_utils.draw_sigma_XU(trajectories_extra[num_good_systems], settings, noise_criterion=1, limA=[0.4, 1.5], limB=[0, 2.0], resolution=resolution, colormap="Oranges", hatch='\\\\')

system_marker = control_utils.plot_systems4synth(systems, color=evaltools.rgb_to_hex([0,0,50]))
# system_marker = control_utils.plot_systems4synth(systems_extra, color=evaltools.rgb_to_hex([50,0,0]), idx_to_plot=[num_good_systems])

system_bound = ax.add_artist(Ellipse((1.0,1.05), 1.45, 0.7, angle=-70, fill=False, linestyle='--', linewidth=1, color='blue'))
ax.set_xlabel(r'$A$')
ax.set_ylabel(r'$B$')
ax.set_xlim([0, 2.0])
ax.set_ylim([0, 2.0])
# ax.hlines([0], xmin=[-10], xmax=[10], color='black')
# ax.vlines([0], ymin=[-10], ymax=[10], color='black')
ax.grid(True, linestyle=":", linewidth=0.5)
ax.legend(
    [stable_region, sigma_good[0], system_marker, system_bound],
    [r'Stable region $\Sigma_K$', r'Uncertainty set $\Sigma_{\tau_i}$', r'Samples $\theta_i$', r'Domain $\hat \Theta$'],
    loc=[0,1.05],
    borderaxespad=0., ncol=2, columnspacing=0.6
)

# plt.show()

fig.subplots_adjust(bottom=0.15, top=0.8, left=0.14, right=.95)
plt.savefig(os.path.join(basepath, 'figures','sigma_regions_1d.pdf'))
plt.savefig(os.path.join(basepath, 'figures','sigma_regions_1d.pgf'), format='pgf', dpi=300)
plt.close()