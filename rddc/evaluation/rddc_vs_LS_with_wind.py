
import matplotlib as mpl
mpl.use('pgf')

import rddc.evaluation.tools as evaltools
c, params = evaltools.get_colors_and_plot_params('CDC_paper')  # specify font size etc.,
mpl.rcParams.update(params)
import matplotlib.pyplot as plt
from rddc.run.simulation import get_reference, get_absolute_trajectories
from rddc.run.settings.simulation import get_settings
import json
import os
import rddc.evaluation.tools as evaltools
import seaborn as sns

basepath = '.'

## GET SETTINGS AND SET APPEARENCE
settings = get_settings()
with open(os.path.join('rddc','tools','RWTHcolors.json')) as json_file:
        RWTHcolors = json.load(json_file)
colornames75 = ['blau75', 'gruen75', 'rot75', 'orange75', 'violett75']
colornames25 = ['blau25', 'gruen25', 'rot25', 'orange25', 'violett25']
colors75 = [evaltools.rgb01_to_hex(RWTHcolors[color]) for color in colornames75]
colors25 = [evaltools.rgb01_to_hex(RWTHcolors[color]) for color in colornames25]
grey = evaltools.rgb01_to_hex(RWTHcolors['schwarz50'])
black = evaltools.rgb01_to_hex(RWTHcolors['schwarz100'])
deep_sns_colors = [evaltools.rgb01_to_hex(color) for color in sns.color_palette("deep", n_colors=5)]

## GET TRAJECTORY DATA
seeds = [settings['seed']+i+settings['N_synth'] for i in range(settings['N_test'])]
seeds = [605, 606, 607, 608, 609]
rddc_paths = [  os.path.join('data', settings['name'], settings['suffix'], 
    'test_' + settings['testSettings']['traj'] + '_direct' + '_seed' + str(seed) + '_reference.npy'
    )
    for seed in seeds]
sysid_paths = [  os.path.join('data', settings['name'], settings['suffix'], 
    'test_' + settings['testSettings']['traj'] + '_indirect' + '_seed' + str(seed) + '_reference.npy'
    )
    for seed in seeds]
rddc_trajectories = get_absolute_trajectories(settings, rddc_paths)
sysid_trajectories = get_absolute_trajectories(settings, sysid_paths)
reference = get_reference(settings, rddc_paths[0])

(fig_width_in, fig_height_in) = evaltools.get_size(245, subplots=(1,1), fraction=1, ratio='golden')
fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in*1.1))
for trajId in range(len(sysid_trajectories)):
    x = sysid_trajectories[trajId][0,:]
    y = sysid_trajectories[trajId][1,:]
    ax.plot(x,y,'--', linewidth=0.8, color=grey, zorder=2)
sysIdLine, = ax.plot([],[],'--', linewidth=0.8, color=grey,)
for trajId in range(len(rddc_trajectories)):
    x = rddc_trajectories[trajId][0,:]
    y = rddc_trajectories[trajId][1,:]
    ax.plot(x,y,'-',linewidth=0.8, color=deep_sns_colors[trajId], zorder=2)
rddcLine, = ax.plot([],[],'-k',linewidth=0.8,)
ref_x = reference[0, :]
ref_y = reference[1, :]
refLine, = ax.plot(ref_x,ref_y,'--k',linewidth=1.5, zorder=2)
ax.axis([-0.05, 1.6, -0.05, 1.6])
ax.set_yticks([0.0, 0.5, 1.0, 1.5])
ax.set_box_aspect(1)
ax.grid(True, linewidth=0.5, zorder=1, linestyle=":")
fig.legend([sysIdLine, rddcLine, refLine], ['LS-LQR', 'RDDC', 'Reference'], loc='outside right upper')

fig.subplots_adjust(bottom=0.12, top=0.98, left=0.05, right=.7)
plt.savefig(os.path.join(basepath, 'figures','rddc_vs_ls.pdf'))
plt.savefig(os.path.join(basepath, 'figures','rddc_vs_ls.pgf'), format='pgf', dpi=300)
plt.close()