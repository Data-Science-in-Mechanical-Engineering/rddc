
import matplotlib as mpl
mpl.use('pgf')

import rddc.evaluation.tools as evaltools
c, params = evaltools.get_colors_and_plot_params('CDC_paper')  # specify font size etc.,
mpl.rcParams.update(params)
import matplotlib.pyplot as plt
from rddc.run.simulation import get_reference, get_absolute_trajectories
# from rddc.run.settings.simulation_sof import get_settings as get_settings_sof
# from rddc.run.settings.simulation_ceddc import get_settings as get_settings_ceddc
import json
import os
import rddc.evaluation.tools as evaltools
import seaborn as sns
from rddc.tools.files import get_simulation_trajectory_path
import argparse
import importlib

basepath = '.'

def parse_arguments():
    parser = argparse.ArgumentParser(description='Trajectory plot for simulated drones')
    parser.add_argument('--mode',       required=True,  type=str,   help='Mode to run the code in', metavar='', choices=['sof', 'ceddc'])
    args = parser.parse_args()
    return args

args = parse_arguments()
module = importlib.import_module('rddc.run.settings.simulation' + '_' + args.mode)

settings = module.get_settings()
filename = 'rddc_quad_demo_'+args.mode
if args.mode in ['ceddc']:
    name = 'sim\_1'
else:
    name = 'sim\_N'

## GET SETTINGS AND SET APPEARENCE
with open(os.path.join('rddc','tools','RWTHcolors.json')) as json_file:
        RWTHcolors = json.load(json_file)
colornames75 = ['blau75', 'gruen75', 'rot75', 'orange75', 'violett75']
colornames25 = ['blau25', 'gruen25', 'rot25', 'orange25', 'violett25']
colors75 = [evaltools.rgb01_to_hex(RWTHcolors[color]) for color in colornames75]
colors25 = [evaltools.rgb01_to_hex(RWTHcolors[color]) for color in colornames25]
grey = evaltools.rgb01_to_hex(RWTHcolors['schwarz50'])
black = evaltools.rgb01_to_hex(RWTHcolors['schwarz100'])
deep_sns_colors = [evaltools.rgb01_to_hex(color) for color in sns.color_palette("deep", n_colors=15)]

## GET TRAJECTORY DATA
if 'traj_filename' in settings['testSettings']:
    rddc_paths = [settings['testSettings']['traj_filename']+'_reference.npy']
else:
    rddc_paths = [get_simulation_trajectory_path(settings, 'test', 'rddc', reference=True)+'.npy']
rddc_trajectories = get_absolute_trajectories(settings, rddc_paths)
reference = get_reference(settings, rddc_paths[0])

(fig_width_in, fig_height_in) = evaltools.get_size(180, subplots=(1,1), fraction=1, ratio='golden')
fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in*1.1))
for trajId in range(5): #choosing 5 systems that are definetely not used for training
    x = rddc_trajectories[50+trajId][0,:]
    y = rddc_trajectories[50+trajId][1,:]
    ax.plot(x,y,'-',linewidth=0.8, color=deep_sns_colors[trajId], zorder=2)
rddcLine, = ax.plot([],[],'-k',linewidth=0.8,)
ref_x = reference[0, :]
ref_y = reference[1, :]
refLine, = ax.plot(ref_x,ref_y,'--k',linewidth=1.5, zorder=2)
ax.axis('equal')
ax.set(ylim = (-0.1, 1.4))
ax.set(xlim = (-0.05, 2.2))
ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
ax.set_yticks([0.0, 0.5, 1.0])
# ax.set_box_aspect(1)
ax.set(xlabel=('position x [m]'))
ax.set(ylabel=('position y [m]'))
ax.grid(True, linewidth=0.5, zorder=1, linestyle=":")
ax.legend([rddcLine, refLine], [name, 'Reference'], loc='upper left')

fig.subplots_adjust(bottom=0.24, top=0.98, left=0.19, right=.95)
plt.savefig(os.path.join(basepath, 'figures',filename+'.pdf'))
plt.savefig(os.path.join(basepath, 'figures',filename+'.pgf'), format='pgf', dpi=300)
plt.close()