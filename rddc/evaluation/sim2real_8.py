
import matplotlib as mpl
mpl.use('pgf')

import rddc.evaluation.tools as evaltools
c, params = evaltools.get_colors_and_plot_params('CDC_paper')  # specify font size etc.,
mpl.rcParams.update(params)
import matplotlib.pyplot as plt
from rddc.evaluation.tools import get_absolute_trajectory_experiment, get_reference_experiment
from rddc.run.settings.simulation_rddc import get_settings as get_settings_rddc
from rddc.run.settings.simulation_ceddc import get_settings as get_settings_ceddc
import json
import os
import rddc.evaluation.tools as evaltools
import seaborn as sns
from rddc.tools.files import get_simulation_trajectory_path

basepath = '.'

## GET SETTINGS AND SET APPEARENCE
# settings_exp = get_settings_rddc()
# settings_ceddc = get_settings_ceddc()
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
case_names = [
    '8_011000_10Hz_0.0_controller_rddc_sim_like_exp_10Hz_15sys_w0.002',
    '8_010001_10Hz_0.0_controller_rddc_sim_like_exp_10Hz_15sys_w0.002',
    '8_012001_10Hz_0.0_controller_rddc_sim_like_exp_10Hz_15sys_w0.002',
    '8_000002_10Hz_0.0_controller_rddc_sim_like_exp_10Hz_15sys_w0.002', # crash
    '8_100000_10Hz_0.0_controller_rddc_sim_like_exp_10Hz_15sys_w0.002',
    '8_103010__10Hz_0.0_controller_rddc_sim_like_exp_10Hz_15sys_w0.002',
    '8_003012__10Hz_0.0_controller_rddc_sim_like_exp_10Hz_15sys_w0.002',
    '8_003010_10Hz_0.0_controller_rddc_sim_like_exp_10Hz_15sys_w0.002',
    '8_000011_10Hz_0.0_controller_rddc_sim_like_exp_10Hz_15sys_w0.002',
    '8_102010_10Hz_0.0_controller_rddc_sim_like_exp_10Hz_15sys_w0.002',
    '8_000000_10Hz_0.0_controller_rddc_sim_like_exp_10Hz_15sys_w0.002',
]
case_paths = [os.path.join(basepath, 'data', 'experiment', 'experiment_recording', case_name) for case_name in case_names]
trajectories = [get_absolute_trajectory_experiment(path) for path in case_paths]
reference = get_reference_experiment(case_paths[1])

(fig_width_in, fig_height_in) = evaltools.get_size(245, subplots=(1,1), fraction=1, ratio='golden')
fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in*1.05))
for trajId in range(len(trajectories)):
    x = trajectories[trajId]['state'][0,:]
    y = trajectories[trajId]['state'][1,:]
    ax.plot(x,y,'-', linewidth=0.8, color=deep_sns_colors[trajId], zorder=2)
trajLine, = ax.plot([],[],'-', linewidth=0.8, color=grey,)
ref_x = reference[:, 0]
ref_y = reference[:, 1]
refLine, = ax.plot(ref_x,ref_y,'--k',linewidth=1.5, zorder=2)
ax.axis('equal')
ax.set(ylim = (-0.7, 1.0))
ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax.set_yticks([-0.5, 0.0, 0.5, 1.0])
# ax.set_box_aspect(1)
ax.set(xlabel=('position x [m]'))
ax.set(ylabel=('position y [m]'))
ax.grid(True, linewidth=0.5, zorder=1, linestyle=":")
ax.legend([trajLine, refLine], ['sim2real', 'Reference'], loc='upper center')

fig.subplots_adjust(bottom=0.2, top=0.95, left=0.2, right=0.95)
plt.savefig(os.path.join(basepath, 'figures','sim2real_8.pdf'))
plt.savefig(os.path.join(basepath, 'figures','sim2real_8.pgf'), format='pgf', dpi=300)
plt.close()