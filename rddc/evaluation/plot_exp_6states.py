import matplotlib as mpl
mpl.use('pgf')

import rddc.evaluation.tools as evaltools
c, params = evaltools.get_colors_and_plot_params('CDC_paper')  # specify font size etc.,
mpl.rcParams.update(params)
import numpy as np
import matplotlib.pyplot as plt
from rddc.experiment.dmitrii_drones.settings_test import get_settings
import json
import os
import rddc.evaluation.tools as evaltools
import seaborn as sns

def get_time_series(paths):
    values = [np.load(path, allow_pickle=True).item()['state'] for path in paths]
    times = [np.load(path, allow_pickle=True).item()['time'] for path in paths]
    output_values = list()
    output_times = list()
    for (value, time) in zip(values, times):
        idx_nonzero = np.logical_and(np.abs(value[0,:]) > 1e-6, np.abs(value[1,:]) > 1e-6)
        time = time[idx_nonzero]
        value = value[:, idx_nonzero]
        output_values.append(value)
        output_times.append(time)
    return output_values, output_times

ctrl_configurations = {
    'exp1'          : ['exp1_controller_rddc_nom',          10],
    'exp2'          : ['exp2_controller_rddc_N14_T200+',    10],
    's2r1'          : ['s2r1_controller_rddc',              10],
    's2r4'          : ['s2r4_controller_rddc',              10],
    's2r4_5deg'     : ['s2r4_controller_rddc_drpy5deg',     10],
    's2r5_0.5deg'   : ['s2r5_controller_rddc_drpy0.5deg',   50],
    # 's2r5'          : ['s2r5_controller_rddc',              50],
}
# weight_configuration = '220000'
weight_configuration = '200020'
# weight_configuration = '000023'
# weight_configuration = '020003'
fan_configuration = 'x'
session_name = 'session2'
plot_file_name = 'exp_6states_'+session_name + '_' + weight_configuration + fan_configuration

def get_path_and_outcome(base, traj, input_noise, weight_conf, fan_conf, ctrl):
    raw_path =  base + '/' + \
                traj + '_' + \
                weight_conf + fan_conf + '_' + \
                str(ctrl_configurations[ctrl][1]) + 'Hz_' + \
                str(input_noise) + '_' + \
                ctrl_configurations[ctrl][0]
    if os.path.isdir(raw_path + '_crash'):
        return raw_path + '_crash', False
    elif os.path.isdir(raw_path + '_pass'):
        return raw_path + '_pass', True

basepath = os.getcwd()
settings = get_settings()
with open(os.path.join('rddc','tools','RWTHcolors.json')) as json_file:
        RWTHcolors = json.load(json_file)
# colornames75 = ['blau75', 'gruen75', 'rot75', 'orange75', 'violett75']
# colornames25 = ['blau25', 'gruen25', 'rot25', 'orange25', 'violett25']
# colors75 = [evaltools.rgb01_to_hex(RWTHcolors[color]) for color in colornames75]
# colors25 = [evaltools.rgb01_to_hex(RWTHcolors[color]) for color in colornames25]
grey = evaltools.rgb01_to_hex(RWTHcolors['schwarz50'])
black = evaltools.rgb01_to_hex(RWTHcolors['schwarz100'])
deep_sns_colors = [evaltools.rgb01_to_hex(color) for color in sns.color_palette("deep", n_colors=15)]

test_paths = [
    get_path_and_outcome(basepath + '/data/experiment/'+session_name+'/','line', 0.0, weight_configuration, fan_configuration, ctrl)[0]
    for ctrl in ctrl_configurations
]
outcomes = [
    get_path_and_outcome(basepath + '/data/experiment/'+session_name+'/','line', 0.0, weight_configuration, fan_configuration, ctrl)[1]
    for ctrl in ctrl_configurations
]
abs_traj_paths = [os.path.join(test_path, 'absolute_trajectory.npy') for test_path in test_paths]
states, times = get_time_series(abs_traj_paths)

ref_path = os.path.join(test_paths[0], 'reference_trajectory.npy')
ref = np.load(ref_path, allow_pickle=True)

(fig_width_in, fig_height_in) = evaltools.get_size(245, subplots=(1,1), fraction=1, ratio='golden')
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6, figsize=(fig_width_in, fig_height_in*2.3), sharex=True)
for trajId in range(len(states)):
    x = states[trajId][0,:]
    y = states[trajId][1,:]
    vx = states[trajId][3,:]
    vy = states[trajId][4,:]
    r = states[trajId][6,:]*180/np.pi
    p = states[trajId][7,:]*180/np.pi
    t = times[trajId]
    ax1.plot(t,x,'-',linewidth=0.5, color=deep_sns_colors[trajId], zorder=2)
    ax2.plot(t,y,'-',linewidth=0.5, color=deep_sns_colors[trajId], zorder=2)
    ax3.plot(t,vx,'-',linewidth=0.5, color=deep_sns_colors[trajId], zorder=2)
    ax4.plot(t,vy,'-',linewidth=0.5, color=deep_sns_colors[trajId], zorder=2)
    ax5.plot(t,r,'-',linewidth=0.5, color=deep_sns_colors[trajId], zorder=2)
    ax6.plot(t,p,'-',linewidth=0.5, color=deep_sns_colors[trajId], zorder=2)
    if outcomes[trajId]:
        ax1.scatter([t[-1]], [x[-1]], marker='.', s=5, linewidths=1, color=deep_sns_colors[trajId])
        ax2.scatter([t[-1]], [y[-1]], marker='.', s=5, linewidths=1, color=deep_sns_colors[trajId])
        ax3.scatter([t[-1]], [vx[-1]], marker='.', s=5, linewidths=1, color=deep_sns_colors[trajId])
        ax4.scatter([t[-1]], [vy[-1]], marker='.', s=5, linewidths=1, color=deep_sns_colors[trajId])
        ax5.scatter([t[-1]], [r[-1]], marker='.', s=5, linewidths=1, color=deep_sns_colors[trajId])
        ax6.scatter([t[-1]], [p[-1]], marker='.', s=5, linewidths=1, color=deep_sns_colors[trajId])
    else:
        ax1.scatter([t[-1]], [x[-1]], marker='x', s=5, linewidths=1, color=deep_sns_colors[trajId])
        ax2.scatter([t[-1]], [y[-1]], marker='x', s=5, linewidths=1, color=deep_sns_colors[trajId])
        ax3.scatter([t[-1]], [vx[-1]], marker='x', s=5, linewidths=1, color=deep_sns_colors[trajId])
        ax4.scatter([t[-1]], [vy[-1]], marker='x', s=5, linewidths=1, color=deep_sns_colors[trajId])
        ax5.scatter([t[-1]], [r[-1]], marker='x', s=5, linewidths=1, color=deep_sns_colors[trajId])
        ax6.scatter([t[-1]], [p[-1]], marker='x', s=5, linewidths=1, color=deep_sns_colors[trajId])
# trajLine, = ax.plot([],[],'-k',linewidth=0.8,)
# refLine, = ax.plot(ref[:, 0],ref[:, 1],'--k',linewidth=1.5, zorder=2)
# ax.axis('equal')
ax1.set(ylim = (-1.9, 1.9))
ax2.set(ylim = (-1.9, 1.9))
ax3.set(ylim = (-2.5, 2.5))
ax4.set(ylim = (-2.5, 2.5))
ax5.set(ylim = (-12.5, 12.5))
ax6.set(ylim = (-12.5, 12.5))
# ax.set(xlim = (-1.05, 2))
# ax.set_xticks([-1.0, 0.0, 1.0, 2.0])
ax1.set_yticks([-1.0, 0.0, 1.0])
ax2.set_yticks([-1.0, 0.0, 1.0])
ax3.set_yticks([-2.0, -1.0, 0.0, 1.0, 2.0])
ax4.set_yticks([-2.0, -1.0, 0.0, 1.0, 2.0])
ax5.set_yticks([-10.0, -5.0, 0.0, 5.0, 10.0])
ax6.set_yticks([-10.0, -5.0, 0.0, 5.0, 10.0])
# ax.set_box_aspect(1)
# ax1.set(xlabel=('time [s]'))
ax6.set(xlabel=('time [s]'))
ax1.set(ylabel=('pos x [m]'))
ax2.set(ylabel=('pos y [m]'))
ax3.set(ylabel=('vel x [m/s]'))
ax4.set(ylabel=('vel y [m/s]'))
ax5.set(ylabel=('roll [deg]'))
ax6.set(ylabel=('pitch [deg]'))
ax1.grid(True, linewidth=0.5, zorder=1, linestyle=":")
ax2.grid(True, linewidth=0.5, zorder=1, linestyle=":")
ax3.grid(True, linewidth=0.5, zorder=1, linestyle=":")
ax4.grid(True, linewidth=0.5, zorder=1, linestyle=":")
ax5.grid(True, linewidth=0.5, zorder=1, linestyle=":")
ax6.grid(True, linewidth=0.5, zorder=1, linestyle=":")
# ax.legend([trajLine, refLine], ['RDDC', 'Reference'], loc='right')

fig.subplots_adjust(bottom=0.1, top=0.98, left=0.2, right=.95)
plt.savefig(os.path.join(basepath, 'figures', plot_file_name+'.pdf'))
plt.savefig(os.path.join(basepath, 'figures', plot_file_name+'.pgf'), format='pgf', dpi=300)
plt.close()