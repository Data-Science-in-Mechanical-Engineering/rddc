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

def get_trajectories(paths):
    trajectories = [np.load(path, allow_pickle=True).item()['state'] for path in paths]
    output_trajectories = list()
    for trajectory in trajectories:
        idx_nonzero = np.logical_and(np.abs(trajectory[0,:]) > 1e-6, np.abs(trajectory[1,:]) > 1e-6)
        trajectory = trajectory[:, idx_nonzero]
        output_trajectories.append(trajectory)
    return output_trajectories

ctrl_configurations = {
    'exp1'          : ['exp1_controller_rddc_nom',          10,     'exp_1'],
    'exp2'          : ['exp2_controller_rddc_N14_T200+',    10,     'exp_N'],
    's2r1'          : ['s2r1_controller_rddc',              10,     's2r_1'],
    's2r4'          : ['s2r4_controller_rddc',              10,     's2r_N'],
    's2r4_5deg'     : ['s2r4_controller_rddc_drpy5deg',     10,     's2r_N_n'],
    's2r5_0.5deg'   : ['s2r5_controller_rddc_drpy0.5deg',   50,     's2r_hf_n'],
    # 's2r5'          : ['s2r5_controller_rddc',              50],
}
weight_configuration = '220000'
# weight_configuration = '200020'
# weight_configuration = '000023'
# weight_configuration = '020003'
fan_configuration = 'x'
session_name = 'session2'
plot_file_name = 'exp_traj_'+session_name + '_' + weight_configuration + fan_configuration

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
trajectories = get_trajectories(abs_traj_paths)

ref_path = os.path.join(test_paths[0], 'reference_trajectory.npy')
ref = np.load(ref_path, allow_pickle=True)

(fig_width_in, fig_height_in) = evaltools.get_size(245, subplots=(1,1), fraction=1, ratio='golden')
fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in*1.15))
trajLines = list()
trajNames = [ctrl[2] for ctrl in ctrl_configurations.values()]
for trajId in range(len(trajectories)):
    x = trajectories[trajId][0,:]
    y = trajectories[trajId][1,:]
    trajLines.append(ax.plot(x,y,'-',linewidth=0.8, color=deep_sns_colors[trajId], zorder=2)[0])
    if outcomes[trajId]:
        ax.scatter([x[-1]], [y[-1]], marker='.', s=5, linewidths=1)
    else:
        ax.scatter([x[-1]], [y[-1]], marker='x', s=5, linewidths=1)
trajLine, = ax.plot([],[],'-k',linewidth=0.8,)
refLine, = ax.plot(ref[:, 0],ref[:, 1],'--k',linewidth=1.5, zorder=2)
ax.axis('equal')
ax.set(ylim = (-1.1, 2))
ax.set(xlim = (-1.05, 2))
ax.set_xticks([-1.0, 0.0, 1.0, 2.0])
ax.set_yticks([-1.0, 0.0, 1.0])
# ax.set_box_aspect(1)
ax.set(xlabel=('position x [m]'))
ax.set(ylabel=('position y [m]'))
ax.grid(True, linewidth=0.5, zorder=1, linestyle=":")
ax.legend(trajLines+[refLine], trajNames+['Reference'], loc='lower right')

fig.subplots_adjust(bottom=0.2, top=0.98, left=0.15, right=.95)
plt.savefig(os.path.join(basepath, 'figures', plot_file_name+'.pdf'))
plt.savefig(os.path.join(basepath, 'figures', plot_file_name+'.pgf'), format='pgf', dpi=300)
plt.close()