import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
import os
import xml.etree.ElementTree as et
from scipy.stats import multivariate_normal

def J_as_dict(J):
    return {'ixx':J[0,0], 'ixy':J[0,1], 'ixz':J[0,2], 'iyy':J[1,1], 'iyz':J[1,2], 'izz':J[2,2]}

def J_from_extra_mass(m, pos, form, size):
    """
    calculates the moment of inertia generated by adding an
    extra mass of specified form at a specified position to a drone
    """
    x,y,z = pos
    if form in ['cylinder']:
        radius, length = size
        J0 = np.diag([
            0.5 * m * radius**2,
            0.25 * m * radius**2 + 1/12 * m * length**2,
            0.25 * m * radius**2 + 1/12 * m * length**2,
        ])
    elif form in ['ball']:
        radius = size[0]
        J0 = np.diag([
            0.4 * m * radius**2,
            0.4 * m * radius**2,
            0.4 * m * radius**2
        ])
    elif form in ['box']:
        dx, dy, dz = size
        J0 = np.diag([
            1/12 * m * (dy**2 + dz**2),
            1/12 * m * (dx**2 + dz**2),
            1/12 * m * (dx**2 + dy**2),
        ])

    Jsteiner = m*np.array([
        [ y**2 + z**2,  -x*y,           -x*z         ],
        [-x*y,           x**2 + z**2,   -z*y         ],
        [-x*z,          -z*y,            x**2 + y**2 ]
    ])

    J = J0 + Jsteiner

    return J


def update_urdf_mass_and_inertia(URDFPATH, NEW_URDFPATH, extra_load):
    """
    extra_load is a dict with fields {mass:float, position:array_like, form:str, size:array_like}
    """
    tree = et.parse(URDFPATH)
    root = tree.getroot()

    properties = root.find("properties")
    thrust2weight = float(properties.get('thrust2weight'))
    if thrust2weight is None:
        print("thrust2Weight element not found in the properties.")
        raise ValueError

    base_link = root.find(".//link[@name='base_link']")
    if base_link is None:
        print("Base_link not found in the URDF file.")
        raise ValueError

    inertial = base_link.find('inertial')
    mass_element = inertial.find('mass')
    inertia_element = inertial.find('inertia')
    if mass_element is None or inertia_element is None:
        print("Mass or inertia element not found in the base_link.")
        raise ValueError

    center_of_mass_link = root.find(".//link[@name='center_of_mass_link']")
    com_inertial = center_of_mass_link.find('inertial')
    origin_element = com_inertial.find('origin')
    if origin_element is None:
        print("Origin element not found in the center_of_mass_link.")
        raise ValueError

    mass = float(mass_element.get('value'))
    original_mass = mass
    mass += extra_load['mass']
    mass_element.set('value', str(mass))

    dJ = J_as_dict(extra_load['J'])
    for axis in ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']:
        inertia_value = float(inertia_element.get(axis))
        inertia_value += dJ[axis]
        inertia_element.set(axis, str(inertia_value))

    com_xyz = np.array([float(x) for x in origin_element.get('xyz').split()])
    com_xyz = (com_xyz*original_mass + np.array(extra_load['position'])*extra_load['mass']) / mass
    com_xyz_str = ' '.join(str(x) for x in com_xyz)
    origin_element.set('xyz', com_xyz_str)

    properties.set('thrust2weight', str(thrust2weight * original_mass / mass))

    with open(NEW_URDFPATH, 'wb') as file:
        file.write(b'<?xml version="1.0" ?>\n')
        tree.write(file)

def get_load_sample_normal(rnd, mean, variance, form='ball', size=[0.0]):
    """
    samples extra mass to be added to a drone from a multivariate normal distribution
    @param rnd: random number generator instance
    @param mean: [mass, pos_x, pos_y, pos_z]
    @param variance: covariance matrix [mass, pos_x, pos_y, pos_z] x [mass, pos_x, pos_y, pos_z]
    @param form: form of the extra load
    @param size: size of the extra load

    @output extra_load: dictionary with the sampled parameters, ready to be used in settings
    """
    sample = multivariate_normal(mean, variance, seed=rnd)
    return {'mass':sample[0], 'position':np.array([sample[1], sample[2], sample[3]]), 'form':form, 'size':size}

def get_load_sample_box(rnd, mass_range, pos_size, form='ball', size=[0.0]):
    """
    samples extra mass to be added to a drone
    mass comes from a uniform distribution over the mass range
    position comes from a uniform distribution over the cube
    with the given size centered on the quadcopter's CoM
    @param rnd: random number generator instance
    @param mass_range: mass range, as array_like with two entries
    @param pos_size: half of the cube's edge length
    @param form: form of the extra load
    @param size: size of the extra load

    @output extra_load: dictionary with the sampled parameters, ready to be used in settings
    """
    mass = mass_range[0] + rnd.random()*(mass_range[1] - mass_range[0])
    pos = (2 * rnd.random(3) - 1) * pos_size
    return {'mass':mass, 'position':pos, 'form':form, 'size':size}

def get_load_sample_realistic(rnd, mass_range, displacement_planar=0.04, displacement_vert=0.01, form='cylinder', size=[0.005, 0.01]):
    """
    samples extra mass to be added to a drone
    mass comes from a uniform distribution over the mass range
    It can be positioned only along the wings (x - configuration)
    @param rnd: random number generator instance
    @param mass_range: mass range, as array_like with two entries
    @param displacement_planar: how far can the weight be displaced along the wings
    @param displacement_vert: how far willthe weight be displaced above or below the drone
    @param form: form of the extra load
    @param size: size of the extra load
    """
    mass = mass_range[0] + rnd.random()*(mass_range[1] - mass_range[0])
    pos = np.array([0, 0, 0])
    pos[0] = (2*rnd.random()-1) * displacement_planar
    if rnd.random()>0.5:
        pos[1] = pos[0]
    else:
        pos[1] = -pos[0]
    if rnd.random()>0.5:
        pos[2] = displacement_vert
    else:
        pos[2] = -displacement_vert
    return {'mass':mass, 'position':pos, 'form':form, 'size':size}


def plot_states_and_targets(logger, reference, pwm = False):
    """
    fork of the original logger's plot function
    also contains plots of the reference trajectories
    """
    #### Loop over colors and line styles ######################
    #plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) + cycler('linestyle', ['-', '--', ':', '-.'])))
    fig, axs = plt.subplots(6, 2)
    # t = np.arange(0, logger.timestamps.shape[1]/logger.LOGGING_FREQ_HZ, 1/logger.LOGGING_FREQ_HZ)
    t = logger.timestamps[0]
    # target_t = np.arange(reference['timestamps'].shape[1]) / reference['frequency']
    target_t = reference['timestamps'][0]

    #### Column ################################################
    col = 0

    #### XYZ ###################################################
    row = 0
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 0, :], label="drone_"+str(j))
        axs[row, col].plot(target_t, reference['targets'][j, 0, :], label="reference_"+str(j))
        axs[row, col].plot(target_t, reference['orig_targets'][j, 0, :], label="orig_reference_"+str(j))
    axs[row, col].set_xlabel('time')
    axs[row, col].set_ylabel('x (m)')

    row = 1
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 1, :], label="drone_"+str(j))
        axs[row, col].plot(target_t, reference['targets'][j, 1, :], label="reference_"+str(j))
        axs[row, col].plot(target_t, reference['orig_targets'][j, 1, :], label="orig_reference_"+str(j))
    axs[row, col].set_xlabel('time')
    axs[row, col].set_ylabel('y (m)')

    row = 2
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 2, :], label="drone_"+str(j))
        axs[row, col].plot(target_t, reference['targets'][j, 2, :], label="reference_"+str(j))
        axs[row, col].plot(target_t, reference['orig_targets'][j, 2, :], label="orig_reference_"+str(j))
    axs[row, col].set_xlabel('time')
    axs[row, col].set_ylabel('z (m)')

    #### RPY ###################################################
    row = 3
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 6, :], label="drone_"+str(j))
        axs[row, col].plot(target_t, reference['targets'][j, 6, :], label="reference_"+str(j))
        axs[row, col].plot(target_t, reference['orig_targets'][j, 6, :], label="orig_reference_"+str(j))
    axs[row, col].set_xlabel('time')
    axs[row, col].set_ylabel('r (rad)')
    row = 4
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 7, :], label="drone_"+str(j))
        axs[row, col].plot(target_t, reference['targets'][j, 7, :], label="reference_"+str(j))
        axs[row, col].plot(target_t, reference['orig_targets'][j, 7, :], label="orig_reference_"+str(j))
    axs[row, col].set_xlabel('time')
    axs[row, col].set_ylabel('p (rad)')
    row = 5
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 8, :], label="drone_"+str(j))
        axs[row, col].plot(target_t, reference['targets'][j, 8, :], label="reference_"+str(j))
        axs[row, col].plot(target_t, reference['orig_targets'][j, 8, :], label="orig_reference_"+str(j))
    axs[row, col].set_xlabel('time')
    axs[row, col].set_ylabel('y (rad)')

    # #### Ang Vel ###############################################
    # row = 6
    # for j in range(logger.NUM_DRONES):
    #     axs[row, col].plot(t, logger.states[j, 9, :], label="drone_"+str(j))
    #     axs[row, col].plot(t, reference['targets'][j, 9, :], label="reference_"+str(j))
    #     axs[row, col].plot(t, reference['orig_targets'][j, 9, :], label="orig_reference_"+str(j))
    # axs[row, col].set_xlabel('time')
    # axs[row, col].set_ylabel('wx')
    # row = 7
    # for j in range(logger.NUM_DRONES):
    #     axs[row, col].plot(t, logger.states[j, 10, :], label="drone_"+str(j))
    #     axs[row, col].plot(t, reference['targets'][j, 10, :], label="reference_"+str(j))
    #     axs[row, col].plot(t, reference['orig_targets'][j, 10, :], label="orig_reference_"+str(j))
    # axs[row, col].set_xlabel('time')
    # axs[row, col].set_ylabel('wy')
    # row = 8
    # for j in range(logger.NUM_DRONES):
    #     axs[row, col].plot(t, logger.states[j, 11, :], label="drone_"+str(j))
    #     axs[row, col].plot(t, reference['targets'][j, 11, :], label="reference_"+str(j))
    #     axs[row, col].plot(t, reference['orig_targets'][j, 11, :], label="orig_reference_"+str(j))
    # axs[row, col].set_xlabel('time')
    # axs[row, col].set_ylabel('wz')

    #### Time ##################################################
    # row = 9
    # axs[row, col].plot(t, t, label="time")
    # axs[row, col].set_xlabel('time')
    # axs[row, col].set_ylabel('time')

    #### Column ################################################
    col = 1

    #### Velocity ##############################################
    row = 0
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 3, :], label="drone_"+str(j))
        axs[row, col].plot(target_t, reference['targets'][j, 3, :], label="reference_"+str(j))
        axs[row, col].plot(target_t, reference['orig_targets'][j, 3, :], label="orig_reference_"+str(j))
    axs[row, col].set_xlabel('time')
    axs[row, col].set_ylabel('vx (m/s)')
    row = 1
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 4, :], label="drone_"+str(j))
        axs[row, col].plot(target_t, reference['targets'][j, 4, :], label="reference_"+str(j))
        axs[row, col].plot(target_t, reference['orig_targets'][j, 4, :], label="orig_reference_"+str(j))
    axs[row, col].set_xlabel('time')
    axs[row, col].set_ylabel('vy (m/s)')
    row = 2
    for j in range(logger.NUM_DRONES):
        axs[row, col].plot(t, logger.states[j, 5, :], label="drone_"+str(j))
        axs[row, col].plot(target_t, reference['targets'][j, 5, :], label="reference_"+str(j))
        axs[row, col].plot(target_t, reference['orig_targets'][j, 5, :], label="orig_reference_"+str(j))
    axs[row, col].set_xlabel('time')
    axs[row, col].set_ylabel('vz (m/s)')

    #### RPY Rates #############################################
    row = 3
    for j in range(logger.NUM_DRONES):
        rdot = np.hstack([0, (logger.states[j, 6, 1:] - logger.states[j, 6, 0:-1]) * logger.LOGGING_FREQ_HZ ])
        # axs[row, col].plot(t, logger.states[j, 10, :], label="drone_"+str(j))
        axs[row, col].plot(t, rdot, label="drone_"+str(j))
        axs[row, col].plot(target_t, reference['targets'][j, 9, :], label="reference_"+str(j))
        axs[row, col].plot(target_t, reference['orig_targets'][j, 9, :], label="orig_reference_"+str(j))
    axs[row, col].set_xlabel('time')
    axs[row, col].set_ylabel('rdot (rad/s)')
    row = 4
    for j in range(logger.NUM_DRONES):
        pdot = np.hstack([0, (logger.states[j, 7, 1:] - logger.states[j, 7, 0:-1]) * logger.LOGGING_FREQ_HZ ])
        # axs[row, col].plot(t, logger.states[j, 11, :], label="drone_"+str(j))
        axs[row, col].plot(t, pdot, label="drone_"+str(j))
        axs[row, col].plot(target_t, reference['targets'][j, 10, :], label="reference_"+str(j))
        axs[row, col].plot(target_t, reference['orig_targets'][j, 10, :], label="orig_reference_"+str(j))
    axs[row, col].set_xlabel('time')
    axs[row, col].set_ylabel('pdot (rad/s)')
    row = 5
    for j in range(logger.NUM_DRONES):
        ydot = np.hstack([0, (logger.states[j, 8, 1:] - logger.states[j, 8, 0:-1]) * logger.LOGGING_FREQ_HZ ])
        # axs[row, col].plot(t, logger.states[j, 12, :], label="drone_"+str(j))
        axs[row, col].plot(t, ydot, label="drone_"+str(j))
        axs[row, col].plot(target_t, reference['targets'][j, 11, :], label="reference_"+str(j))
        axs[row, col].plot(target_t, reference['orig_targets'][j, 11, :], label="orig_reference_"+str(j))
    axs[row, col].set_xlabel('time')
    axs[row, col].set_ylabel('ydot (rad/s)')

    ### This IF converts RPM into PWM for all drones ###########
    #### except drone_0 (only used in examples/compare.py) #####
    # for j in range(logger.NUM_DRONES):
    #     for i in range(12,16):
    #         if pwm and j > 0:
    #             logger.states[j, i, :] = (logger.states[j, i, :] - 4070.3) / 0.2685

    #### RPMs ##################################################
    # row = 6
    # for j in range(logger.NUM_DRONES):
    #     axs[row, col].plot(t, logger.states[j, 12, :], label="drone_"+str(j))
    # axs[row, col].set_xlabel('time')
    # if pwm:
    #     axs[row, col].set_ylabel('PWM0')
    # else:
    #     axs[row, col].set_ylabel('RPM0')
    # row = 7
    # for j in range(logger.NUM_DRONES):
    #     axs[row, col].plot(t, logger.states[j, 13, :], label="drone_"+str(j))
    # axs[row, col].set_xlabel('time')
    # if pwm:
    #     axs[row, col].set_ylabel('PWM1')
    # else:
    #     axs[row, col].set_ylabel('RPM1')
    # row = 8
    # for j in range(logger.NUM_DRONES):
    #     axs[row, col].plot(t, logger.states[j, 14, :], label="drone_"+str(j))
    # axs[row, col].set_xlabel('time')
    # if pwm:
    #     axs[row, col].set_ylabel('PWM2')
    # else:
    #     axs[row, col].set_ylabel('RPM2')
    # row = 9
    # for j in range(logger.NUM_DRONES):
    #     axs[row, col].plot(t, logger.states[j, 15, :], label="drone_"+str(j))
    # axs[row, col].set_xlabel('time')
    # if pwm:
    #     axs[row, col].set_ylabel('PWM3')
    # else:
    #     axs[row, col].set_ylabel('RPM3')

    #### Drawing options #######################################
    for i in range (6):
        for j in range (2):
            axs[i, j].grid(True)
            axs[i, j].legend(loc='upper right',
                        frameon=True
                        )
    fig.subplots_adjust(left=0.06,
                        bottom=0.05,
                        right=0.99,
                        top=0.98,
                        wspace=0.15,
                        hspace=0.0
                        )
    if logger.COLAB: 
        plt.savefig(os.path.join('results', 'output_figure.png'))
    else:
        plt.show()