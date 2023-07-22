import numpy as np
#### Target trajectories for an experiment ##########################################

#### Definitions for different kinds of lemniscate ##################################

def get_trajectory_gerono(height, radius, num_points, period):
    """
    Lemniscate as a Lissagous figure in xy plane at a given height
    x = R   * cos(  phi)
    y = R/2 * sin(2 phi)
    """
    pos_soll=np.zeros((num_points,3))
    vel_soll=np.zeros((num_points,3))
    for i in range(num_points):
        t = i/num_points * period
        phi = t/period * 2*np.pi + np.pi/2
        pos_soll[i] = [ radius   * np.cos(phi),\
                        radius/2 * np.sin(2*phi),\
                        height]
        vel_soll[i] = [-2*np.pi/period * radius * np.sin(phi),\
                        2*np.pi/period * radius * np.cos(2*phi),\
                        0.0]
    return pos_soll, vel_soll

def get_trajectory_hover(height, num_points):
    """
    Hovering at a constant height
    """
    pos_soll=np.zeros((num_points,3))
    vel_soll=np.zeros((num_points,3))
    for i in range(num_points):
        pos_soll[i] = [ 0.0, 0.0, height]
        vel_soll[i] = [ 0.0, 0.0, 0.0]
    return pos_soll, vel_soll

def get_trajectory_line(start, finish, num_points, duration):
    """
    going fron start point to finish point at constant speed
    """
    pos_soll=np.zeros((num_points,3))
    vel_soll=np.zeros((num_points,3))
    for i in range(num_points):
        progress = i / (num_points-1)
        pos_soll[i] = start * (1-progress) + finish * progress
        vel_soll[i] = (finish - start) / duration
    return pos_soll, vel_soll