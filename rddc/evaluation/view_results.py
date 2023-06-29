import os
import numpy as np

path = os.path.join('data','dean_test','N_synth=85-sigma=1-T=500-bound=0.0005-assumedBound=0.001-seed=564-')

settings = np.load(os.path.join(path, 'settings.npy'), allow_pickle=True).item()
systems4synth = np.load(os.path.join(path, 'systems4synth.npy'), allow_pickle=True).item()['systems4synth']
trajectories4synth = np.load(os.path.join(path, 'trajectories4synth.npy'), allow_pickle=True).item()['trajectories4synth']
K = np.load(os.path.join(path, 'controller.npy'), allow_pickle=True).item()['controller']
systems4test = np.load(os.path.join(path, 'systems4test.npy'), allow_pickle=True).item()['systems4test']
results = np.load(os.path.join(path, 'results.npy'), allow_pickle=True).item()

print("Done! Place a breakpoint to see the variables")