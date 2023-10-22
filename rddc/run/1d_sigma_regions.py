"""
Uses a manually picked set of 1d systems to synthesize a controller
to use it later in the sigma region plot
"""

from rddc.run import main
import numpy as np
from rddc.run.settings.dean_1d_sigma_regions import get_settings

def run_single(settings, systems):
    main.run(settings)
    trajectories4synth = main.generate_trajectories4synth(settings, systems4synth=systems)
    K = main.synthesize_controller(settings, trajectories4synth=trajectories4synth)

    main.save_necessary(settings,
        systems4synth=systems,
        trajectories4synth=trajectories4synth,
        K=K,
    )

    print(f"Systems used:\n{systems4synth}\n\n")
    print(f"controller created:\n{K}\n")

if __name__=='__main__':
    settings = get_settings()
    settings['suffix'] = 'working'
    systems4synth = [
        [np.atleast_2d(0.9), np.atleast_2d(1.4)],
        [np.atleast_2d(0.85), np.atleast_2d(1.0)],
        [np.atleast_2d(1.1), np.atleast_2d(0.68)],
        [np.atleast_2d(1.3), np.atleast_2d(0.55)],
    ]
    run_single(settings, systems4synth)

    settings['suffix'] = 'extra'
    systems4synth.append([np.atleast_2d(0.72), np.atleast_2d(1.48)])
    run_single(settings, systems4synth)
