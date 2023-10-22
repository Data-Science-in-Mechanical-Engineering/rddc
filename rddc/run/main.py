import numpy as np
import os
from rddc.tools import controller_synthesis, testing, trajectory, systems, files

# Since random number generator is created multiple times for different functions,
# this extraSeed avoids it being the same for these functions
extraSeed = np.arange(10)


def save_settings(settings):
    path = os.path.join('data', settings['name'], settings['suffix'])
    files.save_dict_npy(os.path.join(path, 'settings.npy'), settings)

def create_sys_dist(settings, save=False):
    path = os.path.join('data', settings['name'], settings['suffix'])

    # Create random number generator and system distribution
    sys_dist = systems.create_sys_dist(
        seed    = settings['seed'],
        A       = settings['A'],
        B       = settings['B'],
        sigma   = settings['sigma']
    )
    if save:
        files.save_dict_npy(os.path.join(path, 'sys_dist.npy'), {'sys_dist' : sys_dist})

    return sys_dist


def create_systems4synth(settings, sys_dist=None, save=False):
    # create the path and random number generator based on settings
    path = os.path.join('data', settings['name'], settings['suffix'])
    rnd = np.random.default_rng(settings['seed'] + extraSeed[0])

    if sys_dist is None:
        sys_dist = np.load(os.path.join(path,'sys_dist.npy'), allow_pickle=True).item()['sys_dist']

    systems4synth = systems.sample_MN(
        N_sys               = settings['N_synth'],
        trunc_threshold     = settings['trunc_threshold'],
        controllability_tol = settings['controllability_tol'],
        sys_dist            = sys_dist,
        rnd                 = rnd
    )

    if save:
        files.save_dict_npy(os.path.join(path, 'systems4synth.npy'), {'systems4synth': systems4synth})

    return systems4synth


def generate_trajectories4synth(settings, systems4synth=None, save=False):
    path = os.path.join('data', settings['name'], settings['suffix'])

    rnd = np.random.default_rng(settings['seed'] + extraSeed[1])

    if systems4synth is None:
        # extract saved systems for controller synthesis
        systems4synth = np.load(os.path.join(path,'systems4synth.npy'), allow_pickle=True).item()['systems4synth']

    # Generate trajectories with a preliminary controller
    trajectories4synth = trajectory.generate_trajectories_synth(
        systems             = systems4synth,
        N_traj_per_sys      = 1,
        T                   = settings['T'],
        noiseInfo           = settings,
        rnd                 = rnd,
        K                   = settings['K_prelim'],
        start               = settings['init_state'],
        inputNoiseAmplitude = settings['input_noise_amplitude']
    )

    if save:
        files.save_dict_npy(os.path.join(path, 'trajectories4synth.npy'), {'trajectories4synth': trajectories4synth})

    return trajectories4synth


def synthesize_controller(settings, trajectories4synth=None, save=False):
    path = os.path.join('data', settings['name'], settings['suffix'])

    if trajectories4synth is None:
        trajectories4synth = np.load(os.path.join(path,'trajectories4synth.npy'), allow_pickle=True).item()['trajectories4synth']

    func = getattr(controller_synthesis, settings['algorithm'])
    if len(trajectories4synth)>=1:
        K = func(
            trajectories    = trajectories4synth,
            noiseInfo       = settings,
            perfInfo        = settings,
            verbosity       = settings['output_verbosity'],
        )
    else: #no trajectories available
        K = np.nan
    if save:
        files.save_dict_npy(os.path.join(path, 'controller.npy'), {'controller': K})

    return K


def create_systems4test(settings, sys_dist=None, save=False):
    # create the path and random number generator based on settings
    path = os.path.join('data', settings['name'], settings['suffix'])
    rnd = np.random.default_rng(settings['seed'] + extraSeed[2])

    if sys_dist is None:
        sys_dist = np.load(os.path.join(path,'sys_dist.npy'), allow_pickle=True).item()['sys_dist']

    systems4test = systems.sample_MN(
        N_sys               = settings['N_test'],
        trunc_threshold     = settings['trunc_threshold'],
        controllability_tol = settings['controllability_tol'],
        sys_dist            = sys_dist,
        rnd                 = rnd
    )

    if save:
        files.save_dict_npy(os.path.join(path, 'systems4test.npy'), {'systems4test': systems4test})

    return systems4test


def test_closed_loop_stability(settings, systems4test=None, K=None, save=False):
    # create the path based on settings
    path = os.path.join('data', settings['name'], settings['suffix'])

    # extract the saved controller
    if K is None:
        controller = np.load(os.path.join(path,'controller.npy'), allow_pickle=True).item()
        K = controller['controller']

    # Stability testing for unknown systems
    try:
        results = np.load(os.path.join(path,'results.npy'), allow_pickle=True).item()
    except FileNotFoundError:
        results = dict()
    N_stable = None
    if K is not np.nan:
        if systems4test is None:
            systems4test = np.load(os.path.join(path,'systems4test.npy'), allow_pickle=True).item()['systems4test']

        N_stable = testing.test_stability(
            systems             = systems4test,
            K                   = K
        )
        # print(f"N_stable: {N_stable}\n")
    results.update({'N_stable': N_stable})

    if save:
        files.save_dict_npy(os.path.join(path, 'results.npy'), results)

    return N_stable


def test_closed_loop_performance(settings, systems4test=None, K=None, save=False):
    # create the path based on settings
    path = os.path.join('data', settings['name'], settings['suffix'])

    # extract the saved controller
    if K is None:
        K = np.load(os.path.join(path,'controller.npy'), allow_pickle=True).item()['controller']

    try:
        results = np.load(os.path.join(path,'results.npy'), allow_pickle=True).item()
    except FileNotFoundError:
        results = dict()
    costs = None
    if K is not np.nan:
        if systems4test is None:
            systems4test = np.load(os.path.join(path,'systems4test.npy'), allow_pickle=True).item()['systems4test']

        N_traj = 5
        T = 50
        costs = testing.test_performance_empiric(systems4test, K, T, settings, N_traj)
    results.update({'costs':costs})

    if save:
        files.save_dict_npy(os.path.join(path, 'results.npy'), results)

    return costs


def save_necessary(settings, sys_dist=None, systems4synth=None, trajectories4synth=None, K=None, systems4test=None, results=None):
    path = os.path.join('data', settings['name'], settings['suffix'])
    save_settings(settings)
    if sys_dist is not None:
        files.save_dict_npy(os.path.join(path, 'sys_dist.npy'), {'sys_dist' : sys_dist})
    if systems4synth is not None:
        files.save_dict_npy(os.path.join(path, 'systems4synth.npy'), {'systems4synth': systems4synth})
    if trajectories4synth is not None:
        files.save_dict_npy(os.path.join(path, 'trajectories4synth.npy'), {'trajectories4synth': trajectories4synth})
    if K is not None:
        files.save_dict_npy(os.path.join(path, 'controller.npy'), {'controller' : K})
    if systems4test is not None:
        files.save_dict_npy(os.path.join(path, 'systems4test.npy'), {'systems4test': systems4test})
    if results is not None:
        files.save_dict_npy(os.path.join(path, 'results.npy'), results)


def run_modular(settings):
    """
    here, intermediate results are saved immediately after being produced
    new functions take all the intermediate results from the output directory
    Therefore, individual functions can be switched off (via comments),
    given their output is already present in the output drectory
    """
    save_settings(settings)
    create_sys_dist(settings, save=True)
    create_systems4synth(settings, save=True)
    generate_trajectories4synth(settings, save=True)
    synthesize_controller(settings, save=True)
    create_systems4test(settings, save=True)
    test_closed_loop_stability(settings, save=True)
    test_closed_loop_performance(settings, save=True)


def run(settings, save=True):
    """
    here, the results are passed from one function to another to skip I/O and decrease run time
    Saving of function outputs happens in the end and can be disabled for individual outputs
    """
    sys_dist = create_sys_dist(settings)
    systems4synth = create_systems4synth(settings, sys_dist=sys_dist)
    trajectories4synth = generate_trajectories4synth(settings, systems4synth=systems4synth)
    K = synthesize_controller(settings, trajectories4synth=trajectories4synth)
    systems4test = create_systems4test(settings, sys_dist=sys_dist)
    N_stable = test_closed_loop_stability(settings, systems4test=systems4test, K=K)
    # costs = test_closed_loop_performance(settings, systems4test=systems4test, K=K)
    costs = None

    if save:
        results = {'N_stable' : N_stable, 'costs' : costs}
        # feel free to comment some entries out, if no saving is needed
        save_necessary(settings,
            sys_dist=sys_dist,
            systems4synth=systems4synth,
            trajectories4synth=trajectories4synth,
            K=K,
            systems4test=systems4test,
            results=results
        )


def run_variation(run_id, settings, variations, save=True):
    """
    Update the settings with a certain combination of
    variable variations and run

    'save' specifies whether the output files
    should be saved (overwritten)
    """
    settings_var = settings.copy()
    variations_size = [len(_) for _ in variations.values()]
    idx = np.unravel_index(run_id, variations_size)
    var_id = 0
    for name, values in variations.items():
        value = values[idx[var_id]]
        settings_var[name] = value
        settings_var['suffix'] = settings_var['suffix'] + files.get_suffix_part(name, value) + '-'
        var_id = var_id + 1
    run(settings_var, save=save)
    # main.run_modular(settings_var)
    #print(f"Done with job # {run_id}")