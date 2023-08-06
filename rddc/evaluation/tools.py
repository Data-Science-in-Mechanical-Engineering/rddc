import numpy as np
import os
from rddc.tools import files
from PIL import Image
import glob
import json
import colorsys

def rgb01_to_hex(rgb):
    r, g, b = [int(x * 255) for x in rgb]
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(rgb[0], rgb[1], rgb[2])

def get_scalars(settings, variables, variations_all, variations_fixed=dict(), variations_to_fold=list(), basepath='data'):
    """
    returns an array with requested variable names collected from all runs of variations
    some dimensions of variations can either be taken for one fixed value of variation variable,
    or folded (=averaged) along specified variable.
    @input: settings: dictionary with original settings
    @input: variables: list of strings with names of variables to be extracted
    @input: variations_all: dictionary with keys being names of variations variables
            and values being lists of their possible values.
    @input: variations_fixed: dictionary with variation variables that need to be fixed to a certain value
    @input: variations_to_fold: list of variation variable names that need to be averaged out for the output.

    @output: scalars_folded a dictionary with keys being names of desired variables
            and values being N-D arrays values over left variations.
    @output: variations_folded a dictionary specifying variations left after fixing and folding
    """
    variations_nonfixed = {name : value for name, value in variations_all.items() if not(name in variations_fixed)}
    variations_size = [len(_) for _ in variations_nonfixed.values()]
    variations_total = np.prod(variations_size)

    scalars_nonfixed = {name : np.zeros(variations_size) for name in variables}
    for run_id in range(variations_total):
        idx = np.unravel_index(run_id, variations_size)
        var_id = 0
        suffix = ''
        for name, values in variations_all.items():
            if name in variations_fixed:
                value = variations_fixed[name]
            else:
                value = values[idx[var_id]]
                var_id = var_id + 1
            suffix = suffix + files.get_suffix_part(name, value) + '-'
        data = dict()
        for variable in variables:
            path = os.path.join(basepath, settings['name'], suffix, files.get_variable_location(variable, settings))
            new_data = np.load(path, allow_pickle=True).item()
            data.update(new_data)
        for name in scalars_nonfixed:
            scalars_nonfixed[name][idx] = data[name]

    variations_folded = variations_nonfixed.copy()
    scalars_folded = scalars_nonfixed.copy()
    fraction_nonnans = scalars_nonfixed.copy()
    for name_variation in variations_to_fold:
        dim = list(variations_folded.keys()).index(name_variation)
        del variations_folded[name_variation]
        for name_variable in variables:
            # np.nanmean ignores the nans
            scalars_folded[name_variable] = np.nanmean(scalars_nonfixed[name_variable], axis=dim)
            fraction_nonnans[name_variable] = np.mean(1 - np.isnan(scalars_nonfixed[name_variable]), axis=dim)
    
    return scalars_folded, variations_folded, fraction_nonnans

def get_arrays(settings, variables, variations_all, variations_fixed=dict(), basepath='data'):
    """
    returns an array with requested variable names collected from all runs of variations
    some dimensions of variations can either be taken for one fixed value of variation variable,
    or folded along specified variable by averaging them out.
    @input: settings: dictionary with original settings
    @input: variables: list of strings with names of variables to be extracted
    @input: variations_all: dictionary with keys being names of variations variables
            and values being lists of their possible values.
    @input: variations_fixed: dictionary with variation variables that need to be fixed to a certain value
    @input: variations_to_fold: list of variation variable names that need to be averaged out for the output.
    @output: a dictionary with keys being names of desired variables
            and values being N-D arrays values over left variations.
    @output: a dictionary specifying variations left after fixing and folding
    """
    variations_nonfixed = {name : value for name, value in variations_all.items() if not(name in variations_fixed)}
    variations_size = [len(_) for _ in variations_nonfixed.values()]
    variations_total = np.prod(variations_size)

    arrays_nonfixed = {name : np.empty(variations_size, dtype=object) for name in variables}
    for run_id in range(variations_total):
        idx = np.unravel_index(run_id, variations_size)
        var_id = 0
        suffix = ''
        for name, values in variations_all.items():
            if name in variations_fixed:
                value = variations_fixed[name]
            else:
                value = values[idx[var_id]]
                var_id = var_id + 1
            suffix = suffix + files.get_suffix_part(name, value) + '-'
        data = dict()
        for variable in variables:
            path = os.path.join(basepath, settings['name'], suffix, files.get_variable_location(variable, settings))
            new_data = np.load(path, allow_pickle=True).item()
            data.update(new_data)
        for name in arrays_nonfixed:
            arrays_nonfixed[name][idx] = data[name]

    return arrays_nonfixed, variations_nonfixed

def removeImages(img_path):
    for f in glob.glob(img_path):
        os.remove(f)


def createGIF(img_path, save_path, delete_imgs=False, duration=200):
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    all_imgs = sorted(glob.glob(img_path))
    img, *imgs = [Image.open(f) for f in all_imgs]
    img.save(fp=save_path, format='GIF', append_images=imgs,
             save_all=True, duration=duration, loop=0)
    if delete_imgs:
        removeImages(img_path)


def get_colors_and_plot_params(purpose):
    path = '.'
    color_json = glob.glob(path + '/**/RWTHcolors.json', recursive=True)
    with open(color_json[0]) as json_file:
        c = json.load(json_file)

    if purpose in ['CDC_paper']:
        plot_params = {
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.texsystem': 'pdflatex',
        'pgf.rcfonts': True,
        'axes.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.labelsize': 9,
        "legend.fontsize": 9,
        "font.size": 9
        }
    else:
        plot_params = {}

    return c, plot_params


def get_size(width_pt, fraction=1, subplots=(1, 1), ratio='golden'):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    if ratio in ['golden']:
        proportion = (5 ** .5 - 1) / 2
    elif ratio in ['square']:
        proportion = 1

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * proportion * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def get_absolute_trajectory_experiment(path):
    abs_trajectory = np.load(os.path.join(path,'absolute_trajectory.npy'), allow_pickle=True).item()
    return abs_trajectory


def get_reference_experiment(path):
    ref_trajectory = np.load(os.path.join(path,'reference_trajectory.npy'), allow_pickle=True)
    return ref_trajectory


if __name__=="__main__":
    import rddc.run.settings.dean_1d_test as module
    settings = module.get_settings()
    variations = module.get_variations()

    arrays, var = get_scalars(
        settings=settings,
        variables=['controller'],
        variations_all=variations,
        variations_fixed={'bound': 1e-3, 'T':20},
        variations_to_fold=['sigma']
    )

    print(f"Scalar variable array: {arrays}\n")
    print(f"corresponds to the following variations: {var}")

    arrays, var = get_arrays(
        settings=settings,
        variables=['cost_distribution'],
        variations_all=variations,
        variations_fixed={'bound': 1e-2, 'T':10, 'sigma':0.01}
    )

    print(f"Array variable array: {arrays}\n")
    print(f"corresponds to the following variations: {var}")
