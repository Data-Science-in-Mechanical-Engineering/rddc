# Robust Direct Data-Driven Control for Probabilistic Systems

This repository contains the supplementary code for the paper "Robust Direct Data-Driven Control for Probabilistic Systems".

A preprint of the paper can be found on [arXiv](https://arxiv.org/abs/2306.16973).

## Abstract

We propose a data-driven control method for systems with aleatoric uncertainty, such as robot fleets with variations between agents. Our method leverages shared trajectory data to increase the robustness of the designed controller and thus facilitate transfer to new variations without the need for prior parameter and uncertainty estimation. In contrast to existing work on experience transfer for performance, our approach focuses on robustness and uses data collected from multiple realizations to guarantee generalization to unseen ones. Our method is based on scenario optimization combined with recent formulations for direct data-driven control. We derive upper bounds on the minimal amount of data required to provably achieve quadratic stability for probabilistic systems with aleatoric uncertainty and demonstrate the benefits of our data-driven method through a numerical example. We find that the learned controllers generalize well to high variations in the dynamics even when based on only a few short open-loop trajectories. Robust experience transfer enables the design of safe and robust controllers that work ``out of the box'' without additional learning during deployment.

### Citation

If you find our code or paper useful, please consider citing the current preprint

```bibtex
@misc{vonrohr2024robustdirectdatadrivencontrol,
      title={Robust Direct Data-Driven Control for Probabilistic Systems}, 
      author={Alexander von Rohr and Dmitrii Likhachev and Sebastian Trimpe},
      year={2024},
      eprint={2306.16973},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2306.16973}, 
}
```

## How to use the supplementary code

### Software Requirements

* python 3.8
* pipenv
* mosek (preferred) / cvxopt
* latex (e.g. `texlive-latex-recommended` + `texlive-latex-extra`)

### Installation

* Install the basic software requirements
* Clone this repository
* Install the dependencies using pipenv:

```bash
pipenv install
```

### Running the environment

* Navigate to the root of the installation
* activate the virtual environment

```bash
pipenv shell
```

if this command doesn't work, try `python -m pipenv shell`

* Run modules from the root of the installation using `python -m`, e.g.

```bash
python -m rddc.run.1d_sigma_regions
```

### Reproduction of the results

After each simulation / variation run, the data will be stored in `data/`. The figures are always generated from that data using separate scripts and stored in `figures/`.

Make sure to start every attempt at reproducing the results with an empty corresponding data folder.

#### Fig. 2 (Illustrative example of scenario optimization for a fleet of one-dimensional systems)

```bash
python -m rddc.run.1d_sigma_regions
python -m rddc.evaluation.plot_sigma_regions
```

#### Fig. 3 (Synthetic example with different number of observed systems and varying degree of uncertainty)

This figure visualizes a comprehensive parameter study. Recreating the data and reproducing the plots might require significant amount of computational resources (~2000 core-hours). Go to step 3 to recreate the plots with the data from this repository.

* Adjust the number of cores available for computation in `rddc/run/settings/dean_var_sigma_N.py`.
* If desired, adjust the parameters to vary in the same file (function `get_variations()`).

1. Simulate the parameter study: (Skip this if data is already available and stored in `./data/dean_var_sigma_N`)

      ```bash
      python -m rddc.run.synthetic --testcase dean --mode var_sigma_N
      ```

2. Extract the data

      ```bash
      python -m rddc.evaluation.heatmap_sigma_N_extract_data --testcase dean --mode var_sigma_N
      ```

3. Create the plots

      ```bash
      python -m rddc.evaluation.heatmap_sigma_N
      ```

#### Fig. 4 (Synthetic example with different number of observed systems and varying trajectory length)

This figure visualizes a comprehensive parameter study. Recreating the data and reproducing the plots might require significant amount of computational resources (~2000 core-hours). Go to step 3 to recreate the plots with the data from this repository.

* Adjust the number of cores available for computation in `rddc/run/settings/dean_var_T_N.py`.
* If desired, adjust the parameters to vary in the same file (function `get_variations()`).

1. Simulate the parameter study: (Skip this if data is already available and stored in `./data/dean_var_T_N`)

      ```bash
      python -m rddc.run.synthetic --testcase dean --mode var_T_N
      ```

2. Extract the data

      ```bash
      python -m rddc.evaluation.heatmap_T_N_extract_data --testcase dean --mode var_T_N
      ```

3. Create the plots

      ```bash
      python -m rddc.evaluation.heatmap_T_N
      ```

### Troubleshooting

If you encounter issues with paths, check the following files and adjust the variables `basepath` to the global path of your installation:

* `rddc/evaluation/plot_sigma_regions.py`
* `rddc/evaluation/heatmap_sigma_N_extract_data.py`
* `rddc/evaluation/heatmap_sigma_N.py`
* `rddc/evaluation/heatmap_T_N_extract_data.py`
* `rddc/evaluation/heatmap_T_N.py`
