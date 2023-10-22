# Experience Transfer for Robust Direct Data-Driven Control

This repository contains the supplementary code for the paper "Experience Transfer for Robust Direct Data-Driven Control"
A preprint of the paper can be found on [arXiv](https://arxiv.org/abs/2306.16973).

![Trajectories of the quadcopters with the robust controller](https://github.com/Data-Science-in-Mechanical-Engineering/rddc/assets/76944030/a7d45ef1-78b7-44b6-9814-8b7c38e8fadf)

## Abstract

Learning-based control uses data to design efficient controllers for specific systems. When multiple systems are involved, *experience transfer* usually focuses on data availability and controller performance yet neglects robustness to changes between systems. In contrast, this letter explores experience transfer from a robustness perspective. We leverage the transfer to design controllers that are robust not only to the uncertainty regarding an individual agent's model but also to the choice of agent in a fleet.  Experience transfer enables the design of safe and robust controllers that work out of the box for all systems in a heterogeneous fleet.  Our approach combines scenario optimization and recent formulations for direct data-driven control without the need to estimate a model of the system or determine uncertainty bounds for its parameters. We demonstrate the benefits of our data-driven robustification method through a numerical case study and obtain learned controllers that generalize well from a small number of open-loop trajectories in a quadcopter simulation.

### Citation

If you find our code or paper useful, please consider citing the current preprint

```bibtex
@misc{vonrohr2023experience,
      title={Experience Transfer for Robust Direct Data-Driven Control},
      author={Alexander von Rohr and Dmitrii Likhachev and Sebastian Trimpe},
      year={2023},
      eprint={2306.16973},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
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

This figure visualizes a comprehensive parameter study. Recreating the data and reproducing the plots might require significant amount of computational resources (~2000 core-hours).

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

This figure visualizes a comprehensive parameter study. Recreating the data and reproducing the plots might require significant amount of computational resources (~2000 core-hours).

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
