# Experience Transfer for Robust Direct Data-Driven Control

This repository contains the supplementary code for the master thesis "Learning-based Control for Heterogeneous Quadcopter Fleets" written by Dmitrii Likhachev under the supervision of Alexander von Rohr at the Institute for Data Science in Mechanical Engineering of RWTH Aachen University.

## Abstract

Automation through the use of robots has consistently demonstrated efficiency in terms of cost, safety, and the quality of outcomes across various fields. Certain tasks, like terrain exploration and mapping or forest firefighting, are more efficiently executed by deploying fleets of multiple robots. The robots in the fleet might exhibit different dynamical behavior due to factors such as changes in equipment or varying payloads.
In such scenarios, it is desired to have a single universal low-level controller that can stabilize all fleet members instead of manually tuning or developing a controller for each possible configuration. Finding such a controller can be challenging if underlying system's model is unavailable and the underlying uncertainty of the behavior is not quantified.

We propose an algorithm for probabilistically robust control design that does not need to access or estimate the system's model or behavioral uncertainty.
Instead, the algorithm uses the data collected from different fleet members during a test operation or from a simulation. It solves a convex optimization problem based on the data and returns a state-feedback controller that stabilizes all of the observed fleet members. Moreover, the probability of stabilizing the entire fleet, including unseen variations, grows as more members are observed.

We provide theoretical guarantees for the robustification probability based on the number of systems observed. When put to the test through a numerical example, we found that the algorithm generalized faster than our theoretical guarantee predicted. Then, we showcase its performance for simulated and real quadcopters and highlight insights for its potential application for real-world robotic systems.

### Citation

This thesis is based on the following publication. If you find this code or paper useful, please consider citing it:

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
* Get the gym-pybullet-drones submodule

```bash
git submodule update --init`
```

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
python -m rddc.run.simulation --train
```

### Reproduction of the results

After each simulation / variation run, the data will be stored in `data/`. The figures are always generated from that data using separate scripts and stored in `figures/`.

Make sure to start every attempt at reproducing the results with an empty corresponding data folder.

#### Fig. 4.1 (Visualization of scenario optimization for a fleet of four one-dimensional systems)

```bash
python -m rddc.run.1d_sigma_regions
python -m rddc.evaluation.plot_sigma_regions
```

#### Fig. 5.1 (Synthetic example with varying number of observed systems and degree of uncertainty)

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

#### Fig. 5.2 (Synthetic example with varying number of observed systems and trajectory length)

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

#### Fig. 5.3 (Uncertainty sets for varying trajectory lengths)

```bash
python -m rddc.evaluation.plot_sigma_regions_T_seed
```

#### Fig. 5.5 (Tod-down view of simulated test flights)

* Plug `ceddc` or `sof` into `<mode>` to reproduce the __sim_1__ and __sim_N__ respectively
* If desired, adjust the simulation settings (gui, trajectory length, etc.) in `rddc/run/settings/simulation_<mode>.py`.

1. Produce training data and the controller based on it

      ```bash
      python -m rddc.run.simulation --train --K --mode <mode>
      ```

2. Execute the test runs:

      ```bash
      python -m rddc.run.simulation --test --mode <mode>
      ```

3. (Optional) after each run, you can plot the results using:

      ```bash
      python -m rddc.run.simulation --eval --mode <mode>
      ```

4. Produce the figures with:

      ```bash
      python -m rddc.evaluation.rddc_quad_demo --mode <mode>
      ```

### Troubleshooting

If you encounter issues with paths, check the following files and adjust the variables `basepath` to the global path of your installation:

* `rddc/evaluation/plot_sigma_regions.py`
* `rddc/evaluation/rddc_vs_LS_with_wind.py`
* `rddc/evaluation/heatmap_T_N.py`
* `rddc/evaluation/heatmap_sigma_N.py`
