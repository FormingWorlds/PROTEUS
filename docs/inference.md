# Asynchronous Bayesian Optimization for PROTEUS

This project implements parallel-asynchronous Bayesian Optimization (BO) for parameter inference using the PROTEUS planetary evolution simulator. It uses multiple workers to efficiently explore the parameter space and find optimal matches between simulated and observed planetary characteristics.

## Overview

The system performs Bayesian optimization to infer planetary formation parameters by:
1. Running PROTEUS simulations with different parameter combinations
2. Comparing simulated observables (planet radius, mass, transit depth, etc.) with target values
3. Using Gaussian Process surrogates and acquisition functions to guide the search toward optimal parameters
4. Employing multiple parallel workers asynchronously to accelerate the optimization process

## Project Structure

| File               | Description                               |
|:-------------------|:------------------------------------------|
| `main.py`          | Main entry point                          |
| `async_BO.py`      | Parallel BO implementation                |
| `BO.py`            | Single BO step implementation             |
| `objective.py`     | PROTEUS interface and objective function  |
| `plots.py`         | Visualization utilities                   |
| `utils.py`         | Helper functions                          |
| `gen_D_init.py`    | Generate initial data                     |
| `test.py`          | Sanity check script                       |
| `BO_config.toml`   | Configuration file                        |


## Dependencies

The project requires the following Python packages:
- `torch` (PyTorch)
- `botorch` (Bayesian Optimization)
- `gpytorch` (Gaussian Processes)
- `scipy`
- `pandas`
- `matplotlib`
- `toml`
- `numpy`

You also need the PROTEUS simulator installed and accessible via the `proteus` command.

## Configuration

The main configuration is done through the `BO_config.toml` file:

```toml
n_workers = 7                    # Number of parallel workers
kernel = "RBF"                   # Kernel type for GP
max_len = 40                     # Maximum number of evaluations
n_restarts = 10                  # GP optimization restarts
n_samples = 1000                 # Raw samples for acquisition optimization
directory = "output/inference/results/"  # Output directory
ref_config = "input/demos/dummy.toml"    # Reference PROTEUS config
D_init_path = "inference/data/prot.pth"  # Initial data path

[observables]                    # Target observables to match
"R_int" = 7629550.6175
"M_planet" = 7.9643831975e+24
"transit_depth" = 0.00012026905833
"bond_albedo" = 0.25

[parameters]                     # Parameters to optimize (with bounds)
"struct.mass_tot" = [0.5, 3.0]
"struct.corefrac" = [0.3, 0.9]
"atmos_clim.dummy.gamma" = [0.05, 0.95]
"escape.dummy.rate" = [1.0, 1e5]
"interior.dummy.ini_tmagma" = [2000, 4500]
"outgas.fO2_shift_IW" = [-4.0, 4.0]
```

## Usage

### 1. Generate Initial Data

First, create initial training data for the Bayesian optimization:

```bash
python gen_D_init.py
```

This creates `inference/data/prot.pth` with initial parameter-observable pairs.

### 2. Run Optimization

Execute the main optimization script:

```bash
python main.py --config BO_config.toml
```


## How It Works

### Objective Function

The system optimizes an objective function that measures how well simulated observables match target values:

```
J = 1 - ||1 - sim/true||²
```

Where `sim` are the simulated observables and `true` are the target values.

### Parallel Processing

- Multiple workers run simultaneously, each performing BO steps
- Workers share a common dataset but operate independently
- Lock mechanisms prevent race conditions when updating shared data
- Each worker tracks "busy" locations to avoid redundant evaluations

### Bayesian Optimization

- Uses Gaussian Process (GP) models to predict objective values
- Upper Confidence Bound (UCB) acquisition function guides exploration
- Automatic hyperparameter tuning via marginal likelihood optimization

The optimization will run until `max_len` evaluations are completed or manually stopped. Results are continuously saved and can be resumed if needed.


## Output

The system generates several outputs in the `output/inference/` directory:

### Data Files
- `data.pkl`: Final dataset with all evaluated parameters and objectives
- `logs.pkl`: Detailed logs of each BO step
- `Ts.pkl`: Timestamps for performance analysis

### Plots
- `parallel.png`: Timeline showing parallel worker execution
- `t_hist.png`: Distribution of total evaluation times
- `BO_t_hist.png`: Distribution of BO computation times
- `eval_t_hist.png`: Distribution of PROTEUS evaluation times
- `fit_t_hist.png`: Distribution of GP fitting times
- `ac_t_hist.png`: Distribution of acquisition optimization times
- `dist_v_iter.png`: Distance between queries and busy locations
- `reg.png`: Convergence plots (regret vs time/iterations)
- `best_val.png`: Best objective value evolution

### Results Summary
The system prints the final results including:
- Best found parameters
- Corresponding simulated observables
- Comparison with target observables

## Customization

### Adding New Parameters
1. Update the `[parameters]` section in `BO_config.toml`
2. Ensure the parameter names match PROTEUS configuration keys

### Changing Observables
1. Update the `[observables]` section with your target values
2. Make sure these observables are output by PROTEUS


## Performance Considerations

- Set `n_workers` to be less than your CPU core count minus 1
- The system automatically limits thread usage to prevent oversubscription
- PROTEUS evaluation time typically dominates total runtime
