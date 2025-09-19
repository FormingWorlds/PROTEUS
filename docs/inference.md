# Asynchronous Bayesian Optimization for PROTEUS

This project implements parallel-asynchronous Bayesian Optimization (BO) for parameter inference using PROTEUS as the  'simulator'. It uses multiple workers to efficiently explore the parameter space and find optimal matches between simulated and observed planetary characteristics. You can also run this BO inference scheme to refine the results of a grid.

## Overview

The system performs Bayesian optimization to infer planetary formation parameters by:

1. Running PROTEUS simulations with different parameter combinations
2. Comparing simulated observables (planet radius, mass, transit depth, etc.) with target values
3. Using Gaussian Process surrogates and acquisition functions to guide the search toward optimal parameters
4. Employing multiple parallel workers asynchronously to accelerate the optimization process

## Project Structure

These files are contained within the folder `src/proteus/inference/`.

| File               | Description                               |
|:-------------------|:------------------------------------------|
| `inference.py`     | Main entry point                          |
| `async_BO.py`      | Parallel BO implementation                |
| `BO.py`            | Single BO step implementation             |
| `objective.py`     | PROTEUS interface and objective function  |
| `plot.py`          | Visualization utilities                   |
| `utils.py`         | Helper functions for inference scheme     |
| `gen_D_init.py`    | Generate initial data                     |

## Configuration

The main configuration is done through a TOML-formatted configuration file. There are two ways to initialise the inference process:

1. Allowing PROTEUS to randomly sample the parameter space provided in the config.
2. Using the result of a previously-computed grid of models.

To apply case (1), set the config variable `init_samps=4` to use 4 initial samples. You can choose any number greater than 2, but ideally less than 10. Then set `init_grid='none'`.

If you instead wish to initialise under case (2), where a pre-computed grid provides the initial samples, set the config variable `init_grid='outname'` where `outname` is the name of the folder containing the grid inside the shared PROTEUS output folder. Then set `init_samps='none'`.

An example configuration file is shown below.

```toml
# Set seed for reproducibility
seed = 2

# Path to output folder where inference will be saved (relative to PROTEUS output folder)
output = "infer_demo/"

# Path to base (reference) config file relative to PROTEUS root folder
ref_config = "input/demos/dummy.toml"

# Method for initialising the inference scheme (one of these must be 'none')
init_samps = '2'         # Number of random samples if starting from scratch.
init_grid  = 'none' # grid_demo/'   # Path pre-computed grid (relative to PROTEUS output folder)

# Parameters for Bayesian optimisation
n_workers  = 7        # Number of parallel workers
kernel     = "MAT"    # Kernel type for GP, "RBF" | "MAT"
acqf       = "LogEI"  # Acquisition function, "UCB" | "LogEI"
n_steps    = 30       # Total number of evaluations (i.e. BO steps)
n_restarts = 10       # GP optimization restarts
n_samples  = 1000     # Raw samples for acquisition optimization

# Parameters to optimize (with bounds)
[parameters]
"struct.mass_tot" = [0.7, 3.0]
"struct.corefrac" = [0.3, 0.9]
"delivery.elements.H_ppmw" = [6e3, 2e4]
"outgas.fO2_shift_IW" = [-3.0, 5.0]

# Target observables to match by optimisation
[observables]
"R_obs" = 7.9950245489e6
"H2O_vmr" = 0.41
```

## Usage

Execute the main optimisation process by using the PROTEUS command-line interface

```bash
proteus infer --config input/ensembles/example.infer.toml
```

In this case, we randomly sample the parameter space to provide a starting point for the
optimisation. This process must stay open in order to manage the workers.


## How It Works

### Objective Function

The system optimizes an objective function that measures how well simulated observables match target values:

```
J = 1 - ||1 - sim/true||²
```

Where `sim` are the simulated observables and `true` are the target values.
This means that the 'best' value for the objective function is 1. Values closer to 1 represent
better fits, while smaller values (including negative ones) are worse fits.

### Parallel Processing

- Multiple workers run simultaneously, each performing BO steps
- Workers share a common dataset but operate independently
- Lock mechanisms prevent race conditions when updating shared data
- Each worker tracks "busy" locations to avoid redundant evaluations

### Bayesian Optimization

- Uses Gaussian Process (GP) models to predict objective values
- Acquisition function guides exploration-exploitation trade-off on search space
- Automatic hyperparameter tuning via marginal likelihood optimization

The optimization will run until `max_len` evaluations are completed or manually stopped. Results are continuously saved and can be resumed if needed.


## Output

The system generates several outputs in:

### Data Files
- `data.pkl`: Final dataset with all evaluated parameters and objectives
- `logs.pkl`: Detailed logs of each BO step
- `Ts.pkl`: Timestamps for performance analysis
- `init.pkl`: Data used as an initial guess for starting the optimisation

### Plots
The BO scheme will generate many plots upon completion.
Those prefixed with `perf_` diagnose the performance of the optimisation.

- `perf_parallel.png`: Timeline showing parallel worker execution
- `perf_timehist.png`: Distribution of total evaluation times
- `perf_BO_timehist.png`: Distribution of BO computation times
- `perf_eval_timehist.png`: Distribution of PROTEUS evaluation times
- `perf_fit_timehist.png`: Distribution of GP fitting times
- `perf_ac_timehist.png`: Distribution of acquisition optimization times
- `perf_distance_iters.png`: Distance between queries and busy locations
- `perf_regret.png`: Convergence plots (regret vs time/iterations)
- `perf_bestval.png`: Best objective value evolution

Plots prefixed with `result_` show the results of the optimisation.

- `result_correlation.png`: Scatter plot observables for each parameter, at each sample.
- `result_objective.png`: Value of objective `J` for each parameter, at each sample.

### Results Summary
The system prints the final results including:
- Best found parameters
- Corresponding simulated observables
- Comparison with target observables

## Customization

### Adding New Parameters
1. Update the `[parameters]` section in your inference config file
2. Ensure the parameter names match PROTEUS configuration keys

### Changing Observables
1. Update the `[observables]` section with your target values
2. Make sure these observables are output by PROTEUS


## Performance Considerations

- Set `n_workers` to be less than your CPU core count minus 1
- The system automatically limits thread usage to prevent oversubscription
- PROTEUS evaluation time typically dominates total runtime
