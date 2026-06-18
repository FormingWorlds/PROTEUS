# Asynchronous Bayesian Optimization for PROTEUS

This project implements parallel-asynchronous Bayesian Optimization (BO) for parameter inference using PROTEUS as the  'simulator'. It uses multiple workers to efficiently explore the parameter space and find optimal matches between simulated and observed planetary characteristics. You can also run this BO inference scheme to refine the results of a grid.

## Overview

The system performs Bayesian optimization to infer planetary formation parameters by:

1. Running PROTEUS simulations with different parameter combinations
2. Comparing simulated observables (planet radius, mass, transit depth, etc.) with target values
3. Using Gaussian Process surrogates and acquisition functions to guide the search toward optimal parameters
4. Employing multiple parallel workers asynchronously to accelerate the optimization process

??? info "Project structure (developer reference)"

    These files are contained within the folder `src/proteus/inference/`.

    | File               | Description                               |
    |:-------------------|:------------------------------------------|
    | `inference.py`     | Main entry point                          |
    | `transforms.py`    | Functions for transforming and scaling variables |
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

To apply case (1), set the config variable `init_samps=4` to use 4 initial samples. You can choose any number greater than 2, but ideally less than 10. Then set `init_grid='none'`. Set `init_samps=-1` to use the same value as `n_workers`.

If you instead wish to initialise under case (2), where a pre-computed grid provides the initial samples, set the config variable `init_grid='outname'` where `outname` is the name of the folder containing the grid inside the shared PROTEUS output folder. Then set `init_samps='none'`.

An example configuration file is available at `input/inference/example.infer.toml`.

## Usage

Execute the main optimisation process by using the PROTEUS command-line interface

```bash
proteus infer --config input/inference/example.infer.toml
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

The optimization will run until `n_steps` evaluations are completed or manually stopped. Results are continuously saved and can be resumed if needed.

### Acqusition functions

The `acqusition function' is an analytical function that is aware of the current state of the optimisation.
It is used to evaluate the *potential* value of sampling a candidate particular point in the parameter space, to 
help determine where the optimisation should next run PROTEUS. It helps balance the trade-off between exploring new areas and exploiting known good areas to optimize a black-box function efficiently.

* `UCB` - upper confidence bound
* `LogEI` - logarithm of the expected improvement
* `LogPI` - logarithm of the probability of improvement (analogous to log-likelihood)

See docs [here](https://botorch.readthedocs.io/en/latest/acquisition.html).

### Kernels

The kernel is an analytical function used by the Gaussian processes to represent the similarity between model behaviour as a function of the parameter space. It includes the underlying function by capturing the relationships and uncertainties/noise in the data.

* `RBF` - radial basis function
* `MAT1/2` - Materne kernel with $\nu = 1/2$ 
* `MAT3/2` - Materne kernel with $\nu = 3/2$
* `MAT5/2` - Materne kernel with $\nu = 5/2$

See docs [here](https://botorch.readthedocs.io/en/latest/models.html#module-botorch.models.kernels.categorical).

## Output

The system generates several outputs in:

### Data Files
- `data.csv`: Final dataset with all evaluated parameters (`x_*`) and objective values (`y`)
- `logs.csv`: Detailed logs of each BO step
- `Ts.csv`: Timestamps for performance analysis
- `init.csv`: Data used as an initial guess for starting the optimisation

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
