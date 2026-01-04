# Bayesian Inference and Parameter Retrieval

## Overview

PROTEUS implements parallel-asynchronous Bayesian Optimization (BO) for parameter inference. The system:

1. Runs PROTEUS simulations with different parameter combinations
2. Compares simulated observables with target values
3. Uses Gaussian Process surrogates to guide the search toward optimal parameters
4. Employs multiple parallel workers to accelerate optimization

## Quick Start

Execute Bayesian optimization using:

```console
proteus infer --config input/ensembles/example.infer.toml
```

This command will run the inference scheme and must stay open to manage parallel workers.

## Configuration

Create a TOML configuration file to define your inference parameters. Key sections:

```toml
# Set seed for reproducibility
seed = 2

# Path to output folder where inference will be saved
output = "infer_demo/"

# Path to base (reference) config file
ref_config = "input/demos/dummy.toml"

# Method for initialising the inference
init_samps = '2'         # Number of random samples if starting from scratch
init_grid  = 'none'      # Or path to pre-computed grid for initialization

# Parameters for Bayesian optimisation
n_workers  = 7        # Number of parallel workers
kernel     = "MAT"    # Kernel type: "RBF" | "MAT"
acqf       = "LogEI"  # Acquisition function: "UCB" | "LogEI"
n_steps    = 30       # Total number of evaluations
n_restarts = 10       # GP optimization restarts
n_samples  = 1000     # Raw samples for acquisition optimization

# Parameters to optimize (with bounds)
[parameters]
"struct.mass_tot" = [0.7, 3.0]
"struct.corefrac" = [0.3, 0.9]
"delivery.elements.H_ppmw" = [6e3, 2e4]
"outgas.fO2_shift_IW" = [-3.0, 5.0]

# Target observables to match
[observables]
"R_obs" = 7.9950245489e6
"H2O_vmr" = 0.41
```

## Initialization Methods

### Method 1: Random Sampling

Start inference from scratch with random initial samples:

- Set `init_samps` to number of samples (2-10 recommended)
- Set `init_grid = 'none'`

### Method 2: Using Pre-computed Grid

Initialize from results of a previously-computed grid:

- Set `init_grid = 'folder_name'` (path relative to PROTEUS output folder)
- Set `init_samps = 'none'`

This method refines grid results with Bayesian optimization.

## How It Works

### Objective Function

The optimization minimizes the difference between simulated and target observables:

$$J = 1 - ||1 - \text{sim}/\text{true}||^2$$

Where:

- `sim` = simulated observables
- `true` = target observables
- Best value: J = 1 (perfect match)
- Negative values indicate poor fits

### Parallel Processing

- Multiple workers run simultaneously, each performing BO steps
- Workers share a common dataset but operate independently
- Lock mechanisms prevent race conditions
- Each worker tracks "busy" locations to avoid redundant evaluations

### Bayesian Optimization

- Gaussian Process (GP) models predict objective values
- Acquisition function guides exploration-exploitation trade-off
- Automatic hyperparameter tuning via marginal likelihood optimization

## Output Files

Inference results are saved in your specified output folder:

### Data Files

- `data.pkl` - All evaluated parameters and objectives
- `logs.pkl` - Detailed logs of each BO step
- `Ts.pkl` - Timestamps for performance analysis
- `init.pkl` - Initial data for inference

### Performance Plots

Files prefixed with `perf_` diagnose optimization performance:

- `perf_parallel.png` - Timeline of parallel worker execution
- `perf_timehist.png` - Distribution of evaluation times
- `perf_BO_timehist.png` - Distribution of BO computation times
- `perf_eval_timehist.png` - Distribution of PROTEUS evaluation times
- `perf_regret.png` - Convergence plots (regret vs time)
- `perf_bestval.png` - Evolution of best objective value

### Results Plots

Files prefixed with `result_` show optimization results:

- `result_correlation.png` - Scatter plot of observables vs parameters
- `result_objective.png` - Objective value for each parameter at each sample

## Customization

### Adding Parameters

1. Add new parameter bounds to the `[parameters]` section in your config file
2. Ensure parameter names match PROTEUS configuration keys (e.g., `struct.mass_tot`)

### Changing Observables

1. Update the `[observables]` section with your target values
2. Ensure these observables are output by PROTEUS

## Performance Tips

- Set `n_workers` to CPU core count minus 1
- The system automatically limits thread usage to prevent oversubscription
- PROTEUS evaluation time typically dominates total runtime
- Results are continuously saved and can be resumed if interrupted

## Example Configuration

An example inference configuration is available at:

```toml
input/ensembles/example.infer.toml
```

## Related Concepts

For more information on:

- **Parameter grids**: See [Running Grids of Simulations](grid-simulations.md)
- **Configuration options**: See the configuration reference
- **Retrieved parameters**: See [Understanding Output and Results](run-simulation.md)
