# Asynchronous Bayesian Optimization for PROTEUS

This project implements parallel-asynchronous Bayesian Optimization (BO) for parameter inference using PROTEUS as the  'simulator'. It uses multiple workers to efficiently explore the parameter space and find optimal matches between simulated and observed planetary characteristics. You can also run this BO inference scheme to refine the results of a grid.

!!! info "Requires the `inference` extra"
    The scheme is built on PyTorch, BoTorch and GPyTorch, which install
    separately from the rest of PROTEUS:

    ```console
    pip install "fwl-proteus[inference]"
    ```

    See [optional modules](optionalmodules_installation.md#parameter-inference-bayesian-optimisation).

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
    | `failures.py`      | Functions for handling failing simulations |
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

### Observable uncertainties

An optional `[sigma]` table gives the 1-sigma uncertainty of each observable, in the same
units as the value in `[observables]`. When it is present every observable needs an entry,
and each residual is divided by its uncertainty instead of by the target value:

```
J = -log10( sum( ((sim - true) / sigma)^2 ) + 1e-10 )
```

Observables that span orders of magnitude (`atm_kg_per_mol`, `*_vmr`, `*_bar`, `P_surf`, ...)
are compared as `log10` values, and so are element ratios such as `C/O_atm` when `[sigma]` is
given (without it they stay linear, as before). Their uncertainty is converted to log10 units by
first-order propagation, `sigma / (true * ln 10)`. This approximation is only accurate when the uncertainty is
small compared to the value. A warning is logged at start-up when it exceeds 30 % of the value.

The two objectives are on different scales, so compare `J` only between studies that use the same
one. The objective in use is reported at start-up.

### Correlated uncertainties

When the errors of two observables are correlated, for example abundance ratios from one
retrieval, or a surface gravity derived from a measured radius, an optional `[correlation]` table
gives their correlation coefficient. It needs `[sigma]`. Each pair is given once, as a nested table,
and pairs not listed are uncorrelated:

```toml
[correlation.R_obs]
"g_obs" = -0.4
"T_obs" = 0.2
```

The sum of squares is then replaced by the full chi-squared

```
chi2 = u^T R^-1 u,    u = (sim - true) / sigma
```

where `R` is the correlation matrix, with ones on the diagonal. This is the same as `r^T C^-1 r` with
the covariance `C_ij = rho_ij sigma_i sigma_j`. Each coefficient must lie strictly between -1 and 1,
and the matrix as a whole must be positive definite; both are checked at start-up. For an
observable compared as `log10` values the coefficient is used unchanged, since to first order the
conversion to log10 units only rescales each uncertainty.

For element-ratio observables, `correlate_ratios = true` builds the table instead. Each name
with a `/` must be a ratio of two elements ending in `_atm` (`C/O_atm` is C to the power +1, O to the power -1),
and, assuming the same dex error for every element, two ratios correlate by the cosine of their
exponent vectors. One element shared on the same side gives +0.5 (`C/O_atm` with `S/O_atm`), on
opposite sides -0.5 (`C/O_atm` with `O/H_atm`). Observables without a `/` stay uncorrelated. It
needs `[sigma]` and cannot be combined with `[correlation]`. The assumption fixes only the
correlations; each ratio keeps its own `[sigma]`.

The correlations describe the measurement errors, not the way the model links observables: a
model that predicts both radius and gravity from one planet mass already accounts for that link.

### Known true parameters

When the target observables were extracted from a simulation whose parameters you know (a
synthetic retrieval test), an optional `[truth]` table records those parameters so the best fit
can be compared against them:

```toml
[truth]
"interior_struct.core_frac" = 0.325
"outgas.fO2_shift_IW"       = 2.0
```

When it is present every entry of `[parameters]` needs a value, as with `[sigma]`. A value outside
the sampled range is accepted with a warning, since the study cannot recover it. The table does
not change the optimisation; it adds a True column to the results summary and the
`result_parameters.png` plot.

A best fit that matches the observables but not the true parameters is not necessarily a failed
study: different parameter combinations can produce the same observables.

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

With `patience = N` (default 0, off), the study stops earlier, once `N` evaluations in a row have not raised the best objective by more than 0.01, about a 2% lower chi-squared. Small rises add up: the count resets once the best has risen by more than 0.01 in total since the last reset. Runs already in progress finish first. Plateaus of 20 to 40 evaluations before a further improvement are common, so values below about 50 can stop a study too early.

### Acquisition functions

The acquisition function is an analytical function that is aware of the current state of the optimisation.
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
- `failures.csv`: One row per simulation that failed or was excluded, written only when there is at least one (see [Failed and excluded simulations](#failed-and-excluded-simulations))

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
- `result_observables.png`: Final observables of every sample as a ratio to their target. 
- `result_parameters.png`: Only with a `[truth]` table. Every sample, the true value and the best
  fit placed within each parameter's sampled range (in log10 for log-scaled parameters), and the
  best-fit error as a percentage of that range.

### Results Summary
The system prints the final results including:

- Best found parameters
- Corresponding simulated observables
- Comparison with target observables

### Failed and excluded simulations

During the inference run, some PROTEUS simulations might crash or fail, or stop on a status that is excluded in the inference configuration (e.g. maximum runtime reached). Status 29 (planet evaporated) is always excluded. The run carries on when there are failures unless `abort_on_failure` is set to `true` in the inference config. The study then starts no new simulations, but those already running finish first. Each failure is added to `failures.csv` in the output folder as it happens, and all failures are summarised at the end of the study. A study stopped by `abort_on_failure` keeps its `failures.csv` but prints no summary.

## Customization

### Adding New Parameters
1. Update the ``[parameters]`` section in your inference config file
2. Ensure the parameter names match PROTEUS configuration keys

### Changing Observables
1. Update the ``[observables]`` section with your target values
2. Make sure these observables are output by PROTEUS


## Performance Considerations

- Set `n_workers` to be less than your CPU core count minus 1
- The system automatically limits thread usage to prevent oversubscription
- PROTEUS evaluation time typically dominates total runtime
- Workers share prepared spectral files through a cache in the inference run's output folder. Set `spectral_cache = false` in the inference config to turn it off. The study log names the cache in use.

### Reusing one PROTEUS process per worker

By default each evaluation runs as its own `proteus start`, so every sample
pays for importing PROTEUS, loading the Julia environment and compiling AGNI
on its first call. Setting `dispatch = "runner"` in the inference config keeps 
one PROTEUS process alive per worker and reuses it for every evaluation that worker makes.

On the default dispatch each evaluation writes its own `i_<n>_console.log`; 
a reused process instead writes a single `runner_console.log` per worker, covering every simulation that worker ran, because Julia cannot be redirected between simulations. 
Each run's own `proteus_*.log` is unaffected.

```toml
dispatch = "runner"      # "subprocess" (default) or "runner"
runner_max_jobs = 0      # replace the process after this many simulations; 0 keeps it
```

Set `runner_max_jobs` to a positive number to bound how long any one process is kept.
