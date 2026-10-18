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

    | File               | Description                                                           |
    |:-------------------|:----------------------------------------------------------------------|
    | `inference.py`     | Main entry point                                                      |
    | `gen_D_init.py`    | Generate initial data                                                 |
    | `async_BO.py`      | Parallel BO implementation                                            |
    | `BO.py`            | Single BO step implementation                                         |
    | `objective.py`     | PROTEUS interface and objective function                              |
    | `correlation.py`   | Correlated observable uncertainties                                   |
    | `runner.py`        | One PROTEUS process reused across evaluations (`dispatch = "runner"`) |
    | `failures.py`      | Functions for handling failing simulations                            |
    | `transforms.py`    | Functions for transforming and scaling variables                      |
    | `plot.py`          | Performance plots and per-sample results                              |
    | `plot_fit.py`      | Best fit against the targets and true parameters                      |
    | `utils.py`         | Helper functions for inference scheme                                 |

## Configuration

The main configuration is done through a TOML-formatted configuration file. There are two ways to initialise the inference process:

1. Allowing PROTEUS to randomly sample the parameter space provided in the config.
2. Using the result of a previously-computed grid of models.

To apply case (1), set the config variable `init_samps=4` to use 4 initial samples. You can choose any number greater than 2, but ideally less than 10. Then set `init_grid='none'`. Set `init_samps=-1` to use the same value as `n_workers`.

If you instead wish to initialise under case (2), where a pre-computed grid provides the initial samples, set the config variable `init_grid='outname'` where `outname` is the name of the folder containing the grid inside the shared PROTEUS output folder. Then set `init_samps='none'`.

An example configuration file is available at `input/inference/example.infer.toml`.

`spectral_cache`, `abort_on_failure` and `correlate_ratios` take only a TOML `true` or `false`; any other value stops the study at start, before its output folder is emptied.

## Usage

Execute the main optimisation process by using the PROTEUS command-line interface

```bash
proteus infer --config input/inference/example.infer.toml
```

In this case, we randomly sample the parameter space to provide a starting point for the
optimisation. This process must stay open in order to manage the workers.


## How It Works

### Objective Function

The system maximises an objective function that measures how well simulated observables match target values:

```
J = -log10( ||1 - sim/true||² + 1e-10 )
```

Where `sim` are the simulated observables and `true` are the target values.
A perfect match gives the largest possible value, J = 10. J is 0 when the squared relative
errors sum to 1, and negative for worse fits.

### Observable uncertainties

An optional `[sigma]` table gives the 1-sigma uncertainty of each observable, in the same
units as the value in `[observables]`. When it is present, each residual is divided by its uncertainty:

```
J = -log10( sum( ((sim - true) / sigma)^2 ) + 1e-10 )
```

Observables that span orders of magnitude (e.g. `atm_kg_per_mol`) are compared as `log10` values, and so are element mass ratios such as `C/O_atm` when `[sigma]` is given. Their uncertainty can be given in linear space, e.g. `"C/O_atm" = 0.1`; in this case, it is converted to dex to first order via `sigma / (true * ln 10)`. Optionally, it can be given in dex directly as a string, for example `"C/O_atm" = "0.1 dex"`.

!!! warning "Do not compare `J` when using different objectives"
    The two objectives are on different scales, so compare `J` only between inference runs that use the same one. The objective in use is reported at start-up.

### Correlated uncertainties

When the errors of two observables are correlated, for example abundance ratios from one
retrieval, an optional `[correlation]` table gives their correlation coefficient. It needs `[sigma]`. Each pair is given once, as a nested table. Pairs not listed are uncorrelated:

```toml
[correlation.R_obs]
"g_obs" = -0.4
"T_obs" = 0.2
```

The sum of squares is then replaced by the full chi-squared

```
chi2 = u^T R^-1 u,    u = (sim - true) / sigma
```

where `R` is the correlation matrix. Each coefficient must lie strictly between -1 and 1, and the matrix as a whole must be positive definite.

#### Element-ratio observables

For element-ratio observables (such as `C/O_atm` and `O/H_atm`), `correlate_ratios = true` builds the table instead. Assuming the same dex error for every element, two ratios correlate by the cosine of their exponent vectors. One element shared on the same side gives +0.5 (`C/O_atm` with `S/O_atm`); on opposite sides -0.5 (`C/O_atm` with `O/H_atm`). With this parameter set to True, the study refuses to start when the ratio sigmas differ in dex by more than 1%. Give such coefficients in `[correlation]` instead.

When posterior samples, e.g. from a retrieval, exist, their correlation matrix in `[correlation]` is the best input; `correlate_ratios` can be set when nothing better is known.

### Student-t errors

`objective` selects the error model: `"chi-2"` (the default, Gaussian errors) or `"student-t"`. With `objective = "student-t"`, each observable's error follows a Student-t distribution with `student_t_dof = nu` degrees of freedom (default 4). It needs `[sigma]` and cannot be combined with `[correlation]` or `correlate_ratios`, because the errors are taken as independent. Each squared residual `u^2` is replaced by

```
(nu + 1) * ln(1 + u^2 / nu),    u = (sim - true) / sigma
```

which is `-2 ln` of the Student-t density up to a constant, and tends to `u^2` as `nu` grows. A residual of many sigma then grows only logarithmically, so one observable the model cannot match does not dominate the fit. Small `nu` (3 to 5) gives heavy tails; `nu = 1` is the Cauchy distribution. Under a Student-t, `sigma` is the scale of the error, not its standard deviation, which is `sigma * sqrt(nu / (nu - 2))` for `nu > 2`.

### Known true parameters

When the target observables were extracted from a simulation whose parameters you know (a
synthetic retrieval test), an optional `[truth]` table records those parameters so the best fit can be compared against them:

```toml
[truth]
"interior_struct.core_frac" = 0.325
"outgas.fO2_shift_IW"       = 2.0
```

The table does not change the optimisation; it adds a True column to the results summary and the `result_parameters.png` plot.

### Posterior samples

With `[sigma]`, every objective form (chi-squared, correlated chi-squared, Student-t) is `S = -2 ln L` up to a constant, so `ln L = -10^(-J) / 2` up to a constant. At the end of the study a GP is fitted to `J` of the scored evaluations, and `posterior_samples` (default 2000, `0` is off) parameter samples are drawn from `L` at the GP mean, by importance resampling of 10^6 uniform draws. When the posterior is so narrow that these give fewer effective samples than requested, up to five more rounds draw from a Gaussian mixture centred on the best-weighted draws (with a 10 % uniform share), until they do. The prior is uniform over the sampled ranges, log-uniform for log10-scaled parameters such as `planet.elements.H_budget`. The log reports the effective sample size, with a warning when it stays below the number of samples.

The posterior ignores the GP's own uncertainty, so it is only as good as the GP near the best fit: check that the study converged (`perf_bestval.png`) before using it.

### Parallel Processing

- Multiple workers run simultaneously, each performing BO steps
- Workers share a common dataset but operate independently
- Lock mechanisms prevent race conditions when updating shared data
- Each worker tracks "busy" locations to avoid redundant evaluations

### Bayesian Optimization

- Uses Gaussian Process (GP) models to predict objective values
- Acquisition function guides exploration-exploitation trade-off on search space
- Automatic hyperparameter tuning via marginal likelihood optimization, from fresh hyperparameters at every fit
- A parameter whose fitted lengthscale exceeds 1.5 (in the unit cube) barely changes the GP, which then puts every proposal on one of its bounds; such a parameter is drawn at random instead (logged at debug level)

The optimization will run until `n_steps` evaluations are completed or manually stopped. Results are continuously saved and can be resumed if needed.

With `patience = N` (default 0, off), the study stops earlier, once `N` evaluations in a row have not raised the best objective by more than 0.01. Stalls of tens of evaluations happen, so set patience after a few test runs; too low a value stops a study early.

### Acquisition functions

The acquisition function is an analytical function that is aware of the current state of the optimisation.
It is used to evaluate the *potential* value of sampling a candidate particular point in the parameter space, to
help determine where the optimisation should next run PROTEUS. It helps balance the trade-off between exploring new areas and exploiting known good areas to optimize a black-box function efficiently.

* `UCB` - upper confidence bound
* `LogEI` - logarithm of the expected improvement
* `LogPI` - logarithm of the probability of improvement (analogous to log-likelihood)
* `TS` - Thompson sampling: the maximum of one function drawn from the GP posterior; each worker draws its own, so the workers spread out without the batch forms below

While other workers have runs in progress, `UCB`, `LogEI` and `LogPI` switch to their Monte Carlo batch forms so that they account for those points. BoTorch has no batch form of LogPI, so `LogPI` then uses the plain probability of improvement, which is flatter far from the best point; prefer `LogEI` with several workers.

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
- `posterior.csv`: Only with `[sigma]`. Posterior samples, one column per parameter (see [Posterior samples](#posterior-samples))
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
- `result_posterior.png`: Only with `[sigma]`. Corner plot of the posterior samples.
- `result_parameters.png`: Only with a `[truth]` table. Every sample, the truth and the best fit placed within each parameter's sampled range.

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
one PROTEUS process alive per worker and reuses it for every evaluation that
worker makes.

```toml
dispatch = "runner"      # "subprocess" (default) or "runner"
runner_max_jobs = 0      # replace the process after this many simulations; 0 keeps it
```

Set `runner_max_jobs` to a positive number to bound how long any one process is kept.
