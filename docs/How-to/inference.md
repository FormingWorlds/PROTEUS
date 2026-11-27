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

The system performs Bayesian optimization to infer planet parameters by:

1. Running PROTEUS simulations with different parameter combinations
2. Comparing simulated observables (for example the radius and temperature at the photosphere, or atmospheric element ratios) with target values
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
    | `composite.py`     | One GP per observable (`bo_per_observable`, the posterior)            |
    | `objective.py`     | PROTEUS interface and objective function                              |
    | `correlation.py`   | Correlated observable uncertainties                                   |
    | `noise.py`         | Simulated noise on the observables (`noise = true`)                   |
    | `posterior.py`     | Posterior samples from the GP surrogate                               |
    | `runner.py`        | One PROTEUS process reused across evaluations (`dispatch = "runner"`) |
    | `failures.py`      | Functions for handling failing simulations                            |
    | `transforms.py`    | Functions for transforming and scaling variables                      |
    | `plot.py`          | Performance plots and per-sample results                              |
    | `plot_fit.py`      | Best fit against the targets and true parameters, posterior corner plot |
    | `utils.py`         | Helper functions for inference scheme                                 |

## Configuration

The main configuration is done through a TOML-formatted configuration file. There are two ways to initialise the inference process:

1. Allowing PROTEUS to randomly sample the parameter space provided in the config.
2. Using the result of a previously-computed grid of models.

To apply case (1), set `init_samps` to the number of initial samples and `init_grid = 'none'`. Use more samples than there are parameters; `bo_per_observable` requires at least one more, and refuses to start otherwise. Set `init_samps = -1` to use the same value as `n_workers`.

If you instead wish to initialise under case (2), where a pre-computed grid provides the initial samples, set the config variable `init_grid='outname'` where `outname` is the name of the folder containing the grid inside the shared PROTEUS output folder. Then set `init_samps='none'`.

An example configuration file is available at `input/inference/example.infer.toml`. The study saves the config as given to `copy.infer.toml` in its output folder, so the study can be rerun from that copy.

`spectral_cache`, `abort_on_failure` and `correlate_ratios` take only a TOML `true` or `false`; any other value stops the study at start, before its output folder is emptied.

### Config keys

| Key | Default | Meaning |
|:----|:--------|:--------|
| `seed` | required | Seed of the initial samples, the GP fits and the workers |
| `logging` | required | Log level: `"DEBUG"`, `"INFO"`, `"WARNING"` or `"ERROR"` |
| `output` | required | Output folder, inside the PROTEUS output folder |
| `ref_config` | required | PROTEUS config every evaluation starts from, relative to the PROTEUS folder |
| `init_samps`, `init_grid` | required | Initial samples, see above |
| `n_workers` | required | Parallel workers; must be below the number of CPU cores |
| `n_steps` | required | Evaluations after the initial samples |
| `kernel` | required | GP kernel, see [Kernels](#kernels) |
| `acqf` | required | Acquisition function, see [Acquisition functions](#acquisition-functions) |
| `patience` | `0` (off) | Stop after this many evaluations without improvement |
| `failure_codes` | `[]` | Completion codes left out of the fit, see [Failed and excluded simulations](#failed-and-excluded-simulations) |
| `abort_on_failure` | `false` | Stop starting new simulations at the first failure |
| `child_timeout_s` | `21600` | Wall-clock limit of one simulation in s; 0 or below turns it off |
| `spectral_cache` | `true` | Share prepared spectral files between workers |
| `dispatch`, `runner_max_jobs` | `"subprocess"`, `0` | See [Reusing one PROTEUS process per worker](#reusing-one-proteus-process-per-worker) |
| `objective` | `"chi-2"` | `"chi-2"`, `"student-t"` or `"huber"`; the last two need `[sigma]` |
| `student_t_dof`, `huber_k` | `4`, `1.345` | Parameter of the Student-t or Huber objective |
| `upper_limits` | none | Observables given as upper limits; needs `[sigma]` |
| `correlate_ratios` | `false` | Derive `[correlation]` for element ratios |
| `noise`, `noise_seed` | `false`, `seed` | Add one Gaussian draw of `[sigma]` to the observables |
| `bo_per_observable` | `false` | Optimise with one GP per observable |
| `posterior_samples` | `2000` with `[sigma]`, else `0` | Posterior samples at the end; `0` is off |
| `posterior_per_observable` | `true` | Posterior from one GP per observable |
| `[parameters]` | required | Config key of each parameter and its `[lower, upper]` bounds |
| `[observables]` | required | Helpfile column of each observable and its target value |
| `[sigma]`, `[correlation]`, `[truth]` | none | See the sections below |

An observable is any column of the PROTEUS helpfile, such as `R_obs`, `T_obs` or `atm_kg_per_mol`, or an element mass ratio of the atmosphere written `X/Y_atm`, such as `C/O_atm`. A parameter is any key of the PROTEUS config. Parameters that span orders of magnitude, such as `planet.elements.H_budget`, `C_budget` and `S_budget`, are sampled evenly in log10.

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

!!! warning "Do not compare `J` between objectives"
    The relative, chi-squared, Student-t and Huber objectives are on different scales, so compare `J` only between inference runs that use the same one. The objective in use is reported at start-up.

### Observable uncertainties

An optional `[sigma]` table gives the 1-sigma uncertainty of each observable, in the same
units as the value in `[observables]`. When it is present, each residual is divided by its uncertainty:

```
J = -log10( sum( ((sim - true) / sigma)^2 ) + 1e-10 )
```

Observables that span orders of magnitude (e.g. `atm_kg_per_mol`) are compared as `log10` values, and so are element mass ratios such as `C/O_atm` when `[sigma]` is given. Their uncertainty can be given in linear space, e.g. `"C/O_atm" = 0.1`; in this case, it is converted to dex to first order via `sigma / (true * ln 10)`. Optionally, it can be given in dex directly as a string, for example `"C/O_atm" = "0.1 dex"`. Any other positive observable also takes a sigma in dex, for example `"g_obs" = "0.05 dex"`, which makes the study compare it in log10 instead of linearly; this suits large relative errors, where linear noise could make the value negative.

### Upper limits

A species that was not detected gives an upper limit rather than a value. List such observables in `upper_limits`; their value in `[observables]` is then the limit, and their `[sigma]` its 1-sigma uncertainty:

```toml
upper_limits = ["S/O_atm"]

[observables]
"S/O_atm" = 0.01

[sigma]
"S/O_atm" = "0.2 dex"
```

Instead of `u^2`, an upper limit adds the censored Gaussian term

```
-2 ln Phi(-u),    u = (sim - limit) / sigma
```

where `Phi` is the standard normal cumulative distribution. It is 0 well below the limit, `2 ln 2` at it, and approaches `u^2` above it, so any simulated value well below the limit fits equally well. It applies with every `objective`. Upper limits cannot be combined with `[correlation]` or `correlate_ratios`; the study refuses to start. `noise = true` leaves a limit unchanged, because a limit is a reported value, not a measurement.

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

### Student-t and Huber errors

`objective` selects the error model: `"chi-2"` (the default, Gaussian errors), `"student-t"` or `"huber"`. Both alternatives need `[sigma]`, and both limit how strongly one observable the model cannot match pulls the fit.

With `objective = "student-t"`, each squared residual `u^2` is replaced by

```
(nu + 1) * ln(1 + u^2 / nu),    u = (sim - true) / sigma
```

with `student_t_dof = nu` (default 4). A residual of many sigma then grows only logarithmically. Small `nu` (3 to 5) gives heavy tails; `nu = 1` is the Cauchy distribution, and a large `nu` gives the chi-squared back. Under a Student-t, `sigma` is the scale of the error, not its standard deviation, which is `sigma * sqrt(nu / (nu - 2))` for `nu > 2`.

With `objective = "huber"`, each squared residual becomes

```
u^2                   for |u| <= k
2 k |u| - k^2         for |u| >  k,     u = (sim - true) / sigma
```

with `huber_k = k` (default 1.345). Within `k` sigma it is the chi-squared; beyond, a residual grows linearly. That bounds the pull of an outlier less strongly than a Student-t.

Both combine with `[correlation]` or `correlate_ratios`: the loss then applies to the whitened residuals `z = R^(-1/2) u`.

??? info "Details of the robust objectives"
    Each form is `-2 ln` of its density up to a constant: the Student-t density, and Huber's density, which is Gaussian in the core and Laplace in the tails; its two branches meet with equal value and slope at `|u| = k`. The default `k = 1.345` keeps 95% of the least-squares efficiency when the errors are Gaussian.

    With correlations, the errors are independent Student-t or Huber errors mixed by `R^(1/2)`, which is again `-2 ln L` up to a constant; with a large `nu` or `k` this is the correlated chi-squared. `R^(-1/2)` is the symmetric inverse square root of the correlation matrix. It keeps each `z` closest to its own observable and does not depend on the order of the observables, and an observable correlated with none (such as `T_obs` under `correlate_ratios`) keeps its own residual, so an outlier there is down-weighted as without correlation. An outlier in one correlated ratio spreads over the whitened residuals of that group, so it is bounded less sharply.

### Known true parameters

When the target observables were extracted from a simulation whose parameters you know (a
synthetic retrieval test), an optional `[truth]` table records those parameters so the best fit can be compared against them:

```toml
[truth]
"interior_struct.core_frac" = 0.325
"outgas.fO2_shift_IW"       = 2.0
```

The table does not change the optimisation; it adds a True column to the results summary and the `result_parameters.png` plot, and marks the truth on `result_posterior.png`.

#### Noisy observables

The observables of a simulation carry no noise, so the truth sits at the peak of the likelihood. With `noise = true` (needs `[sigma]`), the study replaces each observable by one Gaussian draw around it, as a real observation would give: linear noise of `sigma` for observables compared linearly, and noise in log10 for those compared in log10 (element ratios, logarithmic variables and observables with a sigma in dex), correlated as in `[correlation]` or `correlate_ratios`. The sigma of a log10 observable is rescaled with its value, so its error in dex stays the same. `noise_seed` (default `seed`) picks the draw, and the log lists the noise-free and the noisy values. Over many draws, the 90 % posterior intervals should hold the truth in about 90 % of cases.

### Posterior samples

With `[sigma]`, every objective is `-2 ln L` up to a constant, where `L` is the likelihood, so the study can sample the posterior of the parameters. At the end of the study it fits one GP per observable to the scored runs, predicts the observables anywhere in the parameter space, and computes `J` from them as for a run. It then draws `posterior_samples` (default 2000, `0` is off) samples from `L` at that prediction, with a prior that is uniform over the sampled ranges (log-uniform for log10-scaled parameters). The samples go to `posterior.csv` and `result_posterior.png`, and the log reports their 5, 50 and 95 % quantiles and the effective sample size, with a warning when it is below the number of samples.

The posterior ignores the GPs' own uncertainty, so it is only as good as the GPs near the best fit: check that the study converged (`perf_bestval.png`) before using it.

With `posterior_per_observable = false`, one GP is fitted to `J` itself, as the optimisation does by default. Near the best fit `J` has steep, narrow valleys that a single GP follows poorly; in test studies the per-observable GPs predicted `J` more accurately and their intervals held the true parameters more often. Fitting them adds minutes to tens of minutes at the end of the study.

??? info "How the samples are drawn"
    `S = 10^(-J) - 1e-10` is `-2 ln L` up to a constant, so `ln L = -10^(-J) / 2` up to a constant. The samples are drawn by importance resampling of 10^6 uniform draws. When the posterior is so narrow that these give fewer effective samples than requested, up to five more rounds draw from a Gaussian mixture centred on the best-weighted draws (with a 10 % uniform share), until they do.

### Parallel Processing

- Multiple workers run simultaneously, each performing BO steps
- Workers share a common dataset but operate independently
- Lock mechanisms prevent race conditions when updating shared data
- Each worker tracks "busy" locations to avoid redundant evaluations

### Bayesian Optimization

- Uses Gaussian Process (GP) models to predict objective values
- Acquisition function guides exploration-exploitation trade-off on search space
- Automatic hyperparameter tuning via marginal likelihood optimization

??? info "Two safeguards on the proposals"
    - A parameter whose fitted lengthscale exceeds 1.5 (in the unit cube) barely changes the GP, which then puts every proposal on one of its bounds; such a parameter is drawn at random instead (logged at debug level).
    - A proposal within 0.005 (in the unit cube) of an evaluated or pending point is replaced by the best acquisition restart further away, or by a random point when there is none.

The optimization runs `n_steps` evaluations after the initial samples, or until it is stopped. `data.csv` is written after every evaluation, but a stopped study cannot be resumed.

With `patience = N` (default 0, off), the study stops earlier, once `N` evaluations in a row have not raised the best objective by more than 0.01. Stalls of tens of evaluations happen, so set patience after a few test runs; too low a value stops a study early.

### One GP per observable

With `bo_per_observable = true` (default `false`), each optimisation step fits one GP per observable instead of one GP to `J`, as the [posterior](#posterior-samples) does by default. Each GP models its observable in the space the objective compares it in (log10 or linear), with its noise learned, and the acquisition scores a candidate by `J` of joint draws from these GPs, computed as for a run. The steep, narrow valley of `J` near the best fit then comes from the formula of `J`, which the single GP has to learn from the runs.

- `acqf` must be `LogEI`, `LogPI` or `UCB`, each in its Monte Carlo form; `TS` is refused.
- `init_samps` must be larger than the number of parameters.
- A run that failed or was excluded enters no GP. An observable compared in log10 whose value is zero, as the gas fractions of an escaped atmosphere are, or a non-finite value, is left out of that observable's GP only; `J` of that run still counts as the best value to improve on.
- `init.csv` and `data.csv` gain one `s_*` column per observable, in the order of `[observables]`, holding the values the GPs are fitted to (NaN where left out).
- A step takes a few seconds longer than with one GP: one fit per observable, and a Monte Carlo acquisition.

### Acquisition functions

The acquisition function is an analytical function that is aware of the current state of the optimisation.
It is used to evaluate the *potential* value of sampling a candidate particular point in the parameter space, to
help determine where the optimisation should next run PROTEUS. It helps balance the trade-off between exploring new areas and exploiting known good areas to optimize a black-box function efficiently.

* `UCB` - upper confidence bound
* `LogEI` - logarithm of the expected improvement
* `LogPI` - logarithm of the probability of improvement
* `TS` - Thompson sampling: the maximum of one function drawn from the GP posterior; each worker draws its own, so the workers spread out without the batch forms below

While other workers have runs in progress, `UCB`, `LogEI` and `LogPI` switch to their Monte Carlo batch forms so that they account for those points. BoTorch has no batch form of LogPI, so `LogPI` then uses the plain probability of improvement, which is flatter far from the best point; prefer `LogEI` with several workers.

See docs [here](https://botorch.readthedocs.io/en/latest/acquisition.html).

### Kernels

The kernel is an analytical function used by the Gaussian processes to represent the similarity between model behaviour as a function of the parameter space. It includes the underlying function by capturing the relationships and uncertainties/noise in the data.

* `RBF` - radial basis function
* `MAT1/2` - Matérn kernel with $\nu = 1/2$
* `MAT3/2` - Matérn kernel with $\nu = 3/2$
* `MAT5/2` - Matérn kernel with $\nu = 5/2$

See docs [here](https://botorch.readthedocs.io/en/latest/models.html).

## Output

The system generates several outputs in:

### Data Files
- `data.csv`: Final dataset with all evaluated parameters (`x_*`) and objective values (`y`), and with `bo_per_observable` the observables (`s_*`)
- `logs.csv`: Detailed logs of each BO step
- `Ts.csv`: Timestamps for performance analysis
- `init.csv`: Data used as an initial guess for starting the optimisation
- `copy.infer.toml`: The inference config as given
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
- `result_posterior.png`: Only with `[sigma]`. Corner plot of the posterior as 68 and 95% credible regions, with every evaluation, the best fit and, if given, the truth on top.
- `result_parameters.png`: Only with a `[truth]` table. Every sample, the truth and the best fit placed within each parameter's sampled range.

### Results Summary
The system prints the final results including:

- Best found parameters
- Corresponding simulated observables
- Comparison with target observables

### Failed and excluded simulations

During the inference run, some PROTEUS simulations might crash or fail, or stop on a completion code listed in `failure_codes` (e.g. 11, maximum runtime reached). Status 29 (planet evaporated) is always excluded. The run carries on when there are failures unless `abort_on_failure` is set to `true` in the inference config. The study then starts no new simulations, but those already running finish first. Each failure is added to `failures.csv` in the output folder as it happens, and all failures are summarised at the end of the study. A study stopped by `abort_on_failure` keeps its `failures.csv` but prints no summary.

A failed or excluded simulation is recorded in `data.csv` with the objective value -20, but the Gaussian process is fitted to the scored simulations only, so that value does not distort the surrogate. The parameters of a failed simulation still count as evaluated, so the optimiser does not propose the same point again.

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
