# Execution and output

The `[params]` section controls code execution, output settings, time-stepping,
and termination criteria. These parameters govern how the simulation runs, not
what it simulates.

See also: [Coupling loop](../../Explanations/coupling_loop.md) for how
time-stepping and convergence work in practice.

## Output settings `[params.out]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str | `"auto"` | Output folder inside `output/`. `"auto"` generates a unique timestamped name (`run_YYYYMMDD_HHMMSS_xxxx`). |
| `logging` | str | `"INFO"` | Log verbosity: `INFO`, `DEBUG`, `ERROR`, `WARNING` |
| `plot_fmt` | str | `"png"` | Plot format: `png` or `pdf` |
| `plot_mod` | int or none | `5` | Plot frequency: `0` = at end only, `n` = every n iterations, `none` = never |
| `write_mod` | int | `1` | Helpfile write frequency: `0` = at end only, `n` = every n iterations |
| `dt_write_rel` | float | `0.0` | Minimum write interval as fraction of elapsed simulation time. Prevents excessive I/O during early rapid evolution. `0` = disabled. |
| `archive_mod` | int or none | `none` | Archive frequency: `0` = at end, `n` = every n iterations, `none` = never |
| `remove_sf` | bool | `false` | Remove SOCRATES spectral files after simulation completes |

## Time-stepping `[params.dt]`

PROTEUS supports three time-stepping methods. The adaptive method is
recommended for production runs; proportional is useful for steady-state
problems; maximum gives a fixed step.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | `"adaptive"` | Time-stepping method: `adaptive`, `proportional`, `maximum` |
| `initial` | float | `30` | Initial time step \[yr] |
| `minimum` | float | `1e4` | Minimum allowed time step \[yr] |
| `minimum_rel` | float | `1e-5` | Minimum relative time step (fraction of current time) |
| `maximum` | float | `1e7` | Maximum allowed time step \[yr] |
| `maximum_rel` | float | `1.0` | Relative cap on the maximum step; the effective maximum is `maximum + maximum_rel * Time`. Set `0.0` for a fixed maximum |

### Adaptive method parameters

These parameters control the adaptive time-stepping when `method = "adaptive"`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `atol` | float | `0.02` | Absolute tolerance on fractional state change per step |
| `rtol` | float | `0.10` | Relative tolerance on fractional state change per step |
| `scale_incr` | float | `1.6` | Step growth factor on successful step (must be > 1) |
| `scale_decr` | float | `0.8` | Step shrink factor on rejected step (must be in (0, 1)) |
| `window` | int | `3` | Number of previous steps to consider for adaptive comparison |
| `max_growth_factor` | float | `0.0` | Cap on step-to-step growth ratio; `0` = disabled |

### Proportional method

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `propconst` | float | `52.0` | Proportionality constant: $\Delta t = t / C$ |

### Mushy zone time-stepping

During the solidification transition (melt fraction between `phi_crit` and
`mushy_upper`), the timestep can be capped to resolve the rapid phase change.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mushy_maximum` | float | `0.0` | Maximum time step during mushy zone \[yr]; `0` = disabled |
| `mushy_upper` | float | `0.99` | Upper melt fraction bound for the mushy regime |
| `hysteresis_iters` | int | `0` | Suppress speed-up for N iterations after a slow-down; `0` = disabled |
| `hysteresis_sfinc` | float | `1.1` | Gentler speed-up factor during hysteresis |

### Spectrum update intervals

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `starspec` | float | `1e8` | Recalculate stellar spectrum every this many years |
| `starinst` | float | `1e2` | Recalculate instellation flux every this many years |

## Termination criteria `[params.stop]`

Each criterion can be independently enabled. The simulation terminates when any
enabled criterion is satisfied. Set `strict = true` to require the criterion
to be satisfied for two consecutive iterations before terminating.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strict` | bool | `false` | Require criteria satisfied on two consecutive iterations |

### Iteration limits `[params.stop.iters]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `true` | Enable iteration count limits |
| `minimum` | int | `5` | Run at least this many iterations before any termination |
| `maximum` | int | `9000` | Terminate after this many iterations |

### Time limits `[params.stop.time]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `true` | Enable simulation time limits |
| `minimum` | float | `1e3` | Run at least this long \[yr] |
| `maximum` | float | `6e9` | Terminate after this time \[yr] |

### Solidification `[params.stop.solid]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `true` | Terminate when mantle solidifies |
| `phi_crit` | float | `0.01` | Stop when global melt fraction falls below this value |
| `freeze_volatiles` | bool | `false` | Freeze outgassing at solidification but continue evolution |

### Radiative equilibrium `[params.stop.radeqm]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `true` | Terminate at radiative equilibrium |
| `atol` | float | `1.0` | Absolute tolerance on $\|F_\mathrm{int} - F_\mathrm{atm}\|$ \[W m$^{-2}$] |
| `rtol` | float | `1e-3` | Relative tolerance on energy balance |

### Atmosphere escape `[params.stop.escape]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `true` | Terminate when atmosphere is lost |
| `p_stop` | float | `3.0` | Stop when surface pressure falls below this value \[bar] |

### Planetary disintegration `[params.stop.disint]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Enable disintegration criteria |
| `roche_enabled` | bool | `true` | Check Roche limit |
| `offset_roche` | float | `0` | Correction to Roche limit \[m] |
| `spin_enabled` | bool | `true` | Check rotational breakup |
| `offset_spin` | float | `0` | Correction to breakup period \[s] |
