# Stabilising PROTEUS simulations

As new regimes of planetary physics are explored, many research projects push the PROTEUS code to its limits. For science, this is great – but sometimes the code struggles with numerical issues. When a simulation fails to converge, systematic adjustments to numerical
settings, module complexity, or initial conditions usually resolve it. This
page contains some tips to help stabilise your simulations.

!!! tip "Check the logs first"
    The PROTEUS output log identifies which module triggered the failure and
    at which iteration. Look for `ERROR` or `WARNING` lines to narrow down
    where to start before trying anything below.

---

## AGNI fails

Try the following in roughly this order.

**1. Reduce atmospheric grid resolution**

```toml
[atmos_clim]
    num_levels = 50     # or even lower
```

**2. Disable optional physics**

Disable all of these first, then re-enable one at a time to identify the
culprit:

```toml
[atmos_clim]
    cloud_enabled    = false
    aerosols_enabled = false

[atmos_clim.agni]
    latent_heat = false     # disable latent heating
    chemistry   = "none"    # disable atmospheric chemistry
    real_gas    = false     # use ideal gas EOS
```

**3. Relax Newton solver tolerances**

Giving AGNI more room to converge might resolve edge cases:

```toml
[atmos_clim.agni]
    solution_atol = 1.0    # default: 0.5  [W m^-2]
    solution_rtol = 0.2    # default: 0.15
```

**4. Increase spectral resolution**

More spectral bands give the radiative transfer solver a better-resolved
spectrum to work with, which can improve convergence, at the cost of
longer runtimes:

```toml
[atmos_clim]
    spectral_bands = "256"    # or "4096" for maximum resolution
```

---

## SPIDER fails 

Reduce the interior grid resolution:

```toml
[interior_energetics]
    num_levels = 50          
```

---

## Tidal Love-number resonances (Obliqua)

If Obliqua's Love-number spectrum (`plot_lovenumber.png`) shows a point
ringed red ("seismic resonance", `Re(k2) > 1.5` or `Im(k2) > 1.0`), or
the orbit/spin state jumps abruptly over one or a few iterations, the
tidal forcing frequency likely crossed one of the interior's own dynamic
normal-mode resonances. These resonances occur at the same time the 
mantle is partially molten, hence to properly resolve them reduce the 
mushy-regime timestep limit: 

```toml
[params.dt]
    mushy_maximum = 1.0e4   # or lower; these resonances only occur while
                            # part of the mantle is molten/mushy, so a
                            # tighter mushy-regime cap gives PROTEUS more
                            # chances to re-sample Obliqua during a crossing
    mushy_upper   = 0.99
```

Alternatively, PROTEUS is able to cap the Love number Obliqua
returns as a safety net. This suppresses divergences that arise at large
timesteps, by limiting the overall tidal response:

```toml
[orbit.obliqua]
    cap_LN = true   # clamp each mode's Re(k2)/Im(k2) to 3x/2x the fluid Love-number limit for its degree n
```


---

## General tips

**Use the `--deterministic` flag**

If a simulation fails on noise-floor floating-point divergence, for example when you get the message `RuntimeError: Aragog retry ladder exhausted` or with T_core jumping warnings, the `--deterministic` flag enforces stricter numerical reproducibility via JAX and XLA settings:

```console
proteus start -c my_config.toml --deterministic
```

!!! warning "Do not enable by default"
    Only enable the `--deterministic`-flag when a config shows noise-floor divergence; do not enable by default.

**For parameter grids**

Simulations that converge in isolation can fail at the edges of a
parameter grid. If grid runs are failing, first identify which parameter
combination is causing the issue using:

```console
proteus grid-summarise -o output/my_grid/ --status error
```

Then apply the relevant fix from this page to that region of parameter space.

