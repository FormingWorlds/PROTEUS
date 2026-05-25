# Quick start: all-dummy run

This tutorial runs PROTEUS with all modules set to "dummy" backends. No
external solvers (AGNI, SPIDER, SOCRATES) are needed; the run completes
in under a minute and exercises the full coupling architecture. Use this
to verify your installation and understand the code flow before moving
to production runs.

## Prerequisites

- PROTEUS installed (`pip install -e ".[develop]"`)
- `FWL_DATA` environment variable set

No external solvers, spectral files, or EOS data are required.

## The configuration file

PROTEUS ships with an all-dummy config at `input/dummy.toml`. Here is what
it sets:

```toml
config_version = "3.0"

[params.out]
    path = "auto"

[params.dt]
    initial  = 1e2
    minimum  = 1e2
    maximum  = 3e7

    [params.stop.time]
        maximum = 1e9
    [params.stop.solid]
        enabled = false
    [params.stop.radeqm]
        enabled = false

[star]
    module = "dummy"

[orbit]
    module        = "dummy"
    semimajoraxis = 0.5

[planet]
    mass_tot      = 1.0
    volatile_mode = "elements"
    [planet.elements]
        H_mode   = "ppmw"
        H_budget = 3e3
        N_budget = 0.0

[interior_struct]
    module         = "dummy"
    core_frac      = 0.55
    core_frac_mode = "radius"

[interior_energetics]
    module = "dummy"

[outgas]
    fO2_shift_IW = 2
    module       = "dummy"

[atmos_clim]
    module     = "dummy"
    surf_state = "fixed"
    rayleigh   = false
    albedo_pl  = 0.1
    [atmos_clim.dummy]
        gamma = 0.7

[escape]
    module = "dummy"
    [escape.dummy]
        rate = 1e6

[atmos_chem]
    module = "dummy"
    when   = "offline"
```

Every module is set to `"dummy"`, meaning:

- **Star**: fixed solar luminosity, no evolution
- **Orbit**: 0.5 AU, weak tidal heating
- **Interior structure**: Noack & Lasbleis (2020)[^cite-noack2020] scaling laws
- **Interior energetics**: parameterised cooling with prescribed solidus/liquidus
- **Outgassing**: fixed composition partitioning
- **Atmosphere**: grey-body opacity ($T_\mathrm{rad} = T_\mathrm{surf} \cdot (1 - \gamma)$)
- **Escape**: constant bulk rate of $10^6$ kg/s
- **Chemistry**: fixed composition

## Running the simulation

```bash
conda activate proteus
proteus start --offline -c input/dummy.toml
```

The `--offline` flag skips data downloads. The run should complete in
30-60 seconds.

## Understanding the output

The run creates a directory inside `output/` with a timestamped name.
Open `runtime_helpfile.csv` to see the time series:

```bash
cd output/run_*
head -5 runtime_helpfile.csv
```

Key columns to look for:

- `Time`: simulation time [yr]; stays at 0 for the first 3 iterations
  (initialisation stage), then advances
- `T_magma`: mantle potential temperature [K]; should decrease over time
- `T_surf`: surface temperature [K]; tracks `T_magma` since `surf_state = "fixed"`
- `Phi_global`: global melt fraction; decreases as the mantle cools
- `F_atm`: outgoing atmospheric flux [W/m$^2$]
- `F_int`: interior heat flux [W/m$^2$]
- `P_surf`: surface pressure [bar]
- `esc_rate_total`: escape rate [kg/s]; constant at $10^6$ kg/s

## What to look for

1. **Time evolution**: After the 3 init iterations, `Time` should increase
   from zero and the timestep should grow adaptively.

2. **Cooling**: `T_magma` decreases monotonically. When it crosses the
   solidus (~1700 K), `Phi_global` drops to zero and the run terminates.

3. **Mass conservation**: `M_planet` should remain constant (within rounding).
   The atmospheric mass changes as volatiles escape, but the total planet
   mass is conserved.

4. **Plots**: Check `plots/plot_global.png` for a multi-panel overview of
   the evolution.

## Next steps

- Try changing `H_budget` to see how the volatile inventory affects
  the atmospheric evolution
- Increase `escape.dummy.rate` to see faster atmospheric loss
- Change `atmos_clim.dummy.gamma` to vary the greenhouse effect
- Move on to the [Earth analogue tutorial](earth_analogue.md) for a
  production-quality run with real physics modules

---

**See also:** [Model description](../Explanations/model.md) | [Coupling loop](../Explanations/coupling_loop.md) | [Configuration reference](../Reference/config/params.md) | [Output format](../Reference/output.md)

[^cite-noack2020]: Noack, L. & Lasbleis, M., *[Parameterisations of interior properties of rocky planets](https://doi.org/10.1051/0004-6361/202037723)*, Astronomy & Astrophysics, 638, A129, 2020. [SciX](https://scixplorer.org/abs/2020A%26A...638A.129N/abstract).
