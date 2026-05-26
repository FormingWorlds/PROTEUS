# Hot rocky super-Earth

This tutorial simulates a 3 M$_\oplus$ rocky planet at 0.05 AU from an
M-dwarf star. The close orbit means high irradiation, strong atmospheric
escape, and rapid magma ocean solidification driven by the competition
between intense stellar heating and interior cooling.

## Configuration

The config file is at `input/tutorials/tutorial_hot_se.toml`:

```toml
config_version = "3.0"

[params.out]
    path = "tutorial_hot_se"

[params.dt]
    initial = 1e2
    minimum = 1e2
    maximum = 1e7

    [params.stop.time]
        maximum = 5e9

[star]
    module  = "mors"
    mass    = 0.3
    age_ini = 0.01
    [star.mors]
        tracks          = "spada"
        spectrum_source = "phoenix"

[orbit]
    semimajoraxis = 0.05

[planet]
    mass_tot         = 3.0
    temperature_mode = "liquidus_super"
    delta_T_super    = 500.0
    volatile_mode    = "elements"
    [planet.elements]
        H_mode   = "oceans"
        H_budget = 3.0
        C_mode   = "C/H"
        C_budget = 1.0
        N_mode   = "N/H"
        N_budget = 0.5
        S_mode   = "S/H"
        S_budget = 1.0

[interior_struct]
    module    = "zalmoxis"
    core_frac = 0.35

[interior_energetics]
    module = "aragog"

[outgas]
    module       = "calliope"
    fO2_shift_IW = 3

[atmos_clim]
    module = "agni"

[escape]
    module = "zephyrus"
    [escape.zephyrus]
        efficiency = 0.15
```

### Key differences from the Earth analogue

- **Star**: 0.3 M$_\odot$ M-dwarf on Spada[^cite-spada2013] tracks, starting at 10 Myr
  (young, XUV-bright)
- **Orbit**: 0.05 AU (receives ~100x Earth's instellation)
- **Planet**: 3 M$_\oplus$ with larger core fraction and higher hydrogen
  inventory (3 oceans)
- **Escape efficiency**: 0.15 (higher for the extended upper atmosphere
  of a close-in planet)

## Running

```bash
conda activate proteus
nohup proteus start --offline -c input/tutorials/tutorial_hot_se.toml \
    > output/tutorial_hot_se/launch.log 2>&1 & disown
```

## What to expect

1. **Rapid early cooling**: The high instellation creates a strong
   greenhouse but the 3x Earth mass means more thermal inertia. The
   magma ocean solidifies in ~50-200 Myr.

2. **Intense escape**: The young M-dwarf emits strong XUV. With 3 oceans
   of hydrogen and a 0.05 AU orbit, atmospheric escape can strip
   significant mass. Watch `esc_rate_total` in the helpfile; it may
   reach $10^8$-$10^9$ kg/s early on.

3. **Atmosphere retention**: Whether the planet retains its atmosphere
   depends on the race between escape and solidification. A thicker
   atmosphere (higher P_surf) has a higher XUV absorption radius but
   also a stronger greenhouse.

4. **Final state**: The planet may end up as a bare rock (desiccated),
   a Venus analogue (thick CO$_2$ atmosphere), or retain some water
   depending on the escape efficiency and volatile inventory.

## Exercises

1. Reduce `H_budget` to 0.5 oceans. Does the planet desiccate?
2. Move the planet to 0.1 AU. How does the escape rate change?
3. Increase `escape.zephyrus.efficiency` to 0.3. How much faster
   does the atmosphere strip?
4. Enable tidal heating: set `orbit.module = "dummy"` with
   `orbit.dummy.H_tide = 1e-7`.

---

**See also:** [Model description](../Explanations/model.md) | [Coupling loop](../Explanations/coupling_loop.md) | [Configuration reference](../Reference/config/params.md) | [Output format](../Reference/output.md)

[^cite-spada2013]: Spada, F., Demarque, P., Kim, Y.C. & Sills, A., *[The radius discrepancy in low-mass stars: single versus binaries](https://doi.org/10.1088/0004-637X/776/2/87)*, The Astrophysical Journal, 776, 87, 2013. [SciX](https://scixplorer.org/abs/2013ApJ...776...87S/abstract).
