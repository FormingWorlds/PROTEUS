# Sub-Neptune

This tutorial simulates a sub-Neptune-class planet using the boundary
interior module coupled with Aragog. Sub-Neptunes have thick
hydrogen/helium envelopes atop a rocky core; the boundary module
provides a parameterised thermal evolution for the envelope while
Aragog handles the rocky interior.

## Configuration

Create `input/tutorial_sub_neptune.toml`:

```toml
config_version = "3.0"

[params.out]
    path = "tutorial_sub_neptune"

[params.dt]
    initial = 1e2
    minimum = 1e2
    maximum = 1e7

    [params.stop.time]
        maximum = 5e9
    [params.stop.solid]
        phi_crit = 0.005

[star]
    module  = "mors"
    mass    = 0.8
    age_ini = 0.01
    [star.mors]
        tracks          = "spada"
        spectrum_source = "phoenix"

[orbit]
    semimajoraxis = 0.1

[planet]
    mass_tot         = 5.0
    temperature_mode = "liquidus_super"
    delta_T_super    = 500.0
    volatile_mode    = "elements"
    [planet.elements]
        H_mode   = "oceans"
        H_budget = 10.0
        C_mode   = "C/H"
        C_budget = 0.5
        N_mode   = "N/H"
        N_budget = 0.3
        S_mode   = "S/H"
        S_budget = 0.5

[interior_struct]
    module    = "zalmoxis"
    core_frac = 0.30

[interior_energetics]
    module = "boundary"

[outgas]
    module       = "calliope"
    fO2_shift_IW = 2

[atmos_clim]
    module     = "agni"
    surf_state = "fixed"

[escape]
    module = "zephyrus"
    [escape.zephyrus]
        efficiency = 0.1
```

### Key features

- **Planet**: 5 M$_\oplus$ with 10 ocean equivalents of hydrogen, creating
  a massive volatile envelope
- **Interior energetics**: The `boundary` module uses a 0-D box model
  (Schaefer et al. 2016) for the mantle thermal evolution. This is
  computationally lighter than Aragog for sub-Neptune-class planets
  where the detailed interior radial structure matters less than the
  envelope evolution.
- **Interior structure**: Zalmoxis still computes the hydrostatic
  density profile with PALEOS EOS, providing the planet radius and
  core-mantle boundary conditions
- **Atmosphere**: `surf_state = "fixed"` locks the surface temperature
  to the interior magma temperature, appropriate when the envelope is
  thick enough to suppress surface-atmosphere decoupling

## Running

```bash
conda activate proteus
nohup proteus start --offline -c input/tutorial_sub_neptune.toml \
    > output/tutorial_sub_neptune/launch.log 2>&1 & disown
```

## What to expect

1. **Large planet radius**: With 10 ocean-equivalents of hydrogen on a
   5 M$_\oplus$ core, the planet radius is significantly larger than
   Earth. Check `R_int` and `R_obs` in the helpfile.

2. **Thick atmosphere**: Surface pressures can reach hundreds to
   thousands of bar. The atmosphere contributes a measurable fraction
   of the total planet mass.

3. **Slow cooling**: The thick envelope acts as an insulating blanket,
   trapping interior heat. The magma ocean solidification timescale is
   longer than for bare rocky planets.

4. **Escape competition**: Whether the planet retains its envelope over
   Gyr timescales depends on the balance between XUV-driven escape and
   the gravitational binding energy of the envelope. At 5 M$_\oplus$
   and 0.1 AU, the planet is near the "radius valley" boundary.

## Exercises

1. Reduce `mass_tot` to 2.0 and `H_budget` to 5.0. Does the planet
   lose its envelope and become a super-Earth?
2. Move the planet to 0.03 AU. How does the stronger irradiation and
   escape affect the final state?
3. Switch to `interior_energetics.module = "aragog"` for a comparison.
   The run will be slower but provides detailed radial temperature and
   melt fraction profiles.
4. Enable the H$_2$-silicate miscibility model: set
   `interior_struct.zalmoxis.global_miscibility = true` and
   `outgas.h2_binodal = true`. How does H$_2$ dissolution into the
   magma affect the envelope mass?
