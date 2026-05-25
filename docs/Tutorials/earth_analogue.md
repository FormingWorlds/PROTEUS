# Earth analogue

This tutorial simulates the thermal and atmospheric evolution of an
Earth-mass rocky planet at 1 AU from a Sun-like star. It uses the
production-quality module combination:
[Aragog](https://proteus-framework.org/aragog/) (interior energetics),
[Zalmoxis](https://proteus-framework.org/Zalmoxis/) (interior structure),
[CALLIOPE](https://proteus-framework.org/CALLIOPE/) (outgassing), and
[AGNI](https://www.h-nicholls.space/AGNI/) (atmosphere climate).

## Prerequisites

- Full PROTEUS installation with AGNI and SOCRATES compiled
- `FWL_DATA` and `RAD_DIR` environment variables set
- Spectral files downloaded (`proteus get spectral`)
- Stellar spectra downloaded (`proteus get phoenix`)

## Configuration

Create a file `input/tutorial_earth.toml`:

```toml
config_version = "3.0"

[params.out]
    path    = "tutorial_earth"
    logging = "INFO"

[params.dt]
    initial = 1e2
    minimum = 1e3
    maximum = 1e7

    [params.stop.time]
        maximum = 5e9

[star]
    module  = "mors"
    mass    = 1.0
    age_ini = 0.1
    [star.mors]
        tracks          = "spada"
        spectrum_source = "phoenix"

[orbit]
    semimajoraxis = 1.0

[planet]
    mass_tot         = 1.0
    temperature_mode = "liquidus_super"
    delta_T_super    = 500.0
    volatile_mode    = "elements"
    [planet.elements]
        H_mode   = "oceans"
        H_budget = 1.0
        C_mode   = "C/H"
        C_budget = 1.0
        N_mode   = "N/H"
        N_budget = 0.5
        S_mode   = "S/H"
        S_budget = 1.0

[interior_struct]
    module    = "zalmoxis"
    core_frac = 0.325

[interior_energetics]
    module = "aragog"

[outgas]
    module       = "calliope"
    fO2_shift_IW = 4

[atmos_clim]
    module = "agni"

[escape]
    module = "zephyrus"
```

### What this configuration does

- **Planet**: 1 M$_\oplus$, starts fully molten (liquidus + 500 K at the CMB),
  1 Earth ocean equivalent of hydrogen, solar-ratio C/N/S
- **Star**: Solar-mass star on Spada evolutionary tracks, starting at 100 Myr
- **Orbit**: 1 AU, no tidal heating
- **Interior**: Zalmoxis computes the hydrostatic density profile with PALEOS
  EOS tables; Aragog solves the mantle thermal evolution with CVODE + JAX
  analytic Jacobian
- **Outgassing**: CALLIOPE computes the gas-melt equilibrium at fO$_2$ = IW+4
  (moderately oxidised, Earth-like)
- **Atmosphere**: AGNI solves the radiative-convective structure with
  SOCRATES radiative transfer
- **Escape**: ZEPHYRUS energy-limited escape driven by stellar XUV flux

## Running

```bash
conda activate proteus
proteus start --offline -c input/tutorial_earth.toml
```

!!! note
    This run takes approximately 30 minutes to 2 hours depending on hardware.
    For long runs, launch in the background:
    ```bash
    nohup proteus start --offline -c input/tutorial_earth.toml \
        > output/tutorial_earth/launch.log 2>&1 & disown
    ```

## Interpreting the results

### Thermal evolution

Open `output/tutorial_earth/plots/plot_global.png`. The top panel shows
the evolution of key temperatures:

- **T_magma** starts near 4500 K (fully molten) and cools over ~100 Myr
  to below the solidus (~1700 K)
- **T_surf** tracks T_magma initially (the magma ocean surface radiates
  directly) but decouples once a thick atmosphere builds up
- **Phi_global** (melt fraction) decreases from ~1 to 0 as the mantle
  solidifies; the transition through the mushy zone is the most
  scientifically interesting phase

### Atmospheric evolution

- **P_surf** builds up as volatiles outgas from the cooling magma ocean,
  reaching tens to hundreds of bar
- **H2O and CO2 partial pressures** show the dominant atmospheric species
- After solidification, atmospheric escape slowly strips the atmosphere
  over Gyr timescales

### Interior structure

Check `plots/plot_interior.png` for radial profiles of temperature,
viscosity, and melt fraction at selected snapshots. The rheological
front (transition from liquid to solid) migrates inward as the mantle
cools.

### Energy balance

`plots/plot_fluxes_global.png` shows the competition between interior
heat flux (F_int, cooling the mantle) and atmospheric outgoing radiation
(F_atm). When F_int > F_atm, the surface heats up; when F_atm > F_int,
it cools. Radiative equilibrium marks the end of the magma ocean phase.

## Exercises

1. **Change the volatile inventory**: Double `H_budget` to 2.0 oceans.
   How does the thicker steam atmosphere affect the cooling timescale?

2. **Change the redox state**: Set `fO2_shift_IW = -2` for a reduced
   mantle. How does the atmospheric composition change (more H$_2$ and
   CO, less H$_2$O and CO$_2$)?

3. **Move the planet closer**: Set `semimajoraxis = 0.3` AU. How does
   the higher instellation affect the surface temperature and escape?

4. **Change the star**: Use `tracks = "baraffe"` with `mass = 0.3` for
   an M-dwarf host star. Download MUSCLES spectra for TRAPPIST-1 and
   set `spectrum_source = "muscles"`, `star_name = "trappist-1"`.

## Next tutorials

- [Hot rocky super-Earth](hot_super_earth.md): high irradiation, rapid escape
- [Reduced H$_2$-rich world](reduced_h2_world.md): low fO$_2$ chemistry
- [Sub-Neptune](sub_neptune.md): boundary interior module
- [Parameter grid sweep](parameter_grid.md): ensemble runs

---

**See also:** [Model description](../Explanations/model.md) | [Coupling loop](../Explanations/coupling_loop.md) | [Configuration reference](../Reference/config/params.md) | [Output format](../Reference/output.md)
