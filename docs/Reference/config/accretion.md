# Accretion

The `[accretion]` section configures protoplanet growth by giant impacts: which
model supplies the impact history, what each impactor carries, and how much
atmosphere a collision removes.

Submodule documentation:
[Morrigan](https://proteus-framework.org/Morrigan/).
See also [Model description](../../Explanations/model.md#accretion-morrigan)
and the [coupling loop](../../Explanations/coupling_loop.md#execution-order-per-iteration).

## Accretion `[accretion]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | str or none | `"none"` | Accretion module: `morrigan` (dynamical model), `timeline` (replay a file), `dummy` (analytical growth), `none` (disabled) |
| `time_offset` | float | `0.0` | Offset applied to every impact time when mapping the timeline onto the PROTEUS time axis \[yr]. Impacts landing at or before the start of the run are discarded with a warning |
| `impactor_volatiles` | str | `"dry"` | Volatile content of each impactor: `dry` (rock and iron only), `match_planet` (the planet's own initial fractional abundances, scaled to the impactor mass), `ppmw` (the per-element budgets below) |
| `impactor_H_ppmw` | float | `0.0` | Hydrogen carried by each impactor \[ppmw of impactor mass] |
| `impactor_C_ppmw` | float | `0.0` | Carbon carried by each impactor \[ppmw of impactor mass] |
| `impactor_N_ppmw` | float | `0.0` | Nitrogen carried by each impactor \[ppmw of impactor mass] |
| `impactor_S_ppmw` | float | `0.0` | Sulfur carried by each impactor \[ppmw of impactor mass] |
| `impactor_O_ppmw` | float | `0.0` | Oxygen carried by each impactor \[ppmw of impactor mass] |
| `atmloss_module` | str or none | `"none"` | Impact atmosphere loss: `constant` (the fixed fraction below), `zephyrus` (the giant-impact erosion scaling of Kegerreis et al. 2020 [^cite-kegerreis2020]), `none` (no loss) |
| `atmloss_frac` | float | `0.0` | Fraction of the atmosphere each impact removes when `atmloss_module = "constant"` \[0, 1] |

One loss fraction governs both bodies at each impact: the target loses that
fraction of its atmosphere, and a volatile-bearing impactor loses the same
fraction of its atmospheric part and delivers the remainder. PROTEUS ships no
impact-loss physics of its own.

The mantle re-melt after an impact is a thermodynamic reset rather than an
energy deposition: it re-applies the run's `planet.temperature_mode` initial
condition to the whole mantle, so how molten the result is follows that
condition. Only `liquidus_super` is fully molten for any planet mass and melting
curve.

### Morrigan `[accretion.morrigan]`

Evolves a system of embryos after disk dispersal with the semi-analytical Monte
Carlo model of Kimura et al. (2025) [^cite-kimura2025] and reports the impacts
experienced by one selected survivor. The host star mass is taken from
`star.mass`, so the dynamical model and the rest of PROTEUS cannot disagree
about it.

!!! note
    Morrigan is an optional module and is not installed with PROTEUS by
    default. Install it with `pip install "fwl-proteus[morrigan]"` before
    setting `accretion.module = "morrigan"`, or as an editable checkout with
    `tools/get_morrigan.sh`. See
    [Installation: optional modules](../../How-to/optionalmodules_installation.md).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | `1` | Random seed for the Monte Carlo. Fixing it makes a history reproducible; sweeping it samples the outcome distribution |
| `num_planets` | int | `10` | Number of embryos the system starts with |
| `masses` | list of float | `[]` | Initial embryo masses \[M$_\oplus$], one per embryo. Empty starts every embryo at `mass_equal` |
| `mass_equal` | float | `0.5` | Initial mass of every embryo \[M$_\oplus$], used when `masses` is empty |
| `eccentricity_init` | float | `0.01` | Initial eccentricity shared by all embryos |
| `inner_edge` | float | `0.1` | Semi-major axis of the innermost embryo \[AU] |
| `spacing` | float | `10.0` | Initial separation between adjacent embryos, in mutual Hill radii |
| `density` | float | `5500.0` | Uniform bulk density used to convert embryo mass to radius \[kg m$^{-3}$] |
| `impact_angle` | float | `45.0` | Impact angle \[deg]; the impact parameter is its sine |
| `evolution_time` | float | `1.0` | Duration of the dynamical evolution \[Gyr] |
| `inner_cutoff` | float | `0.005` | Perihelion inside which an embryo counts as lost to the star \[AU] |
| `selector` | str | `"match_config"` | Which survivor's history to follow: `match_config` (closest initial mass and orbit to the PROTEUS configuration), `mass` (most massive), `semimajoraxis` (final orbit nearest `selector_value`), `id` (embryo index `selector_value`) |
| `selector_value` | float or none | `none` | Target value for the `semimajoraxis` and `id` selectors; ignored otherwise |

Typical `spacing` values are 5 to 15 mutual Hill radii; beyond roughly 30 the
system does not go unstable within a useful evolution time and the run finishes
with no impacts. The 50 accepted here only catches an order-of-magnitude
mistake at configuration load: the layout condition the dynamical model applies
depends on the embryo masses and the host mass, and it refuses a layout that is
too wide by name.

### Timeline `[accretion.timeline]`

Applies a pre-written sequence of impacts instead of deriving one. Every impact
consequence is computed exactly as for a model-derived history, so this
reproduces a published impact history, drives PROTEUS from a history computed
elsewhere, or applies a hand-written sequence for a controlled experiment.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeline_path` | str or none | `none` | Path to the impact timeline file. Environment variables and `~` are expanded |

### Dummy accretion `[accretion.dummy]`

Builds an accretion history from scaling laws rather than by integrating a
system of embryos. The planet approaches an asymptotic mass exponentially,
impacts are placed at evenly spaced times, and each delivers the mass the law
accretes over its interval, so the increments decay with time and the largest
impact is the first. Radii follow the Noack & Lasbleis (2020) mass-radius
scaling [^cite-noack2020], collision velocities combine the pair's mutual escape
velocity with an encounter velocity set by `eccentricity`, and each merged orbit
follows from conserving linear momentum.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mass_accreted` | float | `0.1` | Total mass delivered over the whole timeline \[M$_\oplus$]; the increments are scaled to sum to exactly this |
| `num_impacts` | int | `3` | Number of impacts in the timeline |
| `timescale` | float | `1.0e6` | E-folding time of the accretion law \[yr]. Short compared with `time_last` concentrates the mass in the first impacts; long spreads it evenly |
| `time_last` | float | `5.0e6` | Time of the final impact \[yr]. Impacts are spaced evenly from `time_last / num_impacts` up to this time |
| `eccentricity` | float | `0.05` | Encounter eccentricity \[1], setting both the approach velocity added to the mutual escape velocity and the impactor's orbit |
| `impact_parameter` | float | `0.5` | Impact parameter of every collision \[1], the sine of the impact angle. Zero is head-on, one is grazing |

`timeline_path` is accepted here but is not a parameter of this module: setting
it is refused at configuration load, because it asks to replay a file and would
otherwise be served a generated timeline at default settings. To replay a file,
set `accretion.module = "timeline"` and put the path in
`accretion.timeline.timeline_path`.

## Example

```toml
[accretion]
    module = "morrigan"
    impactor_volatiles = "match_planet"
    atmloss_module = "zephyrus"

    [accretion.morrigan]
        seed = 1
        num_planets = 10
        mass_equal = 0.5
        inner_edge = 0.1
        spacing = 10.0
        evolution_time = 1.0
        selector = "match_config"
```

 [^cite-kimura2025]: Kimura, T., Hoshino, H., Kokubo, E., Matsumoto, Y. & Ikoma, M., *[Semi-analytical model for the dynamical evolution of planetary systems via giant impacts](https://doi.org/10.3847/1538-4357/ade992)*, The Astrophysical Journal, 989, 109, 2025.

 [^cite-kegerreis2020]: Kegerreis, J.A., Eke, V.R., Catling, D.C., Massey, R.J., Teodoro, L.F.A. & Zahnle, K.J., *[Atmospheric erosion by giant impacts onto terrestrial planets: a scaling law for any speed, angle, mass, and density](https://doi.org/10.3847/2041-8213/abb5fb)*, The Astrophysical Journal Letters, 901, L31, 2020.

 [^cite-noack2020]: Noack, L. & Lasbleis, M., *[Parameterisations of interior properties of rocky planets](https://doi.org/10.1051/0004-6361/202037723)*, Astronomy & Astrophysics, 638, A129, 2020. [SciX](https://scixplorer.org/abs/2020A%26A...638A.129N/abstract).
