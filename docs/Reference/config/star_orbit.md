# Star and orbit

The `[star]` section configures the host star model and spectral properties.
The `[orbit]` section configures the planetary orbit, tidal evolution, and
any satellite.

Submodule documentation:
[MORS](https://proteus-framework.org/MORS/) |
[Obliqua](https://github.com/FormingWorlds/Obliqua).
See also [Model description](../../Explanations/model.md#stellar-evolution-mors).

## Stellar model `[star]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | str | `"mors"` | Stellar evolution module: `mors` (age-dependent tracks) or `dummy` (fixed properties) |
| `mass` | float | `1.0` | Stellar mass \[M$_\odot$] |
| `age_ini` | float | `0.1` | Model start age \[Gyr] |
| `bol_scale` | float | `1.0` | Bolometric luminosity scaling factor |

### MORS stellar tracks `[star.mors]`

The MORS module interpolates stellar radius, effective temperature,
luminosity, and XUV flux from pre-computed evolutionary tracks as a function
of stellar age. Two track families are available: Spada[^cite-spada2013] (solar-type) and
Baraffe[^cite-baraffe2015] (low-mass M-dwarfs).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tracks` | str | `"spada"` | Evolution track family: `spada` or `baraffe` |
| `age_now` | float | `4.567` | Observed or estimated stellar age \[Gyr] |
| `rot_pcntle` | float or none | `50.0` | Rotation percentile of stellar population \[0, 100] |
| `rot_period` | float or none | `none` | Rotation period \[days]; overrides `rot_pcntle` if set |
| `spectrum_source` | str | `"phoenix"` | Spectral library: `solar`, `muscles`, `phoenix` |
| `star_name` | str or none | `none` | Named star for solar/muscles lookup (e.g. `"sun"`, `"trappist-1"`) |
| `star_path` | str or none | `none` | Path to custom spectrum file; overrides `spectrum_source` |

### PHOENIX synthetic spectra

These parameters are used when `spectrum_source = "phoenix"`. PHOENIX provides
synthetic spectra on a grid of metallicity, alpha enhancement, and (optionally)
effective temperature.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `phoenix_FeH` | float | `0.0` | Metallicity \[Fe/H]; 0.0 = solar |
| `phoenix_alpha` | float | `0.0` | Alpha enhancement \[$\alpha$/Fe]; 0.0 = solar |
| `phoenix_radius` | float or none | `none` | Stellar radius \[R$_\odot$]; `none` = from MORS tracks |
| `phoenix_log_g` | float or none | `none` | Surface gravity \[log$_{10}$ cgs]; `none` = from MORS tracks |
| `phoenix_Teff` | float or none | `none` | Effective temperature \[K]; `none` = from MORS tracks |

### Dummy star `[star.dummy]`

A fixed-luminosity star with no temporal evolution. Useful for testing and
parameter studies where stellar evolution is not relevant.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Teff` | float | `5772.0` | Effective temperature \[K] |
| `radius` | float or none | `none` | Stellar radius \[R$_\odot$]; if `none`, derived from `Teff` and `mass` when `calculate_radius = true` |
| `calculate_radius` | bool | `false` | Derive radius from mass-luminosity and mass-radius relations |

## Orbital configuration `[orbit]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | str or none | `none` | Tidal heating module: `none` (no tides), `dummy` (fixed heating), `lovepy` (self-consistent Love numbers) |
| `semimajoraxis` | float | `1.0` | Orbital semi-major axis \[AU] |
| `eccentricity` | float | `0.0` | Orbital eccentricity |
| `instellation_method` | str | `"distance"` | How to define the orbit: `distance` (use semi-major axis) or `inst` (use instellation flux) |
| `instellationflux` | float | `1.0` | Instellation flux \[S$_\oplus$] (only used when `method = "inst"`) |
| `zenith_angle` | float | `48.19` | Characteristic zenith angle \[degrees] |
| `s0_factor` | float | `0.375` | Instellation geometric scale factor (accounts for rotation and day-night redistribution) |
| `evolve` | bool | `false` | Evolve semi-major axis and eccentricity via tidal dissipation |
| `axial_period` | float or none | `none` | Planetary rotation period \[hours]; `none` = tidally locked (1:1 spin-orbit resonance) |

### Satellite

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `satellite` | bool | `false` | Include a satellite (moon) |
| `mass_sat` | float | `7.347e22` | Satellite mass \[kg] (default: lunar mass) |
| `semimajoraxis_sat` | float | `3e8` | Satellite orbital semi-major axis \[m] |

### Dummy tides `[orbit.dummy]`

Fixed tidal heating rates, useful for parameter studies.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `H_tide` | float | `0.0` | Fixed tidal power density \[W kg$^{-1}$] |
| `Phi_tide` | str | `"<0.3"` | Inequality defining where tidal heating is applied (melt fraction condition, e.g. `"<0.3"`) |
| `Imk2` | float | `0.0` | Fixed Im($k_2$) Love number (must be $\leq 0$) |

### LovePy tides `[orbit.lovepy]`

Self-consistent tidal heating using viscoelastic Love numbers computed from
the interior rheological profile.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `visc_thresh` | float | `1e9` | Minimum viscosity for tidal heating calculation \[Pa s] |
| `ncalc` | int | `1000` | Number of interior grid points for tidal calculation |

---

**See also:** [Stellar module](../../Explanations/model.md#stellar-evolution-mors) | [Orbit module](../../Explanations/model.md#orbital-evolution-obliqua)

[^cite-spada2013]: Spada, F., Demarque, P., Kim, Y.C. & Sills, A., *[The radius discrepancy in low-mass stars: single versus binaries](https://doi.org/10.1088/0004-637X/776/2/87)*, The Astrophysical Journal, 776, 87, 2013. [SciX](https://scixplorer.org/abs/2013ApJ...776...87S/abstract).

[^cite-baraffe2015]: Baraffe, I., Homeier, D., Allard, F. & Chabrier, G., *[New evolutionary models for pre-main sequence and main sequence low-mass stars down to the hydrogen-burning limit](https://doi.org/10.1051/0004-6361/201425481)*, Astronomy & Astrophysics, 577, A42, 2015. [SciX](https://scixplorer.org/abs/2015A%26A...577A..42B/abstract).
