# Star and orbit

The `[star]` section configures the host star model and spectral properties.
The `[orbit]` section configures the planetary orbit, tidal evolution, and
any satellite.

Submodule documentation:
[MORS](https://proteus-framework.org/MORS/) |
[Obliqua](https://proteus-framework.org/Obliqua/).
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
| `module` | str or none | `none` | Tidal response module: `none` (no tides), `dummy` (fixed heating), `lovepy` (self-consistent Love numbers, solid-only), `obliqua` (self-consistent Love numbers, multi-phase) |
| `semimajoraxis` | float | `1.0` | Orbital semi-major axis \[AU] |
| `eccentricity` | float | `0.0` | Orbital eccentricity |
| `instellation_method` | str | `"distance"` | How to define the orbit: `distance` (use semi-major axis) or `inst` (use instellation flux) |
| `instellationflux` | float | `1.0` | Instellation flux \[S$_\oplus$] (only used when `method = "inst"`) |
| `zenith_angle` | float | `48.19` | Characteristic zenith angle \[degrees] |
| `s0_factor` | float | `0.375` | Instellation geometric scale factor (accounts for rotation and day-night redistribution) |
| `star_planet_model` | str or none | `none` | Star-planet orbit evolution model: `none` (no evolution), `sp0d` (orbit-only, no planet spin tracked), `sp1d` (orbit and planet spin, conserves angular momentum) |
| `axial_period` | float or none | `none` | Planetary rotation period \[hours]; `none` = tidally locked (1:1 spin-orbit resonance) |
| `planet_satellite_model` | str or none | `none` | Planet-satellite orbit evolution model: `none` (no evolution), `ps0d` (orbit-only), `ps1d` (orbit and both spins, conserves angular momentum), `ps1d_evec` (`ps1d` plus the evection resonance angle) |
| `perturber` | str or none | `none` | Body inducing tides on the planet: `none`, `star`, or `satellite` |

### Satellite `[orbit.satellite]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_satellite` | bool | `false` | Include a satellite (moon) |
| `mass_sat` | float | `0.012` | Satellite mass \[M$_\oplus$] (approx. lunar mass) |
| `radius_sat` | float | `0.273` | Satellite radius \[R$_\oplus$] (approx. lunar radius) |
| `axial_period_sat` | float or none | `none` | Satellite initial day length \[hours]; `none` = use orbital period |
| `semimajoraxis_sat` | float | `0.133` | Satellite initial orbital semi-major axis \[AU] |
| `eccentricity_sat` | float | `0.0` | Satellite initial orbital eccentricity |
| `evection_angle` | float | `0.0` | Satellite initial evection angle \[degrees] (only used by `ps1d_evec`) |
| `c_factor_sat` | float | `0.4` | Satellite gyration factor, $\leq 0.4$ |
| `love_number_sat` | str or none | `none` | Path to a netCDF file of precomputed satellite Love numbers (forcing frequency and complex $k_2$ vs. tidal mode), or to a JSON file of interior properties; `none` = not used |

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

### Obliqua tides `[orbit.obliqua]`

Self-consistent, multi-phase tidal response computed from the interior's
solid, mushy, and fluid layers, each handled by a dedicated sub-model. Valid
for arbitrary eccentricity and tidal mode.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_3D` | bool | `false` | Store 3D tidal dissipation information |
| `enforce_ec` | bool | `true` | Enforce energy conservation between Love numbers and the heating profile |
| `optimize_scales` | bool | `false` | Optimize non-dimensional scaling parameters for solid tides |
| `solid_shell` | bool | `true` | Insert an infinitesimal solid shell around the core in fluid only segments |
| `min_frac` | float | `0.05` | Minimal segment radius fraction before smoothing |
| `visc_l` | float | `1e2` | Pure liquid viscosity \[Pa s] |
| `visc_lus` | float | `5e5` | Liquid-Mush handoff viscosity \[Pa s] |
| `visc_s` | float | `1e22` | Pure solid viscosity \[Pa s] |
| `visc_sus` | float | `5e5` | Solid-Mush handoff viscosity \[Pa s] |
| `n` | list of int | `[2]` | Tidal degree (Power(s) of the radial factor $(r/a)^n$) |
| `m` | list of int | `[0, 2]` | Tidal modes |
| `k_min`, `k_max` | int or `"none"` | `"none"` | Fourier index range in mean anomaly; `"none"` = adaptive spectrum |
| `material_mu` | str | `"andrade"` | Rheology model for the complex shear modulus: `andrade`, `maxwell`, or `elastic` |
| `material_k` | str | `"andrade"` | Rheology model for the complex bulk modulus: `andrade`, `maxwell`, or `elastic` |
| `alpha` | float | `0.3` | Andrade power-law exponent |
| `verbosity` | int | `1` | Logging verbosity: 0 (silent), 1 (info), 2 (debug) |
| `module_solid` | str | `"solid0d"` | Solid-tide sub-model: `none`, `solid0d`, `solid1d`, `solid1d-relax`, `solid1d-mush`, `solid1d-mush-relax`, `solid1d-equil-relax` |
| `module_mushy` | str | `"none"` | Mushy-tide sub-model: `none` or `interp` |
| `module_fluid` | str | `"fluid0d"` | Fluid-tide sub-model: `none`, `fluid0d`, `fluid1d` |

#### Solid tides `[orbit.obliqua.solid]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ncalc` | int | `1000` | Number of interpolated interior levels (shooting method) |
| `dr_min` | float | `300` | Minimum radial grid spacing \[m] (Henyey/relaxation method) |
| `dr_max` | float | `3000` | Maximum radial grid spacing \[m] (Henyey/relaxation method) |
| `core` | str | `"liquid"` | Core solution vector: `liquid`, `solid`, or `inertial` |
| `core_props` | str | `"core"` | Core properties to use: `core` or `mantle` |
| `inertial_terms` | bool | `true` | Include inertial terms in the solid-tide solution |
| `bulk_l` | float | `1e9` | Bulk modulus of the liquid phase \[Pa] |
| `porosity_thresh` | float | `3e-2` | Porosity threshold below which melt fraction is set to zero |
| `dbulk_power` | float | `0.5` | Drained bulk modulus power-law scaling exponent |

#### Mushy tides `[orbit.obliqua.mushy]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `b_width` | float | `0.5` | Scale width of the bottom heating decay profile |
| `t_width` | float | `0.03` | Scale width of the top heating decay profile |

#### Fluid tides `[orbit.obliqua.fluid]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sigma_R` | float | `1e-3` | Rayleigh drag in the fluid-mush/solid boundary layers \[s$^{-1}$\] |
| `sigma_R_inf` | float | `0.5` | Rayleigh drag in the pure fluid (times `sigma_R`) |
| `sigma_R_prf` | str | `"exp"` | Radial heating distribution profile: `uniform`, `exp`, `linear`, `quadratic`, `dynamic`, `dynamic_interp` |
| `H_R` | float | `1e4` | Scale height used by the heating profile \[m] |
| `efficiency` | float | `0.3` | Rayleigh drag efficiency at the core interface |

---

**See also:** [Stellar module](../../Explanations/model.md#stellar-evolution-mors) | [Tidal evolution](../../Explanations/model.md#tidal-evolution-obliqua-lovepy) | [Orbital evolution](../../Explanations/model.md#orbital-evolution-proteus-internal)

[^cite-spada2013]: Spada, F., Demarque, P., Kim, Y.C. & Sills, A., *[The radius discrepancy in low-mass stars: single versus binaries](https://doi.org/10.1088/0004-637X/776/2/87)*, The Astrophysical Journal, 776, 87, 2013. [SciX](https://scixplorer.org/abs/2013ApJ...776...87S/abstract).

[^cite-baraffe2015]: Baraffe, I., Homeier, D., Allard, F. & Chabrier, G., *[New evolutionary models for pre-main sequence and main sequence low-mass stars down to the hydrogen-burning limit](https://doi.org/10.1051/0004-6361/201425481)*, Astronomy & Astrophysics, 577, A42, 2015. [SciX](https://scixplorer.org/abs/2015A%26A...577A..42B/abstract).
