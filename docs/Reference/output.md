# Output format

PROTEUS writes simulation output to a directory inside `output/`. This page
documents the structure of that directory and the contents of the main output
files.

## Directory structure

A completed PROTEUS run produces:

```
output/<run_name>/
    runtime_helpfile.csv    # Main time-series output (tab-separated)
    init_coupler.toml       # Copy of the configuration used
    status                  # Exit status code and description
    proteus_00.log          # Log file (subsequent resumes: proteus_01.log, ...)
    plots/                  # Generated diagnostic plots
        plot_global.png
        plot_escape.png
        plot_interior.png
        ...
    data/                   # Module-specific output files
        <iter>_atm.nc       # Atmosphere profiles (NetCDF, per snapshot)
        <iter>_int.nc       # Interior profiles (NetCDF, per snapshot)
        zalmoxis_output.dat # Zalmoxis structure profile (latest)
        spider_eos/         # Cached EOS lookup tables (if Aragog/SPIDER)
        ...
    observe/                # Synthetic observations (if enabled)
        transit_<source>_synthesis.csv   # Transit depth spectra
        eclipse_<source>_synthesis.csv   # Eclipse depth spectra
```

`<source>` is `outgas`, `profile`, or `offchem`, depending on
`observe.source`. With `observe.source = "all"` (default), PROTEUS writes one
pair of files per available source.

## Status codes

The `status` file contains a single integer followed by a human-readable
description. The codes are:

| Code | Meaning |
|------|---------|
| 0 | Started |
| 1 | Running |
| 10 | Completed: mantle solidified |
| 11 | Completed: maximum clock runtime |
| 12 | Completed: maximum iterations reached |
| 13 | Completed: target time reached |
| 14 | Completed: net flux is small (radiative equilibrium) |
| 15 | Completed: atmosphere escaped |
| 16 | Completed: planet disintegrated |
| 20 | Error: configuration |
| 21 | Error: interior module |
| 22 | Error: atmosphere module |
| 23 | Error: stellar module |
| 24 | Error: chemistry module |
| 25 | Error: killed or crashed |
| 26 | Error: tidal module |
| 27 | Error: outgassing module |
| 28 | Error: escape module |

## Helpfile columns

The `runtime_helpfile.csv` is a tab-separated file with one row per coupling
iteration. All quantities use SI units unless noted otherwise. The columns
are grouped by category below.

### Time

| Column | Units | Description |
|--------|-------|-------------|
| `Time` | yr | Simulation time (zero during initialisation stage) |
| `age_star` | yr | Stellar age |
| `runtime` | s | Wall-clock elapsed time |

### Orbital parameters

| Column | Units | Description |
|--------|-------|-------------|
| `semimajorax` | m | Orbital semi-major axis |
| `separation` | m | Orbital separation (accounts for eccentricity) |
| `perihelion` | m | Perihelion distance |
| `eccentricity` | 1 | Orbital eccentricity |
| `orbital_period` | s | Orbital period |
| `axial_period` | s | Planetary rotation period |
| `Imk2` | 1 | Imaginary part of $k_2$ Love number |
| `roche_limit` | m | Roche limit distance |
| `hill_radius` | m | Hill sphere radius |
| `breakup_period` | s | Rotational breakup period |

### Satellite (if enabled)

| Column | Units | Description |
|--------|-------|-------------|
| `semimajorax_sat` | m | Satellite orbital semi-major axis |
| `perigee` | m | Satellite perigee distance |
| `M_sat` | kg | Satellite mass |
| `plan_sat_am` | kg m$^2$ s$^{-1}$ | Planet-satellite angular momentum |

### Planet structure

| Column | Units | Description |
|--------|-------|-------------|
| `R_int` | m | Interior (surface) radius |
| `R_core` | m | Core radius |
| `M_int` | kg | Interior dry mass (mantle + core) |
| `M_planet` | kg | Total planet mass (interior + volatiles) |
| `M_core` | kg | Core mass |
| `M_mantle` | kg | Mantle mass (solid + liquid) |
| `M_mantle_solid` | kg | Solid mantle mass |
| `M_mantle_liquid` | kg | Liquid mantle mass |
| `P_center` | Pa | Central pressure |
| `P_cmb` | Pa | Core-mantle boundary pressure |
| `core_density` | kg m$^{-3}$ | Core density |
| `core_heatcap` | J kg$^{-1}$ K$^{-1}$ | Core heat capacity |
| `gravity` | m s$^{-2}$ | Surface gravitational acceleration |

### Temperatures

| Column | Units | Description |
|--------|-------|-------------|
| `T_surf` | K | Surface temperature |
| `T_magma` | K | Magma ocean potential temperature |
| `T_cmb` | K | Core temperature |
| `T_eqm` | K | Equilibrium temperature (from instellation) |
| `T_skin` | K | Radiative skin temperature |
| `T_star` | K | Stellar effective temperature |

### Energy fluxes

| Column | Units | Description |
|--------|-------|-------------|
| `F_int` | W m$^{-2}$ | Interior heat flux (from mantle to surface) |
| `F_atm` | W m$^{-2}$ | Atmospheric outgoing thermal radiation |
| `F_net` | W m$^{-2}$ | Net surface flux ($F_\mathrm{int} - F_\mathrm{atm}$) |
| `F_olr` | W m$^{-2}$ | Outgoing longwave radiation at TOA |
| `F_sct` | W m$^{-2}$ | Outgoing shortwave (scattered) radiation |
| `F_ins` | W m$^{-2}$ | Instellation (absorbed stellar flux) |
| `F_xuv` | W m$^{-2}$ | XUV flux (10-121.6 nm) at planet |
| `F_tidal` | W m$^{-2}$ | Tidal heating flux |
| `F_radio` | W m$^{-2}$ | Radiogenic heating flux |
| `F_cmb` | W m$^{-2}$ | Core-mantle boundary heat flux |

### Interior state

| Column | Units | Description |
|--------|-------|-------------|
| `Phi_global` | 1 | Global melt fraction (mass-weighted) |
| `Phi_global_vol` | 1 | Global melt fraction (volume-weighted) |
| `RF_depth` | 1 | Rheological front depth (normalised) |
| `T_pot` | K | Mantle potential temperature |
| `boundary_layer_thickness` | m | Thermal boundary layer thickness |

### Stellar properties

| Column | Units | Description |
|--------|-------|-------------|
| `M_star` | kg | Stellar mass |
| `R_star` | m | Stellar radius |
| `age_star` | yr | Stellar age |

### Atmospheric composition

For each gas species (H2O, CO2, N2, H2, CH4, CO, SO2, H2S, NH3, S2):

| Column pattern | Units | Description |
|----------------|-------|-------------|
| `<gas>_bar` | bar | Surface partial pressure |
| `<gas>_vmr` | 1 | Volume mixing ratio at surface |
| `<gas>_kg_atm` | kg | Mass in atmosphere |
| `<gas>_kg_liquid` | kg | Mass dissolved in melt |
| `<gas>_kg_solid` | kg | Mass in solid mantle |
| `<gas>_kg_total` | kg | Total mass (atm + liquid + solid) |
| `<gas>_mol_atm` | mol | Moles in atmosphere |
| `<gas>_mol_total` | mol | Total moles |

### Elemental budgets

For each element (H, C, N, O, S):

| Column pattern | Units | Description |
|----------------|-------|-------------|
| `<elem>_kg_atm` | kg | Elemental mass in atmosphere |
| `<elem>_kg_liquid` | kg | Elemental mass in melt |
| `<elem>_kg_solid` | kg | Elemental mass in solid |
| `<elem>_kg_total` | kg | Total elemental mass |

### Bulk atmosphere

| Column | Units | Description |
|--------|-------|-------------|
| `M_atm` | kg | Total atmospheric mass |
| `M_ele` | kg | Total volatile element mass |
| `P_surf` | bar | Total surface pressure |
| `atm_kg_per_mol` | kg mol$^{-1}$ | Mean molecular weight |

### Redox state

| Column | Units | Description |
|--------|-------|-------------|
| `fO2_shift_IW_derived` | log$_{10}$ | Derived fO2 offset from iron-wustite buffer |
| `O_res` | kg | Oxygen mass-balance residual |

### Escape

| Column | Units | Description |
|--------|-------|-------------|
| `esc_rate_total` | kg s$^{-1}$ | Total bulk escape rate |
| `esc_rate_H` | kg s$^{-1}$ | Hydrogen escape rate |
| `esc_rate_C` | kg s$^{-1}$ | Carbon escape rate |
| `esc_rate_N` | kg s$^{-1}$ | Nitrogen escape rate |
| `esc_rate_O` | kg s$^{-1}$ | Oxygen escape rate |
| `esc_rate_S` | kg s$^{-1}$ | Sulfur escape rate |
| `esc_kg_cumulative` | kg | Cumulative escaped mass |
| `M_vol_initial` | kg | Initial volatile inventory baseline |
| `p_xuv` | bar | XUV absorption pressure level |
| `R_xuv` | m | XUV absorption radius |

### Observables

| Column | Units | Description |
|--------|-------|-------------|
| `R_obs` | m | Observable (transit) radius |
| `T_obs` | K | Observable temperature |
| `p_obs` | bar | Observation pressure level |
| `rho_obs` | kg m$^{-3}$ | Observable bulk density |
| `transit_depth` | 1 | Transit depth |
| `eclipse_depth` | 1 | Eclipse depth |
| `albedo_pl` | 1 | Planetary albedo |
| `bond_albedo` | 1 | Bond albedo |

### Solvus and miscibility (when `global_miscibility` is enabled)

| Column | Units | Description |
|--------|-------|-------------|
| `R_solvus` | m | Solvus radius (H$_2$-silicate miscibility gap boundary) |
| `P_solvus` | Pa | Pressure at the solvus |
| `T_solvus` | K | Temperature at the solvus |
| `X_H2_int` | 1 | H$_2$ mass fraction in the interior |

## Synthetic observation CSV columns

Files in `observe/` are tab-separated. They always contain:

- `Wavelength/um`
- `None/ppm` (baseline spectrum with all included gases)

When `observe.remove_one_gas = true`, additional columns are
included:

- `<SPECIES>_removed/ppm` for each leave-one-out run

### Accretion mode columns (when `temperature_mode = 'accretion'`)

| Column | Units | Description |
|--------|-------|-------------|
| `T_surf_accr` | K | Surface temperature from accretion energy balance |
| `DeltaT_accretion` | K | Accretion-energy temperature contribution |

### Energy conservation diagnostics (Aragog only)

| Column | Units | Description |
|--------|-------|-------------|
| `E_state_J` | J | Total thermal energy of mantle |
| `E_state_cons_J` | J | Conservative thermal energy (frozen-mass frame) |
| `dE_predicted_cons_J` | J | Cumulative predicted energy change from flux integrals |
| `E_residual_cons_J` | J | Cumulative energy residual |
| `E_residual_cons_frac` | 1 | Fractional energy residual |
| `Q_radio_W` | W | Instantaneous radiogenic power |
| `Q_tidal_W` | W | Instantaneous tidal power |
| `solver_residual_J` | J | Entropy-equation flux-divergence self-consistency check (machine-zero by construction; non-zero flags an assembly bug, not integrator quality) |

## Synthetic observation output files

When `observe.module = "petitRADTRANS"`, PROTEUS writes CSV files to
`observe/` for each composition source (`outgas`, `profile`, `offchem`).
Files are named `transit_<source>_synthesis.csv` and
`eclipse_<source>_synthesis.csv`.

The set of files is controlled by `observe.spectrum_type`:

- `both`: writes both transit and eclipse files
- `transit`: writes transit files only
- `eclipse`: writes eclipse files only

All CSV files are tab-separated with a one-line header. The first column is
always the wavelength grid; subsequent columns are depth values in ppm:

| Column | Description |
|--------|-------------|
| `Wavelength/um` | Wavelength \[μm\] on a log-spaced grid |
| `None/ppm` | Baseline depth \[ppm\] — all detected line species included |
| `<SPECIES>_removed/ppm` | Depth \[ppm\] with `<SPECIES>` removed (one column per line species found in the gas mixture) |

The number of `*_removed` columns varies depending on which species have
opacity tables present in `$FWL_DATA/prt/input_data` and exceed the
`clip_vmr` threshold at the time of the calculation.

The `plot_spectra` diagnostic overlays all columns on a two-panel figure:
the upper panel shows transit depths, the lower shows eclipse depths.
The baseline (`None/ppm`) is drawn in black; each removed-species line is
coloured by species identity.

<figure markdown="span">
  ![Example synthetic spectrum: transit and eclipse depth with per-species overlays](../assets/example_spectrum.png)
  <figcaption><b>Example <code>plot_spectra</code> output.</b>
  Upper panel: primary transit depth \[ppm\]. Lower panel: secondary eclipse depth \[ppm\].
  The black line is the baseline (all species present); coloured lines show the spectrum
  with each gas removed in turn.</figcaption>
</figure>

## Diagnostic plots

PROTEUS generates diagnostic plots at intervals controlled by
`params.out.plot_mod`. Available plot types:

| Plot file | Content |
|-----------|---------|
| `plot_global` | Multi-panel overview: temperatures, fluxes, melt fraction, pressures |
| `plot_escape` | Escape rates by element, surface pressure, XUV flux |
| `plot_interior` | Mantle temperature, viscosity, heat flux, melt fraction profiles |
| `plot_interior_cmesh` | Composite mesh visualisation of interior structure |
| `plot_structure` | Interior density, temperature, composition profiles over time |
| `plot_atmosphere` | Atmospheric T(p) profiles at selected snapshots |
| `plot_atmosphere_cbar` | Atmospheric profiles with continuous time colorbar |
| `plot_emission` | Emission spectra at selected snapshots |
| `plot_fluxes_global` | Time evolution of all energy flux components |
| `plot_fluxes_atmosphere` | Net, upwelling, downwelling radiative fluxes |
| `plot_bolometry` | TOA bolometric flux evolution |
| `plot_sflux` | Stellar flux spectrum evolution (colorbar) |
| `plot_sflux_cross` | Stellar flux in wavelength bins over time |
| `plot_orbit` | Semi-major axis and eccentricity evolution |
| `plot_population` | Mass-radius diagram with exoplanet population overlay |
| `plot_visual` | Rendered disk image of planet and star |
| `plot_spectra` | Transit and eclipse depth spectra (baseline + per-species removed overlays) |
| `plot_chem_atmosphere` | Atmospheric chemical species mixing ratios |
