# Interior structure and energetics

PROTEUS separates the interior into two coupled subsystems: **structure**
(hydrostatic equilibrium, density profile, planet radius) and **energetics**
(thermal evolution, melt fraction, heat flux). Each has its own module
selection and parameters.

Submodule documentation:
[Aragog](https://proteus-framework.org/aragog/) |
[SPIDER](https://proteus-framework.org/SPIDER/) |
[Zalmoxis](https://proteus-framework.org/Zalmoxis/).
See also [Model description](../../Explanations/model.md#interior-energetics-aragog-spider-boundary).

## Interior structure `[interior_struct]`

The structure module computes the planet's density profile, radius, and
core-mantle boundary position by solving the hydrostatic equilibrium equation
with an equation of state (EOS).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | str | `"zalmoxis"` | Structure solver: `zalmoxis` (full EOS), `dummy` (scaling laws), `spider` (SPIDER internal) |
| `core_frac` | float | `0.325` | Core fraction (meaning depends on `core_frac_mode`) |
| `core_frac_mode` | str | `"mass"` | How `core_frac` is interpreted: `mass` (mass fraction) or `radius` (radius fraction). The `zalmoxis` module always uses `mass` and ignores `radius` with a warning; `radius` is honoured by the `dummy` and `spider` modules |
| `core_density` | float or "self" | `"self"` | Core density \[kg m$^{-3}$]; `"self"` = computed by the structure solver |
| `core_heatcap` | float or "self" | `"self"` | Core heat capacity \[J kg$^{-1}$ K$^{-1}$]; `"self"` = computed by the structure solver |
| `melting_dir` | str or none | `none` | Melting curve folder name in FWL_DATA (for SPIDER module) |
| `eos_dir` | str or none | `none` | EOS folder name in FWL_DATA (for SPIDER module) |

### Zalmoxis `[interior_struct.zalmoxis]`

Zalmoxis solves the full hydrostatic structure with tabulated EOS (PALEOS,
Chabrier, or Seager 2007). It supports 2-layer (core + mantle) and 3-layer
(core + mantle + ice/water) models.

**Equation of state**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `core_eos` | str | `"PALEOS:iron"` | Core EOS as `"<source>:<material>"` |
| `mantle_eos` | str | `"PALEOS:MgSiO3"` | Mantle EOS as `"<source>:<material>"`; `PALEOS:` uses the unified table, `PALEOS-2phase:` uses separate solid and liquid tables (see the PALEOS mantle EOS notes below) |
| `ice_layer_eos` | str or none | `none` | Ice/water layer EOS; `none` = 2-layer model |
| `mushy_zone_factor` | float | `0.8` | Solidus depression $T_\mathrm{sol}=f\,T_\mathrm{liq}$ \[0.7, 1.0]; the default 0.8 is the Stixrude (2014) solidus-to-liquidus ratio for MgSiO$_3$. Applies to `PALEOS:` unified only; for `PALEOS-2phase:` it is treated as 1.0 |
| `mantle_mass_fraction` | float | `0` | Mantle mass fraction for 3-layer models; `0` = auto ($1 - \mathrm{core\_frac}$) |
| `dry_mantle` | bool | `true` | Structure EOS assumes a dry mantle. Set `false` for melt-fraction-aware dissolved-volatile mixing in the mantle density (per-shell volatile profile). When `false`, only the atmospheric inventory is excluded from the dry-mass target; the dissolved mass stays inside the interior |

**How the PALEOS mantle EOS is applied**

The MgSiO$_3$ mantle EOS resolves through two distinct paths depending on the `<source>` prefix of `mantle_eos`.

With `mantle_eos = "PALEOS:MgSiO3"` (the default), the hydrostatic structure solve uses the PALEOS *unified* MgSiO$_3$ table for the density profile.
The phase-specific property surfaces used by Aragog (density, heat capacity, thermal expansion, adiabatic gradient) and the pressure-entropy lookup tables are built from the PALEOS *two-phase* solid and liquid tables shipped with Zalmoxis when those tables are present, which keeps the properties well resolved across the melting-curve discontinuity that a single unified table interpolates through.
If the two-phase tables are not available, the property surfaces are built from the unified table alone, and the entropy near the melting curve is less reliable.
The liquidus is the analytic PALEOS curve (Belonoshko et al. 2005 below 2.55 GPa, Fei et al. 2021 above, in Simon-Glatzel form), and the solidus is derived as $T_\mathrm{sol}(P) = f\,T_\mathrm{liq}(P)$ with $f$ the `mushy_zone_factor` (default 0.8), the constant solidus-to-liquidus ratio of the Stixrude (2014)[^cite-stixrude2014] MgSiO$_3$ melting parametrization.
The melt fraction then follows from the lever rule between this solidus and liquidus.

With `mantle_eos = "PALEOS-2phase:MgSiO3"`, the solid and liquid tables define the phase boundaries directly.
`mushy_zone_factor` is treated as 1.0 so the solidus coincides with the liquidus, and the latent-heat gap is supplied by the entropy difference between the solid and liquid tables rather than by a fixed temperature depression.

!!! note "Two-phase table versions"
    Two versions of the PALEOS two-phase MgSiO$_3$ tables are in circulation: the set shipped in the Zalmoxis data directory, and the finer-grid set on Zenodo that the reference-data manifest fetches.
    They are generated from the same PALEOS EOS but on different grids, so for exact reproducibility record which set a run used (the shipped tables resolve through the Zalmoxis material registry; the fetched tables land under `FWL_DATA`).

**Grid and solver**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_levels` | int | `150` | Number of radial grid levels |
| `outer_solver` | str | `"newton"` | Outer mass-radius solver: `newton` (recommended) or `picard` (alternative) |
| `use_jax` | bool | `true` | Use JAX backend for the structure solver |
| `use_anderson` | bool | `false` | Anderson Type-II Picard acceleration on the density loop |
| `solver_tol_outer` | float | `3e-3` | Relative tolerance for mass convergence |
| `solver_tol_inner` | float | `1e-4` | Relative tolerance for density convergence |
| `solver_max_iter_outer` | int | `100` | Maximum iterations for mass convergence |
| `solver_max_iter_inner` | int | `100` | Maximum iterations for density convergence |

**Newton solver tuning** (used when `outer_solver = "newton"`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `newton_max_iter` | int | `30` | Maximum Newton iterations |
| `newton_tol` | float | `1e-4` | Newton convergence tolerance |
| `newton_relative_tolerance` | float | `1e-9` | Integrator relative tolerance for Newton path |
| `newton_absolute_tolerance` | float | `1e-10` | Integrator absolute tolerance for Newton path |

**Structure update triggers**

Zalmoxis structure updates are expensive and decoupled from the main coupling
loop. The structure is recomputed when any of these conditions are met:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `update_dphi_abs` | float | `0.05` | Trigger when melt fraction changes by this amount |
| `update_dtmagma_frac` | float | `0.05` | Trigger when $T_\mathrm{magma}$ changes by this fraction |
| `update_dw_comp_abs` | float | `0.05` | Trigger when the relative dissolved-volatile (H$_2$O or H$_2$) mantle mass fraction changes by this amount |
| `update_interval` | float | `1e9` | Maximum time between updates \[yr]; effectively disabled at default |
| `update_min_interval` | float | `0` | Minimum time between updates \[yr] (prevents thrashing) |
| `update_stale_ceiling` | float | `2.5e4` | Time since last successful re-solve that refires a trigger \[yr]; `0` disables |
| `mesh_max_shift` | float | `0.05` | Maximum fractional radius shift per update |
| `mesh_convergence_interval` | float | `10.0` | Convergence relaxation time after mesh update \[yr] |

**Initialisation**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `equilibrate_init` | bool | `true` | Equilibrate structure and composition before the main loop |
| `equilibrate_max_iter` | int | `15` | Maximum equilibration iterations |
| `equilibrate_tol` | float | `0.01` | Equilibration convergence tolerance |

**P-S entropy lookup tables**

These tables are derived from PALEOS EOS data at runtime and cached. They
provide the entropy-to-temperature mapping used by Aragog and SPIDER.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lookup_nP` | int | `1350` | Pressure grid points in lookup tables |
| `lookup_nS` | int | `280` | Entropy grid points in lookup tables |

By default each run derives its own copy of these tables under its output
`data/` directory. Set the `PROTEUS_PS_CACHE_DIR` environment variable to an
absolute path to instead share one derived copy across runs: tables are stored
in a subdirectory keyed by pressure ceiling, resolution, mantle-mass fraction,
table layout, and the resolved mantle-EOS identity, so a run reuses the cache
only when every one of those matches and different equations of state never
collide. This is the mitigation for the per-run disk duplication that a grid or
batch of same-EOS runs would otherwise incur, since all such runs then read one
shared copy. The cache directory is not size-limited or auto-pruned; it grows
with the number of distinct pressure, resolution, and EOS combinations run
against it, so point it at storage sized for the campaign.

**Miscibility** (experimental, not production-ready)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `global_miscibility` | bool | `false` | Enable H$_2$-silicate binodal-aware radial structure. `true` is rejected at config load: it requires the H$_2$-silicate binodal handoff on the Zalmoxis side (Zalmoxis tracker #64), which is not yet implemented |
| `miscibility_max_iter` | int | `10` | Maximum miscibility iterations |
| `miscibility_tol` | float | `0.01` | Miscibility convergence tolerance |

---

## Interior energetics `[interior_energetics]`

The energetics module evolves the mantle thermal state (temperature,
entropy, melt fraction) and computes the interior heat flux that drives
atmospheric evolution.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | str | `"aragog"` | Interior thermal module: `aragog` (default), `spider`, `boundary`, `dummy` |
| `num_levels` | int | `80` | Radial grid levels for the energetics domain |
| `rtol` | float | `1e-10` | ODE solver relative tolerance |
| `atol` | float | `1e-10` | ODE solver absolute tolerance |
| `flux_guess` | float | `-1` | Initial heat flux guess \[W m$^{-2}$]; negative = compute as $\sigma T_\mathrm{magma}^4$ |
| `surface_bc_mode` | str | `"flux"` | Surface BC for SPIDER/Aragog: `flux` (prescribed $F_\mathrm{atm}$ from the atmosphere module) or `grey_body` (native grey-body BC computed inside the interior solver) |

**Transport physics**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trans_conduction` | bool | `true` | Conductive heat transfer |
| `trans_convection` | bool | `true` | Convective heat transfer (mixing length theory) |
| `trans_grav_sep` | bool | `true` | Gravitational separation (Stokes settling) |
| `trans_mixing` | bool | `true` | Chemical mixing flux |

**Heating**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `heat_tidal` | bool | `false` | Tidal heating (requires `orbit.module` $\neq$ `none`) |
| `heat_radiogenic` | bool | `true` | Radiogenic heating |
| `radio_tref` | float | `4.567` | Reference age for decay concentrations \[Gyr] (4.567 = present-day BSE) |
| `radio_Al` | float | `0.0` | $^{26}$Al concentration \[ppmw]; 1.23 = canonical |
| `radio_Fe` | float | `0.0` | $^{60}$Fe concentration \[ppmw] |
| `radio_K` | float | `310.0` | $^{40}$K concentration \[ppmw of element]; BSE value |
| `radio_U` | float | `0.031` | U concentration \[ppmw of element]; BSE value |
| `radio_Th` | float | `0.124` | Th concentration \[ppmw of element]; BSE value |

**Rheology and convection**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rfront_loc` | float | `0.50` | Rheological front centre \[melt fraction] |
| `rfront_wid` | float | `0.20` | Rheological front width \[melt fraction] |
| `grain_size` | float | `0.1` | Crystal settling grain size \[m] |
| `mixing_length` | str | `"nearest"` | MLT length scale: `nearest` (distance to nearest boundary) or `constant` ($D/4$) |
| `kappah_floor` | float | `10.0` | Eddy diffusivity floor \[m$^2$ s$^{-1}$]; prevents MLT freeze |

**Coupling limits**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tmagma_atol` | float | `20.0` | Maximum $T_\mathrm{magma}$ change per PROTEUS step \[K] |
| `tmagma_rtol` | float | `0.02` | Maximum relative $T_\mathrm{magma}$ change per step |

**Ultra-thin boundary layer**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `param_utbl` | bool | `false` | Enable UTBL parameterisation |
| `param_utbl_const` | float | `1e-7` | UTBL scaling constant \[K$^{-1}$] |

**Hydrostatic EOS (Adams-Williamson)**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adams_williamson_rhos` | float | `4078.95` | Surface density \[kg m$^{-3}$] |
| `adams_williamson_beta` | float | `1.1115e-7` | Density gradient \[m$^{-1}$] |
| `adiabatic_bulk_modulus` | float | `260e9` | Adiabatic bulk modulus \[Pa] |

**Phase material properties**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `melt_log10visc` | float | `2.0` | log$_{10}$ molten-silicate viscosity \[Pa s] |
| `solid_log10visc` | float | `22.0` | log$_{10}$ solid-silicate viscosity \[Pa s] |
| `melt_cond` | float | `4.0` | Molten-silicate thermal conductivity \[W m$^{-1}$ K$^{-1}$] |
| `solid_cond` | float | `4.0` | Solid-silicate thermal conductivity \[W m$^{-1}$ K$^{-1}$] |
| `eddy_diffusivity_thermal` | float | `1.0` | Scaling on MLT thermal eddy diffusivity |
| `eddy_diffusivity_chemical` | float | `1.0` | Scaling on MLT chemical eddy diffusivity |
| `latent_heat_of_fusion` | float | `4e6` | Silicate latent heat of fusion \[J kg$^{-1}$] |
| `phase_transition_width` | float | `0.1` | Width of the mushy-zone blend \[melt fraction] |

**Core thermal model**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `core_tfac_avg` | float | `1.147` | $T_\mathrm{avg} / T_\mathrm{cmb}$ ratio from adiabatic gradient |

**Diagnostics**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `write_flux_diagnostics` | bool | `false` | Save per-component flux decomposition to Aragog NetCDF output |

**Constant-properties mode**

Bypasses EOS tables and uses an analytical $T(S) = T_\mathrm{ref} \exp((S - S_\mathrm{ref}) / C_p)$
relationship. Useful for controlled parity tests.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `const_properties` | bool | `false` | Enable constant-properties mode |
| `const_rho` | float | `4000.0` | Constant density \[kg m$^{-3}$] |
| `const_Cp` | float | `1000.0` | Constant heat capacity \[J kg$^{-1}$ K$^{-1}$] |
| `const_alpha` | float | `1e-5` | Constant thermal expansivity \[K$^{-1}$] |
| `const_cond` | float | `4.0` | Constant thermal conductivity \[W m$^{-1}$ K$^{-1}$] |
| `const_log10visc` | float | `2.0` | Constant log$_{10}$ viscosity \[Pa s] |
| `const_T_ref` | float | `3500.0` | Reference temperature \[K] |
| `const_S_ref` | float | `3000.0` | Reference entropy \[J kg$^{-1}$ K$^{-1}$] |

### Aragog `[interior_energetics.aragog]`

Aragog is the default interior thermal evolution module. It solves the
mantle energy equation with CVODE (SUNDIALS) using JAX-derived analytic
Jacobians for robust convergence.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mass_coordinates` | bool | `true` | Use mass-coordinate mesh spacing (gives finer resolution near surface) |
| `backend` | str | `"jax"` | ODE backend: `jax` (CVODE + analytic Jacobian, recommended) or `numpy` (CVODE + finite-difference Jacobian) |
| `solver_method` | str | `"cvode"` | ODE solver: `cvode` (SUNDIALS), `radau` (scipy), `bdf` (scipy) |
| `atol_temperature_equivalent` | float | `1e-8` | Effective temperature-scale absolute tolerance \[K] |
| `phase_smoothing` | str | `"tanh"` | Phase-boundary smoothing: `tanh` or `cubic_hermite` |
| `core_bc` | str | `"energy_balance"` | Core-mantle boundary condition: `energy_balance` (SPIDER bit-parity), `quasi_steady`, `gradient`, or `bower2018` (experimental) |
| `tolerance_struct` | float | `100` | Absolute mass tolerance \[kg] for the interior-radius secant solver |
| `scalar_gravity_override` | bool | `false` | Overwrite the mesh gravity column with a uniform surface scalar; set `true` only for paired scalar-gravity comparisons |
| `phi_step_cap` | float | `0.0` | Per-call melt-fraction step cap. A CVODE root function returns control at the exact time the larger of the global mass-weighted $|\Delta\Phi|$ and the maximum single-cell $|\Delta\varphi|$ reaches this cap, which removes the discontinuous core-temperature drop at crystallisation onset. `0.0` is the schema default and the Aragog wrapper promotes it to `0.1` on the coupled Zalmoxis stack; a positive value overrides that promotion, and `-1.0` is the single off sentinel that keeps the cap off even on Zalmoxis; any other negative, NaN, or infinity is rejected at load |
| `temperature_step_cap` | float | `0.0` | Per-call per-cell temperature step cap \[K]. Shares the same root function as `phi_step_cap` and fires on the maximum single-cell $|\Delta T|$, bounding the core-temperature drop on the solid adiabat just below the solidus where the melt-fraction cap goes blind. `0.0` promotes to `100.0` on the Zalmoxis stack; positive overrides, `-1.0` keeps it off; any other negative, NaN, or infinity is rejected at load |
| `entropy_step_cap` | float | `0.0` | Per-call per-cell entropy step cap \[J kg$^{-1}$ K$^{-1}$] in the native solver variable; same role as `temperature_step_cap` without an EOS lookup in the root function. `0.0` promotes to `100.0` on the Zalmoxis stack; positive overrides, `-1.0` keeps it off; any other negative, NaN, or infinity is rejected at load |
| `phase_boundary_entropy_margin` | float | `200.0` | Phase-boundary proximity band \[J kg$^{-1}$ K$^{-1}$] within which a staggered cell counts as near a solidus or liquidus crossing, tightening the integrator `max_step` so CVODE resolves the stiff two-phase RHS. A positive value is required. Keep it of order a few hundred; a value orders of magnitude above the default makes every cell count as near a boundary at all times, clamping the integrator to 1 yr steps for the whole run and stalling it |

### SPIDER `[interior_energetics.spider]`

SPIDER is the C interior module. It requires PETSc and produces
results consistent with Bower et al. (2018) [^cite-bower2018].

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `solver_type` | str | `"bdf"` | SUNDIALS method: `adams` or `bdf` |
| `matprop_smooth_width` | float | `0.01` | Melt-fraction smoothing window across solidus/liquidus |
| `tolerance_struct` | float | `100` | Absolute mass tolerance \[kg] for the interior-radius secant solver |
| `log_output` | bool | `true` | Write SPIDER solver log output |

### Dummy `[interior_energetics.dummy]`

A parameterised cooling model with prescribed solidus and liquidus.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mantle_tliq` | float | `2700.0` | Liquidus temperature \[K] |
| `mantle_tsol` | float | `1700.0` | Solidus temperature \[K] |
| `mantle_rho` | float | `4550.0` | Mantle density \[kg m$^{-3}$] for the density profile and the fallback mantle-mass estimate; the mantle mass uses $M_\mathrm{int}-M_\mathrm{core}$ when the structure provides it |
| `mantle_cp` | float | `1792.0` | Mantle heat capacity \[J kg$^{-1}$ K$^{-1}$] |
| `heat_internal` | float | `0.0` | Internal heating rate \[W kg$^{-1}$] |

### Boundary `[interior_energetics.boundary]`

A 0-D box model for the mantle thermal evolution based on Schaefer et al. (2016) [^cite-schaefer2016],
with prescribed solidus and liquidus and parameterised convective heat transport.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rtol` | float | `1e-6` | ODE solver relative tolerance |
| `atol` | float | `1e-9` | ODE solver absolute tolerance |
| `T_p_0` | float | `3500.0` | Initial potential temperature \[K] (if Zalmoxis is not active) |
| `T_solidus` | float | `1420.0` | Mantle solidus \[K] |
| `T_liquidus` | float | `2020.0` | Mantle liquidus \[K] |
| `Tsurf_event_change` | float | `20.0` | Maximum surface-$T$ jump per iteration before triggering an event \[K] |
| `critical_rayleigh_number` | float | `1100.0` | Critical Rayleigh number for onset of convection |
| `heat_fusion_silicate` | float | `4e5` | Latent heat of fusion for silicates \[J kg$^{-1}$] |
| `nusselt_exponent` | float | `0.33` | Nusselt-Rayleigh scaling exponent |
| `silicate_heat_capacity` | float | `1200.0` | Silicate heat capacity \[J kg$^{-1}$ K$^{-1}$] |
| `atm_heat_capacity` | float | `1.7e4` | Fallback atmosphere heat capacity \[J kg$^{-1}$ K$^{-1}$] |
| `silicate_density` | float | `4103.0` | Silicate density \[kg m$^{-3}$] |
| `thermal_conductivity` | float | `4.2` | Thermal conductivity \[W m$^{-1}$ K$^{-1}$] |
| `thermal_diffusivity` | float | `1e-6` | Thermal diffusivity \[m$^2$ s$^{-1}$] |
| `thermal_expansivity` | float | `2e-5` | Thermal expansivity \[K$^{-1}$] |
| `viscosity_model` | int | `2` | 1 = constant, 2 = aggregate smooth, 3 = Arrhenius |
| `eta_constant` | float | `100` | Constant viscosity \[Pa s] (model 1) |
| `transition_width` | float | `0.2` | Viscosity transition width \[melt fraction] |
| `eta_solid_const` | float | `1e22` | Solid-end viscosity \[Pa s] (aggregate model) |
| `eta_melt_const` | float | `100` | Melt-end viscosity \[Pa s] (aggregate model) |
| `dynamic_viscosity` | float | `3.8e9` | Arrhenius solid reference dynamic viscosity \[Pa s] |
| `activation_energy` | float | `3.5e5` | Arrhenius solid activation energy \[J mol$^{-1}$] |
| `creep_parameter` | float | `26.0` | Arrhenius creep parameter |
| `viscosity_prefactor` | float | `2.4e-4` | VFT magma-ocean prefactor \[Pa s] |
| `viscosity_activation_temp` | float | `4600.0` | VFT magma-ocean activation temperature \[K] |
| `logging` | bool | `false` | Write diagnostic CSV files |

---

**See also:** [Interior modules](../../Explanations/model.md#interior-energetics-aragog-spider-boundary) | [Structure module](../../Explanations/model.md#interior-structure-zalmoxis) | [Melting curves](../melting_curves.md)

 [^cite-bower2018]: Bower, D.J., Sanan, P. & Wolf, A.S., *[Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets](https://doi.org/10.1016/j.pepi.2017.11.004)*, Physics of the Earth and Planetary Interiors, 274, 49-62, 2018. [SciX](https://scixplorer.org/abs/2018PEPI..274...49B/abstract).

 [^cite-schaefer2016]: Schaefer, L., Wordsworth, R.D., Berta-Thompson, Z. & Sasselov, D., *[Predictions of the atmospheric composition of GJ 1132b](https://doi.org/10.3847/0004-637X/829/2/63)*, The Astrophysical Journal, 829, 63, 2016. [SciX](https://scixplorer.org/abs/2016ApJ...829...63S/abstract).

 [^cite-stixrude2014]: Stixrude, L., *[Melting in super-earths](https://doi.org/10.1098/rsta.2013.0076)*, Philosophical Transactions of the Royal Society A, 372, 20130076, 2014. [SciX](https://scixplorer.org/abs/2014RSPTA.37230076S/abstract).
