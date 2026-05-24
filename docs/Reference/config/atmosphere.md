# Atmosphere and chemistry

The `[atmos_clim]` section configures the atmospheric climate module (radiative-convective
structure and surface energy balance). The `[atmos_chem]` section configures
atmospheric chemistry (photochemical kinetics).

## Atmosphere climate `[atmos_clim]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | str | `"agni"` | Climate module: `agni` (detailed RCE solver), `janus` (fast grey/correlated-k), `dummy` (parameterised) |
| `spectral_group` | str | `"Honeyside"` | Opacity k-table set (shared by AGNI and JANUS) |
| `spectral_bands` | str | `"48"` | Number of wavenumber bands in k-tables |
| `num_levels` | int | `50` | Number of vertical atmosphere levels (minimum 15) |
| `p_top` | float | `1e-6` | Top-of-atmosphere pressure [bar] |
| `p_obs` | float | `0.02` | Observation pressure level [bar] (defines transit radius) |
| `overlap_method` | str | `"ee"` | Gas overlap method: `ro`, `rorr`, `ee` (equivalent extinction) |

**Radiative properties**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rayleigh` | bool | `true` | Enable Rayleigh scattering |
| `cloud_enabled` | bool | `false` | Enable water cloud radiative effects |
| `cloud_alpha` | float | `0.0` | Condensate retention fraction [0, 1] |
| `aerosols_enabled` | bool | `false` | Enable aerosol radiative effects (AGNI only) |
| `albedo_pl` | float or str | `0.0` | Planetary bond albedo: a constant float, or a path to a CSV lookup table |
| `surf_greyalbedo` | float | `0.1` | Grey surface albedo (used when `agni.surf_material = "greybody"`) |

**Surface boundary condition**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `surf_state` | str | `"skin"` | Energy balance scheme: `mixed_layer` (ocean-like), `fixed` (lock to magma temperature), `skin` (conductive skin layer) |
| `surface_d` | float | `0.01` | Conductive skin thickness [m] |
| `surface_k` | float | `2.0` | Conductive skin conductivity [W m$^{-1}$ K$^{-1}$] |

**Solver limits**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tmp_minimum` | float | `0.5` | Temperature floor [K] |

### AGNI `[atmos_clim.agni]`

AGNI is a detailed radiative-convective equilibrium solver written in Julia.
It iteratively solves for the atmospheric temperature-pressure profile that
balances absorbed stellar radiation with outgoing thermal emission and
interior heat flux.

**Physics**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `solve_energy` | bool | `true` | Iterative energy-conserving T(p) profile |
| `convection` | bool | `true` | Convective adjustment (mixing length theory) |
| `conduction` | bool | `true` | Conductive heat transport (Fourier) |
| `sens_heat` | bool | `true` | Sensible heat flux at surface |
| `rainout` | bool | `true` | Volatile condensation and evaporation |
| `oceans` | bool | `true` | Surface liquid water oceans |
| `latent_heat` | bool | `false` | Latent heat from condensation/evaporation |
| `real_gas` | bool | `false` | Real-gas EOS where available |
| `chemistry` | str or none | `"none"` | Atmospheric chemistry: `none` or `eq` (FastChem equilibrium) |
| `mlt_criterion` | str | `"s"` | Convection criterion: `s` (Schwarzschild) or `l` (Ledoux) |

**Surface properties**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `surf_material` | str | `"greybody"` | Surface scattering model; `"greybody"` uses `surf_greyalbedo` |
| `surf_roughness` | float | `1e-3` | Surface roughness scale [m] |
| `surf_windspeed` | float | `2.0` | Characteristic wind speed [m s$^{-1}$] |

**Condensation**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `phs_timescale` | float | `1e6` | Phase change relaxation timescale [s] |
| `evap_efficiency` | float | `0.01` | Raindrop re-evaporation efficiency [0, 1] |

**Solver tuning**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `solution_atol` | float | `0.5` | Newton solver absolute tolerance [W m$^{-2}$] |
| `solution_rtol` | float | `0.15` | Newton solver relative tolerance |
| `psurf_thresh` | float | `0.1` | Skip full RCE when surface pressure is below this [bar] |
| `dx_max` | float | `35.0` | Maximum Newton step per iteration [K] |
| `dx_max_ini` | float | `50.0` | Maximum Newton step during first few PROTEUS loops [K] |
| `max_steps` | int | `200` | Maximum Newton iterations per call |
| `perturb_all` | bool | `false` | Recompute full Jacobian each iteration |
| `ini_profile` | str | `"isothermal"` | Initial T(p) guess: `isothermal`, `loglinear`, `dry_adiabat`, `analytic` |
| `ls_default` | int | `2` | Linesearch method: 0 (none), 1 (golden section), 2 (backtracking) |
| `verbosity` | int | `1` | AGNI log level: 0 (silent), 1 (info), 2 (debug) |

**FastChem equilibrium chemistry** (used when `chemistry = "eq"`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fastchem_floor` | float | `1000.0` | Minimum temperature sent to FastChem [K] |
| `fastchem_maxiter_chem` | int | `60000` | Maximum FastChem iterations (chemistry) |
| `fastchem_maxiter_solv` | int | `20000` | Maximum FastChem iterations (internal solver) |
| `fastchem_xtol_chem` | float | `1e-4` | FastChem convergence tolerance (chemistry) |
| `fastchem_xtol_elem` | float | `1e-4` | FastChem convergence tolerance (elemental) |

### JANUS `[atmos_clim.janus]`

JANUS is a fast 1-D radiative-convective atmosphere model. It is
computationally lighter than AGNI and suitable for parameter sweeps.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `F_atm_bc` | int | `0` | Outgoing flux boundary: 0 (top of atmosphere) or 1 (surface) |
| `tropopause` | str or none | `"none"` | Tropopause scheme: `none`, `skin`, `dynamic` |
| `cloud_alpha` | float | `0.0` | Condensate retention fraction [0, 1] |
| `tmp_maximum` | float | `5000.0` | Solver temperature ceiling [K] |

### Dummy atmosphere `[atmos_clim.dummy]`

A grey-body parameterisation: $T_\mathrm{rad} = T_\mathrm{surf} \cdot (1 - \gamma)$.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gamma` | float | `0.5` | Atmosphere opacity factor; 0 = transparent, 1 = zero OLR |
| `height_factor` | float | `3.0` | Observable height as multiple of scale height |
| `fixed_flux` | float | `-1.0` | If > 0, return this constant $F_\mathrm{atm}$ [W m$^{-2}$] |

---

## Atmospheric chemistry `[atmos_chem]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | str or none | `"none"` | Chemistry module: `none` (skip), `vulcan` (photochemistry), `dummy` |
| `when` | str | `"manually"` | When to run: `manually` (user-triggered), `offline` (at simulation end), `online` (each snapshot) |
| `photo_on` | bool | `true` | Enable photochemistry |
| `Kzz_on` | bool | `true` | Enable eddy diffusion |
| `Kzz_const` | float or none | `none` | Constant $K_{zz}$ [cm$^2$ s$^{-1}$]; `none` = computed profile |
| `moldiff_on` | bool | `true` | Enable molecular diffusion |
| `updraft_const` | float | `0.0` | Constant updraft velocity [cm s$^{-1}$] |

### VULCAN `[atmos_chem.vulcan]`

VULCAN solves photochemical kinetics for atmospheric composition, computing
steady-state mixing ratios from a chemical reaction network.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `network` | str | `"SNCHO"` | Chemical network: `CHO`, `NCHO`, `SNCHO` |
| `ini_mix` | str | `"profile"` | Initial mixing ratios: `profile` (from atmosphere module) or `outgas` (from outgassing) |
| `fix_surf` | bool | `false` | Hold surface mixing ratios fixed |
| `make_funs` | bool | `true` | Generate reaction network functions |
| `yconv_cri` | float | `0.05` | Convergence criterion (mixing ratio change) |
| `slope_cri` | float | `0.0001` | Convergence criterion (rate of change) |
| `clip_fl` | float | `1e-20` | Stellar flux floor [erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$] |
| `clip_vmr` | float | `1e-10` | Neglect species below this VMR |
| `save_frames` | bool | `false` | Save plot frames during iterations |
