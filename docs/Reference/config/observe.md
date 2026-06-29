# Observations

The `[observe]` section configures synthetic observation generation. PROTEUS
can compute transit and eclipse depth spectra from the simulated atmospheric
state using the petitRADTRANS forward model.

The `[accretion]` section is reserved for late accretion modelling (not yet
implemented).

## Synthetic observations `[observe]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | str or none | `"none"` | Observation module: `petitRADTRANS` or `none` (disabled) |
| `source` | str | `"all"` | Composition source selection: `all`, `outgas`, `profile`, or `offchem` |
| `spectrum_type` | str | `"both"` | Spectrum products to generate: `both`, `transit`, or `eclipse` |
| `remove_one_gas` | bool | `true` | Generate leave-one-out spectra (`<SPECIES>_removed/ppm` columns) |
| `clip_vmr` | float | `1e-8` | Minimum VMR below which a species is excluded from the radiative transfer |
| `reference_pressure` | float | `10` | Reference pressure level used to set the planet radius baseline \[bar\] |

### petitRADTRANS `[observe.petitRADTRANS]`

[petitRADTRANS](https://petitradtrans.readthedocs.io/) computes wavelength-dependent
transit and emission depths from the atmospheric composition and
temperature-pressure profile. Its opacity tables are read from
`$FWL_DATA/prt/input_data` (set via the `FWL_DATA` environment variable at
PROTEUS startup).

#### Composition sources

The atmospheric composition passed to petitRADTRANS is controlled by
`observe.source`:

- `all` (default): compute spectra for all available sources
- `outgas`, `profile`, or `offchem`: compute spectra only for the selected source

Available source modes:

| Source | Description |
|--------|-------------|
| `outgas` | Constant VMR with height from the outgassing module (helpfile `*_vmr` columns) |
| `profile` | Full vertical VMR profile from the climate module NetCDF output (`*_vmr` arrays) |
| `offchem` | Photochemical composition from the VULCAN offline chemistry output |

When `observe.source = "all"`, unavailable sources are skipped automatically:
`profile` is skipped when `atmos_clim.module = "dummy"`, and `offchem` is
skipped when `atmos_chem.module` is `none`.

When `observe.source` is set explicitly to `profile` or `offchem`, PROTEUS
raises an error if the required upstream module is unavailable.

The spectrum product set is controlled by `observe.spectrum_type`:

- `both` (default): compute both transit and eclipse spectra
- `transit`: compute only transit spectra
- `eclipse`: compute only eclipse spectra

#### Supported line species

The following gases are searched for in the petitRADTRANS correlated-k
opacity table directory. A species is included only when its corresponding
opacity directory exists under `input_data/opacities/lines/correlated_k/`:

`H2O`, `H2`, `CO2`, `CO`, `CH4`, `SO2`, `H2S`, `O2`, `N2`, `NH3`, `OH`

The species whose opacity file has the broadest wavelength coverage is moved
to the front of the line-species list before calling petitRADTRANS. This helps
avoid opacity-table overlap issues in the forward-model initialisation.

#### Rayleigh and CIA species

Rayleigh scattering and collision-induced absorption (CIA) species are taken
from the constants defined in `proteus.utils.constants`
(`prt_rayleigh_species`, `prt_cia_species`) and are filtered to those present
in the actual gas mixture.

#### Profile interpolation

Before calling petitRADTRANS, the temperature and VMR profiles are
interpolated onto a uniform 100-point log-spaced pressure grid using PCHIP
splines. Temperature values are clipped to the range
\[100.5 K, 3999.5 K\] to stay within petitRADTRANS table bounds.
If the atmosphere profile is stored with decreasing pressure (surface first),
it is reversed to ascending order before interpolation.

#### Leave-one-out species spectra

After computing the baseline spectrum (all species present), PROTEUS can
optionally re-run petitRADTRANS once per line species, removing one
species at a time. This produces additional spectrum columns in the output
CSV that show the contribution of each individual gas to the total signal.

This behaviour is controlled by `observe.remove_one_gas`.
Set it to `false` to write baseline-only spectra.

Set `observe.petitRADTRANS.silent = true` to silence the opacity-loading
console messages emitted by petitRADTRANS when creating `Radtrans` objects.

See [Output format](../../Reference/output.md) for the CSV column layout.

#### Configuration options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `line_opacity_mode` | str | `"c-k"` | Opacity treatment: `"c-k"` (correlated-k, faster) or `"lbl"` (line-by-line, higher spectral resolution) |
| `include_rayleigh` | bool | `true` | Include Rayleigh scattering contributions |
| `include_cia` | bool | `true` | Include collision-induced absorption contributions |
| `silent` | bool | `false` | Suppress petitRADTRANS stdout/stderr during `Radtrans` initialization |

## Late accretion `[accretion]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | str or none | `none` | Late accretion module (reserved for future implementation) |

---

**See also:** [Model description](../../Explanations/model.md) | [Output format](../../Reference/output.md) | [Postprocessing](../../How-to/usage_postprocessing.md)
