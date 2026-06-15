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
| `clip_vmr` | float | `1e-8` | Minimum VMR for species inclusion in the calculation |
| `reference_pressure` | float | `10` | Reference pressure for synthetic spectrum generation [bar] |

### petitRADTRANS `[observe.petitRADTRANS]`

petitRADTRANS computes wavelength-dependent transit and eclipse depths from the
atmospheric composition and temperature-pressure profile. Three input sources
are available:

- **`outgas`**: use the outgassed equilibrium composition only
- **`profile`**: use the full atmospheric T(p) profile from the climate module
- **`offchem`**: use the photochemical composition from VULCAN

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_data_path` | str | `"none"` | Optional path to petitRADTRANS `input_data`. If `None`, the installed package location will be used. |
| `line_opacity_mode` | str | `"c-k"` | Opacity treatment: 'c-k' (correlated-k) or 'lbl' (line-by-line). |
| `include_rayleigh` | bool | `"true"` | Include Rayleigh scattering contributions. |
| `include_cia` | bool | `"true"` | Include collision-induced absorption contributions. |

## Late accretion `[accretion]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | str or none | `none` | Late accretion module (reserved for future implementation) |

---

**See also:** [Model description](../../Explanations/model.md)
