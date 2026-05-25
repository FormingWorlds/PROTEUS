# Observations

The `[observe]` section configures synthetic observation generation. PROTEUS
can compute transit and eclipse depth spectra from the simulated atmospheric
state using the PLATON forward model.

The `[accretion]` section is reserved for late accretion modelling (not yet
implemented).

## Synthetic observations `[observe]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `synthesis` | str or none | `"none"` | Observation module: `platon` or `none` (disabled) |

### PLATON `[observe.platon]`

PLATON computes wavelength-dependent transit and eclipse depths from the
atmospheric composition and temperature-pressure profile. Three input sources
are available:

- **`outgas`**: use the outgassed equilibrium composition only
- **`profile`**: use the full atmospheric T(p) profile from the climate module
- **`offchem`**: use the photochemical composition from VULCAN

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `downsample` | int | `8` | Opacity downsample factor (lower = higher spectral resolution, slower) |
| `clip_vmr` | float | `1e-8` | Minimum VMR for species inclusion in the calculation |

## Late accretion `[accretion]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | str or none | `none` | Late accretion module (reserved for future implementation) |

---

**See also:** [Model description](../../Explanations/model.md)
