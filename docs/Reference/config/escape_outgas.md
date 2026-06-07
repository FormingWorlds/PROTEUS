# Escape and outgassing

The `[escape]` section configures atmospheric escape (mass loss to space).
The `[outgas]` section configures volatile outgassing (partitioning between
interior and atmosphere).

Submodule documentation:
[ZEPHYRUS](https://github.com/FormingWorlds/ZEPHYRUS) |
[CALLIOPE](https://proteus-framework.org/CALLIOPE/) |
[atmodeller](https://github.com/djbower/atmodeller).
See also [Model description](../../Explanations/model.md#atmospheric-escape-zephyrus).

## Atmospheric escape `[escape]`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | str or none | `"zephyrus"` | Escape module: `zephyrus` (energy-limited), `dummy` (fixed rate), `none` (disabled) |
| `reservoir` | str | `"outgas"` | Composition reservoir for escaping gas: `outgas`, `bulk`, `pxuv` |

### ZEPHYRUS `[escape.zephyrus]`

Energy-limited escape: the mass loss rate scales with the XUV flux and escape
efficiency.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Pxuv` | float | `5e-5` | XUV opacity pressure level [bar] |
| `efficiency` | float | `0.1` | Escape efficiency [0, 1] |
| `tidal` | bool | `false` | Include tidal contribution to escape |

### Dummy escape `[escape.dummy]`

A fixed bulk escape rate, useful for testing and parameter studies.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rate` | float | `0.0` | Bulk escape rate [kg s$^{-1}$] |

---

## Outgassing `[outgas]`

The outgassing module computes the thermodynamic equilibrium partitioning of
volatiles between the atmosphere, silicate melt, and solid mantle at the
planetary surface conditions.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | str | `"calliope"` | Outgassing module: `calliope` (Gibbs minimisation), `atmodeller` (simplified), `dummy` (fixed) |
| `fO2_shift_IW` | float | `4.0` | Redox state: fO$_2$ offset from the iron-wustite buffer [log$_{10}$ units] |
| `mass_thresh` | float | `1e16` | Minimum volatile mass threshold [kg] |
| `h2_binodal` | bool | `false` | Enable H$_2$-MgSiO$_3$ miscibility gap model. `true` is rejected at config load: the parameterisation is not production ready |
| `T_floor` | float | `700.0` | Skip outgassing below this temperature [K] |
| `solver_rtol` | float | `1e-4` | Relative mass/equilibrium tolerance |
| `solver_atol` | float | `1e-6` | Absolute mass/equilibrium tolerance |

### CALLIOPE `[outgas.calliope]`

CALLIOPE uses Gibbs free energy minimisation to compute the gas-melt
equilibrium at the planetary surface, handling C-H-N-O-S chemistry with
fO$_2$ buffering.

**Species switches** (set to `false` to exclude a species from the equilibrium)

| Parameter | Default | Species |
|-----------|---------|---------|
| `include_H2O` | `true` | H$_2$O |
| `include_CO2` | `true` | CO$_2$ |
| `include_N2` | `true` | N$_2$ |
| `include_S2` | `true` | S$_2$ |
| `include_SO2` | `true` | SO$_2$ |
| `include_H2S` | `true` | H$_2$S |
| `include_NH3` | `true` | NH$_3$ |
| `include_H2` | `true` | H$_2$ |
| `include_CH4` | `true` | CH$_4$ |
| `include_CO` | `true` | CO |
| `solubility` | `true` | Enable melt-gas partitioning (`false` = all volatiles in atmosphere) |

**Solver**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nguess` | int | `1000` | Maximum number of initial-guess samples for the equilibrium solver |
| `nsolve` | int | `3000` | Maximum number of solver iterations per call |
| `p_guess_max` | float | `1e5` | Upper bound [bar] of the cold-start surface-pressure draw, in `(0, 1e7]`; raise it to seed the solver higher for high-pressure (e.g. sub-Neptune) cases |

### Atmodeller `[outgas.atmodeller]`

An alternative outgassing solver with configurable solubility laws and
real-gas equations of state.

!!! note
    atmodeller is an optional backend and is not installed with PROTEUS by
    default; the standard outgassing module is `calliope`. Install it with
    `pip install "fwl-proteus[atmodeller]"` before setting
    `outgas.module = "atmodeller"`. atmodeller is GPL-3.0 licensed; review
    its terms before installing. See
    [Installation: optional modules](../../How-to/installation.md#optional-modules).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `solver_mode` | str | `"robust"` | Root-finding mode: `robust` (better convergence) or `basic` (faster) |
| `solver_max_steps` | int | `256` | Maximum solver iterations |
| `solver_multistart` | int | `10` | Number of random restarts |
| `include_condensates` | bool | `true` | Enable condensate formation (e.g. graphite) |

**Solubility laws** (set to `"none"` to disable dissolution for a species)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `solubility_H2O` | `"H2O_peridotite_sossi23"` | Water solubility law |
| `solubility_CO2` | `"CO2_basalt_dixon95"` | CO$_2$ solubility law |
| `solubility_H2` | `"H2_basalt_hirschmann12"` | H$_2$ solubility law |
| `solubility_N2` | `"N2_basalt_dasgupta22"` | N$_2$ solubility law |
| `solubility_S2` | `"S2_sulfide_basalt_boulliung23"` | S$_2$ solubility law |
| `solubility_CO` | `"CO_basalt_yoshioka19"` | CO solubility law |
| `solubility_CH4` | `"CH4_basalt_ardia13"` | CH$_4$ solubility law |

**Real gas EOS** (set to `"none"` for ideal gas)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eos_H2O` | `"none"` | Water EOS |
| `eos_CO2` | `"none"` | CO$_2$ EOS |
| `eos_H2` | `"none"` | H$_2$ EOS |
| `eos_CH4` | `"none"` | CH$_4$ EOS |
| `eos_CO` | `"none"` | CO EOS |

---

**See also:** [Escape modules](../../Explanations/model.md#atmospheric-escape-zephyrus) | [Outgassing modules](../../Explanations/model.md#volatile-outgassing-calliope-atmodeller)
