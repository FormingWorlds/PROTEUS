# Planet and volatiles

The `[planet]` section defines the bulk planet properties, initial temperature
profile, and volatile inventory. These parameters set the initial conditions
for the coupled evolution.

## Bulk properties

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mass_tot` | float | `1.0` | Total initial planet mass \[M$_\oplus$] |
| `prevent_warming` | bool | `false` | Require monotonic cooling (clamp $T_\mathrm{magma}$ to previous value if it increases) |
| `R_int_override` | float or none | `none` | Advanced: bypass the radius root finder and force a fixed interior radius \[m]; `none` uses the root finder. Used for SPIDER/Aragog parity runs |

## Initial temperature profile

The `temperature_mode` parameter selects how the initial temperature
distribution is constructed. Different modes anchor the profile at different
reference points.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature_mode` | str | `"liquidus_super"` | See modes table below |
| `tsurf_init` | float | `4000` | Surface temperature \[K] (isothermal, linear, adiabatic modes) |
| `tcmb_init` | float | `6000` | Core-mantle boundary temperature \[K] (adiabatic_from_cmb mode) |
| `tcenter_init` | float | `6000` | Center temperature \[K] (linear mode only) |
| `delta_T_super` | float | `500` | Superliquidus offset at CMB \[K] (liquidus_super mode) |
| `ini_entropy` | float | `3900` | Initial specific entropy \[J/kg/K] (isentropic mode) |
| `ini_dsdr` | float | `-4.698e-6` | Initial entropy gradient \[J/kg/K/m] (isentropic mode) |
| `f_accretion` | float | `0.04` | Accretion heat retention fraction \[0, 1] (accretion mode) |
| `f_differentiation` | float | `0.50` | Differentiation heat retention fraction \[0, 1] (accretion mode) |

### Temperature modes

| Mode | Anchor point | Description |
|------|-------------|-------------|
| `isothermal` | `tsurf_init` | Uniform temperature throughout mantle |
| `linear` | `tsurf_init`, `tcenter_init` | Linear gradient from center to surface |
| `adiabatic` | `tsurf_init` | Adiabat anchored at the surface, integrated downward |
| `adiabatic_from_cmb` | `tcmb_init` | Adiabat anchored at the CMB at a fixed temperature, integrated upward |
| `liquidus_super` | `delta_T_super` | Adiabat anchored at $T_\mathrm{liq}(P_\mathrm{cmb}) + \Delta T_\mathrm{super}$ (default), using the Fei et al. (2021) [^cite-fei2021] MgSiO$_3$ liquidus. Setting $\Delta T_\mathrm{super} = 0$ places the IC exactly on the liquidus. |
| `accretion` | `f_accretion`, `f_differentiation` | Temperature from gravitational accretion and core-mantle differentiation energy retention (White and Li, 2025) |
| `isentropic` | `ini_entropy`, `ini_dsdr` | Entropy-based IC; the interior solver maps $S \to T(P)$ via its EOS table |

## Redox state

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fO2_source` | str | `"user_constant"` | How fO$_2$ is determined: `user_constant` (buffered to `outgas.fO2_shift_IW`), `from_O_budget` (derived from O mass balance) |

When `fO2_source = "user_constant"`, the atmospheric fO$_2$ is buffered at the
iron-wustite offset set by `outgas.fO2_shift_IW`. When `fO2_source = "from_O_budget"`,
the O budget from `planet.elements.O_mode`/`O_budget` is authoritative and the
chemistry solver derives the fO$_2$ that produces the supplied O inventory.
A third value, `from_mantle_redox`, is reserved for a future release and is
rejected at config-load until that work lands.

## Volatile inventory

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `volatile_mode` | str | `"elements"` | How to set the volatile inventory: `elements` (by elemental budgets) or `gas_prs` (by surface partial pressures) |
| `volatile_reservoir` | str | `"mantle"` | Reference mass for ppmw calculations: `mantle` or `mantle+core` |

### Element abundances `[planet.elements]`

Used when `volatile_mode = "elements"`. Each element has a mode (defining
the unit) and a budget (the value in that unit).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `O_mode` | str | `"ic_chemistry"` | `ic_chemistry` (defer to CALLIOPE equilibrium), `ppmw`, `kg`, `FeO_mantle_wt_pct` |
| `O_budget` | float | `0.0` | Oxygen inventory (ignored for `ic_chemistry` mode) |
| `H_mode` | str | `"oceans"` | `oceans` (Earth ocean equivalents), `ppmw`, `kg` |
| `H_budget` | float | `0.0` | Hydrogen inventory |
| `C_mode` | str | `"C/H"` | `C/H` (mass ratio to H), `ppmw`, `kg` |
| `C_budget` | float | `0.0` | Carbon inventory |
| `N_mode` | str | `"N/H"` | `N/H` (mass ratio to H), `ppmw`, `kg` |
| `N_budget` | float | `0.0` | Nitrogen inventory |
| `S_mode` | str | `"S/H"` | `S/H` (mass ratio to H), `ppmw`, `kg` |
| `S_budget` | float | `0.0` | Sulfur inventory |
| `use_metallicity` | bool | `false` | Scale C/N/S from solar metallicity (overrides C/N/S mode+budget) |
| `metallicity` | float | `1000` | Metallicity relative to solar, by mass |

!!! note
    The Python defaults for volatile budgets are zero. The values in
    `all_options.toml` (e.g., `H_budget = 1.0`) show recommended starting
    points for a typical rocky planet.

#### Noble gases

The noble gases He, Ne, Ar, Kr, and Xe are opt-in. A noble gas contributes
only when its inclusion flag in `[outgas.calliope]` (`include_He`, ...) is
`true` and its budget below is positive; a run with no noble budget is
unchanged. Each noble gas is partitioned between the magma ocean and the
atmosphere by a Henry's-law solubility and tracked in the whole-planet mass
balance.

| Field | Type | Default | Description |
|---|---|---|---|
| `He_mode` | str | `"kg"` | `kg`, `ppmw` (relative to the volatile reservoir), or `solar` |
| `He_budget` | float | `0.0` | Helium inventory (units depend on the mode) |
| `Ne_mode`, `Ar_mode`, `Kr_mode`, `Xe_mode` | str | `"kg"` | As `He_mode` |
| `Ne_budget`, `Ar_budget`, `Kr_budget`, `Xe_budget` | float | `0.0` | As `He_budget` |

In `solar` mode the budget is a multiple of the protosolar X/H mass ratio, so
the inventory is `budget * (X/H)_solar * H_kg`; a value of `1.0` gives a
protosolar noble gas complement.

!!! warning
    The `solar` mode is an upper-bound reference, not a realistic default.
    Planetary bodies are depleted in noble gases by orders of magnitude
    relative to solar, so realistic budgets use `solar` with a value far below
    one, or set the inventory directly in `kg` or `ppmw`.

### Partial pressures `[planet.gas_prs]`

Used when `volatile_mode = "gas_prs"`. Sets the initial atmosphere directly
by surface partial pressure for each gas species.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `H2O` | float | `0.0` | \[bar] |
| `CO2` | float | `0.0` | \[bar] |
| `N2` | float | `0.0` | \[bar] |
| `S2` | float | `0.0` | \[bar] |
| `SO2` | float | `0.0` | \[bar] |
| `H2S` | float | `0.0` | \[bar] |
| `NH3` | float | `0.0` | \[bar] |
| `H2` | float | `0.0` | \[bar] |
| `CH4` | float | `0.0` | \[bar] |
| `CO` | float | `0.0` | \[bar] |

---

**See also:** [Model description](../../Explanations/model.md) | [Earth analogue tutorial](../../Tutorials/earth_analogue.md)

 [^cite-fei2021]: Fei, Y., Seagle, C.T., Townsend, J.P., et al., *[Melting and density of MgSiO3 determined by shock compression of bridgmanite to 1254 GPa](https://doi.org/10.1038/s41467-021-21170-y)*, Nature Communications, 12, 876, 2021. [SciX](https://scixplorer.org/abs/2021NatCo..12..876F/abstract).
