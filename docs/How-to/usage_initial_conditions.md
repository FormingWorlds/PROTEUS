# Initial thermal conditions

A PROTEUS run starts the planet as a hot magma ocean and follows it as it
cools. The **initial thermal conditions** fix the mantle's starting
temperature and entropy profile, which is the state the interior solver
evolves forward in time. This page explains what that starting state controls,
how to set it through the `[planet]` section of the configuration file, and
which option to favour.

The parameters described here are listed with their types and defaults in the
[planet and volatiles reference](../Reference/config/planet.md#initial-temperature-profile).
For the physics of the interior modules that consume this state, see the
[model description](../Explanations/model.md).

## What the initial conditions do

PROTEUS does not model planet formation. Instead it begins from a chosen
thermal state and integrates the coupled interior-atmosphere system forward.
The initial conditions therefore set:

- **The starting melt fraction.** A hot enough profile starts the mantle fully
  molten; a cooler profile starts it partially crystallised. The magma-ocean
  stage only exists while melt is present, so a fully molten start is the
  usual intent.
- **The thermal energy budget.** The hotter the initial mantle, the more energy
  has to be radiated away before the planet solidifies, and the longer the
  cooling track.
- **The initial atmosphere.** A hotter mantle outgasses more vigorously, so the
  starting surface pressure and composition depend on the initial temperature.

The chosen mode is converted into an initial entropy (or temperature) profile
that the interior solver (Aragog or SPIDER) carries forward. What the initial
conditions do **not** do is move the long-term endpoint: the planet still cools
toward its solidus or toward radiative balance regardless of where it started.
The initial state sets the transient and the total cooling time, not the
destination.

## How to set the initial state

The initial profile is selected by `planet.temperature_mode`. Each mode anchors
the profile at a different reference point and reads a different companion
parameter:

| Mode | Anchored at | Companion parameter(s) |
|------|-------------|------------------------|
| `adiabatic_from_cmb` (default) | core-mantle boundary | `tcmb_init` |
| `liquidus_super` | core-mantle boundary, above the liquidus | `delta_T_super` |
| `adiabatic` | surface | `tsurf_init` |
| `isothermal` | uniform | `tsurf_init` |
| `linear` | surface and centre | `tsurf_init`, `tcenter_init` |
| `accretion` | accretion energetics | `f_accretion`, `f_differentiation` |
| `isentropic` | entropy directly | `ini_entropy`, `ini_dsdr` |

The two core-mantle-boundary (CMB) modes anchor the adiabat at the base of the
mantle and integrate it upward to the surface. The surface modes do the
opposite, anchoring at the surface and integrating downward. The remaining
modes set the profile from accretion energetics (White and Li, 2025) or from
the specific entropy itself.

A minimal CMB-anchored setup looks like this:

```toml
[planet]
    temperature_mode = "adiabatic_from_cmb"
    tcmb_init        = 6000.0   # [K] adiabat anchor at the core-mantle boundary
```

## What to favour

**Favour a CMB-anchored, fully molten start.** The default
`adiabatic_from_cmb` is the canonical choice and is recommended for most runs.
Anchoring at the core-mantle boundary, where the pressure and therefore the
melting temperature are highest, and integrating upward guarantees that the
whole mantle column starts molten. Surface-anchored modes (`adiabatic`,
`isothermal`) carry the opposite risk: under the current equation of state, an
adiabat pinned at the surface can drop the deep mantle below its liquidus at
`t = 0`, leaving a partially solid base that is not a clean magma-ocean start.

**For an explicitly superliquidus start, use `liquidus_super`.** This mode
anchors the CMB temperature a fixed margin above the silicate liquidus:

$$T_\mathrm{cmb} = T_\mathrm{liq}(P_\mathrm{cmb}) + \Delta T_\mathrm{super}$$

where $T_\mathrm{liq}$ is the Fei et al. (2021) MgSiO$_3$ melting curve and
$\Delta T_\mathrm{super}$ is the user-controlled superliquidus offset
(`delta_T_super`, in K). Because the anchor is set by a third-party melting
curve rather than by a fixed temperature, this mode is EOS-agnostic: it does
not bake in a particular entropy convention, which makes it the right choice
for cross-code comparisons and for guaranteeing a fully molten mantle on
larger super-Earths, where a fixed `tcmb_init` may not clear the elevated
high-pressure liquidus.

```toml
[planet]
    temperature_mode = "liquidus_super"
    delta_T_super    = 500.0   # [K] above the liquidus at the core-mantle boundary
```

The default `delta_T_super = 500` K places the entire mantle comfortably above
the liquidus for Earth-mass and super-Earth planets. Setting
`delta_T_super = 0` anchors the initial adiabat exactly on the liquidus, which
is useful when you want the coolest fully molten start rather than a hot one.

!!! note "Matching a published interior protocol"
    The `isentropic` mode sets the initial specific entropy directly through
    `ini_entropy` and `ini_dsdr`, bypassing the melting-curve lookup. Use it
    when reproducing a reference protocol that specifies the entropy IC, such
    as the [Solar System CHILI intercomparison](../Tutorials/chili_intercomparison.md).

## Choosing the surface modes

The surface-anchored and uniform modes (`adiabatic`, `isothermal`, `linear`)
remain available for controlled tests and for cases where you want to specify
the surface temperature directly through `tsurf_init`. They are not the default
because of the deep-mantle mushy-zone risk described above. Reach for them when
you have a specific reason to fix the surface state rather than the base of the
mantle.

---

**See also:** [Planet and volatiles reference](../Reference/config/planet.md) | [Configuration file](config.md) | [Running and output](usage_running.md) | [Earth analogue tutorial](../Tutorials/earth_analogue.md) | [Model description](../Explanations/model.md)
