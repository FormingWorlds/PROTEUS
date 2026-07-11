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

| Mode | Sets the profile by | Companion parameter(s) |
|------|---------------------|------------------------|
| `liquidus_super` (default) | a fully molten adiabat, superheated above the liquidus | `delta_T_super` |
| `adiabatic_from_cmb` | an adiabat anchored at a fixed CMB temperature | `tcmb_init` |
| `adiabatic` | an adiabat anchored at the surface | `tsurf_init` |
| `isothermal` | a uniform temperature | `tsurf_init` |
| `linear` | a surface-to-centre gradient | `tsurf_init`, `tcenter_init` |
| `accretion` | accretion energetics | `f_accretion`, `f_differentiation` |
| `isentropic` | the specific entropy directly | `ini_entropy`, `ini_dsdr` |

The default `liquidus_super` solves for the single adiabat that is fully molten
throughout the mantle with a controlled superheat margin (see below). The
`adiabatic_from_cmb` mode anchors an adiabat at the base of the mantle and
integrates it upward to the surface; the surface modes anchor at the surface
and integrate downward; the remaining modes set the profile from accretion
energetics (White & Li, 2025) or from the specific entropy itself.

The default needs nothing beyond the mode name, because `delta_T_super` already
defaults to 500 K:

```toml
[planet]
    temperature_mode = "liquidus_super"   # the default; shown here for clarity
    delta_T_super    = 500.0              # [K] above the liquidus at the core-mantle boundary
```

## What to favour

**Use the default `liquidus_super` for most runs.** It starts the mantle on the
coolest single adiabat that is fully molten everywhere, with at least
$\Delta T_\mathrm{super}$ (`delta_T_super`, in K) of superheat above the
silicate liquidus:

$$\min_{P}\,\bigl[\,T_\mathrm{ad}(P) - T_\mathrm{liq}(P)\,\bigr] = \Delta T_\mathrm{super}$$

where $T_\mathrm{ad}$ is the (isentropic) initial adiabat and $T_\mathrm{liq}$
is the configured silicate liquidus. PROTEUS solves for the surface temperature,
and hence the uniform initial entropy, that satisfies this at the
most-constraining depth, checking the superheat against the liquidus actually in
use. This guarantees a fully molten initial state with a known margin for any
planet mass and any melting-curve parameterisation, without you choosing a
surface temperature or entropy by hand. Because the binding depth is shallow,
the solved entropy is essentially independent of planet mass, so a mass grid
starts on a common adiabat.

The default `delta_T_super = 500` K gives a comfortably molten start across the
Earth-mass to ten-Earth-mass range. Setting `delta_T_super = 0` makes the mantle
marginally molten, just touching the liquidus at the binding depth.

!!! note "Requires the silicate liquidus"
    `liquidus_super` evaluates the Fei et al. (2021) liquidus through the
    interior structure module (Zalmoxis), which is part of the standard
    installation. For a run built only from placeholder modules, use
    `adiabatic_from_cmb` instead, which needs no melting-curve lookup.

!!! note "Very deep mantles"
    A sufficiently deep mantle cannot be made molten with an arbitrarily large
    superheat: past a point the deep adiabat would exceed the equation-of-state
    table. If the requested `delta_T_super` cannot be reached, PROTEUS raises
    with the largest achievable superheat rather than using an unphysical
    initial condition; lower `delta_T_super` (or the planet mass) in that case.

**Use `adiabatic_from_cmb` for a fixed CMB temperature.** This mode anchors the
adiabat at a user-set core-mantle-boundary temperature `tcmb_init` and
integrates it upward to the surface:

```toml
[planet]
    temperature_mode = "adiabatic_from_cmb"
    tcmb_init        = 6000.0   # [K] adiabat anchor at the core-mantle boundary
```

It needs no melting curve, so it is also the mode used by the all-dummy
quick-start configuration, which runs without any external structure solver.

**Avoid the surface-anchored modes unless you have a specific reason.** Under
the current equation of state, an adiabat pinned at the surface (`adiabatic`,
`isothermal`) can drop the deep mantle below its liquidus at `t = 0`, leaving a
partially solid base that is not a clean magma-ocean start. The default
`liquidus_super` avoids this by solving for full melt directly, and
`adiabatic_from_cmb` avoids it when you supply a hot enough `tcmb_init`. The
`linear` mode is intended for controlled tests where you set the surface and
centre temperatures directly.

!!! note "Matching a published interior protocol"
    The `isentropic` mode sets the initial specific entropy directly through
    `ini_entropy` and `ini_dsdr`, bypassing the melting-curve lookup. Use it
    when reproducing a reference protocol that specifies the entropy IC, such
    as the [Solar System CHILI intercomparison](../Tutorials/chili_intercomparison.md).

---

**See also:** [Planet and volatiles reference](../Reference/config/planet.md) | [Configuration file](config.md) | [Running and output](usage_running.md) | [Earth analogue tutorial](../Tutorials/earth_analogue.md) | [Model description](../Explanations/model.md)
