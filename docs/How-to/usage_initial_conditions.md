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

$$\min_{P}\,\bigl\lbrack\,T_\mathrm{ad}(P) - T_\mathrm{liq}(P)\,\bigr\rbrack = \Delta T_\mathrm{super}$$

where $T_\mathrm{ad}$ is the (isentropic) initial adiabat and $T_\mathrm{liq}$
is the configured silicate liquidus. PROTEUS solves for the surface temperature,
and hence the uniform initial entropy, that satisfies this at the
most-constraining depth, checking the superheat against the liquidus actually in
use. Where the tables reach it, this gives a fully molten initial state with a
known margin, without you choosing a surface temperature or entropy by hand.
Because the binding depth is shallow, the solved entropy is essentially
independent of planet mass, so a mass grid starts on a common adiabat.

Whether the default `delta_T_super = 500` K is reachable depends on the P-S
tables, the core-mantle boundary pressure and `planet.ini_dsdr`. On
PALEOS-generated tables it is reached at 1 and 10 Earth masses. On the Wolf &
Bower (2018) tables from FWL_DATA, with `ini_dsdr = 0`, it is reached up to a
core-mantle boundary pressure of about 355 GPa. Above that the solve clamps to
a smaller superheat (443 K at 360 GPa, 85 K at 390 GPa), and above about
397 GPa the table liquidus entropy exceeds the table maximum, so no fully
molten state exists and PROTEUS raises. With the Noack & Lasbleis (2020)
pressure and radius estimates and a core mass fraction of 0.325, this means
500 K up to 2.75 Earth masses, 194 K at 3 Earth masses, and a raise above
about 3.1 Earth masses (398 GPa at 3.15 Earth masses). With the default
`ini_dsdr = -4.698e-6` J/kg/K/m the usable entropy is lower by
`|ini_dsdr|` times the mantle thickness (about 19.5 J/kg/K at 3 Earth masses):
2.75 Earth masses clamp to 417 K, 3 Earth masses to 73 K, 3.05 Earth masses
to 3 K, and 3.1 Earth masses (392 GPa) raise.

The molten check uses the uniform entropy of the initial adiabat. With
`ini_dsdr < 0` the initial profile is hotter at depth, so the check is
conservative: it can report a smaller superheat, or raise, where the deepest
node is molten. The clamp warnings also give the superheat at the deepest
node (194 K instead of 73 K at 3 Earth masses above). When the check raises
but the adiabat at the deepest-node entropy is molten, the message names the
`ini_dsdr` allowance as the cause and gives that margin. Setting
`delta_T_super = 0` makes the mantle marginally molten, just touching the
liquidus at the binding depth.

!!! note "Requires a silicate liquidus"
    For every structure module, the initial entropy is solved on the interior
    P-S tables against their own liquidus (`liquidus_P-S.dat`), the same curve
    the interior solver uses for its melt fraction. Where those tables come
    from depends on the structure module:

    - `"zalmoxis"` with a mantle EOS (`interior_struct.zalmoxis.mantle_eos`,
      default `"PALEOS:MgSiO3"`) that is a single PALEOS table name: Zalmoxis
      generates the tables from that EOS, with a liquidus derived from the
      PALEOS (Fei et al. 2021) curve, and Aragog builds its phase-property
      tables from PALEOS too. `interior_struct.melting_dir` is not used.
    - `"spider"` and `"dummy"`, or `"zalmoxis"` when no PALEOS table set is
      generated, which includes a PALEOS mixture (a `+` in `mantle_eos`): the
      tables come from FWL_DATA (SPIDER also falls back to its own lookup
      data), Aragog reads its phase-property tables from the same Wolf & Bower
      (2018) P-S set or `interior_struct.eos_dir`, `interior_struct.melting_dir`
      must be set (the run stops when it is unset, except for
      `interior_energetics.module = "spider"` with `const_properties = true`),
      and PROTEUS derives `liquidus_P-S.dat` from that P-T curve. The Wolf &
      Bower (2018) tables end at 1 TPa; with `"zalmoxis"`, Aragog logs a
      warning when the core-mantle boundary pressure is above that.

    With `"zalmoxis"`, the structure solve also anchors its temperature profile
    on a P-T adiabat that is `delta_T_super` above the P-T liquidus, while the
    initial entropy is solved on the P-S tables. The initial entropy solves
    this anchor again at the core-mantle boundary pressure of the current
    structure, before and after the structure equilibration. When it raises
    there, no molten state exists and the run stops at the initial entropy;
    the same error is raised when the anchor integration fails numerically.
    With a PALEOS mantle and `"spider"` or `"aragog"` energetics, a failed
    anchor in a structure solve or in the adiabat structure re-solve, at the
    estimated or an intermediate pressure, only logs a warning: the structure
    solve uses the last solved anchor for the same `delta_T_super` and mantle
    EOS, or `tcmb_init` otherwise, as its core-mantle boundary temperature,
    and the re-solve keeps the linear-guess structure. The initial entropy
    then decides at the core-mantle boundary pressure of that fallback
    structure. In other cases no later step solves the anchor again, so a
    failed anchor stops the run there. A resumed run that restores the
    entropy snapshot solves neither the anchor nor the initial entropy
    again. When Aragog cannot restore the snapshot (no snapshot, or a
    snapshot length that does not match the mesh), it sets a fresh initial
    entropy, which solves the anchor again at the current core-mantle
    boundary pressure and can raise. When
    the anchor clamps at the PALEOS table below `delta_T_super`, the initial
    entropy is capped at the anchor entropy, and
    a warning names the superheat the anchor reached. The generated P-S tables
    fill cells where PALEOS has no valid state, so on their own they can
    report a superheat that PALEOS does not support. For 10 Earth masses
    (core-mantle boundary at 1474 GPa), `delta_T_super = 5000` K and
    `ini_dsdr = 0`, the capped initial adiabat matches the anchor within
    0.1 K at the surface and at the core-mantle boundary, and is up to 38 K
    colder near 1.25 GPa: at these entropies the first P-S table row above
    1 bar (1.26 GPa) holds cells filled with the 1 bar values. With
    `ini_dsdr < 0` the deepest node sits at the cap and the rest of the
    profile is colder.

    The capped initial entropy is the anchor entropy less the `ini_dsdr`
    allowance, and the molten check uses that uniform value. If the P-S
    adiabat at that entropy is below the P-S liquidus somewhere, PROTEUS
    raises instead of going above the anchor entropy, where the P-S tables
    hold filled cells. When the P-S adiabat at the anchor entropy itself is
    molten, the message names the `ini_dsdr` allowance as the cause; when it
    is not, the message gives its P-S superheat and the anchor superheat
    against the P-T liquidus.

    This cap has two known limits:

    - The cap applies only when the anchor clamps. The anchor measures the
      superheat against the P-T liquidus and the initial entropy against the
      P-S table liquidus, and the two curves differ. For 10 Earth masses the
      anchor reaches a requested `delta_T_super` of up to about 1181 K, but
      at the anchor edge entropy the P-S tables give only about 1135 K of
      superheat (measured with `ini_dsdr = 0`). A request in between puts
      the initial entropy up to about 53 J/kg/K above the anchor edge (at
      1180 K the surface is 35 K and the core-mantle boundary 113 K hotter
      than the anchor at its edge).
    - For 10 Earth masses the anchor edge comes from a narrow band near 1 GPa
      (0.99 to 1.08 GPa) where the PALEOS liquid table has no valid state at
      these temperatures. Whether an adiabat is marked valid depends on
      which pressures the adiabat samples in that band, so the edge, and the
      cap with it, can shift with the core-mantle boundary pressure.

    The superheat is checked exactly at every pressure where the margin can
    change slope: the table pressure nodes, the liquidus-file nodes, and the
    pressures where the liquidus entropy crosses a table entropy node.
    Between two such pressures the bilinear table can let the margin sag by
    up to about 1 K on the PALEOS tables.

    The core-mantle boundary pressure must lie inside the table; PROTEUS raises
    if it is above the table maximum or if the table liquidus is undefined at
    any evaluated pressure. If the requested superheat exceeds what the tables
    reach, the entropy is clamped to the highest usable table entropy (the
    table maximum, lowered by the entropy a negative `ini_dsdr` adds at the
    core-mantle boundary) and a warning reports the achieved superheat. A
    positive `ini_dsdr` is rejected with `liquidus_super`. For a run built only
    from placeholder modules, use `adiabatic_from_cmb` instead, which needs no
    melting-curve lookup.

!!! note "Very deep mantles"
    A sufficiently deep mantle cannot be made molten with an arbitrarily large
    superheat: past a point the adiabat would exceed the equation-of-state
    table. If the requested `delta_T_super` cannot be reached, PROTEUS clamps
    to the largest achievable superheat and logs a warning that reports it. If
    no adiabat inside the table is fully molten at all, PROTEUS raises a
    `RuntimeError` rather than start from a partially molten mantle; lower the
    planet mass or change the equation of state in that case.

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
