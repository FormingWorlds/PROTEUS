# Core evolution

By default Aragog treats the core as a heat reservoir of fixed heat capacity
below the mantle (`core_bc = "energy_balance"`). The `core_module` boundary
condition replaces that reservoir with an energy budget of the core: secular
cooling along the core adiabat, the latent heat of a growing inner core, the
gravitational energy of the light elements it rejects, and radiogenic heat.
This page explains how to switch it on, what it needs, and which output
columns to read.

The parameters are listed with their types and defaults in the
[interior reference](../Reference/config/interior.md#core-evolution-interior_energeticsaragogcore_module),
and the output columns in the [output reference](../Reference/output.md). The
model, its closure choices and its verification are part of the Aragog
documentation:
[core boundary condition modes](https://proteus-framework.org/aragog/Explanations/core_bc.html)
and
[core module verification](https://proteus-framework.org/aragog/Explanations/core_verification.html).

## How to switch it on

```toml
[interior_struct]
module = "zalmoxis"

[interior_energetics]
module = "aragog"

[interior_energetics.aragog]
core_bc = "core_module"
```

Nothing else is needed. Every key of `[interior_energetics.aragog.core_module]`
has a default, and the defaults describe a core of pure iron: the melting
curve is not depressed (`light_element_fraction = 0`) and the inner core
releases no gravitational energy (`alpha_c = 0`, `c_light = 0`). Set these
keys for a core with light elements.

A config without `core_bc` and without the `core_module` table runs the
default `energy_balance` mode.

## What it needs

- **The Zalmoxis structure.** The density profile of the core is fitted to
  the core mass and the central pressure of the structure solution, which
  only Zalmoxis provides. A config with another structure module is rejected
  when it is read.
- **No resolved stable layer with accretion or Radau.** `stratification = true`
  resolves the outer part of the core as a shell in which a stable layer can
  form. It is experimental, and a config that combines it with an accretion
  module or with `solver_method = "radau"` is rejected when it is read.

## What happens in a run

The core temperature is a state of the Aragog solver. It starts at the
temperature of the base of the mantle, so the heat flux through the
core-mantle boundary (CMB) is zero at the start and follows the temperature
contrast between the core and the mantle from then on, positive when the core
is the hotter side. The inner core starts to grow when the core adiabat meets
the melting curve.

With `stratification = true` the flux follows the contrast between the top of
the shell (`core_T_top`) and the base of the mantle, and it is not zero at the
start.

A resumed run reads the core temperature, the fitted core profile and, with
`stratification = true`, the shell temperatures from the interior snapshot.

With an accretion module, a giant impact that re-melts the mantle raises the
core temperature to at least the temperature of the re-melted base, and the
core profile is fitted again to the grown core at the next interior solve. The
heat of that rise is written to `step_dE_impact_core_J`.

## What to read in the output

| Column | Use |
|---|---|
| `T_cmb` | Core temperature: the state of the core that the solver evolves. |
| `T_cmb_node` | Temperature of the base of the mantle at the CMB pressure; `T_cmb - T_cmb_node` is the contrast that sets `F_cmb`, and `core_T_top - T_cmb_node` with `stratification = true`. |
| `core_r_icb`, `core_regime` | Radius of the inner core and the way the core freezes (0 fully liquid, 1 from the centre outwards). |
| `core_dynamo_margin`, `core_B_rms` | Entropy available for a dynamo and an estimate of the field strength. |
| `E_core_residual_frac` | Closure of the core energy ledger: the summed core heat change against the summed CMB heat. |

Check `E_core_residual_frac` first. The Aragog verification page gives the
values measured in coupled runs and how they depend on
`interior_energetics.rtol`.

The `core_*` and `E_core_*` columns are zero in every other `core_bc` mode.
Zero is also a valid value in a `core_module` run (no inner core yet, no field estimate under a
subadiabatic CMB flux), so select runs by their configured `core_bc`, not by
the column values.
