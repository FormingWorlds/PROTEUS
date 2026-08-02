# The coupling loop

The [previous page](code_architecture.md) described the static layout of
PROTEUS's modules. This page explains how they run: the fixed execution order
within each timestep, how modules exchange state through `hf_row`, and how the
simulation advances and terminates. For the broader scientific context of each
module, see [Model description](model.md).

## The helpfile row: data bus between modules

All inter-module communication passes through a single Python dictionary called
`hf_row`. Each module reads the quantities it needs from `hf_row`, runs its
solver, and writes its results back. The main loop then appends the completed
row to the helpfile DataFrame (`hf_all`), which is periodically saved to
`runtime_helpfile.csv`.

This design means modules are loosely coupled: they do not call each other
directly. The orchestrator (`proteus.py`) controls the execution order, and
`hf_row` carries the state.

Key quantities in `hf_row` include:

| Category | Examples | Units |
|----------|----------|-------|
| Time | `Time`, `age_star` | yr |
| Structure | `R_int`, `M_int`, `M_core`, `R_core` | m, kg |
| Thermal | `T_surf`, `T_magma`, `T_cmb` | K |
| Energy fluxes | `F_int`, `F_atm`, `F_net`, `F_ins`, `F_xuv` | W m$^{-2}$ |
| Composition | `H2O_bar`, `CO2_vmr`, `H_kg_total` | bar, 1, kg |
| Orbit | `semimajorax`, `eccentricity` | m, 1 |
| Escape | `esc_rate_total`, `esc_rate_H` | kg s$^{-1}$ |

The full column reference is in the [Output format](../Reference/output.md) page.

## Execution order per iteration

Within each iteration, modules execute in a fixed order. This order matters
for coupling stability: each module sees the most recent output from all
upstream modules.

1. **Interior energetics** (`run_interior`): Evolves the mantle temperature,
   melt fraction, and heat flux using the chosen solver (Aragog, SPIDER,
   boundary, or dummy). Advances simulation time by the interior timestep.

2. **Structure update** (`update_structure_from_interior`): If Zalmoxis is
   active and a structure update is triggered (by elapsed time, melt fraction
   change, or temperature change exceeding configured thresholds), recomputes
   the hydrostatic density profile and planet radius.

3. **Orbit and tides** (`run_orbit`): Updates orbital elements (semi-major
   axis, eccentricity) and computes tidal heating rates. Tidal power is
   distributed radially and passed to the interior module for the next
   iteration.

4. **Stellar evolution** (`update_stellar_quantities`): Interpolates the
   stellar mass, radius, effective temperature, and luminosity from
   pre-computed evolutionary tracks at the current stellar age. Recomputes
   the instellation flux and XUV flux. The stellar spectrum is updated on a
   separate, longer cadence controlled by `params.dt.starspec`.

5. **Atmospheric escape** (`run_escape`): Computes mass loss rates for each
   element (H, C, N, S, O) based on the XUV flux, planet mass, and current
   atmospheric composition. Updates element inventories by debiting the
   escaped mass. Only active after the initialisation stage.

6. **Outgassing** (`run_outgassing`): Given the updated element inventories,
   mantle temperature, and melt fraction, computes the thermodynamic
   equilibrium partitioning of volatiles between atmosphere, melt, and solid.
   Writes partial pressures, mixing ratios, and atmospheric mass to `hf_row`.
   Also calls `update_planet_mass` and `assert_mass_conservation` to verify
   the whole-planet mass budget.

7. **Atmosphere climate** (`run_atmosphere`): Solves the radiative-convective
   structure of the atmosphere using the chosen backend (AGNI, JANUS, or
   dummy). Takes the interior heat flux and atmospheric composition as input;
   returns the surface temperature, outgoing longwave radiation, and Bond
   albedo.

8. **Atmospheric chemistry** (`run_chemistry`): If configured for online mode,
   runs photochemical kinetics (VULCAN) to compute steady-state mixing ratios.
   Most configurations skip this step or run it offline after the simulation.

9. **Housekeeping**: Updates iteration counters, checks convergence criteria,
   writes the helpfile row to `hf_all`, generates plots and archives if
   scheduled.

## Initialisation stage

The first three iterations (iterations 0, 1, 2) are the **initialisation
stage**. During this stage:

- **Time is held at zero.** The simulation clock does not advance. This
  allows modules to exchange boundary conditions and reach a mutually
  consistent state before dynamic evolution begins.

- **Element inventories are recalculated** each iteration based on the
  evolving melt fraction, so the volatile partitioning adjusts to the
  initial structure.

- **Escape is disabled.** No atmospheric mass loss occurs during
  initialisation.

- **The interior solver runs in IC mode**, setting up the initial entropy
  profile and thermal state rather than time-stepping forward.

After iteration 2, the simulation enters the **science stage**: time advances
normally, escape becomes active, and the deadlock detector is armed.

## Time-stepping

PROTEUS uses an adaptive time-stepping scheme controlled by `params.dt.method`:

- **`adaptive`** (default): The timestep grows or shrinks based on how much
  key quantities (temperature, melt fraction, surface pressure) changed in the
  previous step. If changes exceed the tolerance (`params.dt.atol`,
  `params.dt.rtol`), the step shrinks by `scale_decr`; if changes are small,
  the step grows by `scale_incr`. A lookback window (`params.dt.window`)
  smooths the adaptation.

- **`proportional`**: The timestep is proportional to the current simulation
  time: $\Delta t = t / C$ where $C$ is `params.dt.propconst`.

- **`maximum`**: The timestep is always `params.dt.maximum`. Useful for
  steady-state runs.

All methods enforce `params.dt.minimum` and `params.dt.maximum` bounds. During
the mushy zone (melt fraction between `phi_crit` and `mushy_upper`), the
timestep is additionally capped at `params.dt.mushy_maximum` to resolve the
rapid solidification transition.

## Convergence and termination

The simulation terminates when one or more criteria are satisfied for two
consecutive iterations (if `params.stop.strict = true`) or one iteration
(if `false`). Available criteria:

| Criterion | Config section | Condition |
|-----------|---------------|-----------|
| Maximum iterations | `params.stop.iters` | Loop count exceeds `maximum` |
| Maximum time | `params.stop.time` | Simulation time exceeds `maximum` |
| Solidification | `params.stop.solid` | Global melt fraction below `phi_crit` |
| Radiative equilibrium | `params.stop.radeqm` | $\|F_\mathrm{int} - F_\mathrm{atm}\|$ within tolerance |
| Atmosphere loss | `params.stop.escape` | Surface pressure below `p_stop` |
| Disintegration | `params.stop.disint` | Planet inside Roche limit or spinning beyond breakup |

## Mass conservation

PROTEUS enforces whole-planet mass conservation as a runtime invariant. After
each outgassing call, `assert_mass_conservation` verifies:

1. $M_\mathrm{atm} \leq M_\mathrm{planet}$ (atmospheric mass cannot exceed
   total planet mass)
2. $\sum_s m_{s,\mathrm{atm}} = M_\mathrm{vol,atm}$ within a relative tolerance
   of $10^{-6}$, summed over the volatile and noble species (excludes rock vapours)

A violation raises a `RuntimeError` and halts the simulation. This invariant
was introduced as part of the whole-planet oxygen accounting framework to
prevent the mass budget from silently diverging.

With `outgas.vapourise = true`, rock vapour enters the atmosphere without being taken from the interior, so the first mass conservation check is not applied. See [Model description](model.md#whole-planet-mass-is-not-conserved-when-vapourisation-is-enabled).

## Deadlock detection

When using the AGNI atmosphere module, the Newton solver can occasionally fail
to converge while the interior state is effectively frozen (bit-exact
`T_magma` and `Phi_global` between iterations). PROTEUS detects this deadlock
by tracking consecutive iterations where:

- The atmosphere solver did not converge
- The interior state has not changed (within machine precision)
- The atmospheric flux is unchanged (relative tolerance $< 10^{-6}$)

After three consecutive deadlocked iterations, PROTEUS aborts with a
diagnostic message identifying the stuck state.

## Levels from a rejected atmosphere solve

A solver that rejects its solution still returns an atmospheric structure, and
the photospheric and XUV levels read off that structure can sit far outside the
planet. Escape reads $R_\mathrm{xuv}$ from the previous iteration, and the
energy-limited rate goes as $R_\mathrm{xuv}^3$, so an unusable structure would
otherwise become a large mass-loss rate that looks like a physical result.

When the atmosphere solve does not converge, PROTEUS therefore substitutes the
radius, pressure, temperature and gravity of both levels ($R_\mathrm{obs}$,
$p_\mathrm{obs}$, $T_\mathrm{obs}$, $g_\mathrm{obs}$, $R_\mathrm{xuv}$,
$p_\mathrm{xuv}$, $T_\mathrm{xuv}$, $g_\mathrm{xuv}$) together with the volume
mixing ratios at the XUV level, using the values from the most recent converged
solve. A warning reports each such iteration, and the substituted values are
listed at debug level. The levels are carried as a group, so a held radius is
never combined with the pressure, temperature, gravity or composition of a
rejected structure; the quantities derived from the radius, such as
$\rho_\mathrm{obs}$ and the transit depth, are computed from the carried value.
Fluxes and surface state are never carried: the coupling advances on them, and
the deadlock detector above needs to see them stop moving.

The record of converged levels is not written to the output files, so a resumed
run starts without one. If the first solve of such a run is rejected, it falls
back on the last committed row instead, which was written before the run began
and is the state escape would have used in any case; the warning names which of
the two sources it used. Later solves never reach back to the committed rows,
because once a run has begun writing rows, an empty record means those rows carry
rejected levels themselves. A run with nothing to fall back on keeps the levels of
the rejected structure and warns that it is doing so. Modules without a nonlinear
solve (JANUS, the dummy module, and AGNI's transparent and prescribed-temperature
branches) always report convergence and are unaffected.

Each warning counts the consecutive iterations whose levels the run did not
resolve itself, the ones with nothing to fall back on included, and past ten in a
row the run reports it at error level. A streak that long means the interior is
evolving while the levels stand still, which the deadlock detector above cannot
catch: it fires only when the interior has stopped moving too.

Two helpfile columns persist the outcome, so a carried row is identifiable from
the output alone: `atm_converged` records the solve outcome of each row (+1
converged, -1 rejected, 0 before the first atmosphere call), and
`atm_levels_stale` records the number of consecutive iterations without a
converged solve of the run, zero on every converged row.

## The XUV level is limited to the Hill radius

Independently of convergence, the XUV level itself is bounded: gas beyond the
Hill radius is not bound to the planet, so an XUV radius outside it would size
the escape cross-section with material the planet does not hold, and the
energy-limited rate grows as the cube of the excess. With `escape.hill_clamp`
enabled (the default), each atmosphere module limits $R_\mathrm{xuv}$ to
`escape.hill_clamp_frac` of the Hill radius, floored at $R_\mathrm{int}$ since
the solid body is always bound. The level moves as a whole: $p_\mathrm{xuv}$,
$T_\mathrm{xuv}$, $g_\mathrm{xuv}$ and the XUV-level mixing ratios are read at
the clipped radius, so escape never sees a radius from one level combined with
a composition from another. Each engagement is logged as a warning. The
observed level is not clipped; a transit radius beyond the Hill radius is
reported as computed and flagged by the orbit module's existing warning.

## Energy conservation diagnostics

When using the Aragog interior module, PROTEUS tracks cumulative energy
conservation using a frozen-mass framing: the total thermal energy change
of the mantle is compared against the sum of all flux integrals (interior
heat flux, core-mantle boundary flux, radiogenic heating, tidal heating)
accumulated over each solver call.

The diagnostic columns `E_residual_cons_J` and `E_residual_cons_frac` in
the helpfile quantify the residual. Typical values are below 5% of the
total cooling over multi-Myr runs. These columns are written on every run.
For a finer per-component flux decomposition in the Aragog NetCDF output,
set `write_flux_diagnostics = true` (disabled by default).