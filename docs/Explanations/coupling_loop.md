# The coupling loop

PROTEUS evolves a planet by coupling multiple physics modules in a sequential
loop. Each iteration advances the simulation by one timestep, with modules
exchanging boundary conditions through a shared data structure. This page
explains how that loop works and why it is structured the way it is.

For the high-level module overview, see [Model description](model.md). For the
code layout, see [Code architecture](code_architecture.md).

## Architecture diagram

The diagram below shows the full PROTEUS coupling architecture. Each box
represents a physics module; arrows show the data flow between them within
a single iteration.

<object type="image/svg+xml" data="../assets/proteus_architecture.svg" class="arch-diagram arch-diagram--light"></object>
<object type="image/svg+xml" data="../assets/proteus_architecture_darkmode.svg" class="arch-diagram arch-diagram--dark"></object>

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
2. $\sum_s m_{s,\mathrm{atm}} = M_\mathrm{atm}$ within a relative tolerance
   of $10^{-6}$ (species masses sum to the total atmospheric mass)

A violation raises a `RuntimeError` and halts the simulation. This invariant
was introduced as part of the whole-planet oxygen accounting framework to
prevent the mass budget from silently diverging.

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
