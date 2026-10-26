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
| Thermal | `T_surf`, `T_magma`, `T_cmb`, `T_cmb_node` | K |
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

2. **Giant impacts** (`apply_impact`): Applies every impact whose time falls
   within the step just taken. The impactor's rock is added to the planet and
   the interior structure is re-solved at the new mass, its volatiles are
   delivered while part of the target's atmosphere is stripped, the mantle is
   re-melted by raising it to the run's temperature-mode initial condition
   (parts that are already hotter keep their state), and
   the orbit takes the impact's change in semi-major axis and eccentricity.
   How molten the re-melt leaves the mantle follows that initial condition:
   only `planet.temperature_mode = "liquidus_super"` is fully molten for any
   planet mass and melting curve. Runs only when an accretion module is
   configured and an impact is due. When `accretion.atmloss_module = "zephyrus"`,
   the eroded atmosphere fraction is computed by the chosen scaling law
   (`accretion.atmloss_law`, default `roche2026`, or `kegerreis2020`). For
   `roche2026`, the scaling law evaluates collision masses, radii, contact
   velocity, and impact angle from the impact record, and the target atmospheric
   mass fraction $f_\mathrm{atm} = m_\mathrm{atm} / M_\mathrm{planet}$ from
   the running planet state. For `kegerreis2020`, the law evaluates collision
   masses, radii, bulk densities, contact velocity, and impact angle from the
   impact record.

3. **Structure update** (`update_structure_from_interior`): If Zalmoxis is
   active and a structure update is triggered (by elapsed time, melt fraction
   change, or temperature change exceeding configured thresholds), recomputes
   the hydrostatic density profile and planet radius.

4. **Volatile trapping** (`run_trapping`): With `outgas.trap_mode = "front"`,
   buries part of the dissolved volatiles in the mantle that froze over the
   step, or returns buried mass to the melt when the mantle remelts. Runs
   before escape and outgassing, so both see the updated inventories. See
   [Volatile trapping](outgas.md#volatile-trapping-in-the-freezing-mantle).

5. **Orbit and tides** (`run_orbit`): Updates orbital elements (semi-major
   axis, eccentricity) and computes tidal heating rates. Tidal power is
   distributed radially and passed to the interior module for the next
   iteration.

6. **Stellar evolution** (`update_stellar_quantities`): Interpolates the
   stellar mass, radius, effective temperature, and luminosity from
   pre-computed evolutionary tracks at the current stellar age. Recomputes
   the instellation flux and XUV flux. The stellar spectrum is updated on a
   separate, longer cadence controlled by `params.dt.starspec`.

7. **Atmospheric escape** (`run_escape`): Computes mass loss rates for each
   element (H, C, N, S, O) based on the XUV flux, planet mass, and current
   atmospheric composition. Updates element inventories by debiting the
   escaped mass. Only active after the initialisation stage.

8. **Outgassing** (`run_outgassing_and_vapourisation`): Given the updated
   element inventories, mantle temperature, and melt fraction, computes the thermodynamic
   equilibrium partitioning of volatiles between atmosphere, melt, and solid.
   Writes partial pressures, mixing ratios, and atmospheric mass to `hf_row`.
   After this step, on each of the 3 outgassing paths (see
   [the order of the volatile steps](outgas.md#the-order-of-the-volatile-steps)),
   the loop calls `update_planet_mass` and `assert_mass_conservation` to
   check the whole-planet mass budget.

9. **Atmosphere climate** (`run_atmosphere`): Solves the radiative-convective
   structure of the atmosphere using the chosen backend (AGNI, JANUS, or
   dummy). Takes the interior heat flux and atmospheric composition as input;
   returns the surface temperature, outgoing longwave radiation, and Bond
   albedo.

10. **Housekeeping**: Updates iteration counters, checks convergence criteria,
   writes the helpfile row to `hf_all`, generates plots and archives if
   scheduled.

Atmospheric chemistry (`run_chemistry`, VULCAN) is not a step of this loop. It
post-processes atmosphere snapshots after the simulation ends, with
`atmos_chem.mode = "offline"` or with `proteus offchem`; see
[Postprocessing and chemistry](../How-to/usage_postprocessing.md#atmospheric-chemistry-with-vulcan).

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

All methods enforce `params.dt.minimum` (which holds except where atmospheric
escape, the top-cell drain cap, solver retries, mushy-zone limiting, impact
landing steps, and other caps, among them growth, evection, bolometric clamp,
and the final-time cap, shorten the step; escape and the drain cap stop at
`minimum * step_dt_floor_frac`, while other caps can go lower) and
`params.dt.maximum` bounds. During the mushy zone (melt fraction between
`phi_crit` and `mushy_upper`), the timestep is additionally capped at
`params.dt.mushy_maximum` to resolve the rapid solidification transition.
When `interior_energetics.module = 'aragog'` and
`interior_energetics.aragog.top_cell_dt_limit = true` (incompatible with
`interior_energetics.surface_bc_mode = 'grey_body'`), the timestep is further
capped at the top-cell sensible-heat drain timescale $\tau_\mathrm{drain}$
(floored at `params.dt.minimum * escape.step_dt_floor_frac`, which can sit
below `params.dt.minimum`), and the post-step `tmagma_atol`/`tmagma_rtol` clip
(maximum increase $\mathrm{atol} + \mathrm{rtol} \times T_\mathrm{magma,prev}$)
on `T_magma`, `T_surf`, and `Phi_global` is skipped on steps where
$\tau_\mathrm{drain}$ is finite and the step does not exceed
$\tau_\mathrm{drain}$ (staying active when the escape step floor or an
impact-landing snap-forward places the step above $\tau_\mathrm{drain}$, while
`planet.prevent_warming` continues to clamp `T_magma`, `T_surf`, `Phi_global`,
and `F_int` when active). A 1-point EOS temperature lookahead probes
$S_\mathrm{trial} = \max(S_\mathrm{top} - c_{p,\mathrm{eff}}\,\Delta
T_\mathrm{allow} / T_\mathrm{top},\, S_{\min} + 300)$ and replaces
$c_{p,\mathrm{eff}}$ with the chord heat capacity
$T_\mathrm{top}\,(S_\mathrm{top} - S_\mathrm{trial}) / (T_\mathrm{top} -
T_\mathrm{trial})$ whenever $T(P_\mathrm{top}, S_\mathrm{trial})$ falls more
than $\Delta T_\mathrm{allow}$ below $T_\mathrm{top}$ ($T_\mathrm{top} -
T_\mathrm{trial} > \Delta T_\mathrm{allow}$; using $T_\mathrm{mean}$ in the
chord numerator would give a chord $c_p$ smaller by the fraction
$(T_\mathrm{top} - T_\mathrm{trial}) / (2\,T_\mathrm{top})$, about $9\%$ at the
$531.5\,\mathrm{yr}$ state), which fires across a
liquidus crossing and on every step with a mushy top cell at
$\phi_\mathrm{top} > 0.6$, where the tabulated mush `heat_capacity` (a
lever-rule blend) is about $2$ to $2.7$ times $T\,\mathrm{d}S/\mathrm{d}T$
along the EOS ($1.95$ at $\phi_\mathrm{top} = 0.60$ to $2.71$ at the liquidus).
The PALEOS solid table at $P_\mathrm{top} = 0.446\,\mathrm{GPa}$ holds
$T = T_\mathrm{sol} = 1429.77\,\mathrm{K}$ on every node from
$S = 2429.71\,\mathrm{J\,kg^{-1}\,K^{-1}}$, the last node below
$S_\mathrm{sol} = 2503.84\,\mathrm{J\,kg^{-1}\,K^{-1}}$, to the end of the
table. A lookahead trial that reaches or spans this flat part measures a
temperature drop that is too small for its entropy interval, so the chord $c_p$
exceeds the heat taken per kelvin of cooling and the step can exceed
$\Delta T_\mathrm{allow}$. On constructed slice states this gives a linearized
ratio of $1.005$ at $\eta = 0.08$ ($S_\mathrm{top} =
3106.1\,\mathrm{J\,kg^{-1}\,K^{-1}}$, $\phi_\mathrm{top} = 0.605$,
$T_\mathrm{surf}/T_\mathrm{top} = 1.05$) and up to $1.077$ at $\eta = 0.15$
(same $S_\mathrm{top}$, $T_\mathrm{surf}/T_\mathrm{top} = 0.60$). Trials in the
low-entropy clamp of the table stay below $1$ up to $\eta = 0.15$. In the
coupled runs below, trials reach the flat part only near the solidus, where the
chord does not fire, and no step exceeds $\Delta T_\mathrm{allow}$.

Across coupled runs of 2 dummy-atmosphere configurations of a $1\,M_\oplus$
planet with the PALEOS $\mathrm{MgSiO}_3$ EOS and dummy star to $25\,\mathrm{kyr}$
at `top_cell_dt_rel_dT = top_cell_dt_rel_dT_solid = 0.08` and the Aragog defaults
($0.5\,\mathrm{au}$ with `surf_greyalbedo = 0.2`, and $0.7\,\mathrm{au}$ with
`surf_greyalbedo = 0.0`), the maximum per-step $|\Delta T_\mathrm{surf}| /
(\eta_\mathrm{eff} T_\mathrm{surf})$, with $T_\mathrm{surf}$ at step start, is
between $0.68$ and $0.71$ at $0.5\,\mathrm{au}$ (8 runs: 5 starts perturbed at
round-off level and resumed at $1.4\,\mathrm{yr}$, and 3 runs from $t = 0$ with
other PROTEUS and module versions) and $0.69$ at $0.7\,\mathrm{au}$ (1 run from
$t = 0$),
with $0$ steps above $1$ and $0$ late drops (steps where $T_\mathrm{magma}$
falls to the table floor near $302\,\mathrm{K}$). These runs take about
$1850$ rows; with the option off, the same configurations take $50$ rows.
With `atmos_clim.module = 'agni'` in place of the dummy atmosphere and every
other setting of the $0.5\,\mathrm{au}$ configuration unchanged, the $0.08$ run
reaches $25\,\mathrm{kyr}$ in $1357$ to $1360$ rows with a
maximum per-step ratio of $0.47$ in the step that starts at $1241\,\mathrm{yr}$,
$41$ to $62\,\mathrm{yr}$ after the top cell crosses the liquidus (between
$1179$ and $1200\,\mathrm{yr}$), with $0$ steps above $1$ and $0$ late drops
(2 runs of different code versions). With the option off, this configuration
takes about $250$ rows and has $25$ to $31$ steps above $1$, the largest between
$6$ and $10$; these counts differ between the 2 runs, and in one $T_\mathrm{surf}$ falls
from $1373\,\mathrm{K}$ to the $302\,\mathrm{K}$ table floor in one
$160\,\mathrm{yr}$ step.
In an AGNI configuration with `surf_state = 'skin'` (and different albedos,
Rayleigh scattering, observation pressure, and solver settings),
$T_\mathrm{surf}$ is a skin temperature at $0.53$ to $0.94$ of
$T_\mathrm{magma}$ in the $0.08$ run, which reaches $25\,\mathrm{kyr}$ in
$1137$ to $1138$ rows; its largest per-step drop of $T_\mathrm{magma}$ is $0.73$ of the
allowance $0.08\,T_\mathrm{surf}$, while its option-off twin drops to the table
floor in one step.

In a computed unresupplied top-cell estimate that maps the step energy
$q = c_{p,\mathrm{chord}}\,\Delta T_\mathrm{allow}$ back to entropy at a liquid
state just above the liquidus ($t = 531.5\,\mathrm{yr}$, from a separate run of the
$0.5\,\mathrm{au}$ configuration from $t = 0$ with `top_cell_dt_rel_dT_solid = 0.02`,
stopped at $600\,\mathrm{yr}$) and at the worst mushy
step of 2 of these runs, 1 per configuration ($\phi_\mathrm{top} =
0.957$, $\Phi_\mathrm{global} = 0.787$ in both; evaluated on the slice at
the $531.5\,\mathrm{yr}$ top pressure), the linearized drop-to-allowance ratio
with $\Delta S = q / T_\mathrm{top}$ is $0.786$ and $1.000$ (both run states)
at $0.08$, $0.687$ and $0.784$ at the validator maximum $0.15$,
$0.843$ and $0.891$ at $0.30$, and $1.130$ and $1.174$ at $0.50$. Under exact
constant-pressure enthalpy
$\mathrm{d}h = T\,\mathrm{d}S$ ($\int_{S_\mathrm{end}}^{S_\mathrm{top}}
T(P_\mathrm{top}, S)\,\mathrm{d}S = q$), the ratio at the same states is
$0.816$ and $1.043$ at $0.08$, $0.728$ and $0.836$ at $0.15$, and
$1.012$ and $1.257$ at $0.30$.
A grid scan over
$S_\mathrm{top} \in [S_{\min} + 300, 6000]\,\mathrm{J\,kg^{-1}\,K^{-1}}$ in steps of
$5\,\mathrm{J\,kg^{-1}\,K^{-1}}$ and $T_\mathrm{surf}/T_\mathrm{top} \in
\{0.90, 0.95, 0.99, 1.00, 1.02\}$ gives linearized maxima of $1.000$ at
$0.08$, $0.9996$ at $0.15$, $0.993$ at $0.30$, and $1.221$ at $0.50$, and
exact-enthalpy maxima of $1.106$, $1.204$, and $1.370$ at $0.08$, $0.15$, and
$0.30$. Extending the scan to $T_\mathrm{surf}/T_\mathrm{top}$ from $0.50$ to
$1.05$ in steps of $0.05$ raises the linearized maximum to $1.005$ at $0.08$
and $1.077$ at $0.15$ (the 2 plateau states above) and the exact-enthalpy
maximum to $1.110$ at $0.08$ and $1.211$ at $0.15$, both at
$T_\mathrm{surf}/T_\mathrm{top} = 1.05$ in the liquid.

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
| Unconverged atmosphere | none, always active | 150 consecutive iterations without a converged atmosphere solve |

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
evolving while the levels stand still, which the deadlock detector above does not
see, since it fires only when the interior has stopped moving too. A separate
limit covers it: after 150 consecutive iterations without a converged atmosphere
the run ends, whatever the interior is doing. The two are sized for different
things, so the frozen-interior detector still fires at three iterations when
neither side is moving, and the longer count only ends a run whose atmosphere
never comes back on its own.

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