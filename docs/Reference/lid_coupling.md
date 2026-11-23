# Lid coupling details

This page holds the code paths, the guard codes, the resume fields and the run numbers of the sub-grid lid coupling. The physics and the symbols are explained on [Sub-grid lid and the surface skin](../Explanations/lid_coupling.md); read that page first.

Symbols used here:

- $w$: the lid weight, from 0 to 1;
- $T_\mathrm{surf}$: the surface temperature of the last atmosphere call, in K; $T_b$: the lid base, in K;
- $G_\mathrm{skin} = k/d$: the skin conductance, and $G_\mathrm{lid}' = \min(G_\mathrm{lid}, G_\mathrm{skin})$: the capped lid conductance, both in W m$^{-2}$ K$^{-1}$;
- $C$: the cap, in W m$^{-2}$; $F'$: the slope of `F_atm` with the surface temperature, in W m$^{-2}$ K$^{-1}$.

## Code that keeps Rule 1

Every lid branch acts only for $w > 0$. With $w = 0$:

- the atmosphere call receives the skin columns and `T_magma` unchanged (`lid_boundary` in `atmos_clim/common.py`);
- `lid_F_open` is set to `F_atm` (`lid_open_flux`);
- Aragog takes `F_atm` as its surface flux, and the 4 terms of the open part stay unset (`AragogRunner.update_solver` in `interior_energetics/aragog.py`);
- `F_int` is assigned `F_atm` when no lid acts at the start or at the end of the step (`_surface_flux_of_step`).

## Code that keeps Rule 3

- In the atmosphere call, `lid_boundary` caps $G_\mathrm{lid}$ at $G_\mathrm{skin}$, so a lid never conducts more than the open surface. The open part reads its melt at $T_m \ge T_b$, so with $T_\mathrm{surf} \le T_m$ the lid part never loses more than the open part would at the same surface temperature.
- `lid_flux_cap` computes $C$ from the row after the atmosphere step. `_set_lid_surface` passes it to Aragog as `lid_surface_flux_cap`. The held surface is the `surface_temperature` of the solver boundary, the `T_surf` of the row. Without a lid, or for a surface at or above $T_b$ ($C \le 0$), no cap is set.
- `update_solver` passes $F'$ with the cap as `lid_surface_flux_slope`. It is the slope of the open part: the slope after the replacement of an unusable secant and after the cut at the grey secant, so the cap and the open part take one slope ([Slope guard codes](#slope-guard-codes)).
- After the step, where the lid surface ends above the held one, `_correct_lid_cap` takes the secant of the atmosphere flux between the held and the end surface, $(F_\mathrm{atm}(T_\mathrm{end}) - F_\mathrm{atm}) / (T_\mathrm{end} - T_\mathrm{surf})$, at least 0. A secant of 0 (the flux at the end surface not above the held flux, or not finite) keeps the first solve, with its slope and guard code, and the log names it. Otherwise `_correct_lid_cap` cuts that secant at the grey secant from the held to the end surface and runs the step again from its start, once, with the result as $F'$ for the cap and the open part. A cut secant whose difference from $F'$ is at most $10^{-3}$ of the larger of the two changes nothing, so that re-solve is skipped. The log names each re-solve with its wall time and the running total. The flux at $T_\mathrm{end}$ comes from the atmosphere module: the dummy without `fixed_flux` has it as a formula; the other modules have no corrector and take the slope of the rows alone.

**When the cap acts.** With $G_\mathrm{lid} \le G_\mathrm{skin}$ the cap is close to the flux of the held lid at the start of the step, so it acts where the held lid would conduct more than at that start. The two are not equal. $C$ takes $G_\mathrm{lid}$ from the profile shape at the end surface of the last step, and the held lid has another shape at the new `T_surf`. In the first dummy row under a flux ceiling, the held lid conducted 24 % less than $C$. The held lid conducts $k f_s'(0) (T_b - T_\mathrm{surf}) / \delta$, and $f_s'(0)$ rises with the shape parameter $s$ of the profile, so the cap acts while the lid thins and while $s$ rises. Near the onset, where $G_\mathrm{lid} > G_\mathrm{skin}$, the cap lies below the flux of the held lid and can act from the start of the step.

## Slope guard codes

`lid_flux_secant` gives $F'$ from the last 2 rows. The secant is unusable for a surface step below 0.01 K (`LID_SLOPE_DT_MIN`), for a slope of 0 or below, or for one that is not finite; it is then 0. Under a live lid, `lid_fallback_slope` replaces it. `lid_open_slope` then cuts the slope at the slope of a bare grey surface of emissivity 1, and the cap and the open part take the result. The column `lid_slope_guard` records the code on the row where the step ends, not on the row it starts from. The first step of a new run starts without a slope, so the row where it ends has code 0:

| Code | When | Slope of the cap and the open part |
|---|---|---|
| 0 | the secant of the last 2 rows is usable and at or below the cut, or no lid is live | the secant |
| 2 | the slope lies above the cut | the cut |
| 3 | the last secant is unusable; a live lid took the newest usable secant of 2 consecutive earlier rows | that secant, at most the grey tangent |
| 4 | the last secant is unusable and the run has no usable secant; a live lid took the grey tangent $4 \sigma T_\mathrm{surf}^3$ | the grey tangent |
| 5 | as code 3, with that secant above the grey tangent | the grey tangent |

Here $\sigma$ is the Stefan-Boltzmann constant. The cut is the grey secant $\sigma (T_1^4 - T_\mathrm{surf}^4)/(T_1 - T_\mathrm{surf})$ over the same 2 rows, with $T_1$ the surface of the row before. The same cut applies after a cooling and after a warming step. After a cooling step this secant lies above the tangent $4 \sigma T_\mathrm{surf}^3$, so the cut is looser than the slope of a grey surface at $T_\mathrm{surf}$ itself. A slope from an earlier pair of rows has no other end at the current row, so its cut is the grey tangent; a code 3 slope above it records code 5. In a re-solve of Rule 3 the cut is the grey secant from the held surface to the end surface of the step, and `lid_slope_guard` records the code of that secant.

The log line `Lid slope: last secant unusable, ...` names the slope taken and, for codes 3 and 5, its age in rows. Without a live lid no slope goes to Aragog. The grey tangent acts only where the run has no usable secant at all; no row of the stored coupled runs of the Validation page needs a fallback.

## The open-part diagnostic

After the atmosphere step, `run_atmosphere` writes `lid_F_open`, the start value of the open part per open area at the weight of the row:

$$F_\mathrm{open,0} = \frac{F_\mathrm{atm} - w\,L}{1 - w}, \qquad L = G_\mathrm{lid}'\,(T_b - T_\mathrm{surf}),$$

and `F_atm` where no lid acts. $L$ is the flux of the lid part in the call, with its sign: it is inward for a surface above $T_b$. Aragog does not read this column; it takes the open part per total area ([What the interior takes](../Explanations/lid_coupling.md#what-the-interior-takes)).

**The residual of the atmosphere solve.** With $S = G_\mathrm{skin}(T_m - T_\mathrm{surf})$ the skin flux of the open part and $B$ the skin balance of the call, $F_\mathrm{open,0} = S + (F_\mathrm{atm} - B)/(1 - w)$. Where the atmosphere closes the balance of its call, the start value is the skin flux. The column `lid_F_resid` records the residual $F_\mathrm{atm} - B$ per total area on every row; it does not grow near $w = 1$. In coupled 1 Earth-mass runs with AGNI it was at most 0.01 W m$^{-2}$ on the live-lid rows.

Limits of these columns:

- A call whose residual has a size above 1 W m$^{-2}$, or is not finite, logs a warning with the value and the model time; the run goes on. The bound is absolute: a bound relative to `F_atm` would warn on every step where `F_atm` crosses 0.
- Where $1 - w$ is $10^{-9}$ or less, $F_\mathrm{open,0}$ is `F_atm`. $F_\mathrm{open,0}$ tends to `F_atm` as $w$ goes to 0.
- Under a live lid, an `F_atm` or $T_\mathrm{surf}$ that is not finite stops the run with status 22. So does a lid weight that is not finite, a lid conductance that is not finite under a weight above 0, or a node or lid base temperature that is not finite. The check of the lid columns runs before the atmosphere call.

## Resume of a run with the lid

Each interior snapshot (`data/<time>_int.nc`) of a run with the lid stores:

| Snapshot field | Unit | Content |
|---|---|---|
| `lid_delta_state` | m | lid thickness $\delta$, 0 after a hand-over to the mesh |
| `lid_E_state` | J m$^{-2}$ | lid energy $E_\mathrm{lid}$ |
| `lid_handover_time_state` | yr | time of the hand-over, only when one happened |

A run with the lid off writes none of them. A non-finite value is not written, with a warning.

On `proteus start -r`, PROTEUS restores the lid of the snapshot first and then its entropy profile (`restore_lid_start`, then `set_initial_entropy`, in `interior_energetics/aragog.py`). A new solver without a set lid would otherwise start the phantom lid, which refuses a top cell below the full floor gap or in the heat-jump band. If the snapshot entropy does not fit the cells of the resumed mesh, the resume stops with an error when a structure update was pending at the stop. Otherwise PROTEUS logs an error, restarts from the initial entropy profile, and the lid starts as the phantom lid. When the entropy fits:

- A solver without the lid takes nothing.
- If the lid thickness or the lid energy is absent or not finite, PROTEUS logs a warning that names the status of both. The lid then restarts as the phantom lid of Aragog at `lid_delta_lo`; by the Aragog default that is half of `lid_delta_on`, 0.01 m for the default skin.
- Otherwise PROTEUS calls the Aragog `set_initial_lid` with the stored thickness, energy and hand-over time. If Aragog refuses the stored lid, the resume stops with an error that names the snapshot time.

A helpfile written without the lid columns resumes with them read as zero, which means no lid.

## Settings the lid refuses

- **Interior settings.** `interior_energetics.aragog.lid_mush_base = true` is refused at load without `interior_energetics.module = "aragog"`, without `interior_energetics.aragog.rheology.enabled = true`, with an `interior_energetics.surface_bc_mode` other than `"flux"`, with `interior_energetics.aragog.core_bc = "gradient"` or `"bower2018"`, with `interior_energetics.param_utbl = true`, with an `interior_energetics.kappah_floor` that is not above 0, and with `interior_energetics.const_properties = true`.
- **JANUS.** JANUS does not read the lid. With `interior_energetics.module = "aragog"` and `interior_energetics.aragog.lid_mush_base = true`, a config with `atmos_clim.module = "janus"` is refused at load, and so is any `atmos_clim.surf_state` other than `"skin"`. The dummy with `atmos_clim.dummy.fixed_flux > 0` is refused too: it returns its flux without the skin, so the lid does not reach the atmosphere.
- **T_magma clip.** The large-increase clip of `tmagma_atol` and `tmagma_rtol` in the interior wrapper acts on the row `T_magma`, not on the lid branch of the call. A clip in a step where a lid acts (weight above 0 at the start or the end of the step, or at an accepted state of the solver, column `lid_weight_max`) stops the run with an error that names the clipped value: a clipped lid step does not keep the energy rules. Without a lid the run goes on with the clipped value, and the column `T_magma_clipped` is 1 on each row whose step was clipped, else 0.
- **prevent_warming.** `planet.prevent_warming = true` is refused with the lid at load: it raises `F_atm` to at least $10^{-8}$ W m$^{-2}$ and keeps `T_magma` from rising, so the two parts of the surface would lose a flux that the atmosphere did not solve.
- **Saturation skin.** With `atmos_clim.surface_d = "saturation"` the skin is the stack of condensate layers, and it has no thickness while no layer has formed; a condensate layer over the lid is not modelled. A config with the lid and a saturation skin is refused at load. A live lid with a zero skin, as on a resume from a helpfile without `boundary_layer_conductivity`, stops the run with an error, because the open part then has no conductance to blend.

## The lid conductance at the onset

The lid enters at the phantom thickness $\delta_\mathrm{lo} = \delta_\mathrm{on}/2$, so its conductance at the onset is $2 f'(0)\,G_\mathrm{skin}$. For the profile family of the Aragog lid, $f'(0) = 2\sqrt{s}/(\sqrt{\pi}\,\mathrm{erf}\sqrt{s})$ for a growing lid of shape $s > 0$: 1.03 at $s = 0.1$ and 1.34 at $s = 1$; a thinning lid ($s < 0$) has $f'(0) < 1$.

## Setting the onset thickness

`interior_energetics.aragog.lid_delta_on` \[m\]:

- Default none: PROTEUS takes the skin value $k_\mathrm{lid}\,d/k$.
- A finite number above 0: used as given. PROTEUS logs a warning when it differs from the skin value by more than 10 %, for the defaults outside 0.018 to 0.022 m.
- A bool, a string, NaN, 0 or a negative value is refused at config load.

At setup PROTEUS logs the value it passes, `Aragog sub-grid lid onset thickness lid_delta_on = ... m`. The helpfile column `lid_delta_on` holds the value of the solver, 0 without the lid. Aragog refuses an onset thickness at or above 0.7 of the top-cell width, where the lid with the mush base hands over.

## Run numbers

These numbers come from coupled runs of 1 Earth-mass planets. They show how the rules act in practice; they are not test bounds.

### The first lid rows

In two AGNI runs, one at the coupling step that the top-cell drain cap sets ([coupling loop](../Explanations/coupling_loop.md)) and one at half of it, the first lid row had $w = 3.0 \times 10^{-10}$ and $3.6 \times 10^{-10}$, with the held surface 13.7 and 15.3 K below $T_b$. That step therefore booked the full `F_atm` it started from. The first rows with $w$ above 0.5 came 284 and 267 yr later, with the surface 77 and 79 K below $T_b$.

The [Validation page](../Validation/atmos_clim/common.md) stores 2 segments of coupled runs through the lid onset, each run with and without the in-step response of the open part. In the step that switches the lid on, the lid weight rose from 0 to 0.97 in the first segment and to 0.70 in the second; in that step the weight of the second segment reached 0.79 at the largest (column `lid_weight_max`). In the next step it fell back to 0 in all 4 runs; in the first segment with the response the AGNI solve of that row did not converge, so PROTEUS kept the atmosphere levels of the last committed row. The lid first acts there through the floor weight $g$: at the start of that step the held surface lay 13.2 and 31.8 K above $T_b$, so $w_\mathrm{s}$ was 0, and at its end the top node lay 5.8 and 10.6 K below $T_b$. In the first segment with the response, the weight rose again 2 rows later, to 0.89, with the top node 5.4 K below $T_b$; in the run of that segment without the response it stayed 0 on that row.

### Several balances at the onset

The atmosphere call uses the lid weight of the last interior row, so each call has one surface temperature. One row therefore maps the held surface to the next one: the interior step sets $w$ from the held surface, and the atmosphere balance at that $w$ sets the next surface. Where the weight changes fast with the surface, this map can have 3 fixed points. Two are stable: a nearly full lid with a cold surface, and a small weight with a hot surface. The third, between them, is unstable. On such a state the rows move monotonically to the stable branch on the side of the previous row and do not alternate between branches. A jump is still possible where one branch ends (a fold): the rows then move to the other branch, in one direction, over several rows.

On the state of the Aragog test `test_rows_follow_the_branch_of_the_previous_row` (in the Aragog repository; grey atmosphere, top cells 67 J kg$^{-1}$ K$^{-1}$ above the solidus) the stable branches have $w$ = 0.967 and 0.100 and the unstable one $w$ = 0.71. Whether a coupled run meets a state with 3 fixed points is not known. In a coupled dummy run through the onset, the row-to-row changes of $w$ and $T_\mathrm{surf}$ kept their sign; in two AGNI runs with the same weight, those of $w$ and `F_atm` did.

### The guard and the corrector in dummy runs

After a kick of $-5$ K on the held surface, the row slope (87.4 W m$^{-2}$ K$^{-1}$) lies within 2.2 % of the exact slope of the dummy flux (85.5), and the re-solve moves the end surface by 0.001 K.

### The lid against the hand-over

With the mush base the lid hands over at 0.7 of the top-cell width. The lid stays near 0.18 m in the dummy runs and below 0.06 m in the two AGNI onset runs.

### The slope with AGNI

A secant from AGNI needs a second solve at a fixed surface temperature. That solve differs from the flux of the row by up to about $2\times10^{-3}$ at `solution_rtol` 0.01 and $2.5\times10^{-3}$ at 0.15. In one AGNI onset passage at `solution_rtol` 0.01, with steps of 12 to 25 K, a secant from the row flux at the held end and a second solve at the end surface was off by 2.5 to 3.6 %. That is as much as the correction itself (0.3 to 3.4 %).

### Lid onset within a step

Two runs, not stored in the repository, start from one state at 50828.99 yr: one at the normal coupling step and one at half that step. The band of a coupling step is the range between `F_atm` at its start and at its end. A step lies outside the band where its `F_int` is below the smaller or above the larger of the two. The relative gap is that distance divided by the `F_atm` at the start of the step.

| Start of step (yr) | $w$ at start | `F_atm` start (W m$^{-2}$) | `F_atm` end (W m$^{-2}$) | `F_int` (W m$^{-2}$) | Relative gap |
|---|---|---|---|---|---|
| 50913.4 | 0.349 | 26585 | 23457 | 21608 | 7.0 % |
| 50943.8 | 0.957 | 23457 | 21519 | 18161 | 14.3 % |
| 50976.9 | 0.9980 | 21519 | 19722 | 17255 | 11.5 % |
| 51006.7 | 0.99981 | 19722 | 18101 | 16490 | 8.2 % |
| 51038.4 | 0.99997 | 18101 | 16646 | 15602 | 5.8 % |
| 51071.9 | 0.999995 | 16646 | 15424 | 14745 | 4.1 % |
| 51107.3 | 0.999998 | 15424 | 14557 | 14007 | 3.6 % |

The table is the run at the normal step. In every step of the table `F_int` lies below the band. The step from 50943.8 yr also takes the compression work of a structure update of the interior ($-7.74 \times 10^{29}$ J); the other 6 steps lie 3.6 to 11.5 % below the band. Over the steps that start from 50890 to 51300 yr, `F_int` lies below the band in 10 steps: the 7 steps of the table, the step from 51144.4 yr (6.3 %), which takes the compression work of a second structure update ($-1.73 \times 10^{29}$ J), and the 2 steps after it (0.8 and 0.1 %). The gap integral of a run is the sum, over the steps that start from 50890 to 51300 yr, of the signed distance of `F_int` to the band times the length of the step; a step above the band counts positive. In the run of the table, the gap integral is $-1.30 \times 10^{13}$ J m$^{-2}$ at the normal step (mean step 35.6 yr). The time integral of `F_atm` over the same steps, by the trapezoid rule, is $2.11 \times 10^{14}$ J m$^{-2}$, so the gap is 6.2 % of it. At half the step (mean step 17.8 yr) the two integrals are $-6.38 \times 10^{12}$ and $2.15 \times 10^{14}$ J m$^{-2}$, so the gap is 3.0 % of the `F_atm` integral; 2 of its steps lie above the band and add $+7.5 \times 10^{10}$ J m$^{-2}$, and none of the run of the table does. In the half-step run the largest relative gap of a step was 11.1 %, against 14.3 % in the run of the table.

**The start gap.** At the start of a step the surface loses $F_\mathrm{top} = F_\mathrm{atm} + w\,(F_{s,0} - L)$ per total area. Here $F_\mathrm{top}$ is the heat flux that Aragog takes out at its top node, and $F_{s,0}$ is the loss of the held lid at the start state. A positive start gap means that the interior loses more than `F_atm` at the start of the step. $F_{s,0}$ differs from $L$ for 4 reasons:

- Aragog holds the lid surface at least 1 K below $T_b$.
- $L$ takes the smaller of $G_\mathrm{lid}$ and $G_\mathrm{skin}$.
- Aragog solves the lid profile again at the held surface.
- Above the cap, the lid surface carries $F_s$ of [Rule 3](../Explanations/lid_coupling.md#rule-3-the-atmosphere-flux-is-the-largest-loss).

The coupling does not correct this gap, and no helpfile column records it: `lid_F_diff` is `F_int` of the step less `F_atm`, a different quantity. The numbers below come from the run of the table, a 1 Earth-mass run with AGNI and the Aragog lid. For each step the start gap is the top flux that Aragog takes at the first evaluation of the step, $F_\mathrm{atm} + w\,(F_{s,0} - L)$ with $w$, $F_{s,0}$ and $L$ of the start state, less `F_atm`. The Aragog tests of the start gap (`tests/test_lid_open_flux_instep.py`) assert $F_\mathrm{top} - F_\mathrm{atm} = w\,(F_{s,0} - L)$ at the start state in 2 cases:

- a hand-set $L$ of either sign, with the held surface far below $T_b$;
- the clamp branch of the stored live-lid rows: the held surface above $T_b$ (9.6 K in the test, 0.14 and 0.005 K on the stored rows), so Aragog holds the lid surface 1 K below $T_b$, and $L$ below 0.

A further test checks, in both cases and at 2 values of $L$, that the top-cell rate of the JAX right-hand side equals the NumPy rate (relative $10^{-9}$), so the JAX path carries the same start gap.

In the run of the table, take the start gap of each step times the length of that step. This sum holds the start value over the whole step: it gives the size of the term, not the energy that the step lost by it. Over the steps that start from 50890 to 51300 yr, this sums to $+9.63 \times 10^{12}$ J m$^{-2}$: 4.6 % of the time integral of `F_atm` over those steps. Over all 24 steps from 50828.99 to 51700.0 yr it sums to $+9.66 \times 10^{12}$ J m$^{-2}$: 2.0 % of the time integral of `F_atm` over those 24 steps by the trapezoid rule, $4.74 \times 10^{14}$ J m$^{-2}$. The start gap is positive, while the gap of `F_int` to the band, defined at the top of this section, is negative over the steps from 50890 to 51300 yr ($-1.30 \times 10^{13}$ J m$^{-2}$).
