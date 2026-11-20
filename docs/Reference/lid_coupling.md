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
- `_set_lid_surface` passes $F'$ with the cap as `lid_surface_flux_slope`. The cap takes the slope after the replacement of an unusable secant, but before the cut at the grey secant; only the open part takes the cut ([Slope guard codes](#slope-guard-codes)).
- After the step, where the lid surface ends above the held one, `_correct_lid_cap` takes the secant of the atmosphere flux between the held and the end surface, $(F_\mathrm{atm}(T_\mathrm{end}) - F_\mathrm{atm}) / (T_\mathrm{end} - T_\mathrm{surf})$, at least 0. A secant of 0 (the flux at the end surface not above the held flux, or not finite) keeps the first solve, with its slope and guard code, and the log names it. Otherwise `_correct_lid_cap` runs the step again from its start, once, with that secant as $F'$ for the cap and for the open part. A secant within $10^{-3}$ of $F'$ changes nothing, so that re-solve is skipped. The log names each re-solve with its wall time and the running total. The flux at $T_\mathrm{end}$ comes from the atmosphere module: the dummy without `fixed_flux` has it as a formula; the other modules have no corrector and take the slope of the rows alone.

**When the cap acts.** With $G_\mathrm{lid} \le G_\mathrm{skin}$ the cap is close to the flux of the held lid at the start of the step, so it acts where the held lid would conduct more than at that start. The two are not equal. $C$ takes $G_\mathrm{lid}$ from the profile shape at the end surface of the last step, and the held lid has another shape at the new `T_surf`. In the first dummy row under a flux ceiling, the held lid conducted 24 % less than $C$. The held lid conducts $k f_s'(0) (T_b - T_\mathrm{surf}) / \delta$, and $f_s'(0)$ rises with the shape parameter $s$ of the profile, so the cap acts while the lid thins and while $s$ rises. Near the onset, where $G_\mathrm{lid} > G_\mathrm{skin}$, the cap lies below the flux of the held lid and can act from the start of the step.

## Slope guard codes

`lid_flux_secant` gives $F'$ from the last 2 rows. The secant is unusable for a surface step below 0.01 K (`LID_SLOPE_DT_MIN`), for a slope of 0 or below, or for one that is not finite; it is then 0. Under a live lid, `lid_fallback_slope` replaces it. `lid_open_slope` cuts the slope of the open part at the slope of a bare grey surface of emissivity 1. The column `lid_slope_guard` records the result:

| Code | When | Slope of the cap | Slope of the open part |
|---|---|---|---|
| 0 | the secant of the last 2 rows is usable and at or below the cut, or no lid is live | the secant | the secant |
| 2 | the slope lies above the cut | the slope | the cut |
| 3 | the last secant is unusable; a live lid took the newest usable secant of 2 consecutive earlier rows | that secant | that secant, at most the grey tangent |
| 4 | the last secant is unusable and the run has no usable secant; a live lid took the grey tangent $4 \sigma T_\mathrm{surf}^3$ | the grey tangent | the grey tangent |
| 5 | as code 3, with that secant above the grey tangent | that secant | the grey tangent |

Here $\sigma$ is the Stefan-Boltzmann constant. The cut is the grey secant $\sigma (T_1^4 - T_\mathrm{surf}^4)/(T_1 - T_\mathrm{surf})$ over the same 2 rows, with $T_1$ the surface of the row before. The same cut applies after a cooling and after a warming step. After a cooling step this secant lies above the tangent $4 \sigma T_\mathrm{surf}^3$, so the cut is looser than the slope of a grey surface at $T_\mathrm{surf}$ itself. A slope from an earlier pair of rows has no other end at the current row, so its cut is the grey tangent; a code 3 slope above it records code 5. In a re-solve of Rule 3 the cut is the grey secant from the held surface to the end surface of the step, and `lid_slope_guard` records the code of that secant.

The log line `Lid slope: last secant unusable, ...` names the slope taken and, for codes 3 and 5, its age in rows. Without a live lid no slope goes to Aragog. The grey tangent acts only where the run has no usable secant at all; no row of the stored coupled runs of the Validation page needs a fallback.

## The open-part diagnostic

After the atmosphere step, `run_atmosphere` writes `lid_F_open`, the start value of the open part per open area at the weight of the row:

$$F_\mathrm{open,0} = \frac{F_\mathrm{atm} - w\,L}{1 - w}, \qquad L = G_\mathrm{lid}'\,(T_b - T_\mathrm{surf}),$$

and `F_atm` where no lid acts. $L$ is the flux of the lid part in the call, with its sign: it is inward for a surface above $T_b$. Aragog does not read this column; it takes the open part per total area ([What the interior takes](../Explanations/lid_coupling.md#what-the-interior-takes)).

**The residual of the atmosphere solve.** With $S = G_\mathrm{skin}(T_m - T_\mathrm{surf})$ the skin flux of the open part and $B$ the skin balance of the call, $F_\mathrm{open,0} = S + (F_\mathrm{atm} - B)/(1 - w)$. Where the atmosphere closes the balance of its call, the start value is the skin flux. The column `lid_F_resid` records the residual $F_\mathrm{atm} - B$ per total area on every row; it does not grow near $w = 1$. In coupled 1 Earth-mass runs with AGNI it was at most 0.01 W m$^{-2}$ on the live-lid rows.

Limits of these columns:

- A call whose residual has a size above 1 W m$^{-2}$, or is not finite, logs a warning with the value and the model time; the run goes on.
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

- **JANUS.** JANUS does not read the lid. With `interior_energetics.module = "aragog"` and `lid_mush_base = true`, a config with `atmos_clim.module = "janus"` is refused at load, and so is any `atmos_clim.surf_state` other than `"skin"`. The dummy with `atmos_clim.dummy.fixed_flux > 0` is refused too: it returns its flux without the skin, so the lid does not reach the atmosphere.
- **T_magma clip.** The large-increase clip of `tmagma_atol` and `tmagma_rtol` in the interior wrapper acts on the row `T_magma`, not on the lid branch of the call. A clip in a step where a lid acts (weight above 0 at the start or the end of the step, or at an accepted state of the solver, column `lid_weight_max`) stops the run with an error that names the clipped value: a clipped lid step does not keep the energy rules. Without a lid the run goes on with the clipped value, and the column `T_magma_clipped` counts each clip.
- **prevent_warming.** `planet.prevent_warming = true` is refused with the lid at load: it raises `F_atm` to at least $10^{-8}$ W m$^{-2}$ and keeps `T_magma` from rising, so the two parts of the surface would lose a flux that the atmosphere did not solve.
- **Saturation skin.** With `surface_d = "saturation"` the skin is the stack of condensate layers, and it has no thickness while no layer has formed; a condensate layer over the lid is not modelled. A config with the lid and a saturation skin is refused at load. A live lid with a zero skin, as on a resume from a helpfile without `boundary_layer_conductivity`, stops the run with an error, because the open part then has no conductance to blend.

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

In two AGNI runs (drain step $\times 1$ and $\times 0.5$) the first lid row had $w = 3.0 \times 10^{-10}$ and $3.6 \times 10^{-10}$, with the held surface 13.7 and 15.3 K below $T_b$. That step therefore booked the full `F_atm` it started from. The first rows with $w$ above 0.5 came 284 and 267 yr later, with the surface 77 and 79 K below $T_b$.

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
| 50913.4 | 0.349 | 26585 | 23461 | 21608 | 7.0 % |
| 50943.8 | 0.957 | 23461 | 21465 | 18361 | 13.2 % |
| 50976.9 | 0.9977 | 21465 | 19686 | 17223 | 11.5 % |
| 51006.5 | 0.99977 | 19686 | 18072 | 16454 | 8.2 % |
| 51038.0 | 0.99997 | 18072 | 16624 | 15566 | 5.9 % |
| 51071.3 | 0.999993 | 16624 | 15533 | 14717 | 4.9 % |
| 51106.4 | 0.999998 | 15533 | 15197 | 14178 | 6.6 % |

The table is the run at the normal step. In every step of the table `F_int` lies below the band. Over the steps that start from 50890 to 51300 yr, the time integral of the gap is $-1.32 \times 10^{13}$ J m$^{-2}$ at the normal step (mean step 35.1 yr). The time integral of `F_atm` over the same steps, by the trapezoid rule, is $2.28 \times 10^{14}$ J m$^{-2}$, so the gap is 5.8 % of it. At half the step (mean step 17.5 yr) the two integrals are $-6.34 \times 10^{12}$ and $2.15 \times 10^{14}$ J m$^{-2}$, 3.0 %. In that run the largest relative gap of a step was 10.7 %.
