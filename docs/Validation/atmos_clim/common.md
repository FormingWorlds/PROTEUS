# Validation: the lid and the open surface in the atmosphere call

## What this page checks

This page checks how the atmosphere sees a thin solid crust, the sub-grid lid of Aragog, at the surface (`src/proteus/atmos_clim/common.py`).
Under a lid the surface has two parts at one surface temperature: the part covered by the lid and the open part, where the melt conducts through the skin.
Validation here means that we run the code on a case where another source gives the right answer, and the test checks that the code gives that answer within a stated tolerance.
The right answer comes from 3 sources:

- algebra, for the first 2 cases: two conductors in parallel, and a sum of two fluxes, have closed forms;
- a numerical derivative of the surface balance, for the third case;
- stored coupled runs with the response off, on the same code version, for the fourth case.

If this part of the code were wrong, the atmosphere would solve for the wrong surface temperature, or the interior would lose more or less heat than the atmosphere takes.
Each case below says why the check exists, what it tests, how we know the right answer and why it matters; the box "Test details" at the end of each case gives the reference, the tolerance, the test id and the command.

**Terms of this page.**

- Lid weight $w$: the share of the surface under the lid, from 0 (no lid) to 1 (a full lid).
- Conductance $G$, in W m$^{-2}$ K$^{-1}$: the heat flux through a layer for each kelvin of temperature difference across it. $G_\mathrm{skin} = k / d$ is the conductance of the skin, with $k$ its conductivity and $d$ its thickness; $G_\mathrm{lid}$ is that of the lid.
- $T_b$: the temperature at the base of the lid, the solidus with `lid_mush_base`, the setting that runs the lid in PROTEUS. $T_\mathrm{node}$: the temperature of the top node of the interior mesh, under the open part. $T_m = \max(T_\mathrm{node}, T_b)$: the open node as the coupling uses it, held at least at the lid base (a model choice: melt under the open surface is not colder than the solidus). $T_\mathrm{surf}$: the surface temperature.
- $F_\mathrm{atm}$: the net flux that the atmosphere carries away. $C$: the cap, the flux that the last atmosphere balance gave the lid part.

The physics of the lid in the coupling is on the page [Sub-grid lid and the surface skin](../../Explanations/lid_coupling.md).

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/atmos_clim/test_common.py::test_the_call_flux_is_the_parallel_sum_of_the_lid_and_the_open_surface` | Analytical limit: two conductors in parallel at one surface temperature | `lid_boundary` at 110 surface temperatures, 4 weights, 3 lid conductances | 2026-11-08 |
| `tests/atmos_clim/test_common.py::test_the_weighted_parts_sum_to_f_atm` | Analytical limit: the sum of the two weighted parts | `lid_flux_cap` and `lid_open_flux` at 21 weights, 3 ratios $r$ and 2 lid conductances | 2026-11-19 |
| `tests/atmos_clim/test_lid_open_flux_instep.py::test_the_response_is_the_change_of_the_shared_surface` | Derivative of the shared surface balance (central difference) | the response of the run, `lid_open_fields` passed to Aragog's `open_conductance`, at 19 weights and 3 slopes $F'$ | 2026-11-18 |
| `tests/interior_energetics/test_lid_open_flux_coupled.py::test_the_open_part_carries_the_coupled_runs_through_the_live_lid_step` | Stored coupled runs with the response off | 2 segments of 1 Earth-mass runs through the lid onset | 2026-11-20 |

## The atmosphere call sees the lid and the open surface in parallel

**Why this check exists.** AGNI and the dummy atmosphere solve the surface balance against one skin. The code must write the two parts of the surface as one skin with the same flux at every surface temperature. A skin added in series with the lid, or a lid that conducts more than the open surface, gives the atmosphere the wrong flux, and the test sees both.

**What it tests.** The function `lid_boundary` returns the skin thickness $d_\mathrm{call}$ and the melt temperature $T_\mathrm{magma,call}$ of the call. The test computes the flux of that one skin, $\frac{k}{d_\mathrm{call}}(T_\mathrm{magma,call} - T_\mathrm{surf})$, at 110 surface temperatures from 300 to 1390 K. It uses $T_b$ = 1400 K, $T_\mathrm{node}$ = 1500 K (so $T_m = T_\mathrm{node}$), a 10 mm skin with $k$ = 2 W m$^{-1}$ K$^{-1}$ ($G_\mathrm{skin}$ = 200 W m$^{-2}$ K$^{-1}$), the weights 0, 0.3, 0.7 and 1, and a lid conductance of 1/4, 1 and 2 times $G_\mathrm{skin}$. The expected flux is the sum of the two parts,

$$F = w \min(G_\mathrm{lid}, G_\mathrm{skin}) (T_b - T_\mathrm{surf}) + (1 - w)\, G_\mathrm{skin} (T_m - T_\mathrm{surf}).$$

**How we know the right answer.** Two layers side by side, at one surface temperature, conduct in parallel: their fluxes add. The lid conductance is capped at $G_\mathrm{skin}$, so that a lid never loses more heat than the open surface would. The one skin of the call has the conductance $G = w \min(G_\mathrm{lid}, G_\mathrm{skin}) + (1 - w) G_\mathrm{skin}$ and the melt temperature $(w \min(G_\mathrm{lid}, G_\mathrm{skin}) T_b + (1 - w) G_\mathrm{skin} T_m) / G$, the mean of $T_b$ and $T_m$ weighted by the two conductances, so its flux equals $F$ at every $T_\mathrm{surf}$.

**Why it matters.** The atmosphere sets the surface temperature from this flux. The surface temperature then sets the heat the interior loses and the weight of the lid in the next step.

![Flux of the call against the surface temperature for 4 lid weights, with the skin in series as a dashed line](../../assets/validation/atmos_clim_lid_parallel_sum.svg#only-light){ width="80%" }
![Flux of the call against the surface temperature for 4 lid weights, with the skin in series as a dashed line](../../assets/validation/atmos_clim_lid_parallel_sum_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The horizontal axis is the surface temperature in K, the vertical axis the flux of the call in kW m$^{-2}$. The plot shows the lid conductance of $G_\mathrm{skin}/4$. The solid lines are the closed form $F$ for each weight $w$, and the dots are the flux of `lid_boundary`. The dots lie on the lines. The dashed line is context, computed in the test: a skin in series with the lid at $w = 1$, which conducts 20 % less.

??? info "Test details"
    - Reference: analytical limit, two conductors in parallel at one surface temperature.
    - Tolerance: relative $10^{-12}$ at each surface temperature.
    - Discrimination (computed in the test): at $w = 1$ a skin in series with the lid gives 0.8 of $F$ at $G_\mathrm{lid} = G_\mathrm{skin}/4$ and 0.5 at $G_\mathrm{lid} = G_\mathrm{skin}$. A lid without the cap at $G_\mathrm{skin}$ gives 2 times $F$ at $G_\mathrm{lid} = 2 G_\mathrm{skin}$. Each of these fails the main comparison at all 110 surface temperatures.
    - Test id: `tests/atmos_clim/test_common.py::test_the_call_flux_is_the_parallel_sum_of_the_lid_and_the_open_surface`
    - Command: `pytest tests/atmos_clim/test_common.py -k parallel_sum_of_the_lid` (add `--validation-plots` to write the plot pair)

## The two parts of the surface sum to the atmosphere flux

**Why this check exists.** After the atmosphere step the coupling splits `F_atm` into two start fluxes, the lid part and the open part $F_\mathrm{open}$ (the column `lid_F_open`), and their weighted sum must not count a flux twice or lose one. The test fails for 2 wrong forms of the open part:

- an open part equal to $F_\mathrm{atm}$ counts the lid part twice, because $F_\mathrm{atm}$ already holds it;
- an open part equal to the skin flux scaled by $r$, the ratio of $F_\mathrm{atm}$ to the skin balance of the call, loses the share $w C (r - 1)$ of the gap that falls on the lid.

**What it tests.** The test builds a surface with $T_\mathrm{surf}$ = 900 K, the temperatures and skin of the first case, and $G_\mathrm{lid}$ = 100 and 400 W m$^{-2}$ K$^{-1}$ (half of $G_\mathrm{skin}$ and twice it, where the cap at $G_\mathrm{skin}$ acts). It sets $F_\mathrm{atm}$ to $r$ times the skin balance of the call, for $r$ = 0.95, 1 and 1.06, at 21 weights from 0 to 1. The ratio $r$ says how far the atmosphere is from the skin balance; AGNI closes it only to its solver tolerance. The code gives the open part the rest of the atmosphere flux, $F_\mathrm{open} = (F_\mathrm{atm} - w L) / (1 - w)$ per open area, with $L$ the signed flux of the lid part in the call; for a surface below $T_b$, as here, $L$ is the cap $C$ ([explanation](../../Explanations/lid_coupling.md#what-the-interior-takes)). The test checks the sum

$$w C + (1 - w) F_\mathrm{open} = F_\mathrm{atm}.$$

A property test, `test_the_weighted_parts_sum_to_f_atm_on_both_sides_of_the_lid_base` in `tests/atmos_clim/test_lid_open_flux_instep.py`, checks the same sum with the signed lid flux at up to 400 random states, with `F_atm` of either sign and the surface on either side of $T_b$.

**How we know the right answer.** The sum is the flux that the atmosphere carries away, by the definition of the open part, so the case checks that `lid_open_flux` and `lid_flux_cap` use the same lid flux, the cap at $G_\mathrm{skin}$ included. The test computes $C = \min(G_\mathrm{lid}, G_\mathrm{skin}) (T_b - T_\mathrm{surf})$ and the skin balance from these formulas, not from the code. At $r = 1$ the open part is the second term of the parallel sum of the first case, the skin flux $G_\mathrm{skin}(T_m - T_\mathrm{surf})$.

**Why it matters.** The interior loses the sum of the two parts. A flux counted twice would cool the interior too fast under a growing lid, and a flux lost would let it cool too slowly.

![Sum of the two weighted parts divided by the atmosphere flux, against the lid weight, for 3 ratios r](../../assets/validation/atmos_clim_lid_open_part_sum.svg#only-light){ width="80%" }
![Sum of the two weighted parts divided by the atmosphere flux, against the lid weight, for 3 ratios r](../../assets/validation/atmos_clim_lid_open_part_sum_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The horizontal axis is the lid weight $w$, the vertical axis the sum $w C + (1 - w) F_\mathrm{open}$ divided by $F_\mathrm{atm}$, for $G_\mathrm{lid}$ = 100 W m$^{-2}$ K$^{-1}$. The dots are the sum from `lid_flux_cap` and `lid_open_flux`; they lie at 1 for every weight below 1. The dashed lines are the sum with the open part equal to the skin flux scaled by $r$, $1 - w C (r - 1) / F_\mathrm{atm}$, for comparison. At $w = 1$ the open area is 0, and both give the lid part alone, $C / F_\mathrm{atm} = 1/r$.

**Known limit.** The test checks the sum at the lid weight of the row, with the open part per open area that the column `lid_F_open` records. Aragog does not read that column: it takes the open part per total area, $F_\mathrm{atm} - w L$, at its own weight in each evaluation, with no division by $1 - w$ ([explanation](../../Explanations/lid_coupling.md#what-the-interior-takes)). The share of the atmosphere residual in `lid_F_open` grows as $1/(1 - w)$ near $w = 1$; in coupled 1 Earth-mass runs with AGNI it was at most 0.69 W m$^{-2}$ per open area, at $w$ = 0.99951. The column `lid_F_resid` records the residual itself, $F_\mathrm{atm} - B$ per total area. Where the open area $1 - w$ is $10^{-9}$ or less, `lid_F_open` is $F_\mathrm{atm}$ and the sum is $w C + (1 - w) F_\mathrm{atm}$, close to the lid part alone.

??? info "Test details"
    - Reference: analytical limit, the sum of the two weighted parts from the definitions of the cap and the open part.
    - Tolerance: relative $10^{-12}$ at each weight below 1, and $1/r$ to $10^{-12}$ at $w = 1$.
    - Discrimination (computed in the test): at $w = 0.5$ and $G_\mathrm{lid} = G_\mathrm{skin}/2$, an open part equal to $F_\mathrm{atm}$ misses $F_\mathrm{atm}$ by 21 % at $r = 1$, and the skin flux scaled by $r$ misses it by $w C (r - 1)$, 1.7 % at $r = 1.06$.
    - Test id: `tests/atmos_clim/test_common.py::test_the_weighted_parts_sum_to_f_atm`
    - Command: `pytest tests/atmos_clim/test_common.py -k test_the_weighted_parts_sum_to_f_atm` (add `--validation-plots` to write the plot pair)

## The open part follows the melt within the interior step

**Why this check exists.** The open part of the surface starts each interior step at the flux of the atmosphere call. Within the step the melt under it warms or cools, and a fixed start value would let the surface temperature swing from one coupled step to the next. Aragog gives the open part the response $G_\mathrm{eff}$ from the fields that PROTEUS passes (`lid_open_fields`): its loss changes by $G_\mathrm{eff} (T_m - T_{m0})$ per open area. A response that treats the open part as a skin in series with the atmosphere, $G_\mathrm{skin} F' / (G_\mathrm{skin} + F')$, and leaves out the lid part that shares the surface, is too small; the test sees it.

**What it tests.** In the call the lid part and the open part share one surface temperature $T_\mathrm{surf}$. Near the call state the atmosphere flux changes with the surface as $F'$, the slope from the last 2 coupled steps. When the open node warms by $\mathrm{d}T_m$, the shared surface warms by $\mathrm{d}T_\mathrm{surf}$, and the atmosphere takes $F'\, \mathrm{d}T_\mathrm{surf}$ more. With the lid part held, the open part must carry all of it: $(1 - w)\, G_\mathrm{eff} = F'\, \mathrm{d}T_\mathrm{surf} / \mathrm{d}T_m$. The test solves the surface balance at $T_m \pm 10^{-3}$ K and takes the central difference. It compares $(1 - w)\, G_\mathrm{eff}$ of the run path (`lid_open_fields` passed to Aragog's `open_conductance`) with $F'\, \mathrm{d}T_\mathrm{surf}/\mathrm{d}T_m$ at 19 weights from 0.05 to 0.95 and the slopes 0.05, 0.306 and 3 W m$^{-2}$ K$^{-1}$, with $G_\mathrm{lid}$ = 7.27 W m$^{-2}$ K$^{-1}$ and the skin of the first case.

**How we know the right answer.** The derivative of the surface balance gives the closed form $G_\mathrm{eff} = G_\mathrm{skin} F' / (F' + w \min(G_\mathrm{lid}, G_\mathrm{skin}) + (1 - w)\, G_\mathrm{skin})$. The test does not use it: it differentiates the balance itself. On a lid-onset state ($w$ = 0.703, $G_\mathrm{lid}$ = 7.27, $F'$ = 0.306 and $G_\mathrm{skin}$ = 200 W m$^{-2}$ K$^{-1}$) the response is $G_\mathrm{eff}$ = 0.944 W m$^{-2}$ K$^{-1}$ per open area, and $(1 - w)\, G_\mathrm{eff}$ = 0.280 W m$^{-2}$ K$^{-1}$ per total area, the change of the atmosphere flux $F'\, \mathrm{d}T_\mathrm{surf}/\mathrm{d}T_m$. An open part that kept its skin flux $G_\mathrm{skin}(T_m - T_\mathrm{surf})$ while the shared surface moves would change by $(1 - w)\, G_\mathrm{skin} (1 - \mathrm{d}T_\mathrm{surf}/\mathrm{d}T_m)$ = 4.96 W m$^{-2}$ K$^{-1}$ per total area, 17.7 times that 0.280.

**Why it matters.** This response sets how fast the open surface sheds the heat of a warming melt while a lid forms. The next case shows its effect on coupled runs.

![Response of the open part per slope of the atmosphere flux against the lid weight, at 3 slopes, code on the derivative of the surface balance](../../assets/validation/atmos_clim_lid_open_response.svg#only-light){ width="80%" }
![Response of the open part per slope of the atmosphere flux against the lid weight, at 3 slopes, code on the derivative of the surface balance](../../assets/validation/atmos_clim_lid_open_response_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The plot shows $(1 - w)\, G_\mathrm{eff} / F'$ against the lid weight $w$. The lines are the central differences of the surface balance, and the open marks are the code, one colour for each slope $F'$. The value falls below 1 as the lid grows, because the shared surface warms less. The text in the lower left gives the largest difference divided by the tolerance; a pass keeps it below 1.

??? info "Test details"
    - Reference: the derivative of the shared surface balance, by central difference at $\pm 10^{-3}$ K.
    - Tolerance: relative $10^{-7}$ at each weight and slope.
    - Discrimination: the series response without the lid part, $(1 - w)\, G_\mathrm{skin} F' / (G_\mathrm{skin} + F')$, is 47 % below the reference at $w$ = 0.5 and $F'$ = 3 W m$^{-2}$ K$^{-1}$; the test asserts a miss of more than 10 %, and the code with that form fails the main comparison.
    - Test id: `tests/atmos_clim/test_lid_open_flux_instep.py::test_the_response_is_the_change_of_the_shared_surface`
    - Slope guard: `tests/atmos_clim/test_lid_open_flux_instep.py::test_the_grey_secant_cuts_a_cooling_and_a_warming_secant` and `tests/atmos_clim/test_lid_open_flux_instep.py::test_without_a_finite_other_end_the_cut_is_the_grey_tangent` check the cut, and `tests/interior_energetics/test_lid_open_flux_guards.py::test_update_solver_cuts_the_slope_at_the_grey_secant_of_the_last_2_rows` checks it at its call site after a warming and a cooling step.
    - Command: `pytest tests/atmos_clim/test_lid_open_flux_instep.py -k shared_surface` (add `--validation-plots` to write the plot pair)

## Coupled runs through the lid onset

**Why this check exists.** The response of the previous case must also work in a coupled run. In the first step after a lid forms, the interior must lose about what the atmosphere takes, and the surface temperature must not swing. The test fails if a stored run with the response books an interior loss outside the atmosphere flux band of that step, clips the magma temperature, or moves the surface by 2/3 or more of the change in the same step with the response off. The runs with the response off lie inside their band too: the band checks the energy closure of each run, and the change of the surface and of the magma temperature in that step separates the runs.

**What it tests.** Two segments of 1 Earth-mass runs, each from a stored row with a live lid (its weight is above 0). The first segment runs to 10 Myr on a mesh with a 1500 m bottom cell at the core-mantle boundary, the second to 8.82 Myr on a mesh with a 3000 m bottom cell. Each segment is run with the response and with the response off, so that the open part keeps its start value through each step. All rows after the stored row come from one code version. In each pair the 2 runs agree up to the live-lid row, so the step after it shows the effect of the response alone. Later rows differ because the runs took different paths, and the test does not credit them to the response. The `provenance` entry of `tests/data/lid_open_flux/coupled_rows.json` gives the code version, the mesh, the settings and how the response is switched off. The test reads the rows of these runs from that file; it does not run PROTEUS. It asserts:

- in the step after the live-lid row of each run with the response, an interior loss inside the band of the atmosphere flux at the 2 ends of the step, widened by $2.4 \times 10^{-4}$ of the start value: 1.99 W m$^{-2}$ in the band from -0.75 to 4.05 W m$^{-2}$ in the first segment, in the step from 9.440 to 9.611 Myr, and 2.54 W m$^{-2}$ in the band from -0.31 to 4.00 W m$^{-2}$ in the second, in the step from 8.056 to 8.184 Myr; the runs with the response off also lie inside their band, with -0.68 W m$^{-2}$ in the band from -0.75 to 7.76 W m$^{-2}$ and -0.27 W m$^{-2}$ in the band from -0.31 to 6.95 W m$^{-2}$, because a step without the response books about the atmosphere flux at its start;
- in that step, a change of the surface temperature of less than 2/3 of the change with the response off, in the first segment (16.0 K against 27.3 K) and in the second (13.9 K against 22.9 K), the same for the magma temperature (16.2 K against 27.5 K) and (13.9 K against 22.9 K), and a lower magma temperature at the end of the step (1550 K against 1561 K) and (1548 K against 1557 K);
- in every run, no clip of $T_\mathrm{magma}$ and a magma temperature below 1600 K;
- in every run, an energy residual of at most $10^{-5}$ of the content, a sanity bound (the largest stored residual is $7.0 \times 10^{-6}$).

**How we know the right answer.** When the boundary of the interior follows the surface within the step, the interior loss of the step lies between the atmosphere flux at its start and at its end, up to the quadrature floor of the band. A surface under a thin crust over a cooling magma ocean changes smoothly from step to step. In each segment the 2 runs are the same up to the live-lid row, so the difference in the step after it comes from the response alone.

**Why it matters.** The surface temperature sets the outgassing and the next atmosphere step. Swings of the surface and a jump of the magma temperature give wrong outgassing and can stop a run.

![Surface temperature against time in 2 coupled segments, with the response off and with the in-step open part](../../assets/validation/atmos_clim_lid_open_coupled.svg#only-light){ width="80%" }
![Surface temperature against time in 2 coupled segments, with the response off and with the in-step open part](../../assets/validation/atmos_clim_lid_open_coupled_dark.svg#only-dark){ width="80%" }

**How to read the plot.** Both panels show the surface temperature in K against time in Myr, one mark per coupled step, for the run with the response off (dotted) and the run with the response (solid). The dashed vertical line marks the live-lid row. The top panel is the first segment, the bottom panel the second. In each panel the 2 lines lie on each other up to the dashed line. A pass is a solid line that moves less than the dotted line in the step after the dashed line.

**Known limit.** The test checks stored rows of 2 segments and credits the response with the step after the live-lid row only. The later rows show the paths of the runs, not the response: in the first segment with the response the lid is live again on 1 row (9.747 Myr, $w$ = 0.888), and after that step the surface of every run changes by up to 18.9 and 18.1 K between rows in the first segment and 14.8 and 12.7 K in the second, with and without the response. The bound 2/3 lies above the 2 stored ratios, 0.59 and 0.61. In both segments the lid is live on single rows inside a row-to-row swing that the runs without the response show too; where a lid stays live through its onset, `F_int` lies below the band by 4 to 14 % ([Lid onset within a step](../../Reference/lid_coupling.md#lid-onset-within-a-step)). In the first segment with the response, the atmosphere solve at the end of that step did not converge, and PROTEUS kept the atmosphere levels of the last committed row; the upper end of that band comes from this solve. The surface of that row lies 0.02 K below the magma temperature of the interior step, and the magma temperature separates the 2 runs as the surface does. The test does not run PROTEUS, so a change of the code after the stored runs is not seen here. The stored rows come from runs on 2026-11-23 with Aragog 26.11.23. Runs of the same segments on an older version of the lid code wrote the same digits for every stored value, and the `provenance` entry names both builds. The values are read with `read_helpfile_table`, which returns each float as written; the default float parser of pandas reads 57 of the 988 values off by at most 1.6e-15 relative. The response itself is checked by the case above and by the Aragog tests of the open part.

??? info "Test details"
    - Reference: the band of the atmosphere flux of each step, and the stored runs with the response off on the same code version.
    - Tolerance: the bounds in the list above.
    - Test id: `tests/interior_energetics/test_lid_open_flux_coupled.py::test_the_open_part_carries_the_coupled_runs_through_the_live_lid_step`; the test that ties each number of this case to the stored rows: `tests/interior_energetics/test_lid_open_flux_coupled.py::test_the_page_states_the_numbers_of_the_stored_coupled_rows`
    - Command: `pytest tests/interior_energetics/test_lid_open_flux_coupled.py` (add `--validation-plots` to write the plot pair)
