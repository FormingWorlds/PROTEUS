# Validation: the lid and the open surface in the atmosphere call

## What this page checks

This page checks how the atmosphere sees a thin solid crust, the sub-grid lid of Aragog, at the surface (`src/proteus/atmos_clim/common.py`).
Under a lid the surface has two parts at one surface temperature: the part covered by the lid and the open part, where the melt conducts through the skin.
Validation here means that we run the code on a case where another source gives the right answer, and the test checks that the code gives that answer within a stated tolerance.
The right answer comes from 3 sources:

- algebra, for the first 2 cases: two conductors in parallel, and a sum of two fluxes, have closed forms;
- a numerical derivative of the surface balance, for the third case;
- stored coupled runs with a fixed open flux and with the lid off, for the fourth case.

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
| `tests/atmos_clim/test_common.py::test_the_weighted_parts_sum_to_f_atm_less_the_unscaled_cap` | Analytical limit: the sum of the two weighted parts | `lid_flux_cap` and `lid_open_flux` at 21 weights, 3 ratios $r$ and 2 lid conductances | 2026-11-08 |
| `tests/atmos_clim/test_lid_open_flux_instep.py::test_the_response_is_the_change_of_the_shared_surface` | Derivative of the shared surface balance (central difference) | `lid_open_conductance` at 19 weights and 3 slopes $F'$ | 2026-11-18 |
| `tests/interior_energetics/test_lid_open_flux_coupled.py::test_the_open_part_keeps_the_coupled_runs_from_swinging` | Stored coupled runs with the fixed open flux and with the lid off | 2 segments of 1 Earth-mass runs through the lid onset | 2026-11-18 |

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

**Why this check exists.** After the atmosphere step the interior takes two fluxes, the lid part at most the cap $C$ and the open part $F_\mathrm{open}$, and their weighted sum must not count a flux twice or lose one. An open part equal to $F_\mathrm{atm}$ counts the lid part twice, because $F_\mathrm{atm}$ already holds it. An open part without a scale loses the difference between the atmosphere and the skin balance of its call; the test sees both faults.

**What it tests.** The test builds a surface with $T_\mathrm{surf}$ = 900 K, the temperatures and skin of the first case, and $G_\mathrm{lid}$ = 100 and 400 W m$^{-2}$ K$^{-1}$ (half of $G_\mathrm{skin}$ and twice it, where the cap at $G_\mathrm{skin}$ acts). It sets $F_\mathrm{atm}$ to $r$ times the skin balance of the call, for $r$ = 0.95, 1 and 1.06, at 21 weights from 0 to 1. The ratio $r$ says how far the atmosphere is from the skin balance; AGNI closes it only to its solver tolerance. The code scales the open part by $r$: $F_\mathrm{open} = r\, G_\mathrm{skin} (T_m - T_\mathrm{surf})$. This scaled form holds where the skin balance of the call is positive; where it is 0 or below, the open part starts at $F_\mathrm{atm}$ ([explanation](../../Explanations/lid_coupling.md#the-open-part-within-a-step)). The test checks the sum

$$w C + (1 - w) F_\mathrm{open} = F_\mathrm{atm} - w C (r - 1).$$

**How we know the right answer.** Where the atmosphere closes its balance, $r = 1$, the two parts are the two terms of the parallel sum of the first case, so they add to $F_\mathrm{atm}$ exactly. For $r \neq 1$ only the open part is scaled, and the cap of the lid part is not, so the sum misses $F_\mathrm{atm}$ by $w C (r - 1)$. The test computes $C = \min(G_\mathrm{lid}, G_\mathrm{skin}) (T_b - T_\mathrm{surf})$ and the skin balance from these formulas, not from the code.

**Why it matters.** The interior loses the sum of the two parts. A flux counted twice would cool the interior too fast under a growing lid.

![Sum of the two weighted parts divided by the atmosphere flux, against the lid weight, for 3 ratios r](../../assets/validation/atmos_clim_lid_open_part_sum.svg#only-light){ width="80%" }
![Sum of the two weighted parts divided by the atmosphere flux, against the lid weight, for 3 ratios r](../../assets/validation/atmos_clim_lid_open_part_sum_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The horizontal axis is the lid weight $w$, the vertical axis the sum $w C + (1 - w) F_\mathrm{open}$ divided by $F_\mathrm{atm}$, for $G_\mathrm{lid}$ = 100 W m$^{-2}$ K$^{-1}$. The lines are the closed form for each ratio $r$, and the dots are the sum from `lid_flux_cap` and `lid_open_flux`. The dots lie on the lines. At $r = 1$ the sum is $F_\mathrm{atm}$ for every weight.

**Known limit.** For $r \neq 1$ the sum misses $F_\mathrm{atm}$ by $w C (r - 1)$, by design: 1.7 % at $r$ = 1.06 and $w$ = 0.5 here.

??? info "Test details"
    - Reference: analytical limit, the sum of the two weighted parts from the definitions of the cap and the open part.
    - Tolerance: relative $10^{-12}$ at each weight.
    - Discrimination (computed in the test): at $w = 0.5$ and $G_\mathrm{lid} = G_\mathrm{skin}/2$, an open part equal to $F_\mathrm{atm}$ misses $F_\mathrm{atm}$ by 21 % at $r = 1$. An open part without the scale misses $F_\mathrm{atm}$ by 5.7 % and the closed form by 4.0 % at $r = 1.06$, where the closed form is 1.7 % below $F_\mathrm{atm}$; at $r = 1$ the two are equal, so only $r$ = 0.95 and 1.06 tell them apart.
    - Test id: `tests/atmos_clim/test_common.py::test_the_weighted_parts_sum_to_f_atm_less_the_unscaled_cap`
    - Command: `pytest tests/atmos_clim/test_common.py -k less_the_unscaled_cap` (add `--validation-plots` to write the plot pair)

## The open part follows the melt within the interior step

**Why this check exists.** The open part of the surface starts each interior step at the flux of the atmosphere call. Within the step the melt under it warms or cools, and a fixed start value would let the surface temperature swing from one coupled step to the next. The code gives the open part the response $G_\mathrm{eff}$: its loss changes by $G_\mathrm{eff} (T_m - T_{m0})$ per open area. A response that treats the open part as a skin in series with the atmosphere, $G_\mathrm{skin} F' / (G_\mathrm{skin} + F')$, and leaves out the lid part that shares the surface, is too small; the test sees it.

**What it tests.** In the call the lid part and the open part share one surface temperature $T_\mathrm{surf}$. Near the call state the atmosphere flux changes with the surface as $F'$, the slope from the last 2 coupled steps. When the open node warms by $\mathrm{d}T_m$, the shared surface warms by $\mathrm{d}T_\mathrm{surf}$, and the atmosphere takes $F'\, \mathrm{d}T_\mathrm{surf}$ more. With the lid part held, the open part must carry all of it: $(1 - w)\, G_\mathrm{eff} = F'\, \mathrm{d}T_\mathrm{surf} / \mathrm{d}T_m$. The test solves the surface balance at $T_m \pm 10^{-3}$ K and takes the central difference. It compares $(1 - w)\, G_\mathrm{eff}$ of `lid_open_conductance` with $F'\, \mathrm{d}T_\mathrm{surf}/\mathrm{d}T_m$ at 19 weights from 0.05 to 0.95 and the slopes 0.05, 0.306 and 3 W m$^{-2}$ K$^{-1}$, with $G_\mathrm{lid}$ = 7.27 W m$^{-2}$ K$^{-1}$ and the skin of the first case.

**How we know the right answer.** The derivative of the surface balance gives the closed form $G_\mathrm{eff} = G_\mathrm{skin} F' / (F' + w \min(G_\mathrm{lid}, G_\mathrm{skin}) + (1 - w)\, G_\mathrm{skin})$. The test does not use it: it differentiates the balance itself. On a stored lid-onset row of a coupled run ($w$ = 0.703, $F'$ = 0.306 W m$^{-2}$ K$^{-1}$) the response is $G_\mathrm{eff}$ = 0.944 W m$^{-2}$ K$^{-1}$, and $(1 - w)\, G_\mathrm{eff}$ = 0.280.

**Why it matters.** This response sets how fast the open surface sheds the heat of a warming melt while a lid forms. The next case shows its effect on coupled runs.

![Response of the open part per slope of the atmosphere flux against the lid weight, at 3 slopes, code on the derivative of the surface balance](../../assets/validation/atmos_clim_lid_open_response.svg#only-light){ width="80%" }
![Response of the open part per slope of the atmosphere flux against the lid weight, at 3 slopes, code on the derivative of the surface balance](../../assets/validation/atmos_clim_lid_open_response_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The top panel shows $(1 - w)\, G_\mathrm{eff} / F'$ against the lid weight $w$. The lines are the central differences of the surface balance, and the open marks are the code, one colour for each slope $F'$. The value falls below 1 as the lid grows, because the shared surface warms less. The bottom panel shows the difference divided by the tolerance; a pass keeps every mark below 1.

??? info "Test details"
    - Reference: the derivative of the shared surface balance, by central difference at $\pm 10^{-3}$ K.
    - Tolerance: relative $10^{-7}$ at each weight and slope.
    - Discrimination: the series response without the lid part, $(1 - w)\, G_\mathrm{skin} F' / (G_\mathrm{skin} + F')$, is 47 % below the reference at $w$ = 0.5 and $F'$ = 3 W m$^{-2}$ K$^{-1}$; the test asserts a miss of more than 10 %, and the code with that form fails the main comparison.
    - Test id: `tests/atmos_clim/test_lid_open_flux_instep.py::test_the_response_is_the_change_of_the_shared_surface`
    - Command: `pytest tests/atmos_clim/test_lid_open_flux_instep.py -k shared_surface` (add `--validation-plots` to write the plot pair)

## Coupled runs through the lid onset

**Why this check exists.** The response of the previous case must also work in a coupled run. Two stored runs of a 1 Earth-mass planet keep the open flux fixed through each interior step. In the first segment the surface temperature changes by up to 53.7 K between coupled steps after the lid forms, and in one step the interior takes up 1655 W m$^{-2}$ at the surface. In the second segment the surface temperature changes by up to 330.5 K between steps, and the magma temperature reaches 4378 K. The test fails if a stored run with the response has a largest surface step of half that of its run with the fixed open flux or more, takes up more than 1 W m$^{-2}$ at the surface, or has a magma temperature of 1600 K or more.

**What it tests.** Two segments of 1 Earth-mass runs, each from a stored state shortly before the lid forms. The first segment runs from 9.226 to 10 Myr on a mesh with a 1500 m cell at the core. It is run 3 times from the same state: with the open flux fixed through each step, with the response and the solidus floor, and with the lid off. The second segment runs from 7.897 to 9 Myr on a mesh with a 3000 m cell. It compares the run with the fixed open flux that wrote the state at 7.897 Myr with a restart from that state with the response and the floor. In both segments the runs with the fixed open flux and with the lid off come from an earlier code version, which also lacks a change of the CMB flux of at most 0.18 %. The `provenance` entry of `tests/data/lid_open_flux/coupled_rows.json` gives the code version, the mesh and the settings of each run. The test reads the rows of these runs from `tests/data/lid_open_flux/coupled_rows.json`; it does not run PROTEUS. For each segment it asserts:

- the largest step of $T_\mathrm{surf}$ between rows is below half that of the run with the fixed open flux;
- the interior takes up at most 1 W m$^{-2}$ at the surface ($F_\mathrm{int} > -1$ W m$^{-2}$), where the run with the fixed open flux took up more;
- the magma temperature stays below 1600 K, where the run with the fixed open flux went above it;
- no clip of $T_\mathrm{magma}$, and an energy residual of at most $10^{-5}$ of the content, a sanity bound (the largest stored residual is $7.2 \times 10^{-6}$).

For the first segment the largest step is also below that of the run without a lid.

**How we know the right answer.** A surface under a thin crust over a convecting magma ocean should change smoothly from step to step; the run without a lid, which has no lid onset, gives the scale of the steps of the atmosphere alone. The largest steps are 21.9 K with the response, 43.4 K without a lid and 53.7 K with the fixed open flux in the first segment, and 31.7 K against 330.5 K in the second.

**Why it matters.** The surface temperature sets the outgassing and the next atmosphere step. Swings and a magma temperature far above the liquidus give wrong outgassing and can stop a run.

![Surface temperature against time in 2 coupled segments, with the fixed open flux, with the lid off and with the in-step open part](../../assets/validation/atmos_clim_lid_open_coupled.svg#only-light){ width="80%" }
![Surface temperature against time in 2 coupled segments, with the fixed open flux, with the lid off and with the in-step open part](../../assets/validation/atmos_clim_lid_open_coupled_dark.svg#only-dark){ width="80%" }

**How to read the plot.** Both panels show the surface temperature in K against time in Myr, one mark per coupled step. The top panel is the first segment, with the run with the fixed open flux (dotted), the run with the lid off (dashed) and the run with the response (solid). The bottom panel is the second segment; its vertical axis stops at 1650 K, so the run with the fixed open flux leaves the panel from 8.43 to 8.82 Myr. A pass is a solid line that stays close to the lid-off line and far below the swings of the dotted one.

**Known limit.** The test checks stored rows of 2 segments. The lid is live on 2 rows of the first segment and on 1 row of the second; the response acts only there, and the later rows differ because the runs took different paths from those rows. The test does not run PROTEUS, so a change of the code after the stored runs is not seen here; the response itself is checked by the case above and by the Aragog tests of the open part.

??? info "Test details"
    - Reference: the stored runs with the fixed open flux and with the lid off.
    - Tolerance: the bounds in the list above.
    - Test id: `tests/interior_energetics/test_lid_open_flux_coupled.py::test_the_open_part_keeps_the_coupled_runs_from_swinging`
    - Command: `pytest tests/interior_energetics/test_lid_open_flux_coupled.py` (add `--validation-plots` to write the plot pair)
