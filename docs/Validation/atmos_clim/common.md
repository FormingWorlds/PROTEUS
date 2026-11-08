# Validation: the lid and the open surface in the atmosphere call

## What this page checks

This page checks how the atmosphere sees a thin solid crust, the sub-grid lid of Aragog, at the surface (`src/proteus/atmos_clim/common.py`).
Under a lid the surface has two parts at one surface temperature: the part covered by the lid and the open part, where the melt conducts through the skin.
Validation here means that we run the code on a case where another source gives the right answer, and the test checks that the code gives that answer within a stated tolerance.
Here the right answer comes from algebra: two conductors in parallel, and a sum of two fluxes, have closed forms.
If this part of the code were wrong, the atmosphere would solve for the wrong surface temperature, or the interior would lose more or less heat than the atmosphere takes.
Each case below says why the check exists, what it tests, how we know the right answer and why it matters; the box "Test details" at the end of each case gives the reference, the tolerance, the test id and the command.

**Terms of this page.**

- Lid weight $w$: the share of the surface under the lid, from 0 (no lid) to 1 (a full lid).
- Conductance $G$, in W m$^{-2}$ K$^{-1}$: the heat flux through a layer for each kelvin of temperature difference across it. $G_\mathrm{skin} = k / d$ is the conductance of the skin, with $k$ its conductivity and $d$ its thickness; $G_\mathrm{lid}$ is that of the lid.
- $T_b$: the temperature at the base of the lid, the solidus. $T_\mathrm{node}$: the temperature of the top node of the interior mesh, under the open part. $T_\mathrm{surf}$: the surface temperature.
- $F_\mathrm{atm}$: the net flux that the atmosphere carries away. $C$: the cap, the flux that the last atmosphere balance gave the lid part.

The physics of the lid in the coupling is on the page [Sub-grid lid and the surface skin](../../Explanations/lid_coupling.md).

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/atmos_clim/test_common.py::test_the_call_flux_is_the_parallel_sum_of_the_lid_and_the_open_surface` | Analytical limit: two conductors in parallel at one surface temperature | `lid_boundary` at 110 surface temperatures, 4 weights, 3 lid conductances | 2026-11-08 |
| `tests/atmos_clim/test_common.py::test_the_weighted_parts_sum_to_f_atm_less_the_unscaled_cap` | Analytical limit: the sum of the two weighted parts | `lid_flux_cap` and `lid_open_flux` at 21 weights and 3 ratios $r$ | 2026-11-08 |

## The atmosphere call sees the lid and the open surface in parallel

**Why this check exists.** AGNI and the dummy atmosphere solve the surface balance against one skin. The code must write the two parts of the surface as one skin with the same flux at every surface temperature. A skin added in series with the lid, or a lid that conducts more than the open surface, gives the atmosphere the wrong flux, and the test sees both.

**What it tests.** The function `lid_boundary` returns the skin thickness $d_\mathrm{call}$ and the melt temperature $T_\mathrm{magma,call}$ of the call. The test computes the flux of that one skin, $\frac{k}{d_\mathrm{call}}(T_\mathrm{magma,call} - T_\mathrm{surf})$, at 110 surface temperatures from 300 to 1390 K. It uses $T_b$ = 1400 K, $T_\mathrm{node}$ = 1500 K, a 10 mm skin with $k$ = 2 W m$^{-1}$ K$^{-1}$ ($G_\mathrm{skin}$ = 200 W m$^{-2}$ K$^{-1}$), the weights 0, 0.3, 0.7 and 1, and a lid conductance of 1/4, 1 and 2 times $G_\mathrm{skin}$. The expected flux is the sum of the two parts,

$$F = w \min(G_\mathrm{lid}, G_\mathrm{skin}) (T_b - T_\mathrm{surf}) + (1 - w)\, G_\mathrm{skin} (T_\mathrm{node} - T_\mathrm{surf}).$$

**How we know the right answer.** Two layers side by side, at one surface temperature, conduct in parallel: their fluxes add. The lid conductance is capped at $G_\mathrm{skin}$, so that a lid never loses more heat than the open surface would. The one skin of the call has the conductance $w \min(G_\mathrm{lid}, G_\mathrm{skin}) + (1 - w) G_\mathrm{skin}$ and the weighted mean of $T_b$ and $T_\mathrm{node}$ as its melt temperature, so its flux equals $F$ at every $T_\mathrm{surf}$.

**Why it matters.** The atmosphere sets the surface temperature from this flux. The surface temperature then sets the heat the interior loses and the weight of the lid in the next step.

![Flux of the call against the surface temperature for 4 lid weights, with the skin in series as a dashed line](../../assets/validation/atmos_clim_lid_parallel_sum.svg#only-light){ width="80%" }
![Flux of the call against the surface temperature for 4 lid weights, with the skin in series as a dashed line](../../assets/validation/atmos_clim_lid_parallel_sum_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The horizontal axis is the surface temperature in K, the vertical axis the flux of the call in kW m$^{-2}$. The plot shows the lid conductance of $G_\mathrm{skin}/4$. The solid lines are the closed form $F$ for each weight $w$, and the dots are the flux of `lid_boundary`. The dots lie on the lines. The dashed line is a skin in series with the lid at $w = 1$, which conducts 20 % less.

??? info "Test details"
    - Reference: analytical limit, two conductors in parallel at one surface temperature.
    - Tolerance: relative $10^{-12}$ at each surface temperature.
    - Discrimination: at $w = 1$ a skin in series with the lid gives 0.8 of $F$ at $G_\mathrm{lid} = G_\mathrm{skin}/4$ and 0.5 at $G_\mathrm{lid} = G_\mathrm{skin}$. A lid without the cap at $G_\mathrm{skin}$ gives 2 times $F$ at $G_\mathrm{lid} = 2 G_\mathrm{skin}$.
    - Test id: `tests/atmos_clim/test_common.py::test_the_call_flux_is_the_parallel_sum_of_the_lid_and_the_open_surface`
    - Command: `pytest tests/atmos_clim/test_common.py -k parallel_sum_of_the_lid` (add `--validation-plots` to write the plot pair)

## The two parts of the surface sum to the atmosphere flux

**Why this check exists.** After the atmosphere step the interior takes two fluxes: the lid part at most the cap $C$, and the open part $F_\mathrm{open}$. Their weighted sum must not count a flux twice or lose one. An open part equal to $F_\mathrm{atm}$ counts the lid part twice, because $F_\mathrm{atm}$ already holds it. An open part without a scale loses the difference between the atmosphere and the skin balance of its call. The test sees both.

**What it tests.** The test builds a surface with $G_\mathrm{lid}$ = 100 W m$^{-2}$ K$^{-1}$ (half of $G_\mathrm{skin}$), $T_\mathrm{surf}$ = 900 K and the temperatures of the first case. It sets $F_\mathrm{atm}$ to $r$ times the skin balance of the call, for $r$ = 0.95, 1 and 1.06, at 21 weights from 0 to 1. The ratio $r$ says how far the atmosphere is from the skin balance; AGNI closes it only to its solver tolerance. The code scales the open part by $r$: $F_\mathrm{open} = r\, G_\mathrm{skin} (T_\mathrm{node} - T_\mathrm{surf})$. The test checks the sum

$$w C + (1 - w) F_\mathrm{open} = F_\mathrm{atm} - w C (r - 1).$$

**How we know the right answer.** Where the atmosphere closes its balance, $r = 1$, the two parts are the two terms of the parallel sum of the first case, so they add to $F_\mathrm{atm}$ exactly. For $r \neq 1$ only the open part is scaled, and the cap of the lid part is not, so the sum misses $F_\mathrm{atm}$ by $w C (r - 1)$. This follows from the definitions alone.

**Why it matters.** The interior loses the sum of the two parts. A flux counted twice would cool the interior too fast under a growing lid.

![Sum of the two weighted parts divided by the atmosphere flux, against the lid weight, for 3 ratios r](../../assets/validation/atmos_clim_lid_open_part_sum.svg#only-light){ width="80%" }
![Sum of the two weighted parts divided by the atmosphere flux, against the lid weight, for 3 ratios r](../../assets/validation/atmos_clim_lid_open_part_sum_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The horizontal axis is the lid weight $w$, the vertical axis the sum $w C + (1 - w) F_\mathrm{open}$ divided by $F_\mathrm{atm}$. The lines are the closed form for each ratio $r$, and the dots are the sum from `lid_flux_cap` and `lid_open_flux`. The dots lie on the lines. At $r = 1$ the sum is $F_\mathrm{atm}$ for every weight.

**Known limit.** For $r \neq 1$ the sum misses $F_\mathrm{atm}$ by $w C (r - 1)$, by design: 1.7 % at $r$ = 1.06 and $w$ = 0.5 here. In two AGNI runs through the onset of the lid the sum lay between 0.956 and 1.000 of $F_\mathrm{atm}$.

??? info "Test details"
    - Reference: analytical limit, the sum of the two weighted parts from the definitions of the cap and the open part.
    - Tolerance: relative $10^{-12}$ at each weight.
    - Discrimination: at $w = 0.5$ an open part equal to $F_\mathrm{atm}$ misses $F_\mathrm{atm}$ by 21 % at $r = 1$; an open part without the scale misses it by 5.7 % at $r = 1.06$, where the code is 1.7 % below it.
    - Test id: `tests/atmos_clim/test_common.py::test_the_weighted_parts_sum_to_f_atm_less_the_unscaled_cap`
    - Command: `pytest tests/atmos_clim/test_common.py -k less_the_unscaled_cap` (add `--validation-plots` to write the plot pair)
