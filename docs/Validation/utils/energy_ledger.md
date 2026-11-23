# Validation: the whole-run energy ledger

## What this page checks

This page checks the energy ledger of a whole run (`src/proteus/utils/energy_ledger.py`).
The ledger adds up, step by step, the energy that each part of the planet gains and loses, as the helpfile books it.
A planet that keeps its energy has a residual near 0: the heat lost by the interior equals the heat that leaves to space, plus the terms the run books on purpose.
Validation here means that we run the ledger on stored rows of real runs where another measurement or the physics gives the right answer, and check that the ledger gives that answer.
If the ledger is wrong, it reports a closed run as leaking or hides a real leak, and every closure number of a paper that uses it is wrong.
Each case below says why the check exists, what it tests, how we know the right answer and why it matters; the box "Test details" gives the reference, the tolerance, the test id and the command.

**Terms of this page.**

- Step: the interval between 2 rows of the helpfile.
- $L_\mathrm{int}$: the heat that leaves the interior through its surface in a step, in J. Residuals are given as a fraction of its sum over the window, unless a case names another base.
- $C_\mathrm{cmb}$: the heat that crosses the core-mantle boundary (CMB) in a step, in J, positive out of the core.
- $R_m$, $R_c$, $G$, $R_p$: the residuals of the mantle, of the core, of the coupling to the atmosphere and of the planet, in J. The coupling term $G$ is $L_\mathrm{int}$ minus the loss to space $F_\mathrm{atm} A\,\Delta t$ on the area $A$ of the start row of the step. The planet residual is $R_p = R_m + R_c - G$.
- Throughput: the sum of the absolute values of 6 terms, in J: the change of the mantle heat, $L_\mathrm{int}$, $C_\mathrm{cmb}$, radiogenic and tidal heat, impact heat and booked core heat. The change of the core heat is not in it, also where `core_module` books it. The command prints the throughput as `T`; it is an energy, not a temperature.
- CMB node: the lowest node of the mantle mesh; with `energy_balance` the core temperature follows it.
- Closure of the CMB node: the formula for the slope of the node temperature against entropy, which turns the heat that crosses the CMB into a change of the entropy of the node.
- Mush: the part of the mantle where melt and crystals are both present.
- Structure update: Zalmoxis solves the structure again and the interior moves to the new mesh. The step that ends at the update is in the class "update", the step after it in the class "remap".

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/utils/test_energy_ledger_validation.py::test_the_lid_onset_window_reproduces_its_mantle_and_coupling_terms` | Separate script on the same rows | 25 steps of a 1 Earth-mass run with the sub-grid lid | 2026-11-23 |
| `tests/utils/test_energy_ledger_validation.py::test_core_module_books_the_core_heat_of_each_structure_update` | Core ledger identity | 450 yr, 3 structure updates, `core_module` | 2026-11-23 |
| `tests/utils/test_energy_ledger_validation.py::test_energy_balance_leaves_the_core_heat_jump_of_an_update_in_the_remap_class` | Separate script on the same rows | 450 yr, 3 structure updates, `energy_balance` | 2026-11-23 |
| `tests/utils/test_energy_ledger_validation.py::test_the_slope_closure_of_the_cmb_node_closes_the_core_ledger_in_the_mush` | Core balance of Bower et al. (2018)[^cite-bower2018] Eqs. 19-21 | 1 Earth-mass PALEOS run with 2 closures of the CMB node, to 13.3 and 14.1 kyr | 2026-11-23 |

## The mantle and the coupling terms of the lid onset

**Why this check exists.** When the [sub-grid lid](../../Explanations/lid_coupling.md) forms, the heat leaving the interior and the heat leaving to space are computed by different parts of the code. A ledger that uses the wrong area or misses a mantle term reports a wrong coupling gap here.

**What it tests.** 25 steps of a 1 Earth-mass run with `core_bc = 'energy_balance'` in which the lid forms, every step with a live lid. The window starts at a structure update and holds 2 more. The mantle residual $R_m$ must be $-1.646 \times 10^{-6}$ of $L_\mathrm{int}$ and the coupling term $G$ must be $-5.698 \times 10^{-2}$ of $L_\mathrm{int}$. The core residual of the steps after an update must be $1.356$ of $L_\mathrm{int}$, and that of the steps that neither end at an update nor follow one $-9.93 \times 10^{-4}$.

**How we know the right answer.** A separate script summed the same helpfile columns of this window and found these values. The test also computes $G$ with the loss to space on the area of the end row of each step, not the start row, and asserts that this wrong $G$ falls outside the tolerance.

**Why it matters.** The coupling gap during the lid onset is a known open term of the planet ledger; its size decides whether a paper can quote the energy closure of runs with a lid. In this window the core term after the updates is larger than $L_\mathrm{int}$ itself (1.356 of it): with `energy_balance` the change of the core heat at an update is booked nowhere (see the case of `energy_balance` below).

![Cumulative residuals by term over the lid onset window](../../assets/validation/energy_ledger_lid_onset.svg#only-light){ width="80%" }
![Cumulative residuals by term over the lid onset window](../../assets/validation/energy_ledger_lid_onset_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x axis is the time since the start of the window, in years. The y axis is each residual summed from the start of the window, divided by $L_\mathrm{int}$ of the whole window. The scale is linear near 0 and logarithmic beyond $10^{-6}$. The lines are $R_m$, $G$, and the core residual $R_c$ in 3 classes: the steps after a structure update, the 2 steps that end at one, and the other steps. The core line after an update jumps to 1.50 in the first step, because the window starts at an update, and falls at each of the 2 later updates to 1.356. The test asserts the end values of $R_m$, $G$, the core line after an update and the core line of the other steps.

??? info "Test details"
    - Reference: a separate script that summed the same helpfile columns of this window, $R_m/L_\mathrm{int} = -1.646 \times 10^{-6}$, $G/L_\mathrm{int} = -5.698 \times 10^{-2}$; the core residual by class from an independent sum of the stored rows, $1.3557209386827422$ (after an update) and $-9.932 \times 10^{-4}$ (other steps) of $L_\mathrm{int}$.
    - Tolerance: $10^{-3}$ relative on $R_m$ and $G$, and the area of the end row misses by more; $10^{-9}$ and $10^{-6}$ relative on the 2 core values.
    - Data: `tests/data/energy_ledger/r2rel_onset.json`, rows 872 to 898 of the source run with its sha256; row 872 holds the structure before the update at the start of the window.
    - Test id: `tests/utils/test_energy_ledger_validation.py::test_the_lid_onset_window_reproduces_its_mantle_and_coupling_terms`
    - Command: `pytest tests/utils/test_energy_ledger_validation.py --validation-plots`

## The core ledger of `core_module` across structure updates

**Why this check exists.** At a structure update the core moves to a new pressure profile and its heat content changes. `core_module` books that change in `step_dE_struct_core_J`; a ledger that drops this term reports the change as a core leak.

**What it tests.** A 450 yr run with `core_bc = 'core_module'` and 3 structure updates. The core residual $R_c$ must stay within $10^{-7}$ of the throughput in every step class, while the booked core heat of the updates is $6.873 \times 10^{-3}$ of $L_\mathrm{int}$.

**How we know the right answer.** The core heat change, the CMB heat and the booked terms add up to 0 by construction of the core ledger of `core_module`; the stored rows must show that.

**Why it matters.** Runs with `core_module` can quote a closed core over structure updates.

This case is drawn on the plot of the next case, as the line "R_c, core_module", next to the same updates with `energy_balance`.

??? info "Test details"
    - Reference: the identity of the core ledger of `core_module` (booked core heat equals the core heat change plus the CMB heat).
    - Tolerance: $10^{-7}$ of the throughput in each class (measured $6 \times 10^{-9}$); the booked core heat $6.872932228745487 \times 10^{-3}$ of $L_\mathrm{int}$ to $10^{-9}$ relative; dropping the booked term moves $R_c$ by that much.
    - Data: `tests/data/energy_ledger/cm_updates.json`.
    - Test id: `tests/utils/test_energy_ledger_validation.py::test_core_module_books_the_core_heat_of_each_structure_update`
    - Command: `pytest tests/utils/test_energy_ledger_validation.py`

## The unbooked core heat of `energy_balance` at a structure update

**Why this check exists.** With `core_bc = 'energy_balance'` nothing books the change of the core heat at a structure update, so it stays in $R_c$. The ledger must put it in the steps after an update and report its size, not spread it over the run.

**What it tests.** The same 450 yr setup with `energy_balance`. The core residual of the steps after an update must be $5.948 \times 10^{-3}$ of $L_\mathrm{int}$ (the full value is in the box below), and the steps at an update and the other steps must together hold less than 1 % of that. The planet residual must be $8.526 \times 10^{-3}$ of $L_\mathrm{int}$.

**How we know the right answer.** A separate script summed the same rows with the same arithmetic and found this value. The check pins the bookkeeping of the ledger; it is not an independent prediction of the size of the core heat change.

**Why it matters.** This is the size of the open core term of `energy_balance` runs; a paper that uses this core must quote it. No heat flow carries this change: the core temperature follows the CMB node, and at an update the mantle keeps the entropy of each mass parcel while the node moves to a new pressure, so its temperature changes. The ledger does not split this change into compression and other parts.

![Cumulative core and coupling residuals of an energy_balance run](../../assets/validation/energy_ledger_core_remap.svg#only-light){ width="80%" }
![Cumulative core and coupling residuals of an energy_balance run](../../assets/validation/energy_ledger_core_remap_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x axis is the time in years. The y axis is each residual summed from the start, divided by $L_\mathrm{int}$ of the run, linear near 0 and logarithmic beyond $10^{-6}$. The core line of the steps after an update rises at each of the 3 updates; the core lines of the steps at an update and of the other steps stay near 0. The line "R_c, core_module" is a second run of the same setup with `core_module`, which books the change and stays at 0. The coupling line $G$ drops at the first step of the run, which carries a loss to space but no interior heat, and is flat after it. The test asserts the end value of the core line after updates.

??? info "Test details"
    - Reference: a separate measurement script on the same rows, $5.948278897938286 \times 10^{-3}$ of $L_\mathrm{int}$; the planet residual $8.525868397324384 \times 10^{-3}$ of $L_\mathrm{int}$ from an independent sum of the stored rows.
    - Tolerance: $10^{-9}$ relative (same arithmetic).
    - Data: `tests/data/energy_ledger/eb_updates.json`.
    - Test id: `tests/utils/test_energy_ledger_validation.py::test_energy_balance_leaves_the_core_heat_jump_of_an_update_in_the_remap_class`
    - Command: `pytest tests/utils/test_energy_ledger_validation.py --validation-plots`

## The closure of the CMB node in the mush

**Why this check exists.** In the mush, the heat capacity $c_p$ of the phase blend does not give the slope of the temperature of the CMB node against entropy. A closure with $c_p/T$ then lets the core lose more or less heat than crosses the CMB. The ledger must show that, and show that the slope closure removes it.

**What it tests.** The same 1 Earth-mass PALEOS run with 2 closures of the CMB node: $c_p/T$, and the entropy slope of the phase blend. For each step the ledger gives $R_c/|C_\mathrm{cmb}|$, which is positive when the core loses less heat than crosses the CMB. In the mush, from 1787 yr to 14.1 kyr, the $c_p/T$ run must miss by a median of $0.091 \pm 0.01$ and the slope run by at most $10^{-3}$. The $c_p/T$ run ends at 13.3 kyr.

**How we know the right answer.** The core balance of Bower et al. (2018) Eqs. 19-21 requires the core heat change to equal the CMB heat; the slope closure meets it by construction, $c_p/T$ does not.

**Why it matters.** The core temperature sets the CMB heat flow and the start of core freezing in every run with the default core, `core_bc = 'energy_balance'`.

![Core residual of each step over the CMB heat, for 2 closures of the CMB node](../../assets/validation/energy_ledger_cmb_closure.svg#only-light){ width="80%" }
![Core residual of each step over the CMB heat, for 2 closures of the CMB node](../../assets/validation/energy_ledger_cmb_closure_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x axis is the time in years on a log scale. The y axis is $R_c/|C_\mathrm{cmb}|$ of each step, linear between $-10^{-4}$ and $10^{-4}$. The shaded band is the mush window that the test asserts. The first step of the run, the steps at a structure update and the steps after one are left out. Before the band, while the CMB node is just above the liquidus, the $c_p/T$ run misses by up to about 0.33 with a negative sign: there the core loses more heat than crosses the CMB. The slope run misses by up to about $4 \times 10^{-3}$ there, more than in the band. The 2 steps near 1.3 to 1.6 kyr cross a flat cell of the tables. The test does not assert these steps.

**Known limit.** The rows of the slope run come from an unreleased branch of aragog. They must be written again from a run on an aragog release that has this closure.

??? info "Test details"
    - Reference: Bower et al. (2018) Eqs. 19-21.
    - Tolerance: median $0.091 \pm 0.01$ for $c_p/T$ and at most $10^{-3}$ for the slope in the mush (measured $8.0 \times 10^{-4}$); the smallest $c_p/T$ value exceeds 50 times the largest slope value.
    - Data: `tests/data/energy_ledger/a115_base.json` ($c_p/T$ run on aragog 3a869bb1) and `a115_branch.json` (slope run on aragog fbf15478); the mush window comes from the node entropy of the snapshots, as the helpfile has no melt fraction of the node.
    - Test id: `tests/utils/test_energy_ledger_validation.py::test_the_slope_closure_of_the_cmb_node_closes_the_core_ledger_in_the_mush`
    - Command: `pytest tests/utils/test_energy_ledger_validation.py --validation-plots`

[^cite-bower2018]: D. J. Bower, P. Sanan, A. S. Wolf, *[Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets](https://doi.org/10.1016/j.pepi.2017.11.004)*, Physics of the Earth and Planetary Interiors, 274, 49-62, 2018. [SciX](https://scixplorer.org/abs/2018PEPI..274...49B/abstract).
