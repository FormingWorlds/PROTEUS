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
- $L_\mathrm{int}$: the heat that leaves the interior through its surface in a step, in J. Residuals are given as a fraction of its sum over the window.
- $C_\mathrm{cmb}$: the heat that crosses the core-mantle boundary (CMB) in a step, in J, positive out of the core.
- $R_m$, $R_c$, $G$, $R_p$: the residuals of the mantle, of the core, of the coupling to the atmosphere and of the planet, in J. The coupling term $G$ is $L_\mathrm{int}$ minus the loss to space $F_\mathrm{atm} A\,\Delta t$ on the area $A$ of the start row of the step.
- Throughput $T$: the sum of the absolute values of the booked terms, in J.
- Structure update: Zalmoxis solves the structure again and the interior moves to the new mesh. The step that ends at the update is in the class "update", the step after it in the class "remap".

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/utils/test_energy_ledger_validation.py::test_the_lid_onset_window_reproduces_its_mantle_and_coupling_terms` | Earlier measurement of the same window | 25 steps of a 1 Earth-mass run with the sub-grid lid | 2026-11-23 |
| `tests/utils/test_energy_ledger_validation.py::test_core_module_books_the_core_heat_of_each_structure_update` | Core ledger identity | 450 yr, 3 structure updates, `core_module` | 2026-11-23 |
| `tests/utils/test_energy_ledger_validation.py::test_energy_balance_leaves_the_core_heat_jump_of_an_update_in_the_remap_class` | Separate measurement script on the same rows | 450 yr, 3 structure updates, `energy_balance` | 2026-11-23 |
| `tests/utils/test_energy_ledger_validation.py::test_the_slope_closure_of_the_cmb_node_closes_the_core_ledger_in_the_mush` | Core balance of Bower et al. (2018)[^cite-bower2018] Eqs. 19-21 | 1 Earth-mass PALEOS run with 2 closures of the CMB node, to 14.1 kyr | 2026-11-23 |

## The mantle and the coupling terms of the lid onset

**Why this check exists.** When the sub-grid lid forms, the heat leaving the interior and the heat leaving to space are computed by different parts of the code. A ledger that uses the wrong area or misses a mantle term reports a wrong coupling gap here.

**What it tests.** 25 steps of a 1 Earth-mass run in which the lid forms, every step with a live lid. The mantle residual $R_m$ must be $-1.646 \times 10^{-6}$ of $L_\mathrm{int}$ and the coupling term $G$ must be $-5.698 \times 10^{-2}$ of $L_\mathrm{int}$.

**How we know the right answer.** The same window was measured before this function existed, from the same helpfile columns by a separate script. The loss to space on the area of the end row of a step instead of its start row gives a $G$ outside the tolerance, and the test asserts that too.

**Why it matters.** The coupling gap during the lid onset is a known open term of the planet ledger; its size decides whether a paper can quote the energy closure of runs with a lid.

![Cumulative residuals by term over the lid onset window](../../assets/validation/energy_ledger_lid_onset.svg#only-light){ width="80%" }
![Cumulative residuals by term over the lid onset window](../../assets/validation/energy_ledger_lid_onset_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x axis is the time since the start of the window, in years. The y axis is each residual summed from the start of the window, divided by $L_\mathrm{int}$ of the whole window, on a scale that is linear near 0 and logarithmic beyond $10^{-6}$. The lines are $R_m$, $G$, and the core residual $R_c$ split into the steps after a structure update and the other steps. The test asserts the end values of $R_m$ and $G$; the core lines are context.

??? info "Test details"
    - Reference: the measurement of this window in the lid-open-flux work (script `ledger_window.py`), $R_m/L_\mathrm{int} = -1.646 \times 10^{-6}$, $G/L_\mathrm{int} = -5.698 \times 10^{-2}$.
    - Tolerance: $10^{-3}$ relative on each number; the area of the end row misses by more.
    - Data: `tests/data/energy_ledger/r2rel_onset.json`, rows 873 to 898 of the source run, with its sha256.
    - Test id: `tests/utils/test_energy_ledger_validation.py::test_the_lid_onset_window_reproduces_its_mantle_and_coupling_terms`
    - Command: `pytest tests/utils/test_energy_ledger_validation.py --validation-plots`

## The core ledger of `core_module` across structure updates

**Why this check exists.** At a structure update the core moves to a new pressure profile and its heat content changes. `core_module` books that change in `step_dE_struct_core_J`; a ledger that drops this term reports the change as a core leak.

**What it tests.** A 450 yr run with `core_bc = 'core_module'` and 3 structure updates. The core residual $R_c$ must stay within $10^{-6}$ of the throughput $T$ in every step class, while the booked core heat of the updates is $6.92 \times 10^{-3}$ of $L_\mathrm{int}$.

**How we know the right answer.** The core heat change, the CMB heat and the booked terms add up to 0 by construction of the core ledger of `core_module`; the stored rows must show that.

**Why it matters.** Runs with `core_module` can quote a closed core over structure updates.

This case has no plot: the result is an identity that is 0 to round-off, and the plot of the next case shows the same updates with `energy_balance`.

??? info "Test details"
    - Reference: the identity of the core ledger of `core_module` (booked core heat equals the core heat change plus the CMB heat).
    - Tolerance: $10^{-6}$ of $T$ in each class; dropping the booked term moves $R_c$ by $6.92 \times 10^{-3}$ of $L_\mathrm{int}$.
    - Data: `tests/data/energy_ledger/cm_updates.json`.
    - Test id: `tests/utils/test_energy_ledger_validation.py::test_core_module_books_the_core_heat_of_each_structure_update`
    - Command: `pytest tests/utils/test_energy_ledger_validation.py`

## The unbooked core heat of `energy_balance` at a structure update

**Why this check exists.** With `core_bc = 'energy_balance'` nothing books the core heat change of a structure update, so it stays in $R_c$. The ledger must put it in the steps after an update and report its size, not spread it over the run.

**What it tests.** The same 450 yr setup with `energy_balance`. The core residual of the steps after an update must be $5.948278897938286 \times 10^{-3}$ of $L_\mathrm{int}$, and the other steps must hold less than 1 % of that.

**How we know the right answer.** A separate measurement script summed the same rows in the same way and found this value.

**Why it matters.** This is the size of the open core term of `energy_balance` runs; a paper that uses this core must quote it.

![Cumulative core and coupling residuals of an energy_balance run](../../assets/validation/energy_ledger_core_remap.svg#only-light){ width="80%" }
![Cumulative core and coupling residuals of an energy_balance run](../../assets/validation/energy_ledger_core_remap_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x axis is the time in years. The y axis is each residual summed from the start, divided by $L_\mathrm{int}$ of the run, linear near 0 and logarithmic beyond $10^{-6}$. The core line of the steps after an update rises at each of the 3 updates; the other core steps stay near 0. The flat coupling line is the first step of the run, which carries no interior heat. The test asserts the end value of the core line after updates.

??? info "Test details"
    - Reference: the E2 measurement script on the same rows, $5.948278897938286 \times 10^{-3}$ of $L_\mathrm{int}$.
    - Tolerance: $10^{-9}$ relative (same arithmetic).
    - Data: `tests/data/energy_ledger/eb_updates.json`.
    - Test id: `tests/utils/test_energy_ledger_validation.py::test_energy_balance_leaves_the_core_heat_jump_of_an_update_in_the_remap_class`
    - Command: `pytest tests/utils/test_energy_ledger_validation.py --validation-plots`

## The closure of the CMB node in the mush

**Why this check exists.** In the mush the heat capacity of the phase blend does not give the slope of its temperature against entropy. A closure of the CMB node with $c_p/T$ then lets the core lose more or less heat than crosses the CMB; the ledger must show that, and show that the slope closure removes it.

**What it tests.** The same 1 Earth-mass PALEOS run with 2 closures of the CMB node: $c_p/T$ (aragog 3a869bb1) and the entropy slope of the phase blend (aragog fbf15478). For each step the ledger gives $R_c/|C_\mathrm{cmb}|$, positive when the core loses less heat than crosses the CMB. In the mush, from 1787 yr to 14.1 kyr, the $c_p/T$ run must miss by a median of $0.091 \pm 0.01$ and the slope run by at most $10^{-3}$.

**How we know the right answer.** The core balance of Bower et al. (2018) Eqs. 19-21 requires the core heat change to equal the CMB heat; the slope closure meets it by construction, $c_p/T$ does not.

**Why it matters.** The core temperature sets the CMB heat flow and the start of core freezing in every run with the default core.

![Core residual of each step over the CMB heat, for 2 closures of the CMB node](../../assets/validation/energy_ledger_cmb_closure.svg#only-light){ width="80%" }
![Core residual of each step over the CMB heat, for 2 closures of the CMB node](../../assets/validation/energy_ledger_cmb_closure_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x axis is the time in years on a log scale. The y axis is $R_c/|C_\mathrm{cmb}|$ of each step, linear between $-10^{-4}$ and $10^{-4}$. The shaded band is the mush window that the test asserts. Steps at a structure update and the steps after one are left out. Before the band the CMB node is just above the liquidus; the 2 steps near 1.3 to 1.6 kyr cross a flat cell of the tables and are context, not asserted.

**Known limit.** The rows of the slope run come from a branch of aragog; they are written again from a run on the aragog release with this closure.

??? info "Test details"
    - Reference: Bower et al. (2018) Eqs. 19-21.
    - Tolerance: median $0.091 \pm 0.01$ for $c_p/T$ and at most $10^{-3}$ for the slope in the mush (measured $8.0 \times 10^{-4}$); the smallest $c_p/T$ value exceeds 50 times the largest slope value.
    - Data: `tests/data/energy_ledger/a115_base.json` and `a115_branch.json`; the mush window comes from the node entropy of the snapshots, as the helpfile has no melt fraction of the node.
    - Test id: `tests/utils/test_energy_ledger_validation.py::test_the_slope_closure_of_the_cmb_node_closes_the_core_ledger_in_the_mush`
    - Command: `pytest tests/utils/test_energy_ledger_validation.py --validation-plots`

[^cite-bower2018]: D. J. Bower, P. Sanan, A. S. Wolf, *[Numerical solution of a non-linear conservation law applicable to the interior dynamics of partially molten planets](https://doi.org/10.1016/j.pepi.2017.11.004)*, Physics of the Earth and Planetary Interiors, 274, 49-62, 2018. [SciX](https://scixplorer.org/abs/2018PEPI..274...49B/abstract).
