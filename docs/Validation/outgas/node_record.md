# Validation: `src/proteus/outgas/node_record.py`

## What this page checks

This page checks the per-node record of the layer trapping scheme (`proteus.outgas.node_record`). For every node of the Aragog mesh, the record keeps the melt still held between its crystals, the pore melt, and the volatiles stored at the node. It moves each node between the liquid, mush and solid states from one step to the next. A node in the mush buries the volatiles of its pore melt, and a node that freezes locks what it still holds.

Validation here means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If this part of the code were wrong, a run would keep the wrong share of its water in the mantle, and the atmosphere would get the rest. The cases below run the record over many coupling steps, as a PROTEUS run does.

Each case has the layout of every Validation page: why the check exists, what it tests, how we know the right answer, why it matters, a plot, and a box "Test details". Symbols used on this page:

- $\Phi_c$: the melt fraction at which a node enters the mush from the magma ocean, `interior_energetics.rfront_loc` \[1\];
- $\tau_s$: the time the crystal framework takes to compact and push out its melt, the matrix time \[s\];
- $C$: the mass fraction of water in the melt \[1\];
- $m$: the mass of the node \[kg\];
- $h$: the sub-step of the drainage, in units of $\tau_s$ \[1\];
- $\phi$: the solver's melt volume fraction of a node, its porosity \[1\];
- $D$: the crystal/melt partition coefficient of water, the ratio of its mass fraction in the crystals to that in the melt \[1\].

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/outgas/test_node_record.py::test_a_matrix_limited_node_traps_the_closed_form_share_of_its_melt` | Analytical limit of the matrix branch, $\Phi_c e^{-t/\tau_s}$ | One node held in the mush for 3 matrix times over 52 coupling steps, then frozen. | 2026-11-24 |
| `tests/outgas/test_node_record.py::test_a_node_freezing_within_a_step_keeps_what_the_continuous_drainage_keeps` | Second code: the continuous drainage of one node, integrated by SciPy's LSODA without coupling steps | One node whose porosity falls to zero over 1000 yr, in 1 to 64 coupling steps. | 2026-11-24 |

## A node limited by its crystal framework keeps the exponential share of its melt

**Why this check exists.** The record carries a node's pore melt from step to step and locks what is left when the node freezes. An error in how the record hands the pore melt from one step to the next, or in the depth that sets the matrix time, would change what the node locks. Using the depth of the node's base in place of its centre halves $\tau_s$ and locks $\Phi_c e^{-6}$, and the test fails. A forward-Euler step at the same sub-steps ends 8 % low, and the test requires more than 5 %.

**What it tests.** The top node of a mantle of 6 nodes enters the mush at $\Phi_c = 0.5$. Its melt fraction is then held just below $\Phi_c$, so it stays in the mush, and the crystal framework alone drains it, with a 1 cm grain, a melt of 0.1 Pa s and a mush of $10^{22}$ Pa s. After 3 matrix times, over 52 coupling steps, the node freezes. It must reach the solid state with $\Phi_c e^{-3}$ of its mass as melt, and lock that share of the water it buried on entry, $C\,\Phi_c\,m$.

**How we know the right answer.** With a constant matrix time the pore melt follows $d\psi/dt = -\psi/\tau_s$, so the node keeps $e^{-t/\tau_s}$ of its pore melt, which is $\Phi_c e^{-t/\tau_s}$ of its mass. The front scheme pins the same branch on [Melt drainage across the freezing front](compaction.md#analytic-limit-checked-alongside).

**Why it matters.** A real node spends many coupling steps in the mush. This case shows that the record loses no melt and no water between steps, and that what a node locks follows from the drainage law.

![Pore melt of a node limited by its crystal framework against the exponential decay, over 52 coupling steps](../../assets/validation/node_record_matrix_limited_node.svg#only-light){ width="95%" }
![Pore melt of a node limited by its crystal framework against the exponential decay, over 52 coupling steps](../../assets/validation/node_record_matrix_limited_node_dark.svg#only-dark){ width="95%" }

**How to read the plot.** The left panel shows the pore melt of the node over its entry pore melt, on a log scale, against time in the mush in units of $\tau_s$, here 346 kyr. The line is $e^{-t/\tau_s}$; the circles are the record at every third coupling step. The right panel shows their relative deviation, with the dashed bias $t h^2/6$ that the second-order sub-steps build up at $h = 0.058$, and the tolerance as a line. A pass means that the circles lie on the line and the deviation stays below the tolerance; it follows the dashed bias, so it is integration error.

??? info "Test details"
    - Reference: the matrix branch of the drainage equation, $\Phi_c e^{-t/\tau_s}$, with $\tau_s = \mu_s/(\Delta\rho\,g\,d)$ at the depth $d$ of the node's centre.
    - Tolerance: relative 5e-3 on the locked melt fraction; the record equals the sum of the steps' moves to relative 1e-10.
    - Test id: `tests/outgas/test_node_record.py::test_a_matrix_limited_node_traps_the_closed_form_share_of_its_melt`
    - Command: `pytest -m reference_pinned tests/outgas/test_node_record.py --validation-plots`

## A node that freezes within a step keeps what the continuous drainage keeps

**Why this check exists.** The coupling step of a run is set by the interior and the atmosphere, not by the trapping, and a node can freeze within one or a few steps. If the record held the porosity at one value over a step, or let melt that crystallised flow again in the next step, what a node locks would depend on how many steps its freezing took. The record did both before: it locked 0.93 of the water of this case in one step and 0.49 in 64.

**What it tests.** The top node of a mantle of 6 nodes enters the mush at a melt fraction of 0.3. Its porosity $\phi$ then falls linearly in time to 0 over 1000 yr, in 1, 2, 4, 16 or 64 coupling steps, with a 1 mm grain, a melt of 0.1 Pa s, a mush of $10^{19}$ Pa s and $D = 0.0017$. Each run must lock the share of its buried water that the continuous drainage keeps.

**How we know the right answer.** The continuous problem: the pore melt drains at its own rate relative to itself, the lesser of $w_D/L$ and $1/\tau_s$. While the porosity falls faster than the melt drains, the pore melt equals the porosity and the difference crystallises, its crystals keeping $D$ times the concentration; otherwise the melt drains freely below the porosity. SciPy's LSODA integrates the two regimes without coupling steps or sub-steps, with events where they switch (`tests/helpers/_layer_reference.py`). Its integration is checked against 3 closed forms in `tests/helpers/test_layer_reference.py`.

**Why it matters.** A node may spend only a few coupling steps in the mush. This case shows that what it locks depends on its drainage and its freezing, not on the coupling step.

![Water a node freezing within a step locks, against the number of coupling steps, beside the continuous drainage](../../assets/validation/node_record_freezing_within_a_step.svg#only-light){ width="95%" }
![Water a node freezing within a step locks, against the number of coupling steps, beside the continuous drainage](../../assets/validation/node_record_freezing_within_a_step_dark.svg#only-dark){ width="95%" }

**How to read the plot.** The left panel shows the water the node locks over the water it buried, against the number of coupling steps over its freezing. The line is the continuous drainage, 0.482; the dashed line is the same with crystals that keep the concentration of the melt they grow from, 0.684. The right panel shows the relative deviation of the record in units of 1e-3, with the tolerance as a line. A pass means that every circle lies below the tolerance. The deviation falls as the steps shrink, from 2.4e-3 in one step: the sub-steps of one step each span a tenth of the freezing.

??? info "Test details"
    - Reference: the continuous drainage of `tests/helpers/_layer_reference.py`, LSODA at a relative tolerance of 1e-10.
    - Tolerance: relative 5e-3 for every number of steps and 1e-4 at 64 steps; at 64 steps the water the pore crystals keep, the difference between $D = 0.0017$ and $D = 0$, lies within 10 % of the reference's.
    - Test id: `tests/outgas/test_node_record.py::test_a_node_freezing_within_a_step_keeps_what_the_continuous_drainage_keeps`
    - Command: `pytest -m reference_pinned tests/outgas/test_node_record.py --validation-plots`

## Rules checked alongside

The node-record tests also check these rules, which have no published reference:

- volatiles are conserved between the melt reservoir, the pore melt and the record over a step;
- a node entering the mush from above buries $C\,\Phi_c\,m$ in its pore melt, and crystals grown in liquid nodes take $D\,C$ of their mass, with $D$ the crystal/melt partition coefficient;
- one common factor scales every new burial when the melt holds too little, and the cap counts the melt reservoir plus what the same step returns to it;
- buried volatiles move with the drained melt; on the pore melt the record carries at the entry concentration a node keeps that concentration, while the pore melt itself is richer where part of it crystallised and passed its water on;
- a node that becomes liquid again returns its pore melt and its crystal store, species by species, noble gases included;
- a liquid node returns the share of its crystal store that melts, all of it at a melt fraction of 1, so cycles of cooling and heating do not pump volatiles into the record;
- a solid node that re-enters the mush from below starts with its own pore melt, not the pore melt of the transition, and buries nothing new from the ocean;
- a node that crosses the whole mush within one step, a mush layer under a solid layer, and the nodes at and below a dense-melt node keep their pore melt, the no-drainage bound;
- trapped mass the record does not hold is assigned to the nodes that are not liquid, and a record restarted on the previous step's melt fraction tells the nodes already in the mush from those that leave the ocean.

The round trip of the record through an interior snapshot is tested in `tests/outgas/test_node_record_io.py`.
