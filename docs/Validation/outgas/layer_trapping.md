# Validation: `src/proteus/outgas/layer_trapping.py`

## What this page checks

This page checks the trapping step of the layer scheme, `outgas.trap_mode = "layers"` (`proteus.outgas.layer_trapping`). Each step, it runs the per-node record on the Aragog mesh and moves the volatiles the record buries or returns between the melt and the solid mantle. It also hands the chemistry the melt of the magma ocean alone, the melt the volatiles can still dissolve into.

Validation here means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. The answer here comes from a second code, the front scheme of the same model. If this part of the code were wrong, a run would trap the wrong amount of water in its mantle, or move it without conserving the mass of each element.

The case has the layout of every Validation page: why the check exists, what it tests, how we know the right answer, why it matters, a plot, and a box "Test details". Symbols used on this page:

- $\Phi_c$: the melt fraction at which a node enters the mush from the magma ocean, `interior_energetics.rfront_loc` \[1\];
- $D$: the crystal/melt partition coefficient of water, the ratio of its mass fraction in the crystals to that in the melt \[1\];
- $C$: the mass fraction of water in the melt \[1\];
- $M_\mathrm{mantle}$: the mass of the mantle \[kg\];
- $F_\mathrm{tl}$: the trapped melt fraction of the front scheme, the share of a newly frozen mass that stays as melt between the crystals \[1\].

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/outgas/test_layer_trapping.py::test_undrained_mush_traps_the_front_schemes_no_drainage_bound` | Second code: `trapped_mass` of the front scheme at $F_\mathrm{tl} = \Phi_c$, Sim et al. (2024)[^cite-sim2024], Eq. 6 | A mantle of 8 nodes frozen node by node from the base, with a melt that cannot drain and with one that drains. | 2026-11-20 |

## A mush that cannot drain traps the front scheme's no-drainage bound

**Why this check exists.** When the melt cannot leave the mush, every node keeps the melt it had when it entered the mush, and both schemes must trap the same water. An error in the burial on entry, in the crystals' share, or in how the step moves the record's mass into the solid mantle would change the total. Using the nodes' own entry melt fraction, 0.3, in place of $\Phi_c$ misses by more than 30 %, and leaving out the crystals' share misses by more than 0.1 %; the test fails on both.

**What it tests.** A mantle of 8 nodes starts molten. Each step, the next node from the base enters the mush at a melt fraction of 0.3 and the node below it freezes. Before each step the melt is reset to hold water at a fixed $C$. With a melt of $10^{30}$ Pa s, which cannot drain, the water trapped once every node has frozen must equal the front scheme's sum over the same steps, $(D + \Phi_c(1 - D))\,C\,M_\mathrm{mantle}$. With a melt of 0.1 Pa s, the same mantle must trap less, but more than the crystals' share $D\,C\,M_\mathrm{mantle}$.

**How we know the right answer.** The front scheme traps $[(1 - F_\mathrm{tl})D + F_\mathrm{tl}]\,C$ of each frozen mass, Eq. 6 of Sim et al. (2024)[^cite-sim2024]. At $F_\mathrm{tl} = \Phi_c$, no melt drains, so it is the bound the layer scheme must meet when its melt cannot move. The front scheme's `trapped_mass` is checked on [Solid-phase volatile trapping](trapping.md).

**Why it matters.** The layer scheme must agree with the front scheme where both describe the same physics. This case shows that a run's extra trapping in the layer scheme comes from how its mush drains, not from its bookkeeping.

![Water trapped by the layer scheme against the front scheme's no-drainage bound](../../assets/validation/layer_trapping_no_drainage_bound.svg#only-light){ width="80%" }
![Water trapped by the layer scheme against the front scheme's no-drainage bound](../../assets/validation/layer_trapping_no_drainage_bound_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The plot shows the trapped water over $C\,M_\mathrm{mantle}$ against the crystallised fraction of the mantle, with $\Phi_c = 0.5$ and $D = 0.0017$. The line is the front scheme at its bound. Circles are the layer scheme with a melt that cannot drain; it buries a node's pore melt as the node enters the mush, one step before the front scheme traps it, so the two meet once every node has frozen. Squares are the layer scheme with a melt of 0.1 Pa s; in the step each node freezes, the melt beyond its falling melt fraction crystallises in place and only the rest drains. The dotted line is the crystals' share alone. A pass means that the last circle lies on the line, and the last square lies between the line and the dotted line.

??? info "Test details"
    - Reference: Sim et al. (2024)[^cite-sim2024], Eq. 6, through `proteus.outgas.trapping.trapped_mass`.
    - Tolerance: relative 1e-10 for the melt that cannot drain; the draining melt must trap between $D\,C\,M_\mathrm{mantle}$ and 0.9 of the undrained total.
    - Test id: `tests/outgas/test_layer_trapping.py::test_undrained_mush_traps_the_front_schemes_no_drainage_bound`
    - Command: `pytest -m reference_pinned tests/outgas/test_layer_trapping.py --validation-plots`

## Rules checked alongside

The layer-trapping tests also check these rules, which have no published reference:

- each species and element closes, total = liquid + solid, after a freezing and after a remelting step;
- remelting is decided per node: a node that turns liquid returns its own record, while a node that stays in the mush keeps its own as the global melt fraction rises;
- node masses are rescaled to the mantle mass, so masses that sum to 1.4 $M_\mathrm{mantle}$ bury the right amount;
- the chemistry gets the melt of the liquid nodes for its solve, and the global melt fraction is restored after it, also after an error; a row no layer step wrote, which holds 0 in `trap_Phi_ocean` and `trap_n_mush`, leaves the global melt fraction to the solve;
- the next step takes its melt concentration on that ocean melt, with the mantle mass of the previous solve;
- the mush drains under the `separation_buoyancy` of the Aragog configuration;
- every mass a step returns, by remelting or by drainage, is published in `trap_kg_returned` for escape and desiccation, also on a step that buries more than it returns;
- a return is capped at the trapped reservoir;
- without the phase-boundary densities, the step traps by crystal partitioning alone on freezing and returns the remelted share on remelting, without the burial ledger, so nothing returns twice;
- the diagnostics of a step: the nodes that froze, their trapped fraction and residence, and the size, thickness, front speed and Courant number of the thickest mush layer;
- the record survives a resume through the interior snapshot; without a stored record, it restarts on the melt fraction of the snapshot, so a node that leaves the ocean in the first step buries as in a run that kept its record, also when the mesh changed.

## Pending verification

A coupled comparison of the layer and front schemes on the same Earth-analogue run, with a `trap_mode = "none"` control, is not part of the test suite.

[^cite-sim2024]: S. J. Sim, M. M. Hirschmann, S. Hier-Majumder, *[Volatile and trace element storage in a crystallizing martian magma ocean](https://doi.org/10.1029/2024JE008346)*, Journal of Geophysical Research: Planets, 129, e2024JE008346, 2024.
