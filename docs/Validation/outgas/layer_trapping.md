# Validation: `src/proteus/outgas/layer_trapping.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor
`proteus.outgas.layer_trapping`, the trapping step of
`outgas.trap_mode = 'layers'`. It moves the volatiles the per-node record
buries or returns between the melt and the solid reservoirs.

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/outgas/test_layer_trapping.py::test_undrained_mush_traps_the_front_schemes_no_drainage_bound` | Cross-implementation: `proteus.outgas.trapping.trapped_mass` of the front scheme at `F_tl = Phi_c`, Sim, Hirschmann & Hier-Majumder (2024), JGR Planets 129, e2024JE008346, Eq. 6 | An 8-node mantle frozen node by node from the base through `run_trapping`, at a fixed melt concentration, with a melt that cannot drain. The water it traps equals the front scheme's sum over the same steps to `rel=1e-10`, `(D + Phi_c (1 - D)) C M_mantle` in all. Guards: the nodes' own entry melt fraction 0.3 for `Phi_c`, and a scheme without the crystals' share. With a 0.1 Pa s melt it traps less, but more than the crystals' share `D C M_mantle`. | 2026-10-27 |

## The no-drainage bound

### Plot: no-drainage bound

![Water trapped by the layer scheme against the front scheme's no-drainage bound](../../assets/validation/layer_trapping_no_drainage_bound.svg#only-light){ width="80%" }
![Water trapped by the layer scheme against the front scheme's no-drainage bound](../../assets/validation/layer_trapping_no_drainage_bound_dark.svg#only-dark){ width="80%" }

Trapped water over `C M_mantle` against the crystallised fraction of the mantle, with `Phi_c = 0.5` and `D = 0.0017`. Line: the front scheme at its no-drainage bound. Circles: the layer scheme with a melt of `1e30` Pa s, which cannot drain. It buries a node's pore melt as the node enters the mush, one step ahead of the front scheme, which traps as the mantle crystallises. The two meet once every node has frozen. Squares: a melt of 0.1 Pa s, which drains in the step each node freezes. The last step returns the pore melt of the top node at the surface. Dotted: the crystals' share alone.

- Reference: Sim, Hirschmann & Hier-Majumder (2024), Eq. 6, through `trapped_mass`, validated on [Solid-phase volatile trapping](trapping.md).
- Tolerance: `rel=1e-10`.
- Test id: `tests/outgas/test_layer_trapping.py::test_undrained_mush_traps_the_front_schemes_no_drainage_bound`
- Command: `pytest -m reference_pinned tests/outgas/test_layer_trapping.py --validation-plots`

## Invariants checked alongside

The layer-trapping tests also assert, without a published reference:

- per-species and per-element closure `total = liquid + solid` after a freezing and after a remelting step;
- remelting decided per node: a node that turns liquid returns its own record, while a node that stays in the mush keeps its own as the global melt fraction rises;
- node masses rescaled to the mantle mass, so masses that sum to 1.4 `M_mantle` bury the true amount;
- the chemistry handed the ocean melt of the liquid nodes for its solve and the global melt fraction restored after it, also after an error; the next step takes its concentration on that ocean melt;
- the mass a step returns, by remelting or by drainage, published in `trap_kg_returned` for escape and desiccation;
- a return capped at the trapped reservoir;
- without the phase-boundary densities, crystal partitioning alone on freezing and the remelted share on remelting, without the burial ledger, so nothing returns twice;
- the diagnostics of a step: the nodes that froze, their trapped fraction and residence, and the thickest mush layer's size, thickness, front speed and Courant number;
- the record surviving a resume through the interior snapshot, trapped mass from before the record assigned to the nodes, and the same trapped mass whether a run continues, resumes without a record or changes its mesh.

## Pending verification

A coupled comparison of the layer and front schemes on the same Earth-analogue
run, with a `trap_mode = 'none'` control, is not part of the test suite.
