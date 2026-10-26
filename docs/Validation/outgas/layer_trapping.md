# Validation: `src/proteus/outgas/layer_trapping.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor
`proteus.outgas.layer_trapping`, the trapping step of
`outgas.trap_mode = 'layers'` that moves the volatiles the per-node record
locks or returns between the melt and the solid reservoirs.

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/outgas/test_layer_trapping.py::test_undrained_layers_bury_the_front_schemes_no_drainage_bound` | Cross-implementation: `proteus.outgas.trapping.trapped_mass` of the front scheme at `F_tl = phi_c`, Sim, Hirschmann & Hier-Majumder (2024), JGR Planets 129, e2024JE008346, Eq. 6 | An 8-shell mantle frozen shell by shell from the base through `run_trapping`, at a fixed melt concentration, with a melt that cannot drain. The water it buries equals the front scheme's sum over the same steps to `rel=1e-10`, `(D + phi_c (1 - D)) C M_mantle` in all. Guards: the shells' own entry melt fraction 0.3 for `phi_c`, and a scheme without the crystals' share. With a 100 Pa s melt it buries less, but more than the crystals' share `D C M_mantle`. | 2026-10-26 |

## The no-drainage bound

### Plot: no-drainage bound

![Water buried by the layer scheme against the front scheme's no-drainage bound](../../assets/validation/layer_trapping_no_drainage_bound.svg#only-light){ width="80%" }
![Water buried by the layer scheme against the front scheme's no-drainage bound](../../assets/validation/layer_trapping_no_drainage_bound_dark.svg#only-dark){ width="80%" }

Trapped water over `C M_mantle` against the crystallised fraction of the mantle, with `phi_c = 0.5` and `D = 0.0017`. Line: the front scheme at its no-drainage bound. Circles: the layer scheme with a melt of `1e30` Pa s, which cannot drain. It locks a shell's pore melt as the shell enters the mush, one step ahead of the front scheme, which buries as the mantle crystallises; the two meet once every shell has frozen. Squares: a melt of 100 Pa s, which drains over the `1e4` yr each shell spends in the mush. The last step returns the pore melt of the top shell at the surface. Dotted: the crystals' share alone.

- Reference: Sim, Hirschmann & Hier-Majumder (2024), Eq. 6, through `trapped_mass`, validated on [Solid-phase volatile trapping](trapping.md).
- Tolerance: `rel=1e-10`.
- Test id: `tests/outgas/test_layer_trapping.py::test_undrained_layers_bury_the_front_schemes_no_drainage_bound`
- Command: `pytest -m reference_pinned tests/outgas/test_layer_trapping.py --validation-plots`

## Invariants checked alongside

The layer-trapping tests also assert, without a published reference:

- per-species and per-element closure `total = liquid + solid` after a freezing and after a remelting step;
- remelting decided per node: a shell that turns liquid returns its own record, while a shell that stays in the mush keeps its own as the global melt fraction rises;
- a net return flagged as a remelting step, for escape and desiccation;
- a return capped at the trapped reservoir;
- the shell profiles taken as the mean of the two basic-node values, with the surface gravity and the shell volume times density as fallbacks, and crystal partitioning alone without the phase-boundary densities;
- the record surviving a resume through the interior snapshot, and trapped mass from before the record assigned to the shells.

## Pending verification

A coupled comparison of the layer and front schemes on the same Earth-analogue
run, with a `trap_mode = 'none'` control, is not yet part of the test suite.
