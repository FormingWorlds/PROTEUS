# Validation: `src/proteus/outgas/node_record.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor
`proteus.outgas.node_record`, the per-node record of the layer scheme: the
pore melt of every node and the volatiles buried and locked at it.

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/outgas/test_node_record.py::test_a_matrix_limited_node_traps_the_closed_form_share_of_its_melt` | Analytic limit of the front scheme's matrix branch, `F = Phi_c exp(-t_res / tau_s)`, with `tau_s = mu_s / (drho g d)` | The top node of a 6-node mantle enters the mush at `Phi_c = 0.5` and stays at that melt fraction for three matrix times, over 52 coupling steps at the production sub-step 0.1, then freezes. It reaches the floor with `0.5 exp(-3)` of its mass as melt, to `rel=5e-3`, and locks that share of the water it buried. The record equals the sum of the steps' moves to `rel=1e-10`. The tolerance covers the 1.7e-3 bias of the second-order sub-steps. Guards: the depth of the node's base halves `tau_s`, and forward Euler at the same steps ends 8 percent low. | 2026-10-27 |

## A matrix-limited node

### Plot: matrix-limited node

![Pore melt of a matrix-limited node against the exponential decay](../../assets/validation/node_record_matrix_limited_node.svg#only-light){ width="95%" }
![Pore melt of a matrix-limited node against the exponential decay](../../assets/validation/node_record_matrix_limited_node_dark.svg#only-dark){ width="95%" }

(a) Pore melt of the node over its entry melt, against time in the mush in units of `tau_s` (346 kyr). The node has a 1 cm grain, a 0.1 Pa s melt and a mush of `1e22` Pa s. Circles: the record at every third coupling step. Line: `exp(-t / tau_s)`. (b) Relative deviation from the exponential, against the bias `t h^2 / 6` that the second-order Runge-Kutta scheme builds up at sub-steps of `h = 0.058 tau_s` (dashed). The deviation follows that bias, so it is integration error and not a modelling difference.

- Reference: the matrix-limited branch of the drainage equation, as the front scheme pins it on [Melt drainage across the freezing front](compaction.md#analytic-limit-checked-alongside).
- Tolerance: `rel=5e-3`.
- Test id: `tests/outgas/test_node_record.py::test_a_matrix_limited_node_traps_the_closed_form_share_of_its_melt`
- Command: `pytest -m reference_pinned tests/outgas/test_node_record.py --validation-plots`

The node stays just below the transition melt fraction, so the solver porosity never caps its pore melt and the closed form applies.

## Invariants checked alongside

The node-record tests also assert, without a published reference:

- volatiles conserved between the melt reservoir, the pore melt and the record over a step;
- each store taking its settled burial: `C F m` in the pore melt of an entering node and `D C` of the crystals grown in liquid nodes;
- one common factor on every new burial when the melt holds too little;
- buried volatiles moving with the drained melt at their entry concentration;
- a node that becomes liquid again returning its pore melt and its crystal store, species by species, noble gases included;
- a liquid node returning the share of its crystal store that melts, all of it at `x = 1`, so cycles of cooling and heating do not pump volatiles into the record;
- a mush node, and a solid node re-entering the mush from below, moving the melted share of its crystal store into its pore melt, which then drains;
- a node that crosses the whole mush within one step, a sealed mush layer, and the nodes at and below a dense-melt node keeping their pore melt, the no-drainage bound;
- trapped mass the record does not hold assigned to the nodes that are not liquid.

The round trip of the record through an interior snapshot is tested in `tests/outgas/test_node_record_io.py`.
