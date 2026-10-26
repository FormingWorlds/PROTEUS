# Validation: `src/proteus/outgas/layer_state.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor
`proteus.outgas.layer_state`, the per-node record of the layer drainage: the
pore melt of every node and the volatiles locked and buried at it.

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/outgas/test_layer_state.py::test_a_matrix_limited_node_traps_the_closed_form_share_of_its_melt` | Analytic limit of the front scheme's matrix branch, `F = phi_c exp(-t_res / tau_s)`, with `tau_s = mu_s / (drho g d)` | The top shell of a 6-shell mantle enters the mush at `phi_c = 0.5`, spends three matrix times there over 52 coupling steps and freezes. It reaches the floor with `0.5 exp(-3)` of its mass as melt, to `rel=5e-3`, and buries that share of the water it locked; the record equals the sum of the steps' moves to `rel=1e-10`. The tolerance covers the 1.7e-3 bias of the second-order sub-steps. Guards: the depth of the node's base halves `tau_s`, and forward Euler at the same steps ends 8 percent low. | 2026-10-26 |

## A matrix-limited node

### Plot: matrix-limited node

![Retained pore melt of a matrix-limited node against the exponential decay](../../assets/validation/layer_state_matrix_limited_node.svg#only-light){ width="95%" }
![Retained pore melt of a matrix-limited node against the exponential decay](../../assets/validation/layer_state_matrix_limited_node_dark.svg#only-dark){ width="95%" }

(a) Retained pore melt of the node over its entry melt, against time in the mush in units of `tau_s` (346 kyr), with a 1 cm grain, a 0.1 Pa s melt and a mush of `1e22` Pa s: the record at every third coupling step (circles) and `exp(-t / tau_s)` (line). The last point is the trapped fraction at the step the node freezes. (b) Relative deviation from the exponential, against the bias `t h^2 / 6` that the second-order Runge-Kutta scheme accumulates at sub-steps of `h = 0.058 tau_s` (dashed). The deviation follows that bias, so it is integration error and not a modelling difference.

- Reference: the matrix-limited branch of the drainage equation, as the front scheme pins it on [Melt drainage across the freezing front](compaction.md#analytic-limit-checked-alongside).
- Tolerance: `rel=5e-3`.
- Test id: `tests/outgas/test_layer_state.py::test_a_matrix_limited_node_traps_the_closed_form_share_of_its_melt`
- Command: `pytest -m reference_pinned tests/outgas/test_layer_state.py --validation-plots`

## Invariants checked alongside

The layer-state tests also assert, without a published reference:

- volatiles conserved between the melt reservoir, the pore melt and the buried record over a step;
- locked volatiles moving with the drained melt at their entry concentration;
- a node that becomes liquid again returning its pore melt and its buried mass, species by species, noble gases included;
- a solid node re-entering the mush from below locking nothing new;
- a node that crosses the whole mush within one step, or a sealed run, keeping its entry pore melt, the no-drainage bound;
- trapped mass the record does not hold assigned to the nodes that are not liquid;
- the record round-tripping through an interior snapshot.
