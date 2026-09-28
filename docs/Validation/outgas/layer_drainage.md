# Validation: `src/proteus/outgas/layer_drainage.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
per-node layer drainage of pore melt (`outgas.trap_drainage = 'layers'`)
against an analytic limit, the analytic solution of the kinematic wave, and
the front scheme's drainage integral.

| Test id | Reference | Scope |
|---|---|---|
| `tests/outgas/test_layer_drainage.py::test_single_node_reproduces_the_front_drainage_integral` | Cross-implementation: `proteus.outgas.compaction.drainage_integral` (LSODA, rtol 1e-8) | One node with `L = V / A` drains exactly as the front scheme's parcel, `dphi/dt = -phi min(w_D / L, 1 / tau_s)`. Agreement to 1e-3 over five percolation times; also checks that the melt leaving the node equals what it lost. |
| `tests/outgas/test_layer_drainage.py::test_compaction_limited_node_decays_exponentially` | Analytic limit: `psi = psi0 exp(-t / tau_s)` | Where the matrix controls, the node loses melt at `1 / tau_s`. Agreement to 2e-3 over three e-folds; a forward-Euler update at the same sub-step would miss by 7 percent. |
| `tests/outgas/test_layer_drainage.py::test_uniform_column_follows_the_kinematic_wave_rarefaction` | Analytic solution of `d(psi)/dt + d(K psi^3)/dz = 0` on an impermeable base | Before the rarefaction reaches the top, `M / M0 = 1 - K psi0^2 t / H` to rounding; after it, `M / M0 = (2/3) sqrt(t_a / t)`, reached to 3 percent with 100 nodes, with the first-order convergence of upwind differencing and a retention that coarse meshes overstate. |

## Re-derivation note

The melt volume flux out of the top of node `k` of a contiguous run of mush
nodes is

```
q_k = min( A_k psi_k w_D(psi_k),  q_(k-1) + psi_k V_k / tau_s,k ),   q_(base) = 0
```

For a single node this is `d(psi V)/dt = -psi min(A w_D, V / tau_s)`, which is
the front scheme's equation with `L = V / A`, hence the cross-check. With
`tau_s = 0` the second term never binds and a uniform column with `w_D = K psi^2`
is the scalar conservation law above, whose rarefaction from the impermeable
base is `psi(z, t) = sqrt(z / (3 K t))` for `z < 3 K psi0^2 t`.

## Pending verification

A coupled comparison of the layer and front schemes on the same Earth-analogue
run, with a `trap_mode = 'none'` control, is not yet part of the test suite.
