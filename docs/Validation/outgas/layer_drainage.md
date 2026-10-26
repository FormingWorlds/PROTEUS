# Validation: `src/proteus/outgas/layer_drainage.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor
`proteus.outgas.layer_drainage`, the drainage of pore melt through the mush,
node by node, of `outgas.trap_mode = 'layers'`.

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/outgas/test_layer_drainage.py::test_single_node_reproduces_the_front_drainage_integral` | Cross-implementation: `proteus.outgas.compaction.drainage_integral`, the front scheme's LSODA integral at `rtol=1e-8` | One node with `L = V / A` drains as the front scheme's parcel, `dphi/dt = -phi min(w_D / L, 1 / tau_s)`. Agreement to `rtol=1e-3` at ten times over five percolation times of a 1 Pa s melt, under each `separation_buoyancy`; the melt that left the node equals what it lost to `rel=1e-12`. | 2026-10-26 |
| `tests/outgas/test_layer_drainage.py::test_compaction_limited_node_decays_exponentially` | Analytic limit: `psi = psi0 exp(-t / tau_s)` | Where the matrix controls, the node loses melt at `1 / tau_s`. Agreement to `rel=2e-3` over three e-folds; a forward-Euler update at the same sub-step would land 7 percent low. | 2026-10-26 |
| `tests/outgas/test_layer_drainage.py::test_uniform_column_follows_the_kinematic_wave_rarefaction` | Analytic solution of `d(psi)/dt + d(K psi^3)/dz = 0` on an impermeable base | Before the rarefaction reaches the top, `M / M0 = 1 - K psi0^2 t / H`, matched to `rtol=1e-6` up to half the arrival time `t_a`; after it, `M / M0 = (2/3) sqrt(t_a / t)`, matched at `4 t_a` to 3 percent with 100 nodes, with the first-order convergence of upwind differencing. | 2026-10-26 |

## One node against the front integral

### Plot: single node

![One node of the layer drainage against the front scheme's drainage integral](../../assets/validation/layer_drainage_single_node.svg#only-light){ width="95%" }
![One node of the layer drainage against the front scheme's drainage integral](../../assets/validation/layer_drainage_single_node_dark.svg#only-dark){ width="95%" }

(a) Retained melt over the entry melt of one 60 km node of a 1 Pa s melt, entered at `psi0 = 0.3`: the front scheme's integral (lines) and the layer drainage (circles), under `separation_buoyancy = 'mixture'` and `'solid'`, against time in units of the node's percolation time at entry under that buoyancy. (b) Relative difference of the two. It stays near 3e-4, the error of the second-order sub-steps at a fortieth of the node timescale.

- Reference: `proteus.outgas.compaction.drainage_integral`, validated on [Melt drainage across the freezing front](compaction.md).
- Tolerance: `rtol=1e-3`.
- Test id: `tests/outgas/test_layer_drainage.py::test_single_node_reproduces_the_front_drainage_integral`
- Command: `pytest -m reference_pinned tests/outgas/test_layer_drainage.py --validation-plots`

## A column against the kinematic wave

### Plot: kinematic wave

![Retained melt of a uniform column against the kinematic-wave rarefaction](../../assets/validation/layer_drainage_kinematic_wave.svg#only-light){ width="80%" }
![Retained melt of a uniform column against the kinematic-wave rarefaction](../../assets/validation/layer_drainage_kinematic_wave_dark.svg#only-dark){ width="80%" }

Melt retained in a 100 km column at `psi0 = 0.3` with the flux `K psi^3` and no compaction limit, over the time `t_a = H / (3 K psi0^2)` at which the rarefaction fan from the impermeable base reaches the top (dotted). Line: the exact solution. Circles and squares: the layer drainage on 100 and 50 nodes. Upwind differencing smears the fan, so both meshes retain slightly more than the exact solution after `t_a`, the coarser more.

- Reference: the rarefaction `psi(z, t) = sqrt(z / (3 K t))` for `z < 3 K psi0^2 t` of the scalar conservation law.
- Tolerance: `rtol=1e-6` up to `0.5 t_a`, 3 percent at `4 t_a`.
- Test id: `tests/outgas/test_layer_drainage.py::test_uniform_column_follows_the_kinematic_wave_rarefaction`
- Command: `pytest -m reference_pinned tests/outgas/test_layer_drainage.py --validation-plots`

## Re-derivation note

The melt volume flux out of the top of node `k` of a contiguous run of mush
nodes is

```
q_k = min( A_k psi_k w_D(psi_k),  q_(k-1) + psi_k V_k / tau_s,k ),   q_(base) = 0
```

For a single node this is `d(psi V)/dt = -psi min(A w_D, V / tau_s)`, the
front scheme's equation with `L = V / A`, hence the cross-check. With
`tau_s = 0` the second term never binds, and a uniform column with
`w_D = K psi^2` is the scalar conservation law above.
