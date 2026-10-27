# Validation: `src/proteus/outgas/layer_drainage.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor
`proteus.outgas.layer_drainage`, the drainage of pore melt through the mush,
node by node, of `outgas.trap_mode = 'layers'`. Every case runs at the
production sub-step, `SUBSTEP_FRACTION = 0.1`.

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/outgas/test_layer_drainage.py::test_single_node_reproduces_the_front_drainage_integral` | Cross-implementation: `proteus.outgas.compaction.drainage_integral`, the front scheme's LSODA integral at `rtol=1e-8` | One node with `L = V / A` on the percolation branch, its matrix time set to zero, drains as the front scheme's parcel, `dphi/dt = -phi w_D / L`. Agreement to `rtol=5e-3` at ten times over five percolation times of a 1 Pa s melt, under each `separation_buoyancy`; the largest difference is 3.3e-3. The melt that left the node equals what it lost to `rel=1e-12`. | 2026-10-27 |
| `tests/outgas/test_layer_drainage.py::test_compaction_limited_node_decays_exponentially` | Analytic limit: `psi = psi0 exp(-t / tau_s)` | Where the matrix controls, the node loses melt at `1 / tau_s`. Agreement to `rel=7e-3` over three e-folds in 30 sub-steps; the difference is 5.4e-3. A forward-Euler update at the same sub-step would land 15 percent low. | 2026-10-27 |
| `tests/outgas/test_layer_drainage.py::test_uniform_column_follows_the_kinematic_wave_rarefaction` | Analytic solution of `d(psi)/dt + d(K psi^3)/dz = 0` on an impermeable base | Before the rarefaction reaches the top, `M / M0 = 1 - K psi0^2 t / H`, matched to `rtol=1e-6` up to half the arrival time `t_a`. After it, `M / M0 = (2/3) sqrt(t_a / t)`, matched at `4 t_a` to 3 percent with 100 nodes, with the first-order convergence of upwind differencing. | 2026-10-27 |

## One node against the front integral

### Plot: single node

![One node of the layer drainage against the front scheme's drainage integral](../../assets/validation/layer_drainage_single_node.svg#only-light){ width="95%" }
![One node of the layer drainage against the front scheme's drainage integral](../../assets/validation/layer_drainage_single_node_dark.svg#only-dark){ width="95%" }

(a) Pore melt over the entry melt of one 60 km node of a 1 Pa s melt, entered at `psi0 = 0.3`: the front scheme's integral (lines) and the layer drainage (circles). Both are shown under `separation_buoyancy = 'mixture'` and `'solid'`, against time in units of the node's percolation time at entry under that buoyancy. (b) Relative difference of the two, at most 3.3e-3, the error of the second-order sub-steps at a tenth of the node timescale. The band is the tolerance.

- Reference: `proteus.outgas.compaction.drainage_integral`, validated on [Melt drainage across the freezing front](compaction.md).
- Tolerance: `rtol=5e-3`.
- Test id: `tests/outgas/test_layer_drainage.py::test_single_node_reproduces_the_front_drainage_integral`
- Command: `pytest -m reference_pinned tests/outgas/test_layer_drainage.py --validation-plots`

This case covers the percolation branch only. On the matrix branch a single node does not match the front: its matrix time uses the depth of its centre, `L / 2`, so it is twice the front's `mu_s / (drho g L)`. The matrix branch is checked against its own analytic limit below and on [the node record page](node_record.md).

## A compaction-limited node

The permeability of one node is made so large that it never binds, so the
matrix alone sets the rate. Three e-folds are integrated in 30 sub-steps of
`h = 0.1 tau_s`. The second-order error, `30 h^3 / 6 = 5.0e-3`, matches the
measured 5.4e-3.

- Reference: the analytic decay `psi0 exp(-t / tau_s)`.
- Tolerance: `rel=7e-3`.
- Test id: `tests/outgas/test_layer_drainage.py::test_compaction_limited_node_decays_exponentially`
- Command: `pytest -m reference_pinned tests/outgas/test_layer_drainage.py`

## A column against the kinematic wave

### Plot: kinematic wave

![Pore melt of a uniform column against the kinematic-wave rarefaction](../../assets/validation/layer_drainage_kinematic_wave.svg#only-light){ width="80%" }
![Pore melt of a uniform column against the kinematic-wave rarefaction](../../assets/validation/layer_drainage_kinematic_wave_dark.svg#only-dark){ width="80%" }

Melt left in a 100 km column at `psi0 = 0.3` with the flux `K psi^3` and no compaction limit. Time runs in units of `t_a = H / (3 K psi0^2)`, when the rarefaction fan from the impermeable base reaches the top (dotted). Line: the exact solution. Circles and squares: the layer drainage on 100 and 50 nodes. Upwind differencing smears the fan, so both meshes keep slightly more than the exact solution after `t_a`, the coarser more.

- Reference: the rarefaction `psi(z, t) = sqrt(z / (3 K t))` for `z < 3 K psi0^2 t` of the scalar conservation law.
- Tolerance: `rtol=1e-6` up to `0.5 t_a`, 3 percent at `4 t_a`.
- Test id: `tests/outgas/test_layer_drainage.py::test_uniform_column_follows_the_kinematic_wave_rarefaction`
- Command: `pytest -m reference_pinned tests/outgas/test_layer_drainage.py --validation-plots`

## Invariants checked alongside

The drainage tests also assert, without a published reference:

- the melt of a column plus what left it conserved, and every pore melt kept in `[0, 1]`;
- volatiles carried at the concentration of the node they leave, so a column of uniform concentration keeps it as it drains;
- pore melt beyond the solver porosity not flowing;
- a dense-melt node sealing itself and the mush nodes below it in its mush layer.

## Re-derivation note

The melt volume flux out of the top of node `k` of a mush layer is

```
q_k = min( A_k psi_k w_D(psi_k),  q_(k-1) + psi_k V_k / tau_s,k ),   q_(base) = 0
```

For a single node with `tau_s = 0` the second term never binds, and this is
`d(psi V)/dt = -psi A w_D`, the front scheme's percolation equation with
`L = V / A`, hence the cross-check. A uniform column with `w_D = K psi^2` and
`tau_s = 0` is the scalar conservation law above.
