# Validation: `src/proteus/outgas/compaction.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor
`proteus.outgas.compaction`, the melt drainage of the front scheme.

| Test id | Reference | Scope |
|---|---|---|
| `tests/outgas/test_compaction.py::test_mobility_reproduces_the_interior_solver_in_every_regime` | Cross-implementation check against the interior solver: `aragog.eos.entropy_phase.EntropyPhaseEvaluator.relative_velocity` (fwl-aragog), the three-regime permeability of Bower et al. (2018), section 2.1 | Calls the solver's own separation velocity on a stand-in mush and backs the mobility out of it, for three grain sizes and porosities spanning all three regimes and both blends. Agreement is `rtol=2e-4` from a porosity of 0.05 up and `1.5e-3` below, where the solver's soft porosity clip (width 1e-3) lifts the porosity by about `1e-6 / (4 phi)`. Also pins the drainage velocity as the solver's velocity times `(1 - phi)`. |
| `tests/outgas/test_compaction.py::test_drainage_matches_an_independent_quadrature_over_five_martian_fronts` | Analytic limits and a reference calculation: the closed-form timescales `tau_D = L eta_m / ((1 - phi) drho g F(phi))` and `tau_s = mu_s / (drho g L)`; the separable form of the drainage equation, `t_res = int_F^phi_c max(tau_D, tau_s) dphi / phi`, evaluated by quadrature; and its series solution in the Rumpf-Gupte regime, `H(phi) = sum_k phi^(k - 4.5) / (k - 4.5)` falling by `C t_res / L` | A 60 km martian front entered at `phi_c = 0.3` (`drho = 330` kg/m3, `g = 3.711` m/s2), crossed at 70 and 5 cm/yr, for five sets of grain size, melt viscosity and mush viscosity. The shipped model, `(1 - phi)` included, matches the quadrature to `rel=1e-6` in all ten trapped fractions and the series to `1e-5` in the three Rumpf-Gupte cases; the fractions are pinned at `rel=1e-4` (0.27165 and 0.18134 down to 0.0094342 and 0.0025379) and the timescale ratios likewise (`tau_D / t_res` from 8.1653 to 8.1653e-5, `tau_s / t_res` 0.50314, 50.314 and 0.0050314). Dropping `(1 - phi)` from `w_D` shifts the percolation-limited fractions by more than 0.007. |

## Analytic limit checked alongside

`test_drainage_is_limited_by_the_slower_of_the_two_processes` pins the
matrix-limited case to its closed form `phi_c exp(-t_res / tau_s)` at
`rel=1e-6`.
