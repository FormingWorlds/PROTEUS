# Validation: `src/proteus/interior_chem/partitioning.py`

This page tracks the `@pytest.mark.reference_pinned` tests of the Fe-Mg exchange partitioning used by
`planet.interior_chem.partition_coefficients = 'mgsio3_kd'` of the melt-redox tracker
([explanation](../../Explanations/redox.md)). Last comparison: 2026-11-03.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/interior_chem/test_partitioning.py::test_bridgmanite_kd_matches_the_dragulet_stixrude_2024_fit` | Dragulet & Stixrude (2024), GRL 51, e2023GL107979 | Fig. 4 (fit parameters and 4000 K curve), Section 3 text | K_D at 4000 K between 0.25 and 0.33 at 25 GPa, at most 0.1 at 100 GPa, and 0.0486 at 136 GPa (the fit evaluated by hand, rel 2e-3). Guards: k_B in place of R underflows K_D to 0; a flipped sign of dS multiplies it by 18. |
| `tests/interior_chem/test_partitioning.py::test_the_two_fits_agree_with_the_triple_point_values_of_the_2026_paper` | Dragulet & Stixrude (2026), arXiv:2609.13527 | Section 3.1 (K_D 0.057 +- 0.007 bridgmanite, 0.086 +- 0.021 post-perovskite at 180 GPa) | At 180 GPa and 6430 K, the temperature at which the bridgmanite fit gives 0.057, bridgmanite within 0.002 of 0.057 and post-perovskite within the stated 0.021 of 0.086 (observed 0.0836). Guard: the step is upward by more than 0.02, so a swap of the two phases fails. |
| `tests/interior_chem/test_partitioning.py::test_small_batch_steps_converge_to_fractional_crystallisation` | Dragulet & Stixrude (2024) Eq. 12; (2026) Eq. 9 | Fractional-crystallisation ODE, written per formula unit in the (Mg,Fe)SiO3 basis and integrated with `scipy.integrate.solve_ivp` | 2000 batch steps of `fe2_into_solid_kd` at K_D = 0.05 reproduce the ODE melt X_Fe at 20% melt left to rel 2e-3; 5 steps miss it by more than 0.5%. |
| `tests/interior_chem/test_redox.py::test_mgsio3_kd_replaces_only_the_lower_mantle_ferrous_coefficient` | Dragulet & Stixrude (2024), Fig. 4 | Cross-implementation: hand-built D(Fe2+) from the published fit and the starting melt | At 30 GPa / 3600 K and 50 GPa / 4800 K the tracker's initial D(Fe2+) matches the hand-built value to rel 1e-6; shallow D(Fe2+) and every D(Fe3+) equal the `'mgsio3'` values. |

## What is not validated

- The fits themselves carry large, correlated uncertainties (post-perovskite dH +- 52%, dS +- 87%), and the
  covariance is not published; the tests pin central values only.
- Temperatures below 3000 K (bridgmanite) and below 6000 K (post-perovskite) are clamped, not extrapolated.
  The upper bridgmanite limit is 6500 K, 500 K above its fitted data, so that the 2026 paper's own triple-point
  value is reproduced.
- No coupled run with `'mgsio3_kd'` has been compared against a published Fe3+/FeT evolution: the K_D studies
  give no Fe3+ partitioning, and the D(Fe3+) of `'mgsio3'` is kept.
