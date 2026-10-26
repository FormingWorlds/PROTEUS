# Validation: `src/proteus/interior_chem/eos_deng.py`

This page tracks the `@pytest.mark.reference_pinned` tests of the pressure term of the Fe3+/Fe2+ equilibria:
the integral of the FeO -> FeO1.5 volume change of the melt, for the Deng et al. (2020) fourth-order
Birch-Murnaghan model (`model = 'deng2020'`, the default) and for the Keane + Deng-thermal lookup table
(`model = 'keane_table'`, `planet.interior_chem.redox_dV_eos`). The marker is registered in `pyproject.toml`. Last
comparison of the `keane_table` and Komabayashi tests: 2026-10-23; the comparison dates of the two `deng2020` tests were not
recorded when they were added.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/interior_chem/test_eos_deng.py::test_oxidation_integral_reproduces_the_deng_values_of_zhang_2024` | Zhang et al. (2024), Sci. Adv. 10, eadp1752 | Supplementary Fig. S5 (Deng 12.5 mol% FeO* values), Table S1 (P, T) | The `deng2020` oxidation integral at 38, 43 and 71 GPa within 3% of the values read from the figure. Guard: dropping the factor 1/2 doubles the value. |
| `tests/interior_chem/test_eos_deng.py::test_reaction_volume_change_is_positive_near_the_surface_and_negative_at_depth` | Schaefer et al. (2024), JGR Planets 129, e2023JE008262 | Metal-saturation results (no metal in the 500 km models, metal at the base of the whole-mantle models) | The disproportionation integral changes sign with depth at 2500 K. |
| `tests/interior_chem/test_eos_deng.py::test_lange_carmichael_feo_reproduces_the_archived_disproportionation_integral` | Schaefer et al. (2024) archived code, `deltaGFeOFeO15_Deng.m` | `VdPFeO_LC` (Lange & Carmichael 1987 volume, Murnaghan K0 = 30.33 GPa, K' = 4) | With `feo_volume = 'lange_carmichael'` the disproportionation integral at 5-130 GPa, 2000-4500 K within 2e-4 of a numpy port of the archived function (observed 6e-5, the liquid-Fe trapezoid). Guard: the Komabayashi FeO volume misses by more than 4 kJ/mol beyond 5 GPa (log10 K differs by +0.28 at 40 GPa, +2.11 at 100 GPa). Last compared 2026-10-26. |
| `tests/interior_chem/test_eos_deng.py::test_liquid_volumes_reproduce_komabayashi_2014_table3` | Komabayashi (2014), JGR Solid Earth 119, 4164 | Table 2 (Vinet + Anderson-Gruneisen parameters), Table 3 (end-member columns X_Fe = 1 and 0) | Liquid Fe and FeO volumes at 136 GPa, 4000 K and 330 GPa, 5000 K within 0.2 % of Table 3 (observed: 0.1 % at most). Guards: the Lange & Carmichael FeO volume of Schaefer's archived code is 14 % and 37 % too large there, and the volume without thermal expansion more than 1 % off. |
| `tests/interior_chem/test_eos_deng.py::test_keane_table_matches_deng_inside_the_fpmd_coverage` | Deng et al. (2020), Nat. Commun. 11, 2007, Fig. 1 (FPMD points) | Cross-implementation: `keane_table` against `deng2020` | Inside the FPMD coverage of the 12.5 mol% Fe3+ melt (3000 K to 74 GPa, 4000 K to 94 GPa), at 5-80 GPa, the two oxidation integrals agree to within 0.15 in int dV dP / (R T ln10) (0.113 at most observed). Guard: doubling the table value misses by more than 1 at 50 GPa and above. |

## The `keane_table` model

The table `FWL_DATA/interior_chem/keane_deng_table.npz` (1.9 MB, kept out of the repository by the 500 KB
pre-commit limit; the `keane_table` tests skip when it is absent) holds the oxidation integral of
P(V, T) = P_Keane(V) + (a - b x + c x^2)/1000 (T - 3000 K), x = V/V0, for the 12.5 mol% Fe melts
Mg14Fe2Si16O48 and Mg14Fe2Si16O49, on a grid uniform in log10 P (1 bar - 4 TPa) and log10 T
(1000 - 20000 K), 100 points per decade. The Keane cold curve (Attia et al. 2026, A&A 713, A225, Eq. 12) and the
Deng thermal term are fitted jointly to the 73 FPMD points read from the vector graphics of Deng et al. (2020)
Fig. 1 (2000, 3000 and 4000 K; the 2500 K isotherm listed in their Methods is not plotted or tabulated). The fit
reproduces those points with an RMS pressure residual of 0.09 GPa (Fe2+) and 0.08 GPa (Fe3+), against 0.47 and
0.10 GPa for the published Deng parameters. Bilinear interpolation in (log10 P, log10 T) reproduces direct
evaluation to 0.012 log units of fO2 at the 99th percentile (0.05 at most) on 300 random points.

Beyond the FPMD coverage the table is an extrapolation. Across plausible cold-curve and thermal forms that fit
the same points equally well, dV at 4 TPa varies by a factor of about 5, so values above about 100 GPa carry that
uncertainty. Above 4206 K the fit has no liquid at 1 bar; the table integral then evaluates dV at the hottest
temperature with a liquid volume along the low-pressure part of the path.

## The FeO and Fe terms of the disproportionation integral

`int_dV_dP` (3FeO = 2FeO1.5 + Fe) is 2 int dV_melt dP - int V_FeO dP + int V_Fe dP (Schaefer et al. 2024 Eqs 11-12).
Both liquids follow Komabayashi (2014): Vinet at 298 K (Eq. 3) with the Anderson-Gruneisen thermal expansivity
(Eq. 5) evaluated at the 298 K volume, Table 2 parameters (kappa = 0.56 for liquid Fe, 1.4 for liquid FeO), integrated
from 1 bar on 400 nodes. Liquid FeO replaces the Lange & Carmichael (1987) volume of Schaefer's archived code
(`VdPFeO_LC`), which follows their Eq. 11 text but not their code; the archived form overestimates V_FeO by 14 % at
136 GPa and 4000 K and grows without bound with temperature. Komabayashi's database extends to 350 GPa and 7000 K;
beyond that both terms are extrapolated and a warning is logged once per process.
