# Validation: `src/proteus/interior_chem/redox.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
melt Fe3+/Fe2+ tracker against Schaefer, Pahlevan and Elkins-Tanton (2024),
JGR Planets 129, e2023JE008262, their archived code and output (Stanford
Digital Repository, Schaefer 2023), and independent calibrations. The marker
is registered in `pyproject.toml`. Last comparison: 2026-10-22.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/interior_chem/test_redox.py::test_crystallisation_step_replays_the_schaefer_2024_run` | Schaefer et al. (2024) archived Earth code, `output.mat` of the 1000 km model, initial Fe3+/FeT = 0.1, run 570 | Section 2.5, Table 3, Table S2; fixture `tests/interior_chem/fixtures/schaefer2024_1000km_replay.csv` | Feeds the bulk solid/melt D of FeO and FeO1.5 and the solid fraction of each of the 96 steps of that run to `_fe_into_new_solid` and checks that the liquid Fe3+/FeT matches at every step to 1e-10 (8e-14 observed), and that the fixture is Schaefer's equilibrium mass balance (to 1e-10). Guards: D times the melt concentration before the step misses by 1.7e-3, Rayleigh within each step by 8e-4, and a constant D(Fe2+) = 0.85, or the two D swapped, by more than 0.05. |
| `tests/interior_chem/test_redox.py::test_bridgmanite_coefficients_follow_schaefer_2024_in_the_lower_mantle` | Same run | Section 3.1, p. 11 (Fe3+/FeT flat to decreasing across the lower mantle and transition zone) | The tracker's own bridgmanite coefficients on Schaefer's crystallisation sequence: at P >= 22 GPa Fe3+/FeT falls monotonically, by 60-100 % of the drop in their run (74 %), within 5e-3 of it. Guard: D(Fe2+) = 0.85 makes it rise instead. |
| `tests/interior_chem/test_redox.py::test_pyroxene_coefficients_give_the_shallow_rise_of_schaefer_2024` | Same run | Section 3.1, p. 11 (Fe3+/FeT increases below 15 GPa) | The tracker's pyroxene coefficients make Fe3+/FeT rise monotonically from 15 to 2.5 GPa, by 1-3 times the rise in their run (2.3 times); the minimum lies between their 15 GPa and the 22 GPa cutoff. Guard: the pyroxene D(FeO) of their model run (0.345) gives a fall. |
| `tests/interior_chem/test_redox.py::test_bulk_prescription_follows_the_schaefer_2024_run` | Same run | Fig. S3 (bulk D(Fe3+) per layer); Section 3.1 | With `planet.interior_chem.partition_coefficients = "bulk_BSE"` the tracker follows run 570 within 0.01 at every step (their 1000-draw mean within 0.005) and turns at 12-18 GPa as their run does. Guard: the `mgsio3` coefficients miss by more than 0.02 and turn at 22 GPa. |
| `tests/interior_chem/test_redox.py::test_the_pressure_term_moves_metal_formation_away_from_the_coolest_cell` | Schaefer et al. (2024) | Section 3.1.1 | Metal forms at depth, not in the shallow cool melt, once the reaction volume change is included. |
| `tests/interior_chem/test_redox.py::test_hirschmann_iw_buffer_matches_oneill_eggins_at_one_bar_and_the_solid_volume_slope` | Hirschmann (2021) against O'Neill and Eggins (2002) | Eq 11 of O'Neill and Eggins (2002) | The IW buffer at 1 bar and its pressure slope from the solid molar volumes. |

## Partition coefficients

The tracker uses only the MgSiO3 phases: bridgmanite at P >= 22 GPa and a
1:1 mix of clinopyroxene and orthopyroxene below. All D are solid/melt
ratios of mass concentration, the basis of Schaefer et al. (2024).

| Phase | D(Fe2+) | Source | D(Fe3+) | Source |
|---|---|---|---|---|
| Bridgmanite | 0.44 +- 0.19 | Table S1 (experimental D(FeO), n = 20) | 0.75 | Table 3, from Boujibar et al. (2016), Huang et al. (2023), Kuwahara and Nakada (2023) |
| Clinopyroxene | 0.38 +- 0.23 | Table S1 (n = 40; Mallmann and O'Neill 2009) | 0.45 | Table 3, Mallmann and O'Neill (2009) |
| Orthopyroxene | 0.69 +- 0.46 | Table S1 (n = 20; Mallmann and O'Neill 2009) | 0.70 x D_cpx | Table 3, Eq 6 |
| Cpx/Opx 1:1 | 0.535 | Mean; equal Cpx and Opx fractions in their 2.5-15 GPa layer | 0.383 | Mean |

The sign of D(Fe3+) - D(Fe2+) decides whether a crystallising phase lowers
or raises the melt Fe3+/FeT. It is positive for bridgmanite (0.31) and
negative for the pyroxene mix (-0.15), which gives the fall in the lower
mantle and the rise above it of Schaefer et al. (2024, Section 3.1).

Schaefer et al. (2024) use Table S1 only to derive the exchange
coefficients Kd(Fe/Mg) = D(FeO)/D(MgO) of their Table S2; their model
applies the Kd, and D(FeO) follows from the mass balance against the
magma-ocean liquid. In their archived 1000 km run (f_0 = 0.1) the median
D(FeO) is 0.385 for bridgmanite, 0.33 for Cpx and 0.36 for Opx, lower than
Table S1 because their liquid is richer in MgO than the experimental melts
(D(MgO) of Opx 1.24 there against 2.36 in Table S1). With those values the
pyroxene mix has D(Fe2+) = 0.345 < D(Fe3+) and Fe3+/FeT falls at every
depth, so they do not give the shallow rise. Both sets lie within the
Table S1 uncertainties, which straddle D(Fe3+) for the pyroxenes: the sign
of the shallow trend is not constrained by these data. The bulk
D(FeO) = 0.85 they quote in Section 2.5 is for fertile upper-mantle
peridotite, not for any of these minerals.

## The `bulk_BSE` prescription

`planet.interior_chem.partition_coefficients = "bulk_BSE"` gives each cell the bulk D of
the full mineral assemblage of its Schaefer et al. (2024) layer, the medians
over all 1000 draws of their archived 1000 km run:

| Layer | D(Fe2+) | D(Fe3+) | D(Fe3+), Table 3 x mineral fractions |
|---|---|---|---|
| P >= 22 GPa | 0.45 | 0.83 | 0.71 |
| 18-22 GPa | 0.59 | 1.60 | 1.59 |
| 15-18 GPa | 0.42 | 1.14 | 1.14 |
| 2.5-15 GPa | 0.39 | 0.30 | 0.29 |
| P < 2.5 GPa | 0.97 | 0.76 | depends on the spinel temperature |

D(Fe3+) matches their Fig. S3. In the lower mantle it lies above the Table 3
value because their draws of D(Fe3+) are truncated at zero. The medians use
all draws: restricting them to the 469 draws whose mass balance converges at
every step biases D(Fe3+) (0.96 in the lower mantle), because convergence
depends on the draw. On the crystallisation sequence of run 570 this
prescription stays within 0.0096 of that run and within 0.0045 of their
1000-draw mean at every step.

## Offset of the `mgsio3` prescription

Schaefer's layers also hold magnesiowuestite, majorite, ringwoodite,
wadsleyite, olivine, garnet and spinel. With the coefficients above on the
crystallisation sequence of run 570, the liquid Fe3+/FeT follows the shape
of that run (a fall, then a rise) but differs from it by:

| Pressure | Largest difference | Cause |
|---|---|---|
| >= 22 GPa | +0.0034 (drop 74 % of theirs) | their bulk D(Fe3+) averages truncated random draws (0.87 against 0.75) and their layer holds 5 % magnesiowuestite |
| 15-22 GPa | +0.030 | wadsleyite, ringwoodite and majorite have D(Fe3+) = 1.2-1.6; here the pyroxenes already raise Fe3+/FeT, so the minimum moves from 15 to 22 GPa |
| 2.5-15 GPa | +0.042 | the rise is 2.3 times theirs and starts from a higher value |
| < 2.5 GPa | +0.028 at the last step | as above; their spinel layer is absent |

Final Fe3+/FeT: 0.122 here against 0.093 in their run. The agreement is
qualitative: the direction of change matches in the lower mantle and below
15 GPa, the transition-zone fall is missing, and the amplitudes differ by
up to a factor of about 2. This offset belongs to the mineralogy and the
choice of D(Fe2+), not to the bookkeeping, which the first test checks
with Schaefer's own coefficients. A coupled run compared with their results
carries this offset plus whatever the coupling adds.

## Re-derivation note

Schaefer equilibrates the solid formed in a step with the residual liquid.
For a species with partition coefficient D and a crystallised mass fraction
s of the magma ocean, the moles left in the liquid scale by
`(1 - s) / (1 - s + s D)`, and the tracker applies the same factor in each
cell, so on a one-cell replay the two agree to round-off. Taking up D times
the melt concentration before the step instead, a factor `1 - s D`, differs
by `s**2 D (1 - D) / (1 - s + s D)` per step; in run 570, where s grows from
0.008 to 0.16 as the magma ocean shrinks, that puts Fe3+/FeT off by up to
1.7e-3, and it goes negative for `s > 1 / D`.

## Regenerating the fixture

```
python tests/interior_chem/fixtures/make_schaefer2024_replay.py \
    "archived data - revised/code/Earth/output.mat"
```

The script refuses a run whose mass balance does not close to 1e-9 at every
step; in about half of the archived runs the MATLAB solver left at least one
step unconverged.
