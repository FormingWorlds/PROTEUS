# vulcan_agni.py Validation

## What this page checks

This page checks the coupling of the VULCAN chemistry to the AGNI climate in offline chemistry (`atmos_chem.vulcan.couple_agni = true`). VULCAN computes the composition of the atmosphere for a fixed temperature profile. AGNI computes the temperature profile for a fixed composition. Coupled, PROTEUS solves the climate of an atmosphere snapshot again each time the chemistry reaches a steady state, from the composition VULCAN found, until the two agree.

If this coupling is wrong, the chemistry is computed on a temperature profile that does not belong to its composition, and nothing reports it. Each case below says why the check exists, what it tests, how we know the right answer and why it matters. These are integration tests that need the Julia AGNI installation; they skip unless `PROTEUS_CI_NIGHTLY=1` is set, as in the nightly workflow.

## Source under test
`src/proteus/atmos_chem/vulcan_agni.py` (`SnapshotClimate`) and the coupled branch of `run_vulcan` in `src/proteus/atmos_chem/vulcan.py`

## Tests

| Test ID | Reference | What is pinned | Last compared |
|---|---|---|---|
| `test_integration_vulcan_agni_climate::test_climate_update_with_the_snapshot_composition_returns_the_snapshot` | The AGNI solve of the evolution that wrote the snapshot | With the snapshot's own composition, the climate update returns the snapshot temperature within 1 K; with H2O and CO2 swapped it moves by more than 10 K | 2026-11-13 |
| `test_integration_vulcan_agni_climate::test_coupled_run_ends_where_chemistry_and_climate_agree` | AGNI's climate for VULCAN's final composition, and radiative-convective equilibrium | VULCAN ends at a steady state with the temperature settled; its CO is the coupled climate's, and its temperature differs from that climate by the recorded last change, at most 1 %; the net flux of that climate is uniform with height to 1 % | 2026-11-13 |

## The snapshot is a fixed point of the climate update

**Why this check exists.** Chemistry that leaves the composition as it was must leave the climate as it was. If the climate update set AGNI up differently from the evolution (another stellar spectrum, another surface condition, a state not loaded from the snapshot), it would return a different climate for the same gases.

**What it tests.** The test solves one atmosphere with AGNI as the evolution does: a 10 bar H2O-CO2-H2 atmosphere over a 1500 K surface, with banded radiation. It writes the snapshot, builds the climate update from it, and gives the update the snapshot's own composition. It also gives the update the same atmosphere with the H2O and CO2 abundances swapped, to show that the update does respond to the composition.

**How we know the right answer.** The reference is the AGNI solve that wrote the snapshot. AGNI stops when the flux residual is small enough, so solving a converged state again moves the temperature a little: 0.15 K in this case, for the same AGNI atmosphere with or without reloading the snapshot. The tolerance is 1 K. Swapping H2O and CO2 moves the temperature by about 27 K.

**Why it matters.** The coupled chemistry starts from the climate of the evolution, so any difference here would show up as a change that the chemistry did not cause.

![Temperature of the snapshot and of the climate update with the same and with swapped gases](../../assets/validation/atmos_chem_vulcan_agni_fixed_point.svg#only-light){ width="70%" }
![Temperature of the snapshot and of the climate update with the same and with swapped gases](../../assets/validation/atmos_chem_vulcan_agni_fixed_point_dark.svg#only-dark){ width="70%" }

**How to read the plot.** The y-axis is pressure, with the surface at the bottom, and the x-axis the temperature of the climate update minus that of the snapshot. The grey band is the tolerance of 1 K. The circles are the update with the same gases, and they lie inside the band. The dashed line is the update with H2O and CO2 swapped, which leaves the band by tens of kelvin.

??? info "Test details"
    - Reference: the AGNI solve of the evolution that wrote the snapshot.
    - Tolerance: 1 K in every layer; the swapped composition must move the temperature by more than 10 K.
    - Test id: `tests/integration/test_integration_vulcan_agni_climate.py::test_climate_update_with_the_snapshot_composition_returns_the_snapshot`
    - Command: `PROTEUS_CI_NIGHTLY=1 pytest tests/integration/test_integration_vulcan_agni_climate.py --validation-plots`

## A coupled run ends where chemistry and climate agree

**Why this check exists.** The coupled run must stop at a state that both models accept: a steady state of the chemistry for that temperature, and a climate in energy balance for that composition. A coupling that stopped early, or gave AGNI the wrong composition, would end somewhere else.

**What it tests.** VULCAN runs the CHO chemistry with photochemistry on the same snapshot, with `couple_agni`. At 1500 K the chemistry turns part of the CO2 and H2 into CO, a gas the snapshot does not contain. The test checks that:

- VULCAN ends at a steady state, after a climate update that changed the temperature by less than `agni_T_rtol`;
- the CO given to AGNI is VULCAN's;
- VULCAN saves the temperature it is steady for, from before the last climate update, so it differs from the coupled climate by exactly the change of that update, which is at most `agni_T_rtol`;
- the net upward flux of the coupled climate is the same at every height.

**How we know the right answer.** In radiative-convective equilibrium no energy builds up in any layer, so the net flux is the same at every height. The converged AGNI states of this case hold it uniform to 0.3 % (snapshot) and 0.05 % (coupled); the test allows 1 %.

**Why it matters.** This is the state the chemistry results describe: its temperature profile belongs to its composition.

![Temperature and CO mixing ratio of the coupled run](../../assets/validation/atmos_chem_vulcan_agni_coupled_run.svg#only-light){ width="80%" }
![Temperature and CO mixing ratio of the coupled run](../../assets/validation/atmos_chem_vulcan_agni_coupled_run_dark.svg#only-dark){ width="80%" }

**How to read the plot.** Both panels have pressure on the y-axis, with the surface at the bottom. On the left, the thick line is the coupled climate and the circles VULCAN's final temperature, which lie on it within the last change (at most 1 %), except above about 0.01 bar, where VULCAN is held at the 180 K floor (dotted line). On the right, the thick line is VULCAN's CO mixing ratio on the AGNI layers and the circles the CO that AGNI solved the climate with.

??? info "Test details"
    - Reference: AGNI's climate for VULCAN's final composition; radiative-convective equilibrium.
    - Tolerance: last climate update at most `agni_T_rtol` = 1 %; CO relative 1e-3; VULCAN temperature against the coupled climate equal to that last change (relative 1e-6); spread of the net flux at most 1 % of its mean.
    - Test id: `tests/integration/test_integration_vulcan_agni_climate.py::test_coupled_run_ends_where_chemistry_and_climate_agree`
    - Command: `PROTEUS_CI_NIGHTLY=1 pytest tests/integration/test_integration_vulcan_agni_climate.py --validation-plots`
