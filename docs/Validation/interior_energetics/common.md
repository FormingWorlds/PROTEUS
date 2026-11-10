# Validation: `src/proteus/interior_energetics/common.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
shared interior-energetics helpers against a published source, an analytical
limit, or a cross-implementation comparison. The marker is registered in
`pyproject.toml`.

These helpers are the implementation of radiogenic decay and of the linear
crystallisation front that the `boundary` backend uses, so the pins here cover
that backend.

| Test id | Reference | Scope | Last compared |
|---|---|---|---|
| `tests/interior_energetics/test_common.py::test_radiogenic_specific_heating_halves_over_one_half_life` | Analytical limit of exponential decay | Pins the decay exponent over three successive Th-232 half-lives. Guards against a sign-flipped exponent, an amplifying `exp(+lambda t)` regression, and an unconverted ppmw scale. | 2026-11-10 |
| `tests/interior_energetics/test_boundary.py::test_radioactive_heating_matches_values_frozen_before_the_shared_helper` | Cross-implementation: the boundary backend's previous hand-rolled decay loop | Confirms that delegating to the shared helper left boundary's radiogenic heating unchanged to within one unit in the last place. | 2026-11-10 |
| `tests/interior_energetics/test_boundary.py::test_solidification_front_matches_values_frozen_before_the_shared_helper` | Cross-implementation: the boundary backend's previous inline front expressions | Confirms that the solidification radius and its temperature derivative survive delegation exactly. | 2026-11-10 |

## Isotope inventory

`radnuc_data` (`src/proteus/utils/constants.py`) supplies the isotopic
abundances, heat productions and half-lives. Heat productions and half-lives
for Al-26 and Fe-60 are from Ruedas (2017); the remaining values are the
standard tabulated constants.

The shared helper covers all six isotopes: Al-26, Fe-60, K-40, Th-232, U-235
and U-238, with a single elemental uranium concentration feeding both uranium
isotopes. This matches the sets built by `interior_energetics/spider.py` and
`interior_energetics/aragog.py`. Before the helper existed, `boundary.py`
silently dropped Al-26 and Fe-60, so a run that configured `radio_Al` received
no aluminium-26 heating on that backend while SPIDER and Aragog gave it the
dominant early source.
