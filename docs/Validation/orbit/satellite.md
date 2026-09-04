# Validation: `src/proteus/orbit/satellite.py`

This page tracks the `@pytest.mark.reference_pinned` tests that anchor the
behaviour of `proteus.orbit.satellite` against a published source.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/orbit/test_satellite.py::test_update_satellite_angular_momentum_matches_korenaga_2023_eq60` | Korenaga (2023) Icarus 400, 115564, Eq. 60 (orbital component cross-checked against Touma and Wisdom 1994) | n/a | Pins the spin-plus-orbital decomposition on the present-day Earth-Moon configuration. Asserts sign and the 1e34-1e35 kg m^2 / s order of magnitude expected from Korenaga's Eq. 60. |
| `tests/integration/test_slow_orbit_evection_ctl.py::test_resonance_capture_occurs_within_expected_time_window` | Rufu and Canup (2020) Figure 3 evection-resonance capture case, as reproduced in `src/proteus/orbit/evection_notebook.ipynb` / `plot_evection.png` (capture at t~2.4e4 yr) | n/a | Drives the real `evolve_orbit_satellite(model='ps1d_evec')` under a Mignard constant-time-lag tidal spectrum from the paper's Figure-3-calibrated initial conditions; asserts eccentricity crosses 0.1 within t in [2e4, 4e4] yr. Compared against a real run reaching t=2.58e4 yr, e=0.111. |
| `tests/integration/test_slow_orbit_evection_ctl.py::test_peak_eccentricity_matches_reference_location` | Same source as above (peak e~0.72-0.724 at a'~11.89 R_earth, t~5.2e4 yr) | n/a | Same driver; asserts the eccentricity peak reaches >= 0.6 at a' in [10, 13] R_earth. Compared against a real run's observed peak of e=0.755 at a'=11.88 R_earth, t=5.12e4 yr -- the peak a' matches the reference to <0.1%. The post-peak contraction phase (a' declining to ~10.2 R_earth by t=1e5 yr in the reference) was not reached in that run and is not asserted here. |

## Re-derivation note

The orchestrator populates `plan_sat_am` from

```
L_total = I_planet * omega_planet + M_satellite * sqrt(G (M_planet + M_sat) a_sat)
```

with `I_planet = (2/5) M_planet R_planet^2` for a solid sphere. This is the
Korenaga (2023) Eq. 60 form, equivalent to the textbook reduced-mass
orbital angular momentum `L_orb = mu * sqrt(G (M_pl + M_sat) a)` in the
`M_sat << M_planet` limit (where `mu = M_pl M_sat / (M_pl + M_sat) -> M_sat`).
For the Earth-Moon system the substitution carries a 1.2% relative error.

For Earth parameters, the two components evaluate to

| Component | Value (kg m^2 / s) |
|---|---|
| Spin term `I_planet * omega_planet` (24-hr rotation, 1 M_earth, R_earth) | ~7.05e33 |
| Orbital term `M_sat * sqrt(G (M_pl + M_sat) a)` (Earth-Moon distance, lunar mass) | ~2.89e34 |
| **Total `L_planet+sat`** | **~3.60e34** |

The orbital component agrees with Touma and Wisdom (1994), who report the
present-day Earth-Moon orbital angular momentum as ~2.85e34 kg m^2 / s.
The total system angular momentum (spin + orbital) is ~3.6e34 kg m^2 / s,
which is what `plan_sat_am` carries through the simulation. The
reference-pinned test brackets the total in `[1e34, 1e35]` kg m^2 / s; a
regression that swaps `M_sat` for `M_planet` in the orbital prefactor
would land at ~2.4e36 kg m^2 / s, well outside the bracket.

## Correctness of the orbital prefactor

The orbital prefactor in Eq. 60 is the satellite mass `M_M`, not the
planet mass `M_E`. The two forms differ by a factor of `M_planet /
M_satellite`, which is ~81 for the Earth-Moon system and ~larger for any
configuration with a relatively lighter satellite. The reference-pinned
test discriminates between the two forms via the `1e34 < L < 1e35`
bracket and the `pytest.approx` pin on the Eq. 60 value.
