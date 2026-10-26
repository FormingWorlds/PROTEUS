# Validation: `src/proteus/orbit/satellite.py`

## What this page checks

This page checks the planet-satellite tides of `proteus.orbit.satellite`: the angular momentum of the planet and its satellite, and the capture of the satellite into the evection resonance. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If this part of the code is wrong, the spin of the planet and the orbit of the satellite evolve wrongly. Each case below says why the check exists, what it tests, how we know the right answer and why it matters; a box "Test details" ends each case.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/orbit/test_satellite.py::test_ps0d_bootstrap_am_uses_satellite_mass_not_planet_mass` | Korenaga (2023) Icarus 400, 115564, Eq. 60 (orbital component cross-checked against Touma and Wisdom 1994) | n/a | Pins the spin-plus-orbital decomposition on the present-day Earth-Moon configuration. Asserts sign and the 1e34-1e35 kg m^2 / s order of magnitude expected from Korenaga's Eq. 60. |
| `tests/integration/test_slow_orbit_evection_ctl.py::test_resonance_capture_occurs_within_expected_time_window` | Rufu and Canup (2020) Figure 3 evection-resonance capture case, as reproduced in `src/proteus/orbit/evection_notebook.ipynb` / `plot_evection.png` (capture at t~2.4e4 yr) | n/a | Drives the real `evolve_orbit_satellite(model='ps1d_evec')` under a Mignard constant-time-lag tidal spectrum from the paper's Figure-3-calibrated initial conditions; asserts eccentricity crosses 0.1 within t between `2e4` and `4e4` yr. Compared against a real run reaching t=2.58e4 yr, e=0.111. |
| `tests/integration/test_slow_orbit_evection_ctl.py::test_peak_eccentricity_matches_reference_location` | Same source as above (peak e~0.72-0.724 at a'~11.89 R_earth, t~5.2e4 yr) | n/a | Same driver; asserts the eccentricity peak reaches >= 0.6 at a' between `10` and `13` R_earth. Compared against a real run's observed peak of e=0.755 at a'=11.88 R_earth, t=5.12e4 yr; the peak a' matches the reference to <0.1%. The post-peak contraction phase (a' declining to ~10.2 R_earth by t=1e5 yr in the reference) was not reached in that run and is not asserted here. |

## Angular momentum of the Earth-Moon system

**Why this check exists.** The total angular momentum is the spin of the planet plus the orbit of the satellite, and the orbital term must use the satellite mass. The planet mass in its place makes the total about 81 times too large for the Earth-Moon system, without an error message.

**What it tests.** On its first step, `ps0d` computes the total angular momentum `plan_sat_am` with Eq. 60 of Korenaga (2023): $L = I \Omega + M_\mathrm{sat} \sqrt{G (M_p + M_\mathrm{sat})\, a}$, with $I$ the moment of inertia and $\Omega$ the spin rate of the planet, $M_p$ and $M_\mathrm{sat}$ the masses and $a$ the semi-major axis of the satellite. The test uses the present Earth-Moon system and compares the result with Eq. 60 evaluated in the test, without a call into the source. It also checks that the planet-mass version differs by more than 50 times, and that $L$ lies between 1e34 and 1e35 kg m$^2$/s.

**How we know the right answer.** Eq. 60 of Korenaga (2023); its orbital term agrees with the present Earth-Moon value of Touma and Wisdom (1994), about 2.85e34 kg m$^2$/s (see the derivation below).

**Why it matters.** `plan_sat_am` is the conserved angular momentum that the satellite model keeps through the run.

??? info "Test details"
    - Reference: Korenaga (2023) Icarus 400, 115564, Eq. 60 (orbital component cross-checked against Touma and Wisdom 1994).
    - Tolerance: relative 1e-6 against Eq. 60; more than 50 times from the planet-mass version; range 1e34 to 1e35 kg m^2/s.
    - Test id: `tests/orbit/test_satellite.py::test_ps0d_bootstrap_am_uses_satellite_mass_not_planet_mass`
    - Command: `pytest tests/orbit/test_satellite.py::test_ps0d_bootstrap_am_uses_satellite_mass_not_planet_mass`

## Capture into the evection resonance

**Why this check exists.** In the evection resonance the precession of the lunar orbit matches the orbital period of the Earth, and the eccentricity of the satellite orbit grows. A wrong tidal rate or a wrong resonance term changes when the capture happens, or stops it, without an error message.

**What it tests.** The test runs `evolve_orbit_satellite` with `model='ps1d_evec'` and a constant-time-lag tidal response, from the initial conditions of the Figure 3 case of Rufu and Canup (2020). The eccentricity must cross 0.1 between 2e4 and 4e4 yr.

**How we know the right answer.** Rufu and Canup (2020), Figure 3, as reproduced in `src/proteus/orbit/evection_notebook.ipynb`, captures at about 2.4e4 yr. A real run of the model crossed 0.1 at 2.58e4 yr.

**Why it matters.** The orbit wrapper calls `evolve_orbit_satellite` for the satellite models.

**Known limit.** The test checks a time window, not the full eccentricity history.

??? info "Test details"
    - Reference: Rufu and Canup (2020), Figure 3 evection-resonance capture case.
    - Tolerance: the crossing of e = 0.1 between 2e4 and 4e4 yr.
    - Test id: `tests/integration/test_slow_orbit_evection_ctl.py::test_resonance_capture_occurs_within_expected_time_window`
    - Command: `pytest tests/integration/test_slow_orbit_evection_ctl.py::test_resonance_capture_occurs_within_expected_time_window` (slow tier)

## Peak eccentricity in the evection resonance

**Why this check exists.** The resonance drives the eccentricity to a peak at a given distance of the satellite. A wrong resonance term moves the peak or lowers it without an error message.

**What it tests.** With the same run as above, the peak eccentricity must reach at least 0.6, at a scaled distance $a'$ between 10 and 13 Earth radii.

**How we know the right answer.** The reference case peaks at e of about 0.72 to 0.724 at $a'$ of about 11.89 Earth radii, at about 5.2e4 yr. A real run peaked at e = 0.755 at $a'$ = 11.88 Earth radii and 5.12e4 yr.

**Why it matters.** It checks the strength of the resonance in the model, not only its timing.

**Known limit.** The contraction after the peak ($a'$ falling to about 10.2 Earth radii by 1e5 yr in the reference) is not asserted.

??? info "Test details"
    - Reference: Rufu and Canup (2020), Figure 3 (peak e about 0.72 to 0.724 at a' about 11.89 R_earth, t about 5.2e4 yr).
    - Tolerance: peak e of at least 0.6 at a' between 10 and 13 R_earth.
    - Test id: `tests/integration/test_slow_orbit_evection_ctl.py::test_peak_eccentricity_matches_reference_location`
    - Command: `pytest tests/integration/test_slow_orbit_evection_ctl.py::test_peak_eccentricity_matches_reference_location` (slow tier)

## Re-derivation note

The orchestrator populates `plan_sat_am` from

```
L_total = I_planet * omega_planet + M_satellite * sqrt(G (M_planet + M_sat) a_sat)
```

with `I_planet = (2/5) M_planet R_planet^2` for a solid sphere. This is the
Korenaga (2023) Eq. 60 form, equivalent to the textbook reduced-mass
orbital angular momentum `L_orb = mu * sqrt(G (M_pl + M_sat) a)` in the
`M_sat << M_planet` limit (where `mu = M_pl M_sat / (M_pl + M_sat) -> M_sat`).
For the Earth-Moon system the substitution has a 1.2% relative error.

For Earth parameters, the two components evaluate to

| Component | Value (kg m^2 / s) |
|---|---|
| Spin term `I_planet * omega_planet` (24-hr rotation, 1 M_earth, R_earth) | ~7.05e33 |
| Orbital term `M_sat * sqrt(G (M_pl + M_sat) a)` (Earth-Moon distance, lunar mass) | ~2.89e34 |
| **Total `L_planet+sat`** | **~3.60e34** |

The orbital component agrees with Touma and Wisdom (1994), who report the
present-day Earth-Moon orbital angular momentum as ~2.85e34 kg m^2 / s.
The total system angular momentum (spin + orbital) is ~3.6e34 kg m^2 / s,
which is what `plan_sat_am` holds through the simulation. The
reference-pinned test brackets the total in `[1e34, 1e35]` kg m^2 / s; a
regression that swaps `M_sat` for `M_planet` in the orbital prefactor
would land at ~2.4e36 kg m^2 / s, well outside the bracket.

## Eq. 58-59: rotation and semi-major-axis rate equations

Korenaga (2023) Eq. 58 gives the planet's spin-down rate,

```
dOmega/dt = -E_tide_dot / (I*Omega + G*M_pl*M_sat*I / (a*(L - I*Omega)))
```

where `E_tide_dot` is the tidal power dissipated in the planet (positive,
W) and `L` is the conserved total angular momentum from Eq. 60. The minus
sign makes the spin slow whenever tidal energy is dissipated, matching
the expectation that dissipation transfers angular momentum from the
planet's spin to the satellite's orbit. Eq. 59 for the semi-major axis,

```
da/dt = -2*I*a / (L - I*Omega) * dOmega/dt
```

follows directly from differentiating the Eq. 60 closure at constant `L`
and solving for `da/dt`: whenever the planet's spin slows, the orbit
expands, provided `L > I*Omega` (the prograde-satellite regime PROTEUS
targets).

## Correctness of the orbital prefactor

The orbital prefactor in Eq. 60 is the satellite mass `M_M`, not the
planet mass `M_E`. The two forms differ by a factor of `M_planet /
M_satellite`, which is ~81 for the Earth-Moon system and ~larger for any
configuration with a relatively lighter satellite. The reference-pinned
test discriminates between the two forms via the `1e34 < L < 1e35`
bracket and the `pytest.approx` pin on the Eq. 60 value.
