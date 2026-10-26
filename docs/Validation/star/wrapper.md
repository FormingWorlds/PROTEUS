# wrapper.py Validation

## What this page checks

This page checks 2 steps of the star wrapper, `src/proteus/star/wrapper.py`: the equilibrium temperature of the planet and the orbital distance that sets the stellar flux at the planet. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If this part of the code is wrong, every run gets the wrong stellar flux at the planet or the wrong equilibrium temperature. Each case below says why the check exists, what it tests, how we know the right answer and why it matters. A box "Test details" ends each case.

## Reference-pinned tests

| Test ID | Reference | What is pinned |
|---|---|---|
| `test_wrapper::test_update_equilibrium_temperature_pins_stefan_boltzmann_closed_form` | Stefan-Boltzmann law: T_eqm = ((1 - A) * S * s0_factor / sigma)^0.25 | Earth-like equilibrium temperature ~254 K at S = 1361 W/m^2, albedo = 0.3, s0_factor = 0.25 |
| `test_wrapper::test_flux_weighted_distance_pins_the_orbit_averaged_inverse_square_law` | Analytical limit: the orbital average of the inverse-square law, `<1/r^2> = 1 / (a^2 sqrt(1 - e^2))` | The distance carrying that flux, `a (1 - e^2)^(1/4)`, equal to `0.774597 a` at `e = 0.8` |

## Equilibrium temperature of an Earth-like planet

**Why this check exists.** The equilibrium temperature is the fourth root of an energy balance. A wrong root, for example a cube root or a fifth root, gives a wrong temperature without an error message.

**What it tests.** `update_equilibrium_temperature` computes $T_\mathrm{eqm} = ((1 - A)\, S\, f / \sigma)^{1/4}$, with $S$ the stellar flux at the planet, $A$ the albedo of the planet, $f$ the absorption factor `s0_factor` and $\sigma$ the Stefan-Boltzmann constant. At $S$ = 1361 W/m$^2$, $A$ = 0.3 and $f$ = 0.25 (the disk-averaged factor), the test expects about 254 K. A cube root would give about 1613 K and a fifth root about 84 K; the test checks that the result is more than 50 K away from both.

**How we know the right answer.** The value follows from the Stefan-Boltzmann law for a planet in radiative balance. The test checks the closed form to relative 1e-12, with $\sigma$ from the code, and the value 254 K to within 2 K, which catches a $\sigma$ about 2.2 % too small or 4.2 % too large.

**Why it matters.** `update_stellar_quantities` calls it in the stellar step of the main loop and writes `T_eqm` to the helpfile.

??? info "Test details"
    - Reference: Stefan-Boltzmann law, T_eqm = ((1 - A) * S * s0_factor / sigma)^0.25 (analytical limit).
    - Tolerance: relative 1e-12 against the closed form; absolute 2 K on 254 K; more than 50 K from the cube-root and fifth-root values.
    - Test id: `tests/star/test_wrapper.py::test_update_equilibrium_temperature_pins_stefan_boltzmann_closed_form`
    - Command: `pytest tests/star/test_wrapper.py::test_update_equilibrium_temperature_pins_stefan_boltzmann_closed_form`

## Orbit-averaged distance for the stellar flux

**Why this check exists.** On an eccentric orbit the distance to the star changes, and the stellar flux goes as $1/r^2$. An average of the wrong quantity, such as the mean distance, gives a wrong flux without an error message.

**What it tests.** `flux_weighted_distance` returns the distance that gives the orbit-averaged flux. Averaging $1/r^2$ over one orbit gives $1 / (a^2 \sqrt{1 - e^2})$, with $a$ the semi-major axis and $e$ the eccentricity. So the distance with that flux is $a (1 - e^2)^{1/4}$, which is $0.774597\, a$ at $e = 0.8$. The test also checks that the result is positive and shorter than $a$.

**How we know the right answer.** The orbit average of the inverse-square law is an analytical result. The time-averaged separation $a (1 + e^2 / 2)$, which `hf_row['separation']` holds, gives $1.32\, a$ at $e = 0.8$ instead. The 2 distances give fluxes that differ by a factor of 2.904, which the test checks.

**Why it matters.** `update_instellation` uses this distance for both the bolometric and the XUV flux. The source docstring notes that the time-averaged separation remains the right quantity for the Roche-limit checks and the orbit plots, so both distances are in use.

??? info "Test details"
    - Reference: orbital average of the inverse-square law, <1/r^2> = 1 / (a^2 sqrt(1 - e^2)) (analytical limit).
    - Tolerance: relative 1e-12 on a (1 - e^2)^(1/4); relative 1e-3 on the flux ratio 2.904; more than 0.3 a from 1.32 a.
    - Test id: `tests/star/test_wrapper.py::test_flux_weighted_distance_pins_the_orbit_averaged_inverse_square_law`
    - Command: `pytest tests/star/test_wrapper.py::test_flux_weighted_distance_pins_the_orbit_averaged_inverse_square_law`

## Coverage

The Stefan-Boltzmann test verifies the closed-form equilibrium temperature calculation against the analytical relation. Discrimination guards assert that a wrong exponent (cube root or fifth root instead of fourth root) would land at ~1613 K or ~84 K, both well outside the tolerance band around the correct ~254 K value.

The orbital-average test anchors the distance that `update_instellation` uses for both the bolometric and the XUV flux. Averaging `1/r^2` over one orbit gives `1 / (a^2 sqrt(1 - e^2))`, so the equivalent distance is `a (1 - e^2)^(1/4)`. The competing convention, the time-averaged separation `a (1 + e^2 / 2)` that `hf_row['separation']` carries, gives `1.32 a` at `e = 0.8` instead; the two disagree in the resulting flux by a factor of 2.904, which is what the discrimination guard pins. That separation remains the right quantity for the Roche-limit checks and the orbit plots, so both distances are live and the test records which belongs where.

## Last verified
2026-09-29
