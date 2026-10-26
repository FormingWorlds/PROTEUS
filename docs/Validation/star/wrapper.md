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

![Equilibrium temperature against stellar flux for albedo 0.3](../../assets/validation/star_wrapper_equilibrium_temperature.svg#only-light){ width="80%" }
![Equilibrium temperature against stellar flux for albedo 0.3](../../assets/validation/star_wrapper_equilibrium_temperature_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the stellar flux at the planet and the y-axis the equilibrium temperature on a log scale. The planet has an albedo of 0.3 and the disk-averaging factor `s0_factor` = 0.25. The line is `update_equilibrium_temperature`, and the squares are the closed form with the 1/4 power (context). The dashed and dotted lines are the cube-root and fifth-root forms that the test rejects. The filled circle is the asserted code value at 1361 W/m$^2$, and the hollow square the 254 K pin; the test allows 2 K. A pass means that the circle sits on the line and in the square, far from the 2 rejected forms.

??? info "Test details"
    - Reference: Stefan-Boltzmann law, T_eqm = ((1 - A) * S * s0_factor / sigma)^0.25 (analytical limit).
    - Tolerance: relative 1e-12 against the closed form; absolute 2 K on 254 K; more than 50 K from the cube-root and fifth-root values.
    - Test id: `tests/star/test_wrapper.py::test_update_equilibrium_temperature_pins_stefan_boltzmann_closed_form`
    - Command: `pytest -m reference_pinned tests/star/test_wrapper.py --validation-plots`

## Orbit-averaged distance for the stellar flux

**Why this check exists.** On an eccentric orbit the distance to the star changes, and the stellar flux goes as $1/r^2$. An average of the wrong quantity, such as the mean distance, gives a wrong flux without an error message.

**What it tests.** `flux_weighted_distance` returns the distance that gives the orbit-averaged flux. Averaging $1/r^2$ over one orbit gives $1 / (a^2 \sqrt{1 - e^2})$, with $a$ the semi-major axis and $e$ the eccentricity. So the distance with that flux is $a (1 - e^2)^{1/4}$, which is $0.774597\, a$ at $e = 0.8$. The test also checks that the result is positive and shorter than $a$.

**How we know the right answer.** The orbit average of the inverse-square law is an analytical result. The time-averaged separation $a (1 + e^2 / 2)$, which `hf_row['separation']` holds, gives $1.32\, a$ at $e = 0.8$ instead. The 2 distances give fluxes that differ by a factor of 2.904, which the test checks.

**Why it matters.** `update_instellation` uses this distance for the bolometric flux, and with the Spada tracks for the XUV flux too. The source docstring notes that the time-averaged separation remains the right quantity for the Roche-limit checks and the orbit plots, so both distances are in use.

![Flux-weighted distance over the semi-major axis against eccentricity](../../assets/validation/star_wrapper_flux_weighted_distance.svg#only-light){ width="80%" }
![Flux-weighted distance over the semi-major axis against eccentricity](../../assets/validation/star_wrapper_flux_weighted_distance_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the eccentricity $e$ and the y-axis the distance over the semi-major axis. The line is `flux_weighted_distance`, and the squares are $(1 - e^2)^{1/4}$ (context). The dashed line is the time-averaged separation $1 + e^2/2$, which the test rejects. The filled circle is the asserted code value at $e$ = 0.8, 0.774597. A pass means that the circle sits on the line, below 1 and far from the dashed line.

??? info "Test details"
    - Reference: orbital average of the inverse-square law, <1/r^2> = 1 / (a^2 sqrt(1 - e^2)) (analytical limit).
    - Tolerance: relative 1e-12 on a (1 - e^2)^(1/4); relative 1e-3 on the flux ratio 2.904; more than 0.3 a from 1.32 a.
    - Test id: `tests/star/test_wrapper.py::test_flux_weighted_distance_pins_the_orbit_averaged_inverse_square_law`
    - Command: `pytest -m reference_pinned tests/star/test_wrapper.py --validation-plots`

## Coverage

The Stefan-Boltzmann test verifies the closed-form equilibrium temperature calculation against the analytical relation. Discrimination guards assert that a wrong exponent (cube root or fifth root instead of fourth root) would land at ~1613 K or ~84 K, both well outside the tolerance band around the correct ~254 K value.

The orbital-average test anchors the distance that `update_instellation` uses for both the bolometric and the XUV flux. Averaging `1/r^2` over one orbit gives `1 / (a^2 sqrt(1 - e^2))`, so the equivalent distance is `a (1 - e^2)^(1/4)`. The competing convention, the time-averaged separation `a (1 + e^2 / 2)` that `hf_row['separation']` carries, gives `1.32 a` at `e = 0.8` instead; the two disagree in the resulting flux by a factor of 2.904, which is what the discrimination guard pins. That separation remains the right quantity for the Roche-limit checks and the orbit plots, so both distances are live and the test records which belongs where.

## Last verified
2026-09-29
