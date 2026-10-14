# wrapper.py Validation

## Source under test
`src/proteus/star/wrapper.py`

## Reference-pinned tests

| Test ID | Reference | What is pinned |
|---|---|---|
| `test_wrapper::test_update_equilibrium_temperature_pins_stefan_boltzmann_closed_form` | Stefan-Boltzmann law: T_eqm = ((1 - A) * S * s0_factor / sigma)^0.25 | Earth-like equilibrium temperature ~254 K at S = 1361 W/m^2, albedo = 0.3, s0_factor = 0.25 |
| `test_wrapper::test_flux_weighted_distance_pins_the_orbit_averaged_inverse_square_law` | Analytical limit: the orbital average of the inverse-square law, `<1/r^2> = 1 / (a^2 sqrt(1 - e^2))` | The distance carrying that flux, `a (1 - e^2)^(1/4)`, equal to `0.774597 a` at `e = 0.8` |

## Coverage

The Stefan-Boltzmann test verifies the closed-form equilibrium temperature
calculation against the analytical relation. Discrimination guards assert that a
wrong exponent (cube root or fifth root instead of fourth root) would land at
~1613 K or ~84 K, both well outside the tolerance band around the correct
~254 K value.

The orbital-average test anchors the distance that `update_instellation` uses
for both the bolometric and the XUV flux. Averaging `1/r^2` over one orbit gives
`1 / (a^2 sqrt(1 - e^2))`, so the equivalent distance is `a (1 - e^2)^(1/4)`.
The competing convention, the time-averaged separation `a (1 + e^2 / 2)` that
`hf_row['separation']` carries, gives `1.32 a` at `e = 0.8` instead; the two
disagree in the resulting flux by a factor of 2.904, which is what the
discrimination guard pins. That separation remains the right quantity for the
Roche-limit checks and the orbit plots, so both distances are
live and the test records which belongs where.

## Last verified
2026-09-29
