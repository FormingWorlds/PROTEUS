# wrapper.py Validation

## Source under test
`src/proteus/star/wrapper.py`

## Reference-pinned tests

| Test ID | Reference | What is pinned |
|---|---|---|
| `test_wrapper::test_update_equilibrium_temperature_pins_stefan_boltzmann_closed_form` | Stefan-Boltzmann law: T_eqm = ((1 - A) * S * s0_factor / sigma)^0.25 | Earth-like equilibrium temperature ~254 K at S = 1361 W/m^2, albedo = 0.3, s0_factor = 0.25 |

## Coverage

The test verifies the closed-form equilibrium temperature calculation against
the analytical Stefan-Boltzmann relation. Discrimination guards assert that a
wrong exponent (cube root or fifth root instead of fourth root) would land at
~1613 K or ~84 K, both well outside the tolerance band around the correct
~254 K value.

## Last verified
2026-05-24
