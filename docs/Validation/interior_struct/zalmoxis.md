# zalmoxis.py Validation

## Source under test
`src/proteus/interior_struct/zalmoxis.py` (the `liquidus_super` initial-condition path:
`solve_superliquidus_adiabat`, `_resolve_zalmoxis_temperature_mode`,
`_resolve_zalmoxis_cmb_temperature`, `load_zalmoxis_configuration`).

## Reference-pinned tests

| Test ID | Reference | What is pinned |
|---|---|---|
| `test_liquidus_super_ic::TestPaleosLiquidusSourceOfTruth::test_paleos_liquidus_135gpa` | Fei et al. (2021), Nature Communications, 12, 876 (doi:10.1038/s41467-021-21170-y) | The MgSiO$_3$ liquidus evaluated by the `liquidus_super` IC solve at 135 GPa: `T_liq = 6000 * (135 / 140)^0.26 ~ 5943.5 K`, pinned to `abs=1.0` K. A drift in any fit constant (`6000`, `0.26`, `140` GPa reference) fails immediately. An exponent guard asserts that swapping `0.26` for `0.5` would land ~50 K low, well outside the tolerance. |

## Coverage

The `liquidus_super` mode solves for the coolest adiabat that is fully molten
everywhere with at least $\Delta T_\mathrm{super}$ of superheat above the
liquidus, where the liquidus comes from the Zalmoxis `melting_curves` fit: the
Belonoshko et al. (2005) branch below 2.55 GPa and the Fei et al. (2021) branch
above it. The reference-pinned test fixes the Fei et al. (2021) high-pressure
value at the 1 M$_\oplus$ reference pressure so any change to the upstream
Zalmoxis fit forces a re-validation of the superheat the solve evaluates.
Companion tests in the same file check branch continuity at the 2.55 GPa
crossover and monotonicity of the liquidus across the magma-ocean pressure
range.

## Last verified
2026-06-12
