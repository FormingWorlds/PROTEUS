# zalmoxis.py Validation

## What this page checks

This page checks the liquidus that the `liquidus_super` initial condition of the Zalmoxis structure uses for a PALEOS mantle. The liquidus is the temperature above which a rock is fully molten at a given pressure. The `liquidus_super` mode starts the mantle on the coolest adiabat that is fully molten everywhere, with a margin of superheat above the liquidus. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If the liquidus is wrong, the initial temperature profile of the mantle is wrong. The case below says why the check exists, what it tests, how we know the right answer and why it matters. A box "Test details" ends the case.

## Source under test
`src/proteus/interior_struct/zalmoxis.py` (the `liquidus_super` initial-condition path:
`solve_superliquidus_adiabat`, `_resolve_zalmoxis_temperature_mode`,
`_resolve_zalmoxis_anchor_temperatures`, `load_zalmoxis_configuration`).

## Reference-pinned tests

| Test ID | Reference | What is pinned |
|---|---|---|
| `test_liquidus_super_ic::TestPaleosLiquidusSourceOfTruth::test_paleos_liquidus_135gpa` | Fei et al. (2021), Nature Communications, 12, 876 (doi:10.1038/s41467-021-21170-y) | The MgSiO$_3$ liquidus that the `liquidus_super` IC solve uses, evaluated with `paleos_liquidus` at 135 GPa: `T_liq = 6000 * (135 / 140)^0.26 ~ 5943.5 K`, pinned to `abs=1.0` K. A drift in any fit constant (`6000`, `0.26`, `140` GPa reference) fails immediately. An exponent guard asserts that swapping `0.26` for `0.5` would land ~50 K low, well outside the tolerance. |

## MgSiO3 liquidus at 135 GPa

**Why this check exists.** The high-pressure liquidus is a power law with 3 fit constants. A drift in any of them, or a wrong exponent, moves the liquidus and with it the initial temperature profile, without an error message.

**What it tests.** The Fei et al. (2021) branch of the MgSiO$_3$ liquidus is $T_\mathrm{liq} = 6000\,(P / 140\ \mathrm{GPa})^{0.26}$ K. At 135 GPa this gives about 5943.5 K. The test evaluates `zalmoxis.melting_curves.paleos_liquidus` at 135 GPa and checks this value. It also checks that the result is positive, and that an exponent of 0.5 in place of 0.26, which gives about 5892 K, is more than 30 K away.

**How we know the right answer.** Fei et al. (2021), Nature Communications, 12, 876, give the fit and its constants; the value at 135 GPa follows by hand.

**Why it matters.** For a PALEOS mantle, the `liquidus_super` solve in `src/proteus/interior_struct/zalmoxis.py` takes its liquidus from `paleos_liquidus`.

**Known limit.** The test calls the Zalmoxis function directly; it does not run the `liquidus_super` solve of PROTEUS.

??? info "Test details"
    - Reference: Fei et al. (2021), Nature Communications, 12, 876 (doi:10.1038/s41467-021-21170-y).
    - Tolerance: absolute 1.0 K on 5943.53 K; more than 30 K from the exponent-0.5 value.
    - Test id: `tests/interior_struct/test_liquidus_super_ic.py::TestPaleosLiquidusSourceOfTruth::test_paleos_liquidus_135gpa`
    - Command: `pytest tests/interior_struct/test_liquidus_super_ic.py -k test_paleos_liquidus_135gpa`

## Coverage

The `liquidus_super` mode solves for the coolest adiabat that is fully molten
everywhere with at least $\Delta T_\mathrm{super}$ of superheat above the
liquidus. For a PALEOS mantle the liquidus comes from the Zalmoxis
`melting_curves` fit: the Belonoshko et al. (2005) branch below 2.55 GPa and the
Fei et al. (2021) branch above it. For a WolfBower2018 or RTPress100TPa mantle it
is the melting curve that `interior_struct.melting_dir` selects, from P-T files in
`FWL_DATA`; without `melting_dir` it is the same fit as for a PALEOS mantle.
The reference-pinned test fixes the Fei et al. (2021) high-pressure
value at the 1 M$_\oplus$ reference pressure so any change to the upstream
Zalmoxis fit forces a re-validation of the superheat the solve evaluates.
Companion tests in the same file check branch continuity at the 2.55 GPa
crossover and monotonicity of the liquidus across the magma-ocean pressure
range.

## Last verified
2026-10-16
