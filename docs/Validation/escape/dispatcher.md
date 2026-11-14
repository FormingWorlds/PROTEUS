# dispatcher.py Validation

## What this page checks

This page checks `src/proteus/escape/dispatcher.py`, the coupling of PROTEUS to the escape regime dispatcher of ZEPHYRUS. The dispatcher chooses how the atmosphere escapes at each step and returns a mass loss rate for each element. ZEPHYRUS validates the physics of each regime in its own documentation. This page checks what PROTEUS adds: the planet state and the atmosphere column it hands over, the units of the settings, and the rates it writes back. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If this coupling is wrong, every escape rate of a run is wrong while the ZEPHYRUS tests still pass.

## Source under test
`src/proteus/escape/dispatcher.py` (the settings conversion, the atmosphere column handed to the dispatcher, and the helpfile record of its result).

## Reference-pinned tests

| Test ID | Reference | What is pinned |
|---|---|---|
| `test_dispatcher::test_energy_limited_state_matches_the_energy_limited_formula` | Energy-limited escape, $\dot M = \epsilon \pi R_\mathrm{xuv}^2 F_\mathrm{xuv} R_p / (G M_p)$ (Erkaev et al. 2007, without the tidal factor), as the released `zephyrus.escape.EL_escape` evaluates it | The rate of a state the dispatcher labels energy-limited, at three XUV fluxes, to `rel=1e-12`, with the per-element rates summing to the bulk rate. |

## An energy-limited state escapes at the energy-limited rate

**Why this check exists.** PROTEUS builds the planet state that the dispatcher reads: the mass, the radius, the orbit, the XUV flux, and the atmosphere column. A wrong unit or a radius taken from the wrong level changes every rate, and the ZEPHYRUS tests cannot see it.

**What it tests.** A planet of 5 Earth masses and 1.5 Earth radii with an isothermal envelope of hydrogen and helium at 800 K, at 0.1 au from a Sun-like star, at XUV fluxes of 1, 10, and 100 W m⁻². The photon and recombination limits are off, so the hydrodynamic rate is the energy-limited one. The test runs the dispatcher through PROTEUS and checks that it labels the state energy-limited and that the rate equals the energy-limited formula at the radius of the 0.02 bar level of the same column. It also checks that the rates of hydrogen and helium sum to the bulk rate and that the rate is proportional to the XUV flux.

**How we know the right answer.** The energy-limited rate is a closed form. The test evaluates it with the released ZEPHYRUS function `EL_escape` (`scaling = 2`), which ZEPHYRUS validates against Erkaev et al. (2007), at the photospheric radius read from the column.

**Why it matters.** The energy-limited form is the escape rate PROTEUS used before the dispatcher, so this case also ties the dispatcher's rates to the earlier ones on the same state.

**Known limit.** The case checks the coupling on one branch. The other branches (boil-off, recombination- and photon-limited winds, Jeans escape, and Roche-lobe overflow) take the same inputs, but their rates are pinned only in ZEPHYRUS.

![Escape rate against XUV flux for an energy-limited state](../../assets/validation/escape_dispatcher_energy_limited.svg#only-light){ width="80%" }
![Escape rate against XUV flux for an energy-limited state](../../assets/validation/escape_dispatcher_energy_limited_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the XUV flux, and the y-axis the bulk escape rate, both on logarithmic axes. The solid line is the energy-limited formula at the photospheric radius. The circles are the rates of the dispatcher run through PROTEUS. The dashed line is the same formula with $R_\mathrm{xuv}^3$ in place of $R_\mathrm{xuv}^2 R_p$, the form PROTEUS used before the dispatcher, which sits above by $R_\mathrm{xuv}/R_p$. A pass means that the circles lie on the solid line.

??? info "Test details"
    - Reference: Erkaev et al. (2007) energy-limited escape, through `zephyrus.escape.EL_escape` with `scaling = 2`.
    - Tolerance: relative 1e-12 on the rate. The $R_\mathrm{xuv}^3$ form must miss by more than 5%.
    - Test id: `tests/escape/test_dispatcher.py::test_energy_limited_state_matches_the_energy_limited_formula`
    - Command: `pytest -m reference_pinned tests/escape/test_dispatcher.py --validation-plots`
    - Last compared: 13 November 2026.
