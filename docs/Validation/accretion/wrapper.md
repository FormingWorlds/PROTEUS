# wrapper.py Validation

## What this page checks

This page checks the dispatch of the atmospheric loss of a giant impact in `src/proteus/accretion/wrapper.py`. For each impact, the wrapper passes the collision to a loss law in ZEPHYRUS and gets back the eroded fraction of the atmosphere. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If the dispatch is wrong, an impact strips the wrong fraction of the atmosphere. Each pinned case below says why the check exists, what it tests, how we know the right answer and why it matters. A box "Test details" ends each case. The dispatch contract tests after them check the interface without a published reference.

## Source under test
`src/proteus/accretion/wrapper.py` (the impact atmosphere-loss dispatch:
`_impact_loss_fraction` with `accretion.atmloss_module = "zephyrus"`).

## Reference-pinned tests

| Test ID | Reference | What is pinned | Last comparison |
|---|---|---|---|
| `test_wrapper::test_zephyrus_loss_module_evaluates_the_kegerreis_law` | Kegerreis et al. (2020), ApJL 901, L31 (doi:10.3847/2041-8213/abb5fb), Eqn. 1 | The eroded atmosphere fraction the dispatch obtains from `zephyrus.collision.mass_loss` when `atmloss_law = 'kegerreis2020'` for an impact record of two identical Earth-like bodies head-on at their mutual escape speed, where the law collapses to `X = 0.64 * 0.5**0.325 = 0.510911`, pinned to `rel=1e-4`. Two asymmetric events (a half-radius impactor at one eighth the target mass, `b = 0.3`) pin the fraction on both sides of the target/impactor mass assignment (`0.2675` and `0.5258`, `rel=2e-3`), so a dispatch that interchanged the event's target and impactor fields would fail both absolute pins rather than survive as a permutation. | 2026-10-09 |
| `test_wrapper::test_roche2026_oracle_row_through_impact_loss_fraction` | Roche et al. (2026), arXiv:2610.06077 (doi:10.5281/zenodo.23192423), Set A oracle row 1 | The eroded atmosphere fraction returned by `_impact_loss_fraction` for `atmloss_law = 'roche2026'` matches the authors' scaling-law calculation ($X_\mathrm{atm,calc} = 0.562213$) within $5 \times 10^{-5}$ absolute tolerance for an impact record with $v_c = 19.27$ km/s, $b = 0.3, M_t^\mathrm{tot} = 1.007 M_\oplus$, refractory $M_t^r = 0.997 M_\oplus, M_i = 0.249 M_\oplus, R_t = 1.017 R_\oplus, R_i = 0.675 R_\oplus$, and running atmospheric fraction $f_\mathrm{atm} = 0.01002$; the authors' value evaluates nominal grid values $\gamma = 0.2, v/v_\mathrm{esc} = 2.0$ while the dispatch derives them from body masses and contact speed ($v/v_\mathrm{esc} = 2.000434, \gamma = 0.200014$), producing a residual of $-1.47 \times 10^{-5}$. | 2026-10-09 |

## Kegerreis et al. (2020) loss law

**Why this check exists.** The dispatch must pass the right fields of the impact record to the loss law. Interchanging the target and the impactor, or calling the wrong law, gives a wrong eroded fraction without an error message.

**What it tests.** For 2 identical Earth-like bodies colliding head-on at their mutual escape speed, the law collapses to $X = 0.64 \times 0.5^{0.325} = 0.510911$. The test checks this value, that it equals a direct call of `zephyrus.collision.mass_loss`, and 2 asymmetric events (a half-radius impactor at one eighth the target mass, impact parameter $b$ = 0.3) with the bodies in both roles: 0.2675 and 0.5258.

**How we know the right answer.** Eq. 1 of Kegerreis et al. (2020) gives the symmetric value in closed form. The asymmetric values differ, so a dispatch that interchanged the target and the impactor fails both.

**Why it matters.** The wrapper calls `_impact_loss_fraction` for every impact with `accretion.atmloss_module = "zephyrus"`.

![Atmospheric loss fraction against the impactor to target mass ratio](../../assets/validation/accretion_wrapper_kegerreis.svg#only-light){ width="80%" }
![Atmospheric loss fraction against the impactor to target mass ratio](../../assets/validation/accretion_wrapper_kegerreis_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the impactor over target mass, for 2 bodies of equal density that collide at their mutual escape speed. The y-axis is the atmospheric loss fraction $X$. The lines are `_impact_loss_fraction` with the Kegerreis law for impact parameter $b$ = 0 (solid) and 0.3 (dashed), as context. The filled circles are the asserted values for 2 Earth twins (ratio 1, $b$ = 0) and a half-radius impactor (ratio 1/8, $b$ = 0.3). The diamond is the same pair with target and impactor exchanged; it is drawn at ratio 1/8, not at its own ratio 8. The hollow squares are the pinned values 0.510911, 0.2675 and 0.5258, and the cross is the direct `zephyrus.collision.mass_loss` value for the twins. A pass means that each circle and the diamond sit in their squares, and the cross on the twin circle.

??? info "Test details"
    - Reference: Kegerreis et al. (2020), ApJL 901, L31 (doi:10.3847/2041-8213/abb5fb), Eqn. 1.
    - Tolerance: relative 1e-4 on 0.510911; relative 2e-3 on 0.2675 and 0.5258.
    - Test id: `tests/accretion/test_wrapper.py::test_zephyrus_loss_module_evaluates_the_kegerreis_law`
    - Command: `pytest -m reference_pinned tests/accretion/test_wrapper.py --validation-plots`

## Roche et al. (2026) oracle row

**Why this check exists.** The Roche et al. (2026) law takes the refractory mass of the target and quantities derived from the collision. A wrong mass, a wrong derived ratio or a wrong law gives a wrong eroded fraction without an error message.

**What it tests.** For the first row of the Set A oracle of the authors ($v_c$ = 19.27 km/s, $b$ = 0.3, target mass 1.007 Earth masses, impactor 0.249 Earth masses, atmospheric fraction 0.01002), `_impact_loss_fraction` must return the authors' value $X$ = 0.562213. The Kegerreis law and a dispatch with the bodies interchanged must both miss it by more than 0.10.

**How we know the right answer.** The authors' scaling-law calculation of that row. It uses the nominal grid values $\gamma$ = 0.2 and $v/v_\mathrm{esc}$ = 2.0, while the dispatch derives them from the masses and the contact speed ($\gamma$ = 0.200014 and $v/v_\mathrm{esc}$ = 2.000434); this leaves a residual of -1.47e-5, inside the tolerance.

**Why it matters.** It checks the second law the dispatch offers, `atmloss_law = 'roche2026'`.

![Atmospheric loss fraction of the Roche oracle row for 3 dispatches](../../assets/validation/accretion_wrapper_roche_oracle.svg#only-light){ width="80%" }
![Atmospheric loss fraction of the Roche oracle row for 3 dispatches](../../assets/validation/accretion_wrapper_roche_oracle_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The y-axis is the atmospheric loss fraction $X$. The x-axis names 3 ways to compute $X$ for the first reference row of Roche et al. (2026). The dashed line is their value $X$ = 0.562213, marked by the hollow square, and the grey band is 0.10 on each side. The circles are the asserted values: the Roche law through `_impact_loss_fraction`, the Kegerreis law, and the dispatch with target and impactor exchanged. A pass means that the first circle sits in the square (the test allows 5e-5) and the other 2 lie outside the band.

??? info "Test details"
    - Reference: Roche et al. (2026), arXiv:2610.06077 (doi:10.5281/zenodo.23192423), Set A oracle row 1.
    - Tolerance: absolute 5e-5 on 0.562213; the Kegerreis law and the interchanged dispatch more than 0.10 away.
    - Test id: `tests/accretion/test_wrapper.py::test_roche2026_oracle_row_through_impact_loss_fraction`
    - Command: `pytest -m reference_pinned tests/accretion/test_wrapper.py --validation-plots`

## Dispatch contract tests

| Test ID | Reference | What is tested |
|---|---|---|
| `test_wrapper::test_impact_loss_fraction_routes_each_law_and_passes_arguments` | Kegerreis et al. (2020); Roche et al. (2026), arXiv:2610.06077; dispatch interface contract | Law-specific selection via `atmloss_law` ('kegerreis2020' and 'roche2026') routes to `zephyrus.collision.impact_loss`. Collision parameters (contact speed, body masses, radii, bulk densities, impact angle) pass from the impact record, with `roche2026` receiving the event target mass multiplied by $(1 - f_\mathrm{atm})$, and the atmospheric mass fraction $f_\mathrm{atm} = m_\mathrm{atm} / M_\mathrm{planet}$ passes from the running planet state. |
| `test_wrapper::test_rock_vapour_included_in_fatm_and_excluded_from_strip` | PROTEUS volatile mass conservation and outgassing accounting | Rock-forming species in `hf_row` (e.g. `Si_kg_atm`) enter the envelope mass $f_\mathrm{atm}$ seen by the scaling law, but `_target_strip_amounts` never strips rock vapour because `<e>_kg_total` is zero for rock-forming elements and its inventory re-equilibrates with the magma ocean at each step. |
| `test_wrapper::test_roche2026_loss_module_evaluates_real_zephyrus` | Roche et al. (2026), arXiv:2610.06077; ZEPHYRUS collision library | The eroded atmosphere fraction returned by `_impact_loss_fraction` for `atmloss_law = 'roche2026'` matches direct evaluation of `zephyrus.collision.mass_loss_roche2026` to `rel=1e-12` across valid collision parameters ($v_c = 1.2 \times 10^4$ m/s, $b = 0.3$, total event mass $M_t^\mathrm{tot} = 1.0 M_\oplus$, refractory $M_t^r = 0.99 M_\oplus$, $M_i = 0.1 M_\oplus$, $f_\mathrm{atm} = 0.01$). |
| `test_wrapper::test_roche2026_parameter_flags_and_kegerreis_thin_atmosphere_warning` | Kegerreis et al. (2020); Roche et al. (2026), Sect. 4.1 | Out-of-range parameter flags, with the clamp value when one is active, emit one warning per impact for `roche2026` over all fitted ranges applying the 1% evaluation tolerance ($f_\mathrm{atm} \in [0.0099, 0.202]$, $M_{t,\oplus} \in [0.3465, 5.05]$, $\gamma \in [0.099, 0.5]$, $b \in [0, 0.909]$, $R_i/R_t \in [0.00099, 1.02515]$, with $v/v_\mathrm{esc}$ flagged only above 3.03); only the fit terms use the value marked 'evaluated at' while $v_\mathrm{esc}, Q'_R$, and the mass ratio use the raw collision state; suppress the 3% thin-atmosphere warning for `roche2026`; and preserve the 3% thin-atmosphere warning for `kegerreis2020`. |
| `test_wrapper::test_roche2026_stability_clamps_through_impact_loss_fraction` | Roche et al. (2026), arXiv:2610.06077; ZEPHYRUS collision library | Clamps on $M_{t,\oplus}$ ($< 10^{-3}$ and $> 10$), $\gamma$ ($< 10^{-3}$ and $> 0.5$), and $f_\mathrm{atm}$ ($> 0.4$) emit out-of-range flags with clamp annotations ('evaluated at'); sub-escape velocity ($v/v_\mathrm{esc} < 0.99$) emits the 'v_sub_escape' flag with near-field loss held at $v_c = v_\mathrm{esc}$ and marks the loss fraction as extrapolated; clamped $\gamma = 0.5$ evaluation matches direct fit evaluation with $\gamma = 0.5$ and differs from unclamped evaluation ($\gamma = 0.7$) by more than $10^{-3}$. |
| `test_wrapper::test_roche2026_twin_bodies_gamma_clamp_tail` | Roche et al. (2026), arXiv:2610.06077; ZEPHYRUS collision library | Twin bodies ($M_t = M_i$) evaluate $\gamma = 1/(2 - f_\mathrm{atm}) > 0.5$ which is clamped to $0.5$, formatting the union of out-of-range flags and clamped parameters; when collision parameters are out of range but none is clamped ($b = 0.95$ and $f_\mathrm{atm} = 0.005$), the clamp explanation tail is omitted. |
| `test_wrapper::test_roche2026_zero_energy_flag_inside_fitted_range` | Roche et al. (2026), arXiv:2610.06077; ZEPHYRUS collision library | Collisions with high $\gamma$ at $f_\mathrm{atm}$ inside the fitted range emit ('X_FF_zero_energy',) only, using the neutral warning header 'Roche et al. (2026) law:' without extrapolation note or tail. |
| `test_wrapper::test_impact_loss_zero_and_subthreshold_atmosphere_strips_nothing` | PROTEUS outgassing threshold and volatile accounting | When $f_\mathrm{atm} = 0$, real `impact_loss` returns 0.0 and `_target_strip_amounts` returns an empty mapping. When $m_\mathrm{atm}$ is positive but below `outgas.mass_thresh`, volatile stripping is suppressed. |
| `test_wrapper::test_target_mass_mismatch_warning_and_event_mass_dispatch` | PROTEUS coupling contract (dynamical timeline frame vs planet mass) | Collision parameters ($M_t, R_t, M_i, R_i, v_c, b$) passed to `zephyrus.collision.impact_loss` originate from the `ImpactEvent` record ($M_t^\mathrm{tot}$ for kegerreis2020, refractory $M_t^\mathrm{tot}(1 - f_\mathrm{atm})$ for roche2026) while $f_\mathrm{atm}$ is computed from the running planet state ($m_\mathrm{atm} / M_\mathrm{planet}$). A target mass mismatch exceeding 10% between `event.M_target_before` and `hf_row['M_planet']` logs a warning once, whereas matching target mass produces no warning. |
| `test_wrapper::test_roche2026_airless_target_and_trace_atmosphere_jump` | Roche et al. (2026), arXiv:2610.06077; ZEPHYRUS collision library | With `atmloss_law = 'roche2026'`, an airless target ($m_\mathrm{atm} = 0$) returns $f_\mathrm{loss} = 0.0$ exactly, delivering the impactor's full volatile inventory. A target with positive trace atmosphere gives $f_\mathrm{atm}$ below the $10^{-6}$ stability bound, evaluated at $10^{-6}$, and jumps to an evaluated loss fraction depending on collision parameters for target and impactor alike, emitting an out-of-range flag with clamp annotation. |

## Coverage

The dispatch feeds the selected law from the impact record and the running planet
state: collision speed, masses, radii, densities, and angle stay in the frame
the dynamical model produced them in; the record's `v_impact` is the speed at first
contact. Event mass is the total mass. The refractory mass passed to `roche2026` is
the event target mass multiplied by $(1 - f_\mathrm{atm})$, matching the input
convention of the Roche et al. (2026) scaling law. Kegerreis et al. (2020) take the
radii at the base of the atmosphere, the bulk densities without it, and name their
scenarios by atmosphere-free masses; the total event mass changes X by a relative
amount of order $f_\mathrm{atm}$.

The target atmospheric mass fraction $f_\mathrm{atm} = m_\mathrm{atm} / M_\mathrm{planet}$
is evaluated from the running planet state as the sum of `<e>_kg_atm` over `M_planet`
over all elements in `element_list`. This sum includes rock-vapour elements because
vapour adds to the envelope mass the scaling law sees; PROTEUS does not debit
stripped rock vapour because its atmospheric inventory re-equilibrates with the
magma ocean at each step.

The default is `roche2026`. Both laws are calibrated on hydrogen-helium atmospheres:
Kegerreis et al. (2020) used the HM80 equation of state, while Roche et al. (2026,
Sect. 2.1) fitted simulations with H2-He envelopes. For atmospheres of higher mean
molecular weight (H2O, CO2, O2), the shock-driven loss is an upper limit for a
given envelope mass (Roche et al. 2026, Sect. 4.3). The law covers the immediate
shock- and vapour-plume-driven loss only; it neglects later thermally driven loss
(an outflow driven by heat from the post-impact interior, Biersteker & Schlichting
2021), so for primordial H2-He envelopes the total loss can be higher (Roche et
al. 2026, Sect. 4.1); that later loss becomes negligible for envelopes of higher
mean molecular weight. Conversely, envelope miscibility may lead to overestimating
loss from massive young envelopes (Roche et al. 2026, Sect. 4.1).

With `roche2026`, an airless target ($m_\mathrm{atm} = 0$) returns $f_\mathrm{loss} = 0$,
so an impactor delivers its full volatile content. An atmosphere with $f_\mathrm{atm}$
between $10^{-6}$ and below 0.0099 (1 % tolerance) is flagged and extrapolated without
a clamp. Below the $10^{-6}$ stability bound, $f_\mathrm{atm}$ is evaluated at
$10^{-6}$, and the resulting loss fraction depends on the collision parameters. When
parameters are clamped, only the fit terms use the clamped values while $v_\mathrm{esc}$,
$Q'_R$, and the mass ratio use the raw collision state.

When the far-field term predicts loss without impact energy, the 'X_FF_zero_energy'
flag is set; this flag can fire inside the fitted range (for high $\gamma$ at
$f_\mathrm{atm}$ 0.01 to 0.1) and is common below $f_\mathrm{atm}$ 0.01. When only
the zero-energy flag is present (and no out-of-range parameter or clamp is active),
the warning uses the neutral header `Roche et al. (2026) law:` and omits both the
extrapolation note and tail; when out-of-range parameter flags, clamps, or the $f_\mathrm{atm}$
flag are present, the warning reports `Roche et al. (2026) law outside its fitted range:`
with the extrapolation note `(log10 f_atm extrapolation)` and appends the extrapolation tail.
