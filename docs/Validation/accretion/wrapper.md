# wrapper.py Validation

## Source under test
`src/proteus/accretion/wrapper.py` (the impact atmosphere-loss dispatch:
`_impact_loss_fraction` with `accretion.atmloss_module = "zephyrus"`).

## Reference-pinned tests

| Test ID | Reference | What is pinned | Last comparison |
|---|---|---|---|
| `test_wrapper::test_zephyrus_loss_module_evaluates_the_kegerreis_law` | Kegerreis et al. (2020), ApJL 901, L31 (doi:10.3847/2041-8213/abb5fb), Eqn. 1 | The eroded atmosphere fraction the dispatch obtains from `zephyrus.collision.mass_loss` when `atmloss_law = 'kegerreis2020'` for an impact record of two identical Earth-like bodies head-on at their mutual escape speed, where the law collapses to `X = 0.64 * 0.5**0.325 = 0.510911`, pinned to `rel=1e-4`. Two asymmetric events (a half-radius impactor at one eighth the target mass, `b = 0.3`) pin the fraction on both sides of the target/impactor mass assignment (`0.2675` and `0.5258`, `rel=2e-3`), so a dispatch that interchanged the event's target and impactor fields would fail both absolute pins rather than survive as a permutation. | 2026-10-09 |
| `test_wrapper::test_roche2026_oracle_row_through_impact_loss_fraction` | Roche et al. (2026), arXiv:2610.06077 (doi:10.5281/zenodo.23192423), Set A oracle row 1 | The eroded atmosphere fraction returned by `_impact_loss_fraction` for `atmloss_law = 'roche2026'` matches the authors' scaling-law calculation ($X_\mathrm{atm,calc} = 0.562213$) within $2 \times 10^{-4}$ absolute tolerance for an impact record with $v_c = 19.27$ km/s, $b = 0.3$, $M_t^\mathrm{tot} = 1.007 M_\oplus$, refractory $M_t^r = 0.997 M_\oplus$, $M_i = 0.249 M_\oplus$, $R_t = 1.017 R_\oplus$, $R_i = 0.675 R_\oplus$, and running atmospheric fraction $f_\mathrm{atm} = 0.01002$. | 2026-10-09 |

## Dispatch contract tests

| Test ID | Reference | What is tested |
|---|---|---|
| `test_wrapper::test_impact_loss_fraction_routes_each_law_and_passes_arguments` | Kegerreis et al. (2020); Roche et al. (2026), arXiv:2610.06077; dispatch interface contract | Law-specific selection via `atmloss_law` ('kegerreis2020' and 'roche2026') routes to `zephyrus.collision.impact_loss`. Collision parameters (contact speed, body masses, radii, bulk densities, impact angle) pass from the impact record, and the atmospheric mass fraction $f_\mathrm{atm} = m_\mathrm{atm} / M_\mathrm{planet}$ passes from the running planet state. |
| `test_wrapper::test_roche2026_loss_module_evaluates_real_zephyrus` | Roche et al. (2026), arXiv:2610.06077; ZEPHYRUS collision library | The eroded atmosphere fraction returned by `_impact_loss_fraction` for `atmloss_law = 'roche2026'` matches direct evaluation of `zephyrus.collision.mass_loss_roche2026` to `rel=1e-12` across valid collision parameters ($v_c = 1.2 \times 10^4$ m/s, $b = 0.3$, $M_t = 1.0 M_\oplus$, $M_i = 0.1 M_\oplus$, $f_\mathrm{atm} = 0.01$). |
| `test_wrapper::test_roche2026_flags_produce_warnings_and_kegerreis_3pct_absent` | Kegerreis et al. (2020); Roche et al. (2026), Sect. 4.1 | Out-of-range flags, with the clamp value when one is active, emit one warning per impact for `roche2026` over all fitted ranges applying the 1% evaluation tolerance ($f_\mathrm{atm} \in [0.0099, 0.202]$, $M_{t,\oplus} \in [0.3465, 5.05]$, $\gamma \in [0.099, 0.505]$, $b \in [0, 0.909]$, $R_i/R_t \in [0.00099, 1.02515]$, with $v/v_\mathrm{esc}$ flagged only above 3.03); only the fit terms use the clamped value while $v_\mathrm{esc}, Q'_R$, and the mass ratio use the raw collision state; suppress the 3% thin-atmosphere warning for `roche2026`; and preserve the 3% thin-atmosphere warning for `kegerreis2020`. |
| `test_wrapper::test_target_mass_mismatch_warning_and_event_mass_dispatch` | PROTEUS coupling contract (dynamical timeline frame vs planet mass) | Collision parameters ($M_t, R_t, M_i, R_i, v_c, b$) passed to `zephyrus.collision.impact_loss` originate from the `ImpactEvent` record while $f_\mathrm{atm}$ is computed from the running planet state ($m_\mathrm{atm} / M_\mathrm{planet}$). A target mass mismatch exceeding 10% between `event.M_target_before` and `hf_row['M_planet']` logs a warning once, whereas matching target mass produces no warning. |
| `test_wrapper::test_roche2026_airless_target_and_trace_atmosphere_jump` | Roche et al. (2026), arXiv:2610.06077; ZEPHYRUS collision library | With `atmloss_law = 'roche2026'`, an airless target ($m_\mathrm{atm} = 0$) returns $f_\mathrm{loss} = 0.0$ exactly, delivering the impactor's full volatile inventory. A target with positive trace atmosphere gives $f_\mathrm{atm}$ below the $10^{-6}$ stability bound, evaluated at $10^{-6}$, and jumps to an evaluated loss fraction depending on collision parameters for target and impactor alike, emitting an out-of-range flag with clamp annotation. |

## Coverage

The dispatch feeds the selected law from the impact record and the running planet
state: collision speed, masses, radii, densities, and angle stay in the frame
the dynamical model produced them in; the record's `v_impact` is the speed at first
contact. Event mass is the total mass. The refractory mass passed to `roche2026` is
the event target mass multiplied by $(1 - f_\mathrm{atm})$, matching the input
convention of the Roche et al. (2026) scaling law. For `kegerreis2020`, passing the
total event mass introduces an error of order $f_\mathrm{atm}$ because that law is
defined on atmosphere-free bodies.

The target atmospheric mass fraction $f_\mathrm{atm} = m_\mathrm{atm} / M_\mathrm{planet}$
is evaluated from the running planet state as the sum of `<e>_kg_atm` over `M_planet`
over all elements in `element_list`. This sum includes rock-vapour elements because
the loss fraction applies to each element in that same list; excluding vapour would make
$f_\mathrm{atm}$ a fraction of a different mass reservoir than the one the law strips.

An existing configuration with `atmloss_module = "zephyrus"` and no `atmloss_law`
now evaluates `roche2026`. Both laws are calibrated on hydrogen-helium atmospheres:
Kegerreis et al. (2020) used the HM80 equation of state, while Roche et al. (2026,
Sect. 2.1, 4.1) fitted simulations with H2-He envelopes. Because heavier atmospheres
(such as carbon- or oxygen-rich envelopes) are harder to strip, the H2-He scaling laws
provide upper limits on atmospheric loss (Roche et al. 2026, Sect. 4.3).

With `roche2026`, an airless target ($m_\mathrm{atm} = 0$) returns $f_\mathrm{loss} = 0$,
so an impactor delivers its full volatile content. An atmosphere with $f_\mathrm{atm}$
between $10^{-6}$ and 0.01 is flagged and extrapolated without a clamp. Below the
$10^{-6}$ stability bound, $f_\mathrm{atm}$ is evaluated at $10^{-6}$, and the resulting
loss fraction depends on the collision parameters. When parameters are clamped, only the
fit terms use the clamped values while $v_\mathrm{esc}$, $Q'_R$, and the mass ratio use
the raw collision state. In coupled evolution runs such as the Morrigan run and the
four-impact timeline run, $f_\mathrm{atm}$ ranges from 0.013 to 0.035, so real impacts
remain far from the airless-target boundary.

## Coupled behaviour

Coupled simulations evaluate the atmosphere loss dispatch under full planet
evolution for two test configurations: the Morrigan run (a 1.3 M_earth planet following
Morrigan seed 7 with two giant impacts over 3000 yr) and the four-impact timeline run
(four impacts over 300 yr).

In these oxygen-buffered configurations (IW+4 magma-ocean redox state), the
atmosphere prior to each impact is dominated by O2 by mass (species O2 in the run output)
at surface pressures of roughly 2e4 bar and temperatures near 3200 K, which lies outside
the hydrogen-helium calibration regime of both scaling laws (Kegerreis et al. 2020; Roche
et al. 2026, Sect. 2.1, 4.1). Both laws yield comparable post-impact bulk atmospheric masses
because rapid magma-ocean outgassing replenishes the lost oxygen, while the volatile
element inventories and cumulative escape reflect the different erosion fractions.
