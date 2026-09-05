# wrapper.py Validation

## Source under test
`src/proteus/accretion/wrapper.py` (the impact atmosphere-loss dispatch:
`_impact_loss_fraction` with `accretion.atmloss_module = "zephyrus"`).

## Reference-pinned tests

| Test ID | Reference | What is pinned |
|---|---|---|
| `test_wrapper::test_zephyrus_loss_module_evaluates_the_kegerreis_law` | Kegerreis et al. (2020), ApJL 901, L31 (doi:10.3847/2041-8213/abb5fb), Eqn. 1 | The eroded atmosphere fraction the dispatch obtains from `zephyrus.collision.mass_loss` for an impact record of two identical Earth-like bodies head-on at their mutual escape speed, where the law collapses to `X = 0.64 * 0.5**0.325 = 0.510911`, pinned to `rel=1e-4`. Two asymmetric events (a half-radius impactor at one eighth the target mass, `b = 0.3`) pin the fraction on both sides of the target/impactor mass assignment (`0.2675` and `0.5258`, `rel=2e-3`), so a dispatch that interchanged the event's target and impactor fields would fail both absolute pins rather than survive as a permutation. |

## Coverage

The dispatch feeds the law entirely from the impact record, so the collision
speed, masses, radii, densities, and angle stay in the frame the dynamical
model produced them in; the record's `v_impact` is the speed at first contact
and its bodies carry no modelled atmosphere, matching the conventions of the
law (see the ZEPHYRUS validation page for the law's own anchors against the
paper's closed form and its Table 2 simulation suite). The dispatch-level
pins certify the record-to-argument mapping and the returned fraction's
bounds; the law's internal physics is certified in ZEPHYRUS.
