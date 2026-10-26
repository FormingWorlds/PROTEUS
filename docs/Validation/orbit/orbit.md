# Validation: `src/proteus/orbit/orbit.py`

## What this page checks

This page checks the tidal evolution of the orbit in `proteus.orbit.orbit`. Tides raised on the planet by the star change the eccentricity $e$ of the orbit; the model `sp0d` evolves the orbit alone, and `sp1d` also follows the spin of the planet. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If this part of the code is wrong, the eccentricity and the spin of the planet evolve at the wrong rate. Each case below says why the check exists, what it tests, how we know the right answer and why it matters. A box "Test details" ends each case. The `reference_pinned` marker is registered in `pyproject.toml`.

| Test id | Reference | Source page | Scope |
|---|---|---|---|
| `tests/orbit/test_orbit.py::test_sp0d_de_dt_matches_driscoll_barnes_2015_eq16` | Driscoll and Barnes (2015), Astrobiology 15, 739 (DOI 10.1089/ast.2015.1325; arXiv:1509.07452), Eq. 16 | n/a (closed form) | Pins the prefactor `21/2`, the `a^-6.5` exponent, the `R_pl^5` scaling, and the linear-in-`e` dependence of `sp0d`'s tidal eccentricity-damping rate at unit-scale parameters. Also asserts sign and order-of-magnitude, with discrimination guards against `a^5` and `a^7` neighbouring exponents. |
| `tests/orbit/test_orbit.py::test_sp1d_conserves_total_angular_momentum` | Angular-momentum conservation for an isolated two-body system with only internal tidal torques (analytical limit, not paper-specific) | n/a (closed form) | For `sp1d` (which, unlike `sp0d`, tracks the planet's spin), pins that planet-spin + orbital angular momentum is conserved to `rel=1e-6` across a real, non-trivial step (`e`: 0.3 to below 0.1). |
| `tests/orbit/test_orbit.py::test_sp1d_spin_am_gain_matches_orbital_am_loss` | Same conservation law as above, more targeted | n/a (closed form) | The planet's spin AM is ~1e-6 of the system total, so the raw-sum check above is nearly blind to a bug confined to `domega_dt`. This test instead pins `Delta(spin AM) == -Delta(orbital AM)` directly, at a tolerance where the two comparable-magnitude quantities discriminate a coefficient bug the raw-sum check misses. |

## Eccentricity rate of sp0d

**Why this check exists.** The eccentricity rate has a steep power of the semi-major axis and of the planet radius. A wrong exponent or prefactor changes the rate by large factors without an error message.

**What it tests.** Driscoll and Barnes (2015), Eq. 16, give $de/dt = (21/2)\, \mathrm{Im}(k_2)\, M_\star^{3/2} G^{1/2} R_p^5 / (M_p\, a^{13/2})\; e$, with $k_2$ the Love number, $M_\star$ and $M_p$ the masses of the star and the planet, $R_p$ the planet radius and $a$ the semi-major axis. The test sets $\mathrm{Im}(k_2) = M_\star = R_p = M_p = 1$, $a = 2$ and $e = 0.5$, with the real gravitational constant $G$, and compares the rate with the closed form. With $a^5$ in place of $a^{6.5}$ the rate would be more than 100 % off, and with $a^7$ more than 10 %; the test checks both, and that the rate is positive.

**How we know the right answer.** The closed form of Eq. 16 at these inputs.

**Why it matters.** `sp0d` gives the eccentricity evolution of a run with this tidal model.

**Known limit.** The test checks the algebra under the sign convention of the source, not that the convention matches the paper (see the note below).

??? info "Test details"
    - Reference: Driscoll and Barnes (2015), Astrobiology 15, 739, Eq. 16 (closed form).
    - Tolerance: relative 1e-3; the $a^5$ form must miss by more than 100 % and the $a^7$ form by more than 10 %.
    - Test id: `tests/orbit/test_orbit.py::test_sp0d_de_dt_matches_driscoll_barnes_2015_eq16`
    - Command: `pytest tests/orbit/test_orbit.py::test_sp0d_de_dt_matches_driscoll_barnes_2015_eq16`

## Angular momentum of sp1d

**Why this check exists.** Tides move angular momentum between the orbit and the spin of the planet. A wrong coefficient in only one of the equations makes the total change, which no other output shows.

**What it tests.** The test takes one `sp1d` step of 1e5 yr in which the eccentricity falls from 0.3 to below 0.2. The sum of the spin angular momentum of the planet and the orbital angular momentum must stay the same.

**How we know the right answer.** For an isolated 2-body system with only internal tidal torques, the total angular momentum is conserved.

**Why it matters.** `sp1d` gives the spin and the orbit of a run with this tidal model.

**Known limit.** The spin angular momentum is about 1e-6 of the total, so this check cannot see an error in the spin equation alone; the next case covers it.

??? info "Test details"
    - Reference: angular-momentum conservation of an isolated 2-body system (analytical limit).
    - Tolerance: relative 1e-6 on the total.
    - Test id: `tests/orbit/test_orbit.py::test_sp1d_conserves_total_angular_momentum`
    - Command: `pytest tests/orbit/test_orbit.py::test_sp1d_conserves_total_angular_momentum`

## Spin gain equals orbital loss in sp1d

**Why this check exists.** An error in the spin rate `domega_dt` alone changes a quantity that is about 1e-6 of the total, so the case above cannot see it.

**What it tests.** Over the same step, the change of the spin angular momentum must equal minus the change of the orbital angular momentum. Both changes must be larger than 1e30 in size, the spin must gain and the orbit must lose.

**How we know the right answer.** The same conservation law as above, applied to the 2 changes, which are of comparable size.

**Why it matters.** It checks the spin equation of `sp1d` on its own.

??? info "Test details"
    - Reference: angular-momentum conservation, applied to the exchange between spin and orbit (analytical limit).
    - Tolerance: relative 5e-2 between the spin gain and the orbital loss.
    - Test id: `tests/orbit/test_orbit.py::test_sp1d_spin_am_gain_matches_orbital_am_loss`
    - Command: `pytest tests/orbit/test_orbit.py::test_sp1d_spin_am_gain_matches_orbital_am_loss`

## Sign convention note

The paper uses `Im(k2) < 0` for tidal dissipation (Eq. 4 expresses
`-Im(k2)` as the positive dissipation efficiency). The PROTEUS source
takes positive `Imk2` from callers (`run_dummy_tides`, `run_lovepy`),
so the formula evaluated with positive `Imk2` returns positive `de/dt`
and expands the orbit instead of circularizing it. This is documented
in the source docstring as a known science item; the test pins the
algebra under the source convention and does not show that the convention
matches the paper. A sign correction must change every `Imk2` producer in the
ecosystem together.

## Re-derivation note

Driscoll & Barnes (2015) Eq. 16 reads

```
de/dt = (21/2) * Imk2 * Mst^1.5 * G^0.5 * Rpl^5 / (Mpl * a^6.5) * e
```

The test uses `Imk2 = Mst = Rpl = Mpl = 1`, `a = 2` and `e = 0.5` with the real `const_G`, so the closed-form result is `(21/2) * const_G**0.5 * 0.5 / 2**6.5`. A rate with `a^5` in place of `a^6.5` differs from it by more than 100 %, and one with `a^7` by more than 10 %; the test asserts both.
