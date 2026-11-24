"""Monotonic-radius guard: the radius bound of the mass escape and the committed-mass check.

The mass escape keeps a cooling R_int up-step only while the radius step stays within
``_MASS_ESCAPE_RADIUS_SLOPE * dM/M + _MASS_ESCAPE_RADIUS_OFFSET``, with dM/M the relative rise of
the solve target over the retained mass. The structure step warns when the committed M_int misses
its live target by more than the Zalmoxis mass anchor.

See also:
- docs/How-to/testing.md: Running, writing, and marking tests; coverage and CI
"""

from __future__ import annotations

import logging
import math

import pytest

from proteus.interior_energetics.wrapper import (
    _MASS_ESCAPE_RADIUS_OFFSET,
    _MASS_ESCAPE_RADIUS_SLOPE,
    _ZALMOXIS_MASS_ANCHOR_TOL,
)
from tests.interior_energetics.test_wrapper_guard_mass_target import (
    ACCEPT_MSG,
    M_RETAINED,
    R_RETAINED,
    REJECT_MSG,
    _cooling_case,
    _solve_to,
    run_guard,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.timeout(30),
    pytest.mark.usefixtures('_no_core_table'),
]

BOUND_MSG = 'exceeds the radius bound of the mass escape'
MISS_MSG = 'Committed structure misses its mass target'


def _warnings(caplog, key):
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and key in r.getMessage()
    ]


def _bound(dm):
    return _MASS_ESCAPE_RADIUS_SLOPE * dm + _MASS_ESCAPE_RADIUS_OFFSET


@pytest.mark.physics_invariant
def test_a_radius_step_far_above_its_mass_step_is_rejected_with_a_warning(caplog):
    """A +20 % R_int up-step at a +0.31 % rise of the target is not kept by the mass escape.

    The retained mass misses the target by more than the anchor, but the radius step is far above
    any measured mass-driven step, so the guard keeps the retained structure and warns with the
    step, the mass rise and the bound.
    """
    config, hf_row, base = _cooling_case()
    ratio = 1.0 / (1.0 - 3.1e-3)
    msgs, _, _ = run_guard(caplog, hf_row, config, base, _solve_to(ratio, 1.20))

    assert any(REJECT_MSG in m for m in msgs)
    assert not any(ACCEPT_MSG in m for m in msgs)
    warned = _warnings(caplog, BOUND_MSG)
    assert len(warned) == 1
    assert 'dR/R=2.000e-01' in warned[0]
    assert 'dM/M=%.3e' % (ratio - 1.0) in warned[0]
    assert 'bound %.3e' % _bound(ratio - 1.0) in warned[0]
    assert hf_row['R_int'] == R_RETAINED
    assert hf_row['M_int'] == M_RETAINED


@pytest.mark.physics_invariant
def test_a_one_percent_step_at_an_anchor_trip_is_rejected(caplog):
    """A +1.1 % R_int up-step at a +0.31 % rise of the target is refused.

    At an anchor trip a re-expansion is at most 0.941 dM/M plus the largest cooling up-step
    measured with the target inside the anchor, 7.2e-4: about 3.6e-3 here; +1.1 % is 3 times that.
    """
    config, hf_row, base = _cooling_case()
    msgs, _, _ = run_guard(caplog, hf_row, config, base, _solve_to(1.0 / (1.0 - 3.1e-3), 1.011))

    assert any(REJECT_MSG in m for m in msgs)
    assert len(_warnings(caplog, BOUND_MSG)) == 1
    assert hf_row['R_int'] == R_RETAINED


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('dm', 'dr'),
    [
        # dM/M and dR/R of re-expansions measured in coupled runs of a planet of 8.63 Earth masses
        pytest.param(8.9111e-02, 7.0910e-02, id='re-expansion at +8.9 % mass, ratio 0.80'),
        pytest.param(1.6498e-01, 1.5520e-01, id='re-expansion at +16.5 % mass, ratio 0.94'),
        pytest.param(1.9129e-01, 1.7460e-01, id='re-expansion at +19.1 % mass, ratio 0.91'),
        pytest.param(8.9063e-02, 7.5834e-02, id='re-expansion at +8.9 % mass, ratio 0.85'),
        pytest.param(6.9826e-02, 6.2582e-02, id='re-expansion at +7.0 % mass, ratio 0.90'),
    ],
)
def test_measured_mass_steps_pass_the_radius_bound(caplog, dm, dr):
    """Every measured re-expansion of a stuck planet is kept, with no bound warning."""
    config, hf_row, base = _cooling_case()
    msgs, _, _ = run_guard(caplog, hf_row, config, base, _solve_to(1.0 + dm, 1.0 + dr))

    assert any(ACCEPT_MSG in m for m in msgs)
    assert not _warnings(caplog, BOUND_MSG)
    assert hf_row['R_int'] == pytest.approx(R_RETAINED * (1.0 + dr), rel=1e-12)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(('scale', 'kept'), [(1.0 - 1e-9, True), (1.0 + 1e-9, False)])
def test_the_radius_bound_holds_at_its_edge(caplog, scale, kept):
    """A radius step just inside the bound is kept; one just outside is rejected."""
    config, hf_row, base = _cooling_case()
    dm = 1.0e-2
    msgs, _, _ = run_guard(
        caplog, hf_row, config, base, _solve_to(1.0 + dm, 1.0 + _bound(dm) * scale)
    )

    assert any(ACCEPT_MSG in m for m in msgs) is kept
    assert bool(_warnings(caplog, BOUND_MSG)) is not kept


@pytest.mark.physics_invariant
def test_repeated_anchor_trips_keep_each_solves_own_radius(caplog):
    """Each anchor trip of a slowly rising target keeps the radius of its own solve.

    Each solve returns R(M) = R0 (M / M0)^0.8 (1 + 7.2e-4): a mass-driven radius with the largest
    measured artifact. The target rises 0.31 % per step, just above the anchor, so every step is
    an anchor trip. Each step is kept with no bound warning, and after 5 trips the kept radius is
    the last solve's radius.
    """
    config, hf_row, base = _cooling_case()
    artifact = 7.2e-4
    m = M_RETAINED
    for trip in range(5):
        m *= 1.0 / (1.0 - 3.1e-3)

        def _solve(*args, _m=m, **kwargs):
            hr = args[2]
            hr['R_int'] = R_RETAINED * (_m / M_RETAINED) ** 0.8 * (1.0 + artifact)
            hr['M_int_target'] = _m
            hr['M_int'] = _m
            return (3.504e6, None)

        hf_row['Time'] = 1000.0 * (trip + 1)
        msgs, _, _ = run_guard(caplog, hf_row, config, base, _solve)
        assert any(ACCEPT_MSG in m_ for m_ in msgs)
        assert not _warnings(caplog, BOUND_MSG)

    r_mass = R_RETAINED * (m / M_RETAINED) ** 0.8
    assert hf_row['R_int'] == pytest.approx(r_mass * (1.0 + artifact), rel=1e-12)


@pytest.mark.physics_invariant
def test_a_falling_target_with_an_artifact_up_step_warns_about_the_heavier_structure(caplog):
    """A reject against a smaller target keeps a heavier structure, and the step warns.

    The solve target falls 3.5e-3 below the retained mass and the solve up-steps by the largest
    measured artifact, 7.2e-4. The mass escape takes only a larger target, so the guard keeps
    the retained structure, which now misses the live target by more than the anchor.
    """
    config, hf_row, base = _cooling_case()
    ratio = 1.0 - 3.5e-3
    msgs, _, _ = run_guard(caplog, hf_row, config, base, _solve_to(ratio, 1.0 + 7.2e-4))

    assert any(REJECT_MSG in m for m in msgs)
    assert hf_row['M_int'] == M_RETAINED
    assert hf_row['M_int_target'] == M_RETAINED * ratio
    warned = _warnings(caplog, MISS_MSG)
    assert len(warned) == 1
    assert 'M_int=%.6e kg' % M_RETAINED in warned[0]
    assert 'M_int_target=%.6e kg' % (M_RETAINED * ratio) in warned[0]
    assert 'relative miss %.3e' % abs(M_RETAINED / (M_RETAINED * ratio) - 1.0) in warned[0]


@pytest.mark.physics_invariant
def test_a_bound_reject_warns_that_the_structure_misses_its_target(caplog):
    """The bound reject of a too-large radius step leaves the planet off its target, and says so."""
    config, hf_row, base = _cooling_case()
    run_guard(caplog, hf_row, config, base, _solve_to(1.0 / (1.0 - 3.1e-3), 1.20))

    assert len(_warnings(caplog, MISS_MSG)) == 1


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('ratio', 'radius'),
    [
        # the largest cooling up-step measured inside the anchor: target +9.9e-4, radius +7.19e-4
        pytest.param(1.0 + 9.9e-4, 1.0 + 7.19e-4, id='artifact reject inside the anchor'),
        pytest.param(8.010 / 7.355, 1.0709, id='accepted re-expansion'),
        pytest.param(1.0, 1.0 - 1.0e-3, id='accepted down-step'),
    ],
)
def test_no_mass_warning_when_the_committed_structure_meets_its_target(caplog, ratio, radius):
    """Neither an artifact reject inside the anchor nor an accepted solve warns about the mass."""
    config, hf_row, base = _cooling_case()
    run_guard(caplog, hf_row, config, base, _solve_to(ratio, radius))

    assert not _warnings(caplog, MISS_MSG)
    assert abs(hf_row['M_int'] / hf_row['M_int_target'] - 1.0) <= _ZALMOXIS_MASS_ANCHOR_TOL


@pytest.mark.physics_invariant
def test_a_fall_back_after_a_mass_anchor_miss_warns_about_the_new_target(caplog):
    """A solve that misses its 1 % larger target fails the anchor; the fall-back warns.

    The solve succeeds and writes the new target, but its mass stays at the retained mass, so the
    mass-anchor check refuses it and the previous structure comes back against the new target.
    """
    config, hf_row, base = _cooling_case()

    def _solve(*args, **kwargs):
        args[2].update(
            R_int=R_RETAINED * 0.999, M_int_target=M_RETAINED * 1.01, M_int=M_RETAINED
        )
        return (3.504e6, None)

    msgs, _, interior_o = run_guard(caplog, hf_row, config, base, _solve)

    assert any('Falling back to previous structure' in m for m in msgs)
    assert interior_o.structure_stale is True
    assert hf_row['R_int'] == R_RETAINED
    assert hf_row['M_int'] == M_RETAINED
    assert len(_warnings(caplog, MISS_MSG)) == 1


@pytest.mark.physics_invariant
def test_a_fall_back_after_a_raising_solve_keeps_the_old_target_and_does_not_warn(caplog):
    """A solve that raises writes no target, so the restored structure meets its old target."""
    config, hf_row, base = _cooling_case()

    def _fail(*args, **kwargs):
        raise RuntimeError('mock convergence failure')

    msgs, _, _ = run_guard(caplog, hf_row, config, base, _fail)

    assert any('Falling back to previous structure' in m for m in msgs)
    assert hf_row['M_int_target'] == M_RETAINED
    assert not _warnings(caplog, MISS_MSG)


@pytest.mark.physics_invariant
def test_a_non_finite_committed_mass_warns(caplog):
    """A solve that returns a NaN mass on its target commits it; the step warns."""
    config, hf_row, base = _cooling_case()

    def _solve(*args, **kwargs):
        args[2].update(R_int=R_RETAINED * 0.999, M_int_target=M_RETAINED, M_int=float('nan'))
        return (3.504e6, None)

    run_guard(caplog, hf_row, config, base, _solve)

    assert math.isnan(hf_row['M_int'])
    warned = _warnings(caplog, MISS_MSG)
    assert len(warned) == 1
    assert 'M_int=nan kg' in warned[0]
