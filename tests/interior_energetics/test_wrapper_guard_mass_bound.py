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

import numpy as np
import pytest

from proteus.interior_energetics import wrapper as _wrapper
from proteus.interior_energetics.wrapper import _ZALMOXIS_MASS_ANCHOR_TOL
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

# The plan values stand in where the module does not define the bound, so each test fails on its
# own assertion there; test_the_bound_constants_cover_the_measured_steps requires the module's.
SLOPE = getattr(_wrapper, '_MASS_ESCAPE_RADIUS_SLOPE', 2.0)
OFFSET = getattr(_wrapper, '_MASS_ESCAPE_RADIUS_OFFSET', 5.0e-3)
BOUND_MSG = 'exceeds the radius bound of the mass escape'
MISS_MSG = 'Committed structure misses its mass target'


def _warnings(caplog, key):
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and key in r.getMessage()
    ]


def _bound(dm):
    return SLOPE * dm + OFFSET


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
@pytest.mark.parametrize(
    ('dm', 'dr'),
    [
        # dM/M and dR/R of measured mass-driven up-steps (data/fix1r1/steps_*.txt)
        pytest.param(8.9111e-02, 7.0910e-02, id='case 38 first rejected re-expansion'),
        pytest.param(1.6498e-01, 1.5520e-01, id='case 38 re-expansion with the least margin'),
        pytest.param(1.9129e-01, 1.7460e-01, id='old-code resume, largest re-expansion'),
        pytest.param(8.9063e-02, 7.5834e-02, id='fixed resume, first re-solve'),
        pytest.param(6.9826e-02, 6.2582e-02, id='fixed resume, second re-solve'),
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
def test_artifacts_do_not_add_up_over_anchor_trips(caplog):
    """A slowly rising target keeps each solve's own radius, so artifacts do not accumulate.

    Each solve returns R(M) = R0 (M / M0)^0.8 (1 + 7.2e-4): a mass-driven radius with the largest
    measured artifact. The target rises 0.31 % per step, just above the anchor, so every step is
    an anchor trip. After 5 trips the kept radius is the last solve's radius, one artifact above
    the mass-driven radius, and no bound warning fires.
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
        # the artifact reject of the fixed resume: target +9.9e-4, radius +7.19e-4
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
def test_a_fall_back_to_the_previous_structure_warns_when_it_misses_the_new_target(caplog):
    """A failed solve that set a 1 % larger target falls back to the old structure and warns."""
    config, hf_row, base = _cooling_case()

    def _fail(*args, **kwargs):
        args[2]['M_int_target'] = M_RETAINED * 1.01
        raise RuntimeError('mock convergence failure')

    msgs, _, interior_o = run_guard(caplog, hf_row, config, base, _fail)

    assert any('Falling back to previous structure' in m for m in msgs)
    assert interior_o.structure_stale is True
    assert hf_row['M_int'] == M_RETAINED
    assert len(_warnings(caplog, MISS_MSG)) == 1


def test_the_bound_constants_cover_the_measured_steps():
    """The slope and offset hold the measured extremes with a margin of at least 1.5."""
    assert _wrapper._MASS_ESCAPE_RADIUS_SLOPE >= 1.5 * 1.311
    assert _wrapper._MASS_ESCAPE_RADIUS_OFFSET >= 1.5 * 2.23e-3
    assert np.isfinite(_bound(1.0))
