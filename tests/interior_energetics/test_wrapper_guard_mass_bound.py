"""Monotonic-radius guard: the radius bound warning of the mass escape and the committed-mass check.

The mass escape keeps a cooling R_int up-step whenever the retained mass misses a larger target
by more than the Zalmoxis mass anchor. When it is the only escape and the radius step exceeds
``_MASS_ESCAPE_RADIUS_SLOPE * dM/M + _MASS_ESCAPE_RADIUS_OFFSET``, with dM/M the rise of the mass
the solve reached over the retained mass, the step warns. The structure step also warns when the
committed M_int misses its live target by more than the anchor.

See also:
- docs/How-to/testing.md: Running, writing, and marking tests; coverage and CI
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

import numpy as np
import pytest

from proteus.interior_energetics.wrapper import (
    _MASS_ESCAPE_RADIUS_OFFSET,
    _MASS_ESCAPE_RADIUS_SLOPE,
    _ZALMOXIS_MASS_ANCHOR_TOL,
    _warn_committed_mass_miss,
)
from tests.interior_energetics.test_wrapper import _composition_trigger_scenario
from tests.interior_energetics.test_wrapper_guard_mass_target import (
    ACCEPT_MSG,
    HEATING_MSG,
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

BOUND_MSG = 'Mass escape keeps a radius step above its radius bound'
MISS_MSG = 'Committed structure misses its mass target'
BAD_SOLVE_MSG = 'non-finite or non-positive R_int or M_int'
UP_STEPS_CSV = Path(__file__).parent / 'fixtures' / 'mass_escape_up_steps.csv'


def _warnings(caplog, key):
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and key in r.getMessage()
    ]


def _bound(dm):
    return _MASS_ESCAPE_RADIUS_SLOPE * dm + _MASS_ESCAPE_RADIUS_OFFSET


def _measured_up_steps():
    """Return (dM/M, dR/R) of each measured up-step in the fixture file."""
    with UP_STEPS_CSV.open() as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith('#')))
    return [(float(r['dM_over_M']), float(r['dR_over_R'])) for r in rows]


def _misses_around_the_anchor(m_target, side):
    """Committed masses with the largest float miss within the anchor and the next float out."""

    def miss(m):
        return abs(m / m_target - 1.0)

    out = np.inf if side > 0 else 0.0
    m = m_target * (1.0 + side * _ZALMOXIS_MASS_ANCHOR_TOL)
    while miss(m) > _ZALMOXIS_MASS_ANCHOR_TOL:
        m = np.nextafter(m, m_target)
    while miss(np.nextafter(m, out)) <= _ZALMOXIS_MASS_ANCHOR_TOL:
        m = np.nextafter(m, out)
    return float(m), float(np.nextafter(m, out))


@pytest.mark.physics_invariant
def test_a_radius_step_far_above_its_mass_step_is_kept_with_a_warning(caplog):
    """A +20 % R_int up-step at a +0.31 % mass rise is kept by the mass escape, and warns.

    The retained mass misses the target by more than the anchor, so the solve is kept. The radius
    step is far above the bound of that mass rise, so the step warns with both ratios, the bound
    and the masses.
    """
    config, hf_row, base = _cooling_case()
    ratio = 1.0 / (1.0 - 3.1e-3)
    msgs, _, _ = run_guard(caplog, hf_row, config, base, _solve_to(ratio, 1.20))

    assert any(ACCEPT_MSG in m for m in msgs)
    assert not any(REJECT_MSG in m for m in msgs)
    warned = _warnings(caplog, BOUND_MSG)
    assert len(warned) == 1
    assert 'dR/R=2.000000e-01' in warned[0]
    assert 'bound %.6e' % _bound(ratio - 1.0) in warned[0]
    assert '= 2.0 dM/M + 1.0e-03 at the reached dM/M=%.6e' % (ratio - 1.0) in warned[0]
    assert 'target rise %.6e' % (ratio - 1.0) in warned[0]
    assert (
        'retained M_int=%.6e kg, reached M_int=%.6e kg, M_target=%.6e kg'
        % (M_RETAINED, M_RETAINED * ratio, M_RETAINED * ratio)
        in warned[0]
    )
    assert hf_row['R_int'] == pytest.approx(R_RETAINED * 1.20, rel=1e-12)
    assert hf_row['M_int'] == pytest.approx(M_RETAINED * ratio, rel=1e-12)
    assert not _warnings(caplog, MISS_MSG)


@pytest.mark.physics_invariant
def test_a_one_percent_step_at_an_anchor_trip_is_kept_with_a_warning(caplog):
    """A +1.1 % R_int up-step at a +0.31 % mass rise is kept, and warns: the bound there is 0.72 %."""
    config, hf_row, base = _cooling_case()
    msgs, _, _ = run_guard(caplog, hf_row, config, base, _solve_to(1.0 / (1.0 - 3.1e-3), 1.011))

    assert any(ACCEPT_MSG in m for m in msgs)
    assert len(_warnings(caplog, BOUND_MSG)) == 1
    assert hf_row['R_int'] == pytest.approx(R_RETAINED * 1.011, rel=1e-12)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('dry_mantle', [True, False], ids=['dry mantle', 'wet mantle'])
def test_a_fixed_target_at_radius_ratio_4_is_kept_at_the_first_solve(caplog, dry_mantle):
    """A +1 % target with a +4 % radius step is kept at once, and the planet meets the target.

    A radius rise of 4 times the mass rise is above the bound (2.1 %). The cooling dT/T solve is
    kept with a warning, so the structure takes the new target at once. The next solve at the same
    target up-steps by an artifact of 5e-4: the guard rejects it against the kept structure, which
    meets its target, so neither warning fires.
    """
    config, hf_row, base = _cooling_case()
    config.interior_struct.zalmoxis.dry_mantle = dry_mantle
    msgs, _, _ = run_guard(caplog, hf_row, config, base, _solve_to(1.01, 1.04))

    assert any(ACCEPT_MSG in m for m in msgs)
    assert not any(REJECT_MSG in m or 'composition-driven' in m for m in msgs)
    assert len(_warnings(caplog, BOUND_MSG)) == 1
    assert hf_row['R_int'] == pytest.approx(R_RETAINED * 1.04, rel=1e-12)
    assert hf_row['M_int'] == pytest.approx(M_RETAINED * 1.01, rel=1e-12)

    hf_row['Time'] += 1000.0
    msgs, _, _ = run_guard(caplog, hf_row, config, base, _solve_to(1.01, 1.04 * (1.0 + 5e-4)))

    assert any(REJECT_MSG in m for m in msgs)
    assert not _warnings(caplog, BOUND_MSG)
    assert not _warnings(caplog, MISS_MSG)
    assert hf_row['R_int'] == pytest.approx(R_RETAINED * 1.04, rel=1e-12)
    assert hf_row['M_int'] == pytest.approx(M_RETAINED * 1.01, rel=1e-12)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('dm', 'dr'),
    _measured_up_steps(),
    ids=[f'measured up-step {i}' for i in range(len(_measured_up_steps()))],
)
def test_measured_up_steps_are_kept_with_no_bound_warning(caplog, dm, dr):
    """Every R_int up-step measured with a mass rise above the anchor is kept and does not warn."""
    config, hf_row, base = _cooling_case()
    msgs, _, _ = run_guard(caplog, hf_row, config, base, _solve_to(1.0 + dm, 1.0 + dr))

    assert any(ACCEPT_MSG in m for m in msgs)
    assert not _warnings(caplog, BOUND_MSG)
    assert hf_row['R_int'] == pytest.approx(R_RETAINED * (1.0 + dr), rel=1e-12)


def test_the_measured_up_steps_file_holds_112_steps_above_the_anchor():
    """The replayed data set is the full set of 112 measured steps, each a mass-escape case."""
    steps = _measured_up_steps()

    assert len(steps) == 112
    assert all(
        dm > 1.0 / (1.0 - _ZALMOXIS_MASS_ANCHOR_TOL) - 1.0 and dr > 0.0 for dm, dr in steps
    )


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('dm', 'dr', 'warns'),
    [
        # dR/R against 2 dM/M + 1e-3, with the numbers written out
        pytest.param(0.05, 0.08, False, id='+8 % radius at +5 % mass, inside the slope'),
        pytest.param(0.05, 0.102, True, id='+10.2 % radius at +5 % mass, above 0.101'),
        pytest.param(
            3.1e-3, 6.8e-3, False, id='+0.68 % radius at +0.31 % mass, inside the offset'
        ),
        pytest.param(3.1e-3, 7.3e-3, True, id='+0.73 % radius at +0.31 % mass, above 0.72 %'),
    ],
)
def test_the_radius_bound_has_slope_2_and_offset_1e_3(caplog, dm, dr, warns):
    """Each step of the bound dR/R <= 2 dM/M + 1e-3 is kept; only a step above it warns."""
    config, hf_row, base = _cooling_case()
    msgs, _, _ = run_guard(caplog, hf_row, config, base, _solve_to(1.0 + dm, 1.0 + dr))

    assert any(ACCEPT_MSG in m for m in msgs)
    assert bool(_warnings(caplog, BOUND_MSG)) is warns


@pytest.mark.physics_invariant
@pytest.mark.parametrize(('scale', 'warns'), [(1.0 - 1e-9, False), (1.0 + 1e-9, True)])
def test_the_radius_bound_warning_holds_at_its_edge(caplog, scale, warns):
    """A radius step just inside the bound is kept silently; one just outside is kept and warns."""
    config, hf_row, base = _cooling_case()
    dm = 1.0e-2
    msgs, _, _ = run_guard(
        caplog, hf_row, config, base, _solve_to(1.0 + dm, 1.0 + _bound(dm) * scale)
    )

    assert any(ACCEPT_MSG in m for m in msgs)
    assert bool(_warnings(caplog, BOUND_MSG)) is warns


@pytest.mark.physics_invariant
def test_the_bound_uses_the_mass_the_solve_reached(caplog):
    """A solve that lands 2.9e-3 short of a +0.302 % target is judged by the rise it reached.

    The reached rise is 1.1e-4, so the bound is 1.2e-3 and a +0.7 % radius step warns; against
    the target rise the bound would be 7.04e-3 and the step would pass silently. The committed
    mass misses its target by 2.9e-3, inside the anchor, so the committed-mass check stays silent.
    """
    config, hf_row, base = _cooling_case()
    target = M_RETAINED * (1.0 + 3.02e-3)
    reached = target * (1.0 - 2.9e-3)

    def _solve(*args, **kwargs):
        args[2].update(R_int=R_RETAINED * (1.0 + 7.0e-3), M_int_target=target, M_int=reached)
        return (3.504e6, None)

    msgs, _, _ = run_guard(caplog, hf_row, config, base, _solve)

    assert any(ACCEPT_MSG in m for m in msgs)
    warned = _warnings(caplog, BOUND_MSG)
    assert len(warned) == 1
    assert 'reached dM/M=%.6e' % (reached / M_RETAINED - 1.0) in warned[0]
    assert 'target rise %.6e' % (target / M_RETAINED - 1.0) in warned[0]
    assert hf_row['M_int'] == reached
    assert not _warnings(caplog, MISS_MSG)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('radius', 'mass', 'target', 'reason'),
    [
        pytest.param(math.inf, 1.01, 1.01, BAD_SOLVE_MSG, id='infinite radius'),
        pytest.param(math.nan, 1.0, 1.0, BAD_SOLVE_MSG, id='NaN radius'),
        pytest.param(0.0, 1.0, 1.0, BAD_SOLVE_MSG, id='zero radius'),
        pytest.param(-1.0, 1.0, 1.0, BAD_SOLVE_MSG, id='negative radius'),
        pytest.param(1.001, math.nan, 1.01, 'mass-anchor violation', id='NaN mass on a target'),
        pytest.param(0.999, math.nan, None, BAD_SOLVE_MSG, id='NaN mass, no target'),
        pytest.param(0.999, 0.0, None, BAD_SOLVE_MSG, id='zero mass, no target'),
    ],
)
def test_a_non_finite_or_non_positive_solve_falls_back(caplog, radius, mass, target, reason):
    """A solve with an R_int or M_int that is not finite and positive falls back to the old one.

    The structure turns stale and the failure counts; no step is kept and no bound warns. A NaN
    mass on a target fails the mass anchor; without a target the structure check refuses it.
    """
    config, hf_row, base = _cooling_case()

    def _solve(*args, **kwargs):
        args[2].update(
            R_int=R_RETAINED * radius,
            M_int=M_RETAINED * mass,
            M_int_target=0.0 if target is None else M_RETAINED * target,
        )
        return (3.504e6, None)

    msgs, _, interior_o = run_guard(caplog, hf_row, config, base, _solve)

    fell_back = [m for m in msgs if 'Falling back to previous structure' in m]
    assert len(fell_back) == 1
    assert reason in fell_back[0]
    assert hf_row['R_int'] == R_RETAINED
    assert hf_row['M_int'] == M_RETAINED
    assert interior_o.structure_stale is True
    assert interior_o.zalmoxis_fail_count == 1
    assert not any(ACCEPT_MSG in m for m in msgs)
    assert not _warnings(caplog, BOUND_MSG)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('radius', [50.0, 1e300 / R_RETAINED], ids=['50 R', '1e300 m'])
def test_a_finite_absurd_radius_at_an_anchor_trip_is_kept_with_a_warning(caplog, radius):
    """The mass escape keeps every finite solve: a 50-fold or a 1e300 m radius at an anchor trip
    is kept, and the radius bound warning names it."""
    config, hf_row, base = _cooling_case()
    msgs, _, _ = run_guard(caplog, hf_row, config, base, _solve_to(1.01, radius))

    assert hf_row['R_int'] == R_RETAINED * radius
    assert len(_warnings(caplog, BOUND_MSG)) == 1
    assert not any('Falling back to previous structure' in m for m in msgs)


@pytest.mark.physics_invariant
def test_the_heating_escape_keeps_a_large_step_with_no_bound_warning(caplog):
    """Under heating, an anchor trip with a +20 % radius step is kept by the heating escape alone."""
    config, hf_row, base = _cooling_case()
    base[1] = 2700.0  # last structure decision at 2700 K, now 2800 K: heating
    msgs, _, _ = run_guard(caplog, hf_row, config, base, _solve_to(1.0 / (1.0 - 3.1e-3), 1.20))

    assert any(HEATING_MSG in m for m in msgs)
    assert not _warnings(caplog, BOUND_MSG)
    assert hf_row['R_int'] == pytest.approx(R_RETAINED * 1.20, rel=1e-12)


@pytest.mark.physics_invariant
def test_the_wet_composition_escape_keeps_a_large_step_with_no_bound_warning(caplog):
    """A wet composition re-solve with an anchor trip and a +20 % radius step does not warn."""
    config, hf_row, dirs = _composition_trigger_scenario(dry_mantle=False)
    hf_row['M_int_target'] = hf_row['M_int']
    m_ret, r_prev = hf_row['M_int'], hf_row['R_int']

    def _solve(*args, **kwargs):
        args[2].update(R_int=r_prev * 1.20, M_int_target=m_ret * 1.0031, M_int=m_ret * 1.0031)
        return (3.504e6, None)

    msgs, _, _ = run_guard(caplog, hf_row, config, (0.0, 3000.0, 0.5), _solve, dirs=dirs)

    assert any('composition-driven wet' in m for m in msgs)
    assert any(ACCEPT_MSG in m for m in msgs)
    assert not _warnings(caplog, BOUND_MSG)
    assert hf_row['R_int'] == pytest.approx(r_prev * 1.20, rel=1e-12)


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
    assert not _warnings(caplog, BOUND_MSG)
    assert hf_row['M_int'] == M_RETAINED
    assert hf_row['M_int_target'] == M_RETAINED * ratio
    warned = _warnings(caplog, MISS_MSG)
    assert len(warned) == 1
    assert 'M_int=%.6e kg' % M_RETAINED in warned[0]
    assert 'M_int_target=%.6e kg' % (M_RETAINED * ratio) in warned[0]
    assert 'relative miss %.3e' % abs(M_RETAINED / (M_RETAINED * ratio) - 1.0) in warned[0]


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
def test_an_accepted_solve_at_a_miss_of_2p9e_3_does_not_warn(caplog):
    """A down-step solve that lands 2.9e-3 below its target passes the anchor and does not warn."""
    config, hf_row, base = _cooling_case()

    def _solve(*args, **kwargs):
        args[2].update(
            R_int=R_RETAINED * (1.0 - 1.0e-3),
            M_int_target=M_RETAINED,
            M_int=M_RETAINED * (1.0 - 2.9e-3),
        )
        return (3.504e6, None)

    msgs, _, _ = run_guard(caplog, hf_row, config, base, _solve)

    assert not any('Falling back to previous structure' in m for m in msgs)
    assert hf_row['M_int'] == M_RETAINED * (1.0 - 2.9e-3)
    assert not _warnings(caplog, MISS_MSG)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('side', [1.0, -1.0], ids=['mass above target', 'mass below target'])
def test_the_committed_mass_warning_starts_at_the_first_float_beyond_the_anchor(caplog, side):
    """The largest float miss within the anchor does not warn; the next float out warns.

    No float mass gives a miss of exactly 3e-3 (1 +- 3e-3 is not a float), so the two floats on
    either side of the anchor pin the threshold.
    """
    inside, outside = _misses_around_the_anchor(M_RETAINED, side)
    assert abs(inside / M_RETAINED - 1.0) <= _ZALMOXIS_MASS_ANCHOR_TOL
    assert abs(outside / M_RETAINED - 1.0) > _ZALMOXIS_MASS_ANCHOR_TOL

    with caplog.at_level(logging.WARNING):
        _warn_committed_mass_miss({'M_int': inside, 'M_int_target': M_RETAINED})
    assert not _warnings(caplog, MISS_MSG)

    with caplog.at_level(logging.WARNING):
        _warn_committed_mass_miss({'M_int': outside, 'M_int_target': M_RETAINED})
    assert len(_warnings(caplog, MISS_MSG)) == 1


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
@pytest.mark.parametrize(
    ('m_int', 'm_target'),
    [
        pytest.param(-1.0, M_RETAINED, id='negative mass'),
        pytest.param(M_RETAINED, -M_RETAINED, id='negative target'),
    ],
)
def test_a_non_positive_committed_mass_warns(caplog, m_int, m_target):
    """A negative mass or target with both keys set warns."""
    with caplog.at_level(logging.WARNING):
        _warn_committed_mass_miss({'M_int': m_int, 'M_int_target': m_target})

    assert len(_warnings(caplog, MISS_MSG)) == 1
