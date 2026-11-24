"""Monotonic-radius guard: an up-step is kept when the retained structure fails the mass anchor.

Under the super-liquidus adiabat IC the guard in ``update_structure_from_interior`` rejects an
R_int increase while the interior cools. When the retained structure mass misses the larger
target of the new solve by more than the Zalmoxis mass anchor, the retained structure fails the
mass contract of the wrapper, so the larger structure is kept.

See also:
- docs/How-to/testing.md: Running, writing, and marking tests; coverage and CI
"""

from __future__ import annotations

import logging
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from proteus.interior_energetics.wrapper import (
    _ZALMOXIS_MASS_ANCHOR_TOL,
    update_structure_from_interior,
)
from tests.interior_energetics.test_wrapper import (
    _composition_trigger_scenario,
    _dtmagma_scenario,
    _mock_dirs,
    _mock_interior_o,
    _solver_patches,
    _write_eos_dat,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.timeout(30),
    pytest.mark.usefixtures('_no_core_table'),
]

M_EARTH = 5.972e24
# A contracted structure of 7.355 Earth masses; the first re-solve after it is made for 8.010.
M_RETAINED = 7.355 * M_EARTH
R_RETAINED = 1.022808e7
ACCEPT_MSG = 'because the retained structure fails the mass anchor'
HEATING_MSG = 'Accepted R_int increase under active interior heating'
REJECT_MSG = 'Rejected representation-artifact radius increase'


def _target_ratio_at_anchor():
    """Largest ratio r with a float miss abs(M_ret / (M_ret r) - 1) within the anchor."""

    def miss(r):
        return abs(M_RETAINED / (M_RETAINED * r) - 1.0)

    r = 1.0 / (1.0 - _ZALMOXIS_MASS_ANCHOR_TOL)
    while miss(r) > _ZALMOXIS_MASS_ANCHOR_TOL:
        r = np.nextafter(r, 0.0)
    while miss(np.nextafter(r, np.inf)) <= _ZALMOXIS_MASS_ANCHOR_TOL:
        r = np.nextafter(r, np.inf)
    return r


def _cooling_case():
    """A dT/T re-solve from the retained structure while T_magma falls 3000 -> 2800 K."""
    config, hf_row, *base = _dtmagma_scenario()
    config.interior_struct.zalmoxis.dry_mantle = True
    hf_row.update(
        R_int=R_RETAINED,
        M_int=M_RETAINED,
        M_int_target=M_RETAINED,
        M_mantle=3.0e24,
        R_core=3.5e6,
        H2O_kg_liquid=5.0e21,
        H2_kg_liquid=1.0e20,
    )
    return config, hf_row, base


def _solve_to(target_ratio, radius_ratio, set_target=True):
    """Stub solve: R_int up-steps by ``radius_ratio``; the mass lands on its target."""

    def _solve(*args, **kwargs):
        hr = args[2]
        hr['R_int'] = R_RETAINED * radius_ratio
        if set_target:
            hr['M_int_target'] = M_RETAINED * target_ratio
            hr['M_int'] = M_RETAINED * target_ratio
        return (3.504e6, None)

    return _solve


def run_guard(caplog, hf_row, config, base, solve, dirs=None, interior_o=None):
    """Call the wrapper with a stub solve; return the log messages, dirs and interior.

    Without given dirs the output folder is a fresh empty one, so no stray zalmoxis_output.dat
    can make the grid-extent check fire.
    """
    interior_o = interior_o if interior_o is not None else _mock_interior_o()
    caplog.clear()
    s, ns, cp = _solver_patches(side_effect=solve)
    with (
        tempfile.TemporaryDirectory() as out,
        caplog.at_level(logging.INFO),
        s as mock_solver,
        ns,
        cp,
        patch(
            'proteus.interior_energetics.wrapper._use_superliquidus_adiabat_ic',
            return_value=True,
        ),
    ):
        if dirs is None:
            dirs = _mock_dirs()
            dirs['output'] = out
        update_structure_from_interior(dirs, config, hf_row, interior_o, *base)
    mock_solver.assert_called_once()
    return [r.getMessage() for r in caplog.records], dirs, interior_o


@pytest.mark.physics_invariant
def test_up_step_kept_when_target_mass_rose(caplog):
    """A cooling up-step is kept when the retained mass misses the new target by far more.

    The retained structure has 7.355 Earth masses and the new solve is made for 8.010, so the
    retained mass is 8.2 % below the target (the target is 8.9 % above it), far outside the 3e-3
    anchor. The larger structure is kept, the accept line names both masses, their ratio and the
    relative miss, and the success path clears a stale flag that a reject would keep.
    """
    config, hf_row, base = _cooling_case()
    ratio = 8.010 / 7.355
    interior_o = _mock_interior_o()
    interior_o.structure_stale = True
    msgs, dirs, interior_o = run_guard(
        caplog, hf_row, config, base, _solve_to(ratio, 1.0709), interior_o=interior_o
    )

    accepted = [m for m in msgs if ACCEPT_MSG in m]
    assert len(accepted) == 1
    assert 'M_int=%.6e kg' % M_RETAINED in accepted[0]
    assert 'M_target=%.6e kg' % (M_RETAINED * ratio) in accepted[0]
    assert 'ratio %.6f' % ratio in accepted[0]
    assert 'relative miss %.3e' % (1.0 - M_RETAINED / (M_RETAINED * ratio)) in accepted[0]
    assert not any(REJECT_MSG in m for m in msgs)
    assert not any(HEATING_MSG in m or 'composition-driven' in m for m in msgs)
    assert hf_row['R_int'] == pytest.approx(R_RETAINED * 1.0709, rel=1e-12)
    assert hf_row['M_int'] == pytest.approx(M_RETAINED * ratio, rel=1e-12)
    assert dirs['_last_resolve_dR_rel'] == pytest.approx(0.0709, rel=1e-9)
    assert interior_o.zalmoxis_fail_count == 0
    assert interior_o.structure_stale is False


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    'target_ratio',
    [1.0, 1.0 + 1.0e-3, _target_ratio_at_anchor(), 0.95],
    ids=['unchanged', 'inside anchor', 'at anchor', 'fallen'],
)
def test_up_step_still_rejected_without_an_anchor_miss(caplog, target_ratio):
    """The guard's own case stays rejected: the miss is within the anchor, or the target fell.

    The largest target whose float miss stays within the anchor is rejected; a reject keeps a
    stale flag.
    """
    config, hf_row, base = _cooling_case()
    interior_o = _mock_interior_o()
    interior_o.structure_stale = True
    msgs, dirs, interior_o = run_guard(
        caplog,
        hf_row,
        config,
        base,
        _solve_to(target_ratio, 1.0 + 1.0e-3),
        interior_o=interior_o,
    )

    assert any(REJECT_MSG in m for m in msgs)
    assert not any(ACCEPT_MSG in m for m in msgs)
    assert hf_row['R_int'] == pytest.approx(R_RETAINED, rel=1e-12)
    assert dirs['_last_resolve_dR_rel'] == 0.0
    assert interior_o.structure_stale is True


@pytest.mark.physics_invariant
def test_up_step_kept_just_beyond_the_anchor(caplog):
    """A miss a hair beyond the anchor is enough to keep the up-step."""
    config, hf_row, base = _cooling_case()
    ratio = _target_ratio_at_anchor() * (1.0 + 1.0e-9)
    msgs, _, _ = run_guard(caplog, hf_row, config, base, _solve_to(ratio, 1.0 + 1.0e-3))

    assert any(ACCEPT_MSG in m for m in msgs)
    assert hf_row['R_int'] == pytest.approx(R_RETAINED * (1.0 + 1.0e-3), rel=1e-12)


@pytest.mark.physics_invariant
def test_up_step_rejected_when_the_retained_mass_sits_off_its_own_target(caplog):
    """A retained mass 0.1 % below its own unchanged target is inside the anchor: rejected."""
    config, hf_row, base = _cooling_case()
    hf_row['M_int'] = M_RETAINED * (1.0 - 1.0e-3)
    msgs, _, _ = run_guard(caplog, hf_row, config, base, _solve_to(1.0, 1.0 + 1.0e-3))

    assert any(REJECT_MSG in m for m in msgs)
    assert not any(ACCEPT_MSG in m for m in msgs)


@pytest.mark.parametrize('drop', ['M_int', 'M_int_target'])
def test_up_step_rejected_when_a_mass_is_missing(caplog, drop):
    """Without a retained mass or a solve target there is no comparison, so the guard rejects.

    Only R_int is checked: production rows always carry M_int, and a key absent at entry is not
    restored.
    """
    config, hf_row, base = _cooling_case()
    del hf_row[drop]
    solve = _solve_to(1.2, 1.0 + 1.0e-3, set_target=drop != 'M_int_target')
    msgs, _, _ = run_guard(caplog, hf_row, config, base, solve)

    assert any(REJECT_MSG in m for m in msgs)
    assert not any(ACCEPT_MSG in m for m in msgs)
    assert hf_row['R_int'] == pytest.approx(R_RETAINED, rel=1e-12)


def _grid_case(tmp_path, scalar_ratio):
    """A solve for 5 % more mass whose mesh ends 4 km beyond its R_int."""
    config, hf_row, base = _cooling_case()
    (tmp_path / 'data').mkdir()
    eos_path = tmp_path / 'data' / 'zalmoxis_output.dat'
    _write_eos_dat(str(eos_path) + '.prev', np.linspace(hf_row['R_core'], R_RETAINED, 200))

    def _solve(*args, **kwargs):
        hr = args[2]
        hr['R_int'] = R_RETAINED * scalar_ratio
        hr['M_int_target'] = M_RETAINED * 1.05
        hr['M_int'] = M_RETAINED * 1.05
        _write_eos_dat(eos_path, np.linspace(hr['R_core'], hr['R_int'] + 4.0e3, 200))
        return (3.504e6, None)

    dirs = _mock_dirs()
    dirs['output'] = str(tmp_path)
    return config, hf_row, base, _solve, dirs


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('scalar_ratio', 'reject_msg'),
    [(1.0, 'Rejected EOS-grid-extent up-step'), (1.0 + 1.0e-3, REJECT_MSG)],
    ids=['grid only', 'scalar and grid'],
)
def test_a_mesh_beyond_r_int_stays_rejected_with_a_larger_target(
    tmp_path, caplog, scalar_ratio, reject_msg
):
    """A mesh that ends beyond R_int is never handed to Aragog through the mass escape.

    Both the grid-extent-only up-step and the scalar up-step with such a mesh are rejected, the
    previous structure is restored and the step records no radius change.
    """
    config, hf_row, base, solve, dirs = _grid_case(tmp_path, scalar_ratio)
    msgs, dirs, interior_o = run_guard(caplog, hf_row, config, base, solve, dirs=dirs)

    assert not any(ACCEPT_MSG in m for m in msgs)
    assert [m for m in msgs if 'Rejected' in m and reject_msg in m]
    assert len([m for m in msgs if 'Rejected' in m]) == 1
    assert hf_row['R_int'] == pytest.approx(R_RETAINED, rel=1e-12)
    assert hf_row['M_int'] == pytest.approx(M_RETAINED, rel=1e-12)
    assert dirs['_last_resolve_dR_rel'] == 0.0
    assert interior_o.zalmoxis_fail_count == 0


@pytest.mark.physics_invariant
def test_heating_and_an_anchor_miss_log_both_reasons(caplog):
    """A heating re-solve whose target outgrows the retained mass logs both accept lines."""
    config, hf_row, base = _cooling_case()
    base[1] = 2700.0  # last structure decision at 2700 K, now 2800 K: heating
    msgs, _, _ = run_guard(caplog, hf_row, config, base, _solve_to(1.07, 1.06))

    assert any(HEATING_MSG in m for m in msgs)
    assert any(ACCEPT_MSG in m for m in msgs)
    assert hf_row['R_int'] == pytest.approx(R_RETAINED * 1.06, rel=1e-12)


@pytest.mark.physics_invariant
def test_composition_and_an_anchor_miss_log_both_reasons(caplog):
    """A wet composition re-solve whose target outgrows the retained mass logs both lines."""
    config, hf_row, dirs = _composition_trigger_scenario(dry_mantle=False)
    hf_row['M_int_target'] = hf_row['M_int']
    m_ret, r_prev = hf_row['M_int'], hf_row['R_int']

    def _solve(*args, **kwargs):
        args[2].update(R_int=r_prev * 1.06, M_int_target=m_ret * 1.07, M_int=m_ret * 1.07)
        return (3.504e6, None)

    msgs, _, _ = run_guard(caplog, hf_row, config, (0.0, 3000.0, 0.5), _solve, dirs=dirs)

    assert any('composition-driven wet' in m for m in msgs)
    assert any(ACCEPT_MSG in m for m in msgs)
    assert not any(HEATING_MSG in m for m in msgs)
    assert hf_row['R_int'] == pytest.approx(r_prev * 1.06, rel=1e-12)


@pytest.mark.physics_invariant
def test_a_later_artifact_up_step_is_rejected_against_the_kept_structure(caplog):
    """After an accepted re-expansion the guard holds the new structure as its minimum.

    The next cooling solve at the same target with a +1e-3 artifact up-step is rejected and the
    re-expanded structure stays.
    """
    config, hf_row, base = _cooling_case()
    ratio = 8.010 / 7.355
    run_guard(caplog, hf_row, config, base, _solve_to(ratio, 1.0709))
    r_kept = hf_row['R_int']
    hf_row['Time'] = 1000.0
    hf_row['T_magma'] = 2600.0

    def _artifact(*args, **kwargs):
        args[2]['R_int'] = r_kept * (1.0 + 1.0e-3)
        args[2]['M_int_target'] = M_RETAINED * ratio
        args[2]['M_int'] = M_RETAINED * ratio
        return (3.504e6, None)

    msgs, _, _ = run_guard(caplog, hf_row, config, (500.0, 2800.0, 0.80), _artifact)

    assert any(REJECT_MSG in m for m in msgs)
    assert not any(ACCEPT_MSG in m for m in msgs)
    assert hf_row['R_int'] == pytest.approx(r_kept, rel=1e-12)
