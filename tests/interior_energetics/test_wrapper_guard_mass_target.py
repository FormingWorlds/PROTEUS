"""Monotonic-radius guard: an up-step is kept when the structure mass target outgrows the retained structure.

Under the super-liquidus adiabat IC the guard in ``update_structure_from_interior`` rejects an R_int
increase while the interior cools. When the structure mass target of the new solve exceeds the mass
of the retained structure by more than the Zalmoxis mass-anchor tolerance, the retained structure no
longer satisfies the mass contract of the wrapper, so the larger structure is kept.

See also:
- docs/How-to/testing.md: Running, writing, and marking tests; coverage and CI
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from proteus.interior_energetics.common import Interior_t
from proteus.interior_energetics.wrapper import (
    _ZALMOXIS_MASS_ANCHOR_TOL,
    update_structure_from_interior,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.timeout(30),
    pytest.mark.usefixtures('_no_core_table'),
]

M_EARTH = 5.972e24
# Case 38 of the issue: retained structure 7.355 Earth masses, first rejected solve 8.010.
M_RETAINED = 7.355 * M_EARTH
R_RETAINED = 1.022808e7
ACCEPT_MSG = 'Accepted R_int increase after the structure mass target rose'
REJECT_MSG = 'Rejected representation-artifact radius increase'


def _cooling_case():
    """Config, hf_row and baselines of a dT/T re-solve while the interior cools.

    T_magma falls 3000 -> 2800 K and Phi is flat, so neither the heating escape nor the
    wet composition escape of the guard can fire.
    """
    config = MagicMock()
    zal = config.interior_struct.zalmoxis
    zal.update_interval = 1000.0
    zal.update_min_interval = 100.0
    zal.update_dtmagma_frac = 0.03
    zal.update_dphi_abs = 0.05
    zal.update_stale_ceiling = 0.0
    zal.mesh_max_shift = 0.05
    zal.mesh_convergence_interval = 10.0
    zal.num_levels = 50
    zal.dry_mantle = True
    config.planet.temperature_mode = 'isothermal'
    config.interior_energetics.module = 'spider'
    hf_row = {
        'Time': 500.0,
        'T_magma': 2800.0,
        'Phi_global': 0.80,
        'R_int': R_RETAINED,
        'M_int': M_RETAINED,
        'M_int_target': M_RETAINED,
        'gravity': 9.81,
        'M_mantle': 3.0e24,
        'R_core': 3.5e6,
        'H2O_kg_liquid': 5.0e21,
        'H2_kg_liquid': 1.0e20,
    }
    return config, hf_row, 0.0, 3000.0, 0.80


def _dirs(outdir='/tmp/test_output'):
    return {
        'output': str(outdir),
        'spider': '/tmp/spider',
        'mesh_shift_active': False,
        'mesh_convergence_steps': 0,
        'spider_mesh': '/tmp/test_mesh.dat',
        'spider_mesh_prev': '/tmp/test_mesh.dat.prev',
    }


def _interior():
    interior_o = MagicMock(spec=Interior_t)
    interior_o.radius = np.linspace(6.371e6, 3.504e6, 50)
    interior_o.temp = np.full(49, 3000.0)
    interior_o.zalmoxis_fail_count = 0
    interior_o.spider_fail_count = 0
    interior_o.aragog_fail_count = 0
    interior_o.structure_stale = False
    return interior_o


def _run(hf_row, config, baselines, solve, dirs=None):
    """Call the wrapper with a stub solve; return the log messages, dirs and interior."""
    dirs = dirs if dirs is not None else _dirs()
    interior_o = _interior()
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    log = logging.getLogger('fwl.proteus.interior_energetics.wrapper')
    log.addHandler(handler)
    old_level = log.level
    log.setLevel(logging.INFO)
    try:
        with (
            patch('proteus.interior_struct.zalmoxis.zalmoxis_solver', side_effect=solve) as s,
            patch('proteus.interior_energetics.wrapper.np.savetxt'),
            patch('proteus.interior_energetics.wrapper.shutil.copy2'),
            patch(
                'proteus.interior_energetics.wrapper._use_superliquidus_adiabat_ic',
                return_value=True,
            ),
        ):
            update_structure_from_interior(dirs, config, hf_row, interior_o, *baselines)
    finally:
        log.removeHandler(handler)
        log.setLevel(old_level)
    s.assert_called_once()
    return [r.getMessage() for r in records], dirs, interior_o


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


def test_logger_name_matches_the_wrapper():
    """The wrapper logs on the logger this file listens to (a wrong name would silence every test)."""
    import proteus.interior_energetics.wrapper as w

    assert w.log.name == 'fwl.proteus.interior_energetics.wrapper'


@pytest.mark.physics_invariant
def test_up_step_kept_when_target_mass_rose():
    """A cooling up-step is kept when the solve target exceeds the retained mass (case 38 numbers).

    The retained structure has 7.355 Earth masses, the new solve is made for 8.010 and raises R_int
    by 7 %. Keeping the retained structure would leave the run 8.9 % below its own mass target, far
    outside the 3e-3 mass anchor, so the larger structure is kept and the accept line names both
    masses and their ratio.
    """
    config, hf_row, *base = _cooling_case()
    ratio = 8.010 / 7.355
    msgs, dirs, interior_o = _run(hf_row, config, base, _solve_to(ratio, 1.0709))

    accepted = [m for m in msgs if ACCEPT_MSG in m]
    assert len(accepted) == 1
    assert '%.6e' % M_RETAINED in accepted[0]
    assert '%.6e' % (M_RETAINED * ratio) in accepted[0]
    assert '%.4f' % ratio in accepted[0]
    assert not any(REJECT_MSG in m for m in msgs)
    assert hf_row['R_int'] == pytest.approx(R_RETAINED * 1.0709, rel=1e-12)
    assert hf_row['M_int'] == pytest.approx(M_RETAINED * ratio, rel=1e-12)
    assert dirs['_last_resolve_dR_rel'] == pytest.approx(0.0709, rel=1e-9)
    assert interior_o.zalmoxis_fail_count == 0


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    'target_ratio',
    [1.0, 1.0 + 1.0e-3, 1.0 + _ZALMOXIS_MASS_ANCHOR_TOL, 0.95],
    ids=['unchanged', 'inside anchor', 'at anchor', 'fallen'],
)
def test_up_step_still_rejected_without_target_outgrowth(target_ratio):
    """The guard's own case stays rejected: the target is within the anchor of the retained mass or below it.

    At the tolerance itself the comparison is strict, so the up-step is rejected.
    """
    config, hf_row, *base = _cooling_case()
    msgs, dirs, _ = _run(hf_row, config, base, _solve_to(target_ratio, 1.0 + 1.0e-3))

    assert any(REJECT_MSG in m for m in msgs)
    assert not any(ACCEPT_MSG in m for m in msgs)
    assert hf_row['R_int'] == pytest.approx(R_RETAINED, rel=1e-12)
    assert dirs['_last_resolve_dR_rel'] == 0.0


def test_up_step_kept_just_above_the_anchor():
    """A target a hair above the anchor tolerance is enough to keep the up-step."""
    config, hf_row, *base = _cooling_case()
    ratio = (1.0 + _ZALMOXIS_MASS_ANCHOR_TOL) * (1.0 + 1.0e-9)
    msgs, _, _ = _run(hf_row, config, base, _solve_to(ratio, 1.0 + 1.0e-3))

    assert any(ACCEPT_MSG in m for m in msgs)
    assert hf_row['R_int'] == pytest.approx(R_RETAINED * (1.0 + 1.0e-3), rel=1e-12)


@pytest.mark.parametrize('drop', ['M_int', 'M_int_target'])
def test_up_step_rejected_when_a_mass_is_missing(drop):
    """Without a retained mass or a solve target the comparison cannot be made, so the guard rejects."""
    config, hf_row, *base = _cooling_case()
    if drop == 'M_int':
        del hf_row['M_int']
        solve = _solve_to(1.2, 1.0 + 1.0e-3)
    else:
        del hf_row['M_int_target']
        solve = _solve_to(1.2, 1.0 + 1.0e-3, set_target=False)
    msgs, _, _ = _run(hf_row, config, base, solve)

    assert any(REJECT_MSG in m for m in msgs)
    assert not any(ACCEPT_MSG in m for m in msgs)
    assert hf_row['R_int'] == pytest.approx(R_RETAINED, rel=1e-12)


def _write_eos_dat(path, r):
    r = np.atleast_1d(np.asarray(r, dtype=float))
    with open(path, 'w') as fh:
        for ri in r:
            fh.write(f'{ri:.9e} 1.0e9 4000.0 9.8 3000.0\n')


def test_grid_extent_only_up_step_stays_rejected_with_a_rising_target(tmp_path):
    """A mesh beyond a scalar R_int at the minimum is rejected even when the target rose.

    The mass escape applies to the scalar up-step only; keeping a mesh whose outer node lies beyond
    the retained R_int would hand Aragog an inconsistent grid.
    """
    config, hf_row, *base = _cooling_case()
    (tmp_path / 'data').mkdir()
    eos_path = tmp_path / 'data' / 'zalmoxis_output.dat'
    _write_eos_dat(str(eos_path) + '.prev', np.linspace(hf_row['R_core'], R_RETAINED, 200))

    def _solve(*args, **kwargs):
        hr = args[2]
        hr['R_int'] = R_RETAINED
        hr['M_int_target'] = M_RETAINED * 1.1
        hr['M_int'] = M_RETAINED * 1.1
        _write_eos_dat(eos_path, np.linspace(hr['R_core'], R_RETAINED + 4.0e3, 200))
        return (3.504e6, None)

    msgs, _, _ = _run(hf_row, config, base, _solve, dirs=_dirs(tmp_path))

    assert any('Rejected EOS-grid-extent up-step' in m for m in msgs)
    assert not any(ACCEPT_MSG in m for m in msgs)
    assert hf_row['R_int'] == pytest.approx(R_RETAINED, rel=1e-12)
    assert hf_row['M_int'] == pytest.approx(M_RETAINED, rel=1e-12)


hypothesis = pytest.importorskip('hypothesis')
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


@pytest.mark.physics_invariant
@settings(max_examples=60, deadline=None)
@given(
    target_ratio=st.floats(0.9, 1.2),
    radius_step=st.floats(1.0e-6, 0.2),
)
def test_up_step_kept_if_and_only_if_target_outgrew_retained_mass(target_ratio, radius_step):
    """Under cooling the scalar up-step is kept exactly when target > retained mass x (1 + tol)."""
    config, hf_row, *base = _cooling_case()
    msgs, _, _ = _run(hf_row, config, base, _solve_to(target_ratio, 1.0 + radius_step))

    kept = M_RETAINED * target_ratio > M_RETAINED * (1.0 + _ZALMOXIS_MASS_ANCHOR_TOL)
    assert any(ACCEPT_MSG in m for m in msgs) is kept
    assert any(REJECT_MSG in m for m in msgs) is not kept
    expected_R = R_RETAINED * (1.0 + radius_step) if kept else R_RETAINED
    assert hf_row['R_int'] == pytest.approx(expected_R, rel=1e-12)
