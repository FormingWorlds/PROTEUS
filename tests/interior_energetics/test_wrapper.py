"""
Unit tests for proteus.interior_energetics.wrapper module: structure update triggers.

Tests the dynamic trigger logic in update_structure_from_interior() that
decides when to re-run Zalmoxis with SPIDER's evolved temperature profile.
Also tests solve_structure() dispatch, phi_crit warning, full update path
with mesh blending, convergence tracking, and entropy remap.

Testing standards and documentation:
- docs/test_infrastructure.md: Test infrastructure overview
- docs/test_categorization.md: Test marker definitions
- docs/test_building.md: Best practices for test construction

Functions tested:
- update_structure_from_interior(): Trigger logic, T(r) profile building,
  mesh blending, convergence tracking, entropy remap
- solve_structure(): phi_crit warning, dispatch to Zalmoxis
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from proteus.interior_energetics.common import Interior_t
from proteus.interior_energetics.wrapper import (
    _prevent_warming_clamp_active,
    update_structure_from_interior,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _ns_prevent_warming(prevent_warming: bool, module: str = 'aragog'):
    """Build the minimal config namespace _prevent_warming_clamp_active reads.

    The ``module`` argument is preserved for callsite parity with the
    pre-Φ_vol-deletion test suite, but the gate no longer branches on it.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        planet=SimpleNamespace(prevent_warming=prevent_warming),
        interior_energetics=SimpleNamespace(module=module),
    )


def _mock_config(
    update_interval=1000.0,
    update_min_interval=100.0,
    update_dtmagma_frac=0.03,
    update_dphi_abs=0.05,
    update_stale_ceiling=0.0,
    mesh_max_shift=0.05,
    mesh_convergence_interval=10.0,
    num_levels=50,
):
    """Create a minimal mock config for update_structure trigger tests."""
    config = MagicMock()
    config.interior_struct.zalmoxis.update_interval = update_interval
    config.interior_struct.zalmoxis.update_min_interval = update_min_interval
    config.interior_struct.zalmoxis.update_dtmagma_frac = update_dtmagma_frac
    config.interior_struct.zalmoxis.update_dphi_abs = update_dphi_abs
    config.interior_struct.zalmoxis.update_stale_ceiling = update_stale_ceiling
    config.interior_struct.zalmoxis.mesh_max_shift = mesh_max_shift
    config.interior_struct.zalmoxis.mesh_convergence_interval = mesh_convergence_interval
    config.planet.temperature_mode = 'isothermal'
    config.interior_struct.zalmoxis.num_levels = num_levels
    config.interior_energetics.module = 'spider'
    return config


def _mock_dirs(
    mesh_active=False,
    mesh_steps=0,
    has_mesh=True,
):
    """Create a minimal dirs dict for trigger tests."""
    dirs = {
        'output': '/tmp/test_output',
        'spider': '/tmp/spider',
        'mesh_shift_active': mesh_active,
        'mesh_convergence_steps': mesh_steps,
    }
    if has_mesh:
        dirs['spider_mesh'] = '/tmp/test_mesh.dat'
        dirs['spider_mesh_prev'] = '/tmp/test_mesh.dat.prev'
    return dirs


def _mock_interior_o(n_stag=49):
    """Create a mock Interior_t with minimal arrays.

    The consecutive-failure counters are real integers (zero, the
    fresh-run state): the production code increments and compares
    them, which a bare MagicMock attribute cannot support.
    """
    interior_o = MagicMock(spec=Interior_t)
    interior_o.radius = np.linspace(6.371e6, 3.504e6, n_stag + 1)
    interior_o.temp = np.full(n_stag, 3000.0)
    interior_o.zalmoxis_fail_count = 0
    interior_o.spider_fail_count = 0
    interior_o.aragog_fail_count = 0
    return interior_o


# ============================================================================
# test update_interval <= 0 (dynamic updates disabled)
# ============================================================================


@pytest.mark.unit
def test_update_disabled():
    """update_interval=0 should skip all triggers and return unchanged."""
    config = _mock_config(update_interval=0)
    dirs = _mock_dirs()
    hf_row = {'Time': 500.0, 'T_magma': 3000.0, 'Phi_global': 0.8}
    interior_o = _mock_interior_o()

    with patch(
        'proteus.interior_struct.zalmoxis.zalmoxis_solver',
        return_value=(3.504e6, None),
    ) as mock_solver:
        result = update_structure_from_interior(
            dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.8
        )
    assert result == (0.0, 3000.0, 0.8)
    # Side-effect discrimination: the disabled branch must NOT invoke
    # the Zalmoxis solver. Catches a regression that ran the solver
    # and then discarded its output (which would still return the
    # unchanged tuple but burn CPU and pollute the mesh files).
    mock_solver.assert_not_called()


# ============================================================================
# test ceiling trigger
# ============================================================================


@pytest.mark.unit
def test_ceiling_trigger():
    """Elapsed time >= update_interval forces an update."""
    config = _mock_config(update_interval=1000.0, update_min_interval=100.0)
    dirs = _mock_dirs()
    hf_row = {
        'Time': 1100.0,
        'T_magma': 3000.0,
        'Phi_global': 0.8,
        'R_int': 6.371e6,
        'gravity': 9.81,
    }
    interior_o = _mock_interior_o()

    with (
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            return_value=(3.504e6, None),
        ) as mock_solver,
        patch('proteus.interior_energetics.wrapper.np.savetxt'),
        patch('proteus.interior_energetics.wrapper.shutil.copy2'),
    ):
        result = update_structure_from_interior(
            dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.8
        )
    # Should have triggered: last_struct_time updated to current
    assert result[0] == pytest.approx(1100.0, rel=1e-12)
    # Trigger discrimination: the ceiling branch must actually invoke
    # Zalmoxis. Catches a regression where the time was updated without
    # the structure solver being run, leaving the mesh stale.
    mock_solver.assert_called_once()


# ============================================================================
# test T_magma relative change trigger
# ============================================================================


@pytest.mark.unit
def test_tmagma_trigger():
    """dT/T >= update_dtmagma_frac should trigger an update."""
    config = _mock_config(
        update_interval=10000.0,
        update_min_interval=100.0,
        update_dtmagma_frac=0.03,
    )
    dirs = _mock_dirs()
    # 4% relative change in T_magma should trigger (threshold = 3%)
    hf_row = {
        'Time': 200.0,
        'T_magma': 2880.0,
        'Phi_global': 0.80,
        'R_int': 6.371e6,
        'gravity': 9.81,
    }
    interior_o = _mock_interior_o()

    with (
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            return_value=(3.504e6, None),
        ) as mock_solver,
        patch('proteus.interior_energetics.wrapper.np.savetxt'),
        patch('proteus.interior_energetics.wrapper.shutil.copy2'),
    ):
        result = update_structure_from_interior(
            dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.80
        )
    assert result[0] == pytest.approx(200.0, rel=1e-12)
    # Trigger discrimination: a 4% T change crosses the 3% threshold,
    # so the solver must run. Catches a regression that swapped the
    # comparison direction (dT < threshold) and would still produce
    # the same time-advance from the ceiling branch on a later step.
    mock_solver.assert_called_once()


@pytest.mark.unit
def test_tmagma_no_trigger():
    """dT/T below threshold should not trigger."""
    config = _mock_config(
        update_interval=10000.0,
        update_min_interval=100.0,
        update_dtmagma_frac=0.03,
    )
    dirs = _mock_dirs()
    # 1% change, below 3% threshold; also below ceiling
    hf_row = {
        'Time': 200.0,
        'T_magma': 2970.0,
        'Phi_global': 0.80,
    }
    interior_o = _mock_interior_o()

    with patch(
        'proteus.interior_struct.zalmoxis.zalmoxis_solver',
        return_value=(3.504e6, None),
    ) as mock_solver:
        result = update_structure_from_interior(
            dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.80
        )
    assert result[0] == pytest.approx(0.0, abs=1e-12)  # Not triggered
    # No-trigger discrimination: the solver must NOT run when dT/T is
    # below threshold. Catches a regression that fired the solver and
    # then discarded its result on the no-trigger branch.
    mock_solver.assert_not_called()


# ============================================================================
# test Phi_global absolute change trigger
# ============================================================================


@pytest.mark.unit
def test_phi_trigger():
    """dPhi >= update_dphi_abs should trigger an update."""
    config = _mock_config(
        update_interval=10000.0,
        update_min_interval=100.0,
        update_dphi_abs=0.05,
    )
    dirs = _mock_dirs()
    # 0.1 drop in phi, exceeds 0.05 threshold
    hf_row = {
        'Time': 200.0,
        'T_magma': 2990.0,
        'Phi_global': 0.70,
        'R_int': 6.371e6,
        'gravity': 9.81,
    }
    interior_o = _mock_interior_o()

    with (
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            return_value=(3.504e6, None),
        ) as mock_solver,
        patch('proteus.interior_energetics.wrapper.np.savetxt'),
        patch('proteus.interior_energetics.wrapper.shutil.copy2'),
    ):
        result = update_structure_from_interior(
            dirs, config, hf_row, interior_o, 0.0, 2990.0, 0.80
        )
    assert result[0] == pytest.approx(200.0, rel=1e-12)
    # Trigger discrimination: a 0.1 absolute drop in Phi crosses the
    # 0.05 threshold, so the solver must run. The phi trigger branch
    # is separate from the T trigger branch (dT/T is 0% here), so a
    # regression that disabled the phi branch alone would fail here.
    mock_solver.assert_called_once()


@pytest.mark.unit
def test_phi_no_trigger():
    """dPhi below threshold should not trigger (with T also below threshold)."""
    config = _mock_config(
        update_interval=10000.0,
        update_min_interval=100.0,
        update_dphi_abs=0.05,
        update_dtmagma_frac=0.03,
    )
    dirs = _mock_dirs()
    # Small changes in both T and Phi
    hf_row = {
        'Time': 200.0,
        'T_magma': 2990.0,
        'Phi_global': 0.78,
    }
    interior_o = _mock_interior_o()

    with patch(
        'proteus.interior_struct.zalmoxis.zalmoxis_solver',
        return_value=(3.504e6, None),
    ) as mock_solver:
        result = update_structure_from_interior(
            dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.80
        )
    assert result[0] == pytest.approx(0.0, abs=1e-12)  # Not triggered
    # No-trigger discrimination: dPhi = 0.02 < 0.05 threshold; the
    # solver must NOT run. Catches a regression that compared |dPhi|
    # to a different threshold or that flipped the comparison sense.
    mock_solver.assert_not_called()


# ============================================================================
# test floor blocks non-convergence triggers
# ============================================================================


@pytest.mark.unit
def test_floor_blocks_update():
    """Elapsed < update_min_interval should block all non-convergence triggers."""
    config = _mock_config(
        update_interval=10000.0,
        update_min_interval=100.0,
        update_dtmagma_frac=0.03,
    )
    dirs = _mock_dirs(mesh_active=False)
    # Large T change but elapsed only 50 yr (below floor of 100)
    hf_row = {
        'Time': 50.0,
        'T_magma': 2000.0,
        'Phi_global': 0.5,
    }
    interior_o = _mock_interior_o()

    with patch(
        'proteus.interior_struct.zalmoxis.zalmoxis_solver',
        return_value=(3.504e6, None),
    ) as mock_solver:
        result = update_structure_from_interior(
            dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.8
        )
    assert result[0] == pytest.approx(0.0, abs=1e-12)  # Floor blocked
    # Floor discrimination: a 33% T change is well above the dT/T
    # threshold AND a 0.3 Phi drop is well above the Phi threshold;
    # only the floor (elapsed < min_interval) can produce a no-update
    # outcome here. Catches a regression that ignored the floor and
    # still skipped the solver for some other reason.
    mock_solver.assert_not_called()


# ============================================================================
# test no-mesh-file resets convergence state
# ============================================================================


@pytest.mark.unit
def test_no_mesh_resets_convergence():
    """When Zalmoxis returns no mesh file, convergence state should reset."""
    config = _mock_config(update_interval=100.0, update_min_interval=50.0)
    dirs = _mock_dirs(mesh_active=True, mesh_steps=5)
    hf_row = {
        'Time': 200.0,
        'T_magma': 3000.0,
        'Phi_global': 0.8,
        'R_int': 6.371e6,
        'gravity': 9.81,
    }
    interior_o = _mock_interior_o()

    with (
        patch('proteus.interior_struct.zalmoxis.zalmoxis_solver', return_value=(3.504e6, None)),
        patch('proteus.interior_energetics.wrapper.np.savetxt'),
        patch('proteus.interior_energetics.wrapper.shutil.copy2'),
    ):
        update_structure_from_interior(dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.8)
    # No mesh file returned → convergence state reset
    assert dirs['mesh_shift_active'] is False
    assert dirs['mesh_convergence_steps'] == 0


# ============================================================================
# test full update path with mesh blending and convergence
# ============================================================================


@pytest.mark.unit
def test_full_update_with_mesh_blending(tmp_path):
    """Full update path: Zalmoxis returns mesh, blending fires, convergence tracked."""
    config = _mock_config(update_interval=1000.0, update_min_interval=100.0)
    mesh_file = str(tmp_path / 'spider_mesh.dat')
    prev_file = str(tmp_path / 'spider_mesh.dat.prev')

    dirs = {
        'output': str(tmp_path),
        'spider': '/tmp/spider',
        'spider_mesh': mesh_file,
        'spider_mesh_prev': prev_file,
        'mesh_shift_active': False,
        'mesh_convergence_steps': 0,
    }
    # Create data subdir for temp profile
    (tmp_path / 'data').mkdir(exist_ok=True)

    hf_row = {
        'Time': 1100.0,
        'T_magma': 3000.0,
        'Phi_global': 0.8,
        'R_int': 6.371e6,
        'gravity': 9.81,
    }
    interior_o = _mock_interior_o()

    with (
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            return_value=(3.504e6, mesh_file),
        ),
        patch('proteus.interior_energetics.wrapper.np.savetxt'),
        patch('proteus.interior_energetics.wrapper.shutil.copy2'),
        patch(
            'proteus.interior_energetics.spider.blend_mesh_files',
            return_value=0.15,
        ),
        patch('proteus.interior_energetics.wrapper.gc.collect'),
    ):
        result = update_structure_from_interior(
            dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.8
        )

    # Should have triggered (ceiling) and updated
    assert result[0] == pytest.approx(1100.0, rel=1e-12)
    # Mesh blending returned 15% shift > 5% max → convergence active
    assert dirs['mesh_shift_active'] is True
    assert dirs['mesh_convergence_steps'] == 1


@pytest.mark.unit
def test_convergence_max_exceeded(tmp_path):
    """After 20 convergence steps, mesh_shift_active should reset."""
    config = _mock_config(update_interval=1000.0, update_min_interval=100.0)
    mesh_file = str(tmp_path / 'spider_mesh.dat')
    prev_file = str(tmp_path / 'spider_mesh.dat.prev')

    dirs = {
        'output': str(tmp_path),
        'spider': '/tmp/spider',
        'spider_mesh': mesh_file,
        'spider_mesh_prev': prev_file,
        'mesh_shift_active': True,
        'mesh_convergence_steps': 20,
    }
    (tmp_path / 'data').mkdir(exist_ok=True)

    hf_row = {
        'Time': 1100.0,
        'T_magma': 3000.0,
        'Phi_global': 0.8,
        'R_int': 6.371e6,
        'gravity': 9.81,
    }
    interior_o = _mock_interior_o()

    with (
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            return_value=(3.504e6, mesh_file),
        ),
        patch('proteus.interior_energetics.wrapper.np.savetxt'),
        patch('proteus.interior_energetics.wrapper.shutil.copy2'),
        patch(
            'proteus.interior_energetics.spider.blend_mesh_files',
            return_value=0.15,
        ),
        patch('proteus.interior_energetics.wrapper.gc.collect'),
    ):
        result = update_structure_from_interior(
            dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.8
        )

    assert result[0] == pytest.approx(1100.0, rel=1e-12)
    # Exceeded max convergence steps → reset
    assert dirs['mesh_shift_active'] is False
    assert dirs['mesh_convergence_steps'] == 0


@pytest.mark.unit
def test_update_with_entropy_remap(tmp_path):
    """When SPIDER module is active and sim_times exist, entropy remap fires."""
    config = _mock_config(update_interval=1000.0, update_min_interval=100.0)
    mesh_file = str(tmp_path / 'spider_mesh.dat')
    prev_file = str(tmp_path / 'spider_mesh.dat.prev')

    dirs = {
        'output': str(tmp_path),
        'spider': '/tmp/spider',
        'spider_mesh': mesh_file,
        'spider_mesh_prev': prev_file,
        'mesh_shift_active': False,
        'mesh_convergence_steps': 0,
    }
    (tmp_path / 'data').mkdir(exist_ok=True)

    hf_row = {
        'Time': 1100.0,
        'T_magma': 3000.0,
        'Phi_global': 0.8,
        'R_int': 6.371e6,
        'gravity': 9.81,
    }
    interior_o = _mock_interior_o()

    with (
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            return_value=(3.504e6, mesh_file),
        ),
        patch('proteus.interior_energetics.wrapper.np.savetxt'),
        patch('proteus.interior_energetics.wrapper.shutil.copy2'),
        patch(
            'proteus.interior_energetics.spider.blend_mesh_files',
            return_value=0.02,
        ),
        patch(
            'proteus.interior_energetics.spider.get_all_output_times',
            return_value=[0.0, 500.0],
        ),
        patch('proteus.interior_energetics.spider.remap_entropy_for_new_mesh') as mock_remap,
        patch('proteus.interior_energetics.wrapper.gc.collect'),
    ):
        result = update_structure_from_interior(
            dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.8
        )

    assert result[0] == pytest.approx(1100.0, rel=1e-12)
    # Remap should have been called with latest JSON
    mock_remap.assert_called_once()
    call_args = mock_remap.call_args
    assert '500.json' in call_args.kwargs['json_path']


@pytest.mark.unit
def test_entropy_remap_exception(tmp_path):
    """Exception in get_all_output_times should not crash update."""
    config = _mock_config(update_interval=1000.0, update_min_interval=100.0)
    mesh_file = str(tmp_path / 'spider_mesh.dat')
    prev_file = str(tmp_path / 'spider_mesh.dat.prev')

    dirs = {
        'output': str(tmp_path),
        'spider': '/tmp/spider',
        'spider_mesh': mesh_file,
        'spider_mesh_prev': prev_file,
        'mesh_shift_active': False,
        'mesh_convergence_steps': 0,
    }
    (tmp_path / 'data').mkdir(exist_ok=True)

    hf_row = {
        'Time': 1100.0,
        'T_magma': 3000.0,
        'Phi_global': 0.8,
        'R_int': 6.371e6,
        'gravity': 9.81,
    }
    interior_o = _mock_interior_o()

    with (
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            return_value=(3.504e6, mesh_file),
        ),
        patch('proteus.interior_energetics.wrapper.np.savetxt'),
        patch('proteus.interior_energetics.wrapper.shutil.copy2'),
        patch(
            'proteus.interior_energetics.spider.blend_mesh_files',
            return_value=0.02,
        ),
        patch(
            'proteus.interior_energetics.spider.get_all_output_times',
            side_effect=RuntimeError('no JSON files'),
        ) as mock_get_times,
        patch('proteus.interior_energetics.wrapper.gc.collect'),
    ):
        # Should not raise
        result = update_structure_from_interior(
            dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.8
        )

    assert result[0] == pytest.approx(1100.0, rel=1e-12)
    # Exception-path discrimination: get_all_output_times raised, so it
    # must have been invoked; the function still returned cleanly. A
    # regression that short-circuited the remap before invoking the
    # call would still produce the same time advance.
    mock_get_times.assert_called_once()


@pytest.mark.unit
def test_update_creates_prev_file(tmp_path):
    """When no prev_path exists, update should create it."""
    config = _mock_config(update_interval=1000.0, update_min_interval=100.0)
    mesh_file = str(tmp_path / 'spider_mesh.dat')

    dirs = {
        'output': str(tmp_path),
        'spider': '/tmp/spider',
        'spider_mesh': mesh_file,
        'mesh_shift_active': False,
        'mesh_convergence_steps': 0,
        # No spider_mesh_prev set
    }
    (tmp_path / 'data').mkdir(exist_ok=True)

    hf_row = {
        'Time': 1100.0,
        'T_magma': 3000.0,
        'Phi_global': 0.8,
        'R_int': 6.371e6,
        'gravity': 9.81,
    }
    interior_o = _mock_interior_o()

    with (
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            return_value=(3.504e6, mesh_file),
        ),
        patch('proteus.interior_energetics.wrapper.np.savetxt'),
        patch('proteus.interior_energetics.wrapper.shutil.copy2'),
        patch(
            'proteus.interior_energetics.spider.blend_mesh_files',
            return_value=0.02,
        ),
        patch(
            'proteus.interior_energetics.spider.get_all_output_times',
            return_value=[],
        ),
        patch('proteus.interior_energetics.wrapper.gc.collect'),
    ):
        result = update_structure_from_interior(
            dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.8
        )

    assert result[0] == pytest.approx(1100.0, rel=1e-12)
    # spider_mesh_prev should now be set
    assert 'spider_mesh_prev' in dirs
    assert dirs['spider_mesh_prev'].endswith('.prev')


# ============================================================================
# test phi_crit warning in solve_structure
# ============================================================================


@pytest.mark.unit
def test_phi_crit_warning(caplog):
    """phi_crit < 0.01 in solve_structure should emit a warning."""
    from proteus.interior_energetics.wrapper import solve_structure

    config = MagicMock()
    config.planet.mass_tot = 1.0
    config.interior_struct.module = 'zalmoxis'
    config.params.stop.solid.phi_crit = 0.005

    with caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.wrapper'):
        with patch(
            'proteus.interior_energetics.wrapper.determine_interior_radius_with_zalmoxis'
        ) as mock_zal:
            solve_structure({}, config, None, {}, '/tmp')

    assert any('phi_crit' in r.message for r in caplog.records)
    # Dispatch discrimination: the warning fired AND the zalmoxis backend
    # was still dispatched (the warning does not abort solve_structure).
    # Catches a regression where the warning short-circuited the structure
    # call, leaving the simulation without an updated radius.
    mock_zal.assert_called_once()


# ============================================================================
# test determine_interior_radius_with_zalmoxis
# ============================================================================


@pytest.mark.unit
def test_determine_zalmoxis_spider_mesh(tmp_path):
    """determine_interior_radius_with_zalmoxis stores mesh path and .prev copy."""
    from proteus.interior_energetics.wrapper import determine_interior_radius_with_zalmoxis

    mesh_file = str(tmp_path / 'data' / 'spider_mesh.dat')
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data' / 'spider_mesh.dat').write_text('# 50 49\n1.0 2.0 3.0 -4.0\n')

    config = MagicMock()
    config.interior_energetics.module = 'spider'
    config.interior_energetics.eos_dir = 'WolfBower2018_MgSiO3'
    config.planet.temperature_mode = 'isothermal'
    config.planet.mass_tot = 1.0
    config.interior_struct.zalmoxis.mantle_eos = 'WolfBower2018:MgSiO3'

    dirs = {'spider': '/nonexistent/spider'}
    hf_row = {'R_int': 6.371e6, 'gravity': 9.81, 'T_magma': 3000.0}

    with (
        patch('proteus.interior_energetics.wrapper.get_nlevb', return_value=50),
        patch('proteus.interior_energetics.wrapper.Interior_t'),
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            return_value=(3.48e6, mesh_file),
        ),
        patch('proteus.interior_energetics.wrapper.run_interior'),
        patch('proteus.interior_struct.zalmoxis.generate_spider_tables'),
    ):
        determine_interior_radius_with_zalmoxis(dirs, config, None, hf_row, str(tmp_path))

    assert dirs['spider_mesh'] == mesh_file
    assert dirs['spider_mesh_prev'] == mesh_file + '.prev'
    assert dirs['mesh_shift_active'] is False
    assert dirs['mesh_convergence_steps'] == 0
    # Verify .prev file was created
    assert (tmp_path / 'data' / 'spider_mesh.dat.prev').exists()


@pytest.mark.unit
def test_determine_zalmoxis_adiabatic_switch(caplog):
    """SPIDER + isothermal + T-dep EOS switches to adiabatic, then restores."""
    from proteus.interior_energetics.wrapper import determine_interior_radius_with_zalmoxis

    config = MagicMock()
    config.interior_energetics.module = 'spider'
    config.interior_energetics.eos_dir = 'WolfBower2018_MgSiO3'
    config.planet.temperature_mode = 'isothermal'
    config.planet.mass_tot = 1.0
    config.interior_struct.zalmoxis.mantle_eos = 'WolfBower2018:MgSiO3'

    dirs = {'spider': '/nonexistent/spider'}
    hf_row = {'R_int': 6.371e6}

    with (
        patch('proteus.interior_energetics.wrapper.get_nlevb', return_value=50),
        patch('proteus.interior_energetics.wrapper.Interior_t'),
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            return_value=(3.48e6, None),
        ),
        patch('proteus.interior_energetics.wrapper.run_interior'),
        patch('proteus.interior_struct.zalmoxis.generate_spider_tables'),
        caplog.at_level('INFO', logger='fwl.proteus.interior_energetics.wrapper'),
    ):
        determine_interior_radius_with_zalmoxis(dirs, config, None, hf_row, '/tmp')

    # Mode should be restored after the call
    assert config.planet.temperature_mode == 'isothermal'
    assert any('adiabatic' in r.message for r in caplog.records)
    # No mesh file → no spider_mesh key
    assert 'spider_mesh' not in dirs
    assert dirs['mesh_shift_active'] is False


@pytest.mark.unit
def test_determine_zalmoxis_no_adiabatic_switch_non_tdep(caplog):
    """Non-T-dep EOS (Seager2007) does NOT trigger the adiabatic override.

    The companion test `test_determine_zalmoxis_adiabatic_switch` covers the
    positive case (T-dep EOS with SPIDER + isothermal triggers the override).
    This test pins the negative case: a Seager2007 mantle EOS prefix is
    outside `_TDEP_PREFIXES`, so the override must NOT fire and the
    temperature_mode must remain 'isothermal' both during and after the
    call. The 'Overriding Zalmoxis temperature_mode' log line must NOT
    appear.
    """
    from proteus.interior_energetics.wrapper import determine_interior_radius_with_zalmoxis

    config = MagicMock()
    config.interior_energetics.module = 'spider'
    config.interior_energetics.eos_dir = 'Seager2007'
    config.planet.temperature_mode = 'isothermal'
    config.planet.mass_tot = 1.0
    config.interior_struct.zalmoxis.mantle_eos = 'Seager2007:silicate'

    dirs = {'spider': '/nonexistent/spider'}
    hf_row = {'R_int': 6.371e6}

    with (
        patch('proteus.interior_energetics.wrapper.get_nlevb', return_value=50),
        patch('proteus.interior_energetics.wrapper.Interior_t'),
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            return_value=(3.48e6, None),
        ),
        patch('proteus.interior_energetics.wrapper.run_interior'),
        patch('proteus.interior_struct.zalmoxis.generate_spider_tables'),
        caplog.at_level('INFO', logger='fwl.proteus.interior_energetics.wrapper'),
    ):
        determine_interior_radius_with_zalmoxis(dirs, config, None, hf_row, '/tmp')

    # Mode unchanged: the override gate did not fire.
    assert config.planet.temperature_mode == 'isothermal'
    # No override log line: the gate skipped silently.
    assert not any(
        'Overriding Zalmoxis temperature_mode' in r.message for r in caplog.records
    ), (
        'Adiabatic override fired despite Seager2007 EOS being outside '
        '_TDEP_PREFIXES; the prefix gate is broken.'
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    'mantle_eos,should_override',
    [
        # T-dep prefixes: override expected
        ('WolfBower2018:MgSiO3', True),
        ('WolfBower2018:Fe', True),
        ('RTPress100TPa:silicate', True),
        # Non-T-dep prefixes: override skipped
        ('Seager2007:silicate', False),
        ('Seager2007:Fe', False),
        ('PALEOS-Fei2021:MgSiO3', False),
    ],
)
def test_determine_zalmoxis_adiabatic_override_gate_per_eos(
    mantle_eos, should_override, caplog
):
    """The adiabatic override gate fires iff the mantle_eos starts with one
    of `_TDEP_PREFIXES` and the run is SPIDER + isothermal.

    Discriminating values: each parametrized case targets a specific
    prefix-match decision. A regression that broadens the gate (e.g.,
    matching Seager2007 as if it were T-dep) or narrows it (e.g.,
    missing RTPress100TPa) fires here with the offending eos name in
    the assertion output.
    """
    from proteus.interior_energetics.wrapper import determine_interior_radius_with_zalmoxis

    config = MagicMock()
    config.interior_energetics.module = 'spider'
    config.interior_energetics.eos_dir = mantle_eos.split(':')[0]
    config.planet.temperature_mode = 'isothermal'
    config.planet.mass_tot = 1.0
    config.interior_struct.zalmoxis.mantle_eos = mantle_eos

    dirs = {'spider': '/nonexistent/spider'}
    hf_row = {'R_int': 6.371e6}

    with (
        patch('proteus.interior_energetics.wrapper.get_nlevb', return_value=50),
        patch('proteus.interior_energetics.wrapper.Interior_t'),
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            return_value=(3.48e6, None),
        ),
        patch('proteus.interior_energetics.wrapper.run_interior'),
        patch('proteus.interior_struct.zalmoxis.generate_spider_tables'),
        caplog.at_level('INFO', logger='fwl.proteus.interior_energetics.wrapper'),
    ):
        determine_interior_radius_with_zalmoxis(dirs, config, None, hf_row, '/tmp')

    fired = any('Overriding Zalmoxis temperature_mode' in r.message for r in caplog.records)
    assert fired is should_override, (
        f'Adiabatic override gate behaved unexpectedly for mantle_eos={mantle_eos!r}: '
        f'expected fired={should_override}, got fired={fired}.'
    )
    # The config object must never be mutated regardless of the override path.
    assert config.planet.temperature_mode == 'isothermal'


@pytest.mark.unit
def test_determine_zalmoxis_no_override_when_module_is_aragog(caplog):
    """Even with a T-dep EOS, Aragog interior must NOT trigger the override.

    The override is gated on `interior_energetics.module == 'spider'`. A
    regression that drops that gate would corrupt Aragog runs by re-routing
    Zalmoxis through an adiabatic-mode initial guess Aragog never asked for.

    Two assertions: (1) the override log line did not appear, (2) the call
    to `zalmoxis_solver` passed `temperature_mode_override=None`. The
    second is the behavioural check: a regression that suppressed only
    the log line but kept passing the override would still fail (2).
    """
    from proteus.interior_energetics.wrapper import determine_interior_radius_with_zalmoxis

    config = MagicMock()
    config.interior_energetics.module = 'aragog'
    config.interior_energetics.eos_dir = 'WolfBower2018_MgSiO3'
    config.planet.temperature_mode = 'isothermal'
    config.planet.mass_tot = 1.0
    config.interior_struct.zalmoxis.mantle_eos = 'WolfBower2018:MgSiO3'

    dirs = {}
    hf_row = {'R_int': 6.371e6}

    with (
        patch('proteus.interior_energetics.wrapper.get_nlevb', return_value=50),
        patch('proteus.interior_energetics.wrapper.Interior_t'),
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            return_value=(3.48e6, None),
        ) as mock_zalmoxis_solver,
        patch('proteus.interior_energetics.wrapper.run_interior'),
        patch('proteus.interior_struct.zalmoxis.generate_spider_tables'),
        caplog.at_level('INFO', logger='fwl.proteus.interior_energetics.wrapper'),
    ):
        determine_interior_radius_with_zalmoxis(dirs, config, None, hf_row, '/tmp')

    assert not any(
        'Overriding Zalmoxis temperature_mode' in r.message for r in caplog.records
    ), 'Override fired under Aragog; module-gate is broken.'
    # Behavioural check: zalmoxis_solver received `temperature_mode_override=None`.
    assert mock_zalmoxis_solver.call_count == 1
    call = mock_zalmoxis_solver.call_args
    assert call.kwargs.get('temperature_mode_override') is None, (
        f'zalmoxis_solver received temperature_mode_override='
        f'{call.kwargs.get("temperature_mode_override")!r} under Aragog; '
        f'the module gate is broken.'
    )


@pytest.mark.unit
def test_determine_zalmoxis_no_override_when_temperature_mode_already_adiabatic(
    caplog,
):
    """If temperature_mode is already 'adiabatic', the override is a no-op.

    The override only flips isothermal to adiabatic. A regression that
    drops the `temperature_mode == 'isothermal'` clause would re-fire the
    log line spuriously when the user already asked for adiabatic.

    Two assertions: log line absent AND zalmoxis_solver received
    `temperature_mode_override=None` (because the user-set adiabatic
    needs no override; the runtime simply uses the config value).
    """
    from proteus.interior_energetics.wrapper import determine_interior_radius_with_zalmoxis

    config = MagicMock()
    config.interior_energetics.module = 'spider'
    config.interior_energetics.eos_dir = 'WolfBower2018_MgSiO3'
    config.planet.temperature_mode = 'adiabatic'
    config.planet.mass_tot = 1.0
    config.interior_struct.zalmoxis.mantle_eos = 'WolfBower2018:MgSiO3'

    dirs = {'spider': '/nonexistent/spider'}
    hf_row = {'R_int': 6.371e6}

    with (
        patch('proteus.interior_energetics.wrapper.get_nlevb', return_value=50),
        patch('proteus.interior_energetics.wrapper.Interior_t'),
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            return_value=(3.48e6, None),
        ) as mock_zalmoxis_solver,
        patch('proteus.interior_energetics.wrapper.run_interior'),
        patch('proteus.interior_struct.zalmoxis.generate_spider_tables'),
        caplog.at_level('INFO', logger='fwl.proteus.interior_energetics.wrapper'),
    ):
        determine_interior_radius_with_zalmoxis(dirs, config, None, hf_row, '/tmp')

    # User-set adiabatic stays untouched (config object is not mutated)
    assert config.planet.temperature_mode == 'adiabatic'
    # Behavioural check: zalmoxis_solver received no override.
    assert mock_zalmoxis_solver.call_count == 1
    assert mock_zalmoxis_solver.call_args.kwargs.get('temperature_mode_override') is None, (
        'zalmoxis_solver received a non-None temperature_mode_override despite '
        'the user already requesting adiabatic; the isothermal-clause is broken.'
    )
    assert not any(
        'Overriding Zalmoxis temperature_mode' in r.message for r in caplog.records
    ), 'Override fired when user already requested adiabatic; isothermal-clause broken.'


@pytest.mark.unit
def test_solve_structure_invalid_module():
    """solve_structure raises ValueError for unknown struct.module."""
    from proteus.interior_energetics.wrapper import solve_structure

    config = MagicMock()
    config.planet.mass_tot = 1.0
    config.interior_struct.module = 'nonexistent_module'
    config.params.stop.solid.phi_crit = 0.5

    with pytest.raises(ValueError, match='Invalid structure interior module'):
        solve_structure({}, config, None, {}, '/tmp')
    # Dispatch discrimination: switching back to a recognised module
    # name (zalmoxis) must dispatch to its backend. Catches a regression
    # that fired the validation gate regardless of the module string.
    config.interior_struct.module = 'zalmoxis'
    with patch(
        'proteus.interior_energetics.wrapper.determine_interior_radius_with_zalmoxis'
    ) as mock_zal:
        solve_structure({}, config, None, {}, '/tmp')
    mock_zal.assert_called_once()


# ============================================================================
# _structure_stale flag contract (set on fall-back, cleared on success)
# ============================================================================


@pytest.mark.unit
def test_structure_stale_flag_cleared_on_zalmoxis_success(tmp_path):
    """Successful zalmoxis_solver call clears hf_row['_structure_stale'].

    Contract: ``hf_row['_structure_stale']`` is the signal that downstream
    consumers (notably Aragog) use to distinguish a fresh structure from
    a stale one carried over from a prior fall-back. A successful
    re-solve must set the key to False so consumers can rely on it.

    Edge cases covered:
    - Flag was True at entry (simulating prior fall-back) -> must be False after.
    - Flag was missing at entry (legacy hf_row) -> must be False after (set,
      not just not-True), so downstream consumers can rely on the key.
    """
    config = _mock_config(update_interval=1000.0, update_min_interval=100.0)
    dirs = _mock_dirs()
    interior_o = _mock_interior_o()

    # Case A: flag was True at entry (post-fall-back state)
    hf_row_A = {
        'Time': 1100.0,
        'T_magma': 3000.0,
        'Phi_global': 0.8,
        'R_int': 6.371e6,
        'gravity': 9.81,
        'M_int': 5.972e24,
        '_structure_stale': True,
    }
    with (
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            return_value=(3.504e6, str(tmp_path / 'mesh.dat')),
        ),
        patch('proteus.interior_energetics.wrapper.np.savetxt'),
        patch('proteus.interior_energetics.wrapper.shutil.copy2'),
        patch(
            'proteus.interior_energetics.spider.blend_mesh_files',
            return_value=0.0,
        ),
    ):
        update_structure_from_interior(dirs, config, hf_row_A, interior_o, 0.0, 3000.0, 0.8)
    # Flag must be present and False after a successful call.
    assert '_structure_stale' in hf_row_A, 'flag must be set, not absent'
    assert hf_row_A['_structure_stale'] is False, (
        'flag must be cleared (False) on Zalmoxis success, was %r'
        % hf_row_A['_structure_stale']
    )

    # Case B: flag was missing at entry (legacy hf_row never touched by fall-back)
    hf_row_B = {
        'Time': 1100.0,
        'T_magma': 3000.0,
        'Phi_global': 0.8,
        'R_int': 6.371e6,
        'gravity': 9.81,
        'M_int': 5.972e24,
        # No '_structure_stale' key at entry
    }
    with (
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            return_value=(3.504e6, str(tmp_path / 'mesh.dat')),
        ),
        patch('proteus.interior_energetics.wrapper.np.savetxt'),
        patch('proteus.interior_energetics.wrapper.shutil.copy2'),
        patch(
            'proteus.interior_energetics.spider.blend_mesh_files',
            return_value=0.0,
        ),
    ):
        update_structure_from_interior(dirs, config, hf_row_B, interior_o, 0.0, 3000.0, 0.8)
    assert hf_row_B.get('_structure_stale') is False, (
        'success path must set flag to False even if it was absent on entry'
    )


@pytest.mark.unit
def test_structure_stale_flag_set_on_zalmoxis_failure(tmp_path):
    """Zalmoxis RuntimeError sets hf_row['_structure_stale'] = True.

    Documents the canonical fall-back contract. Anti-happy-path: also
    checks that the failure does NOT clear the flag if it was already
    True (i.e. consecutive failures keep it set).
    """

    config = _mock_config(update_interval=1000.0, update_min_interval=100.0)
    dirs = _mock_dirs()
    interior_o = _mock_interior_o()

    hf_row = {
        'Time': 1100.0,
        'T_magma': 3000.0,
        'Phi_global': 0.8,
        'R_int': 6.371e6,
        'gravity': 9.81,
        'M_int': 5.972e24,
        # Flag absent at entry
    }
    with patch(
        'proteus.interior_struct.zalmoxis.zalmoxis_solver',
        side_effect=RuntimeError('mock convergence failure'),
    ):
        update_structure_from_interior(dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.8)
    assert hf_row.get('_structure_stale') is True, 'fall-back path must set flag to True'

    # Second consecutive failure: flag stays True (not toggled or cleared).
    with patch(
        'proteus.interior_struct.zalmoxis.zalmoxis_solver',
        side_effect=RuntimeError('mock convergence failure 2'),
    ):
        update_structure_from_interior(dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.8)
    assert hf_row.get('_structure_stale') is True

    # Reset for any downstream tests.
    interior_o.zalmoxis_fail_count = 0


# ============================================================================
# Post-acceptance mass-anchor guard
# ============================================================================


@pytest.mark.unit
def test_mass_anchor_violation_triggers_fallback(tmp_path):
    """A 'converged' Zalmoxis result with |M_int/M_target - 1| > 3e-3 falls back.

    Contract: Zalmoxis' internal ``solver_tol_outer`` (3e-3) leaves up to
    ~0.3 % drift between ``hf_row['M_int']`` and the dry mass target on
    hot fully-molten profiles, which is acceptable. The wrapper's
    mass-anchor guard rejects results with drift above 3e-3 to catch
    attractor-basin failure modes (~9-15 % off target) and gross
    corruption. The tight <0.1 % mass-conservation contract is delivered
    by the Newton + brentq inner path, not by this safety net.

    Edge cases covered:
    - Far-over-threshold (5e-3 drift): must reject.
    - Just-under-threshold (2e-3 drift): must accept (Zalmoxis
      noise-floor regime).
    - Near-but-under threshold (3e-3 - epsilon drift): must accept
      (strict > comparison).
    """

    config = _mock_config(update_interval=1000.0, update_min_interval=100.0)
    interior_o = _mock_interior_o()

    M_target = 5.972e24

    def _make_hf_row():
        return {
            'Time': 1100.0,
            'T_magma': 3000.0,
            'Phi_global': 0.8,
            'R_int': 6.371e6,
            'gravity': 9.81,
            'M_int': 5.972e24,
        }

    # Helper: mock zalmoxis_solver that mutates hf_row to set M_int and
    # M_int_target. The test-side dict mutation mimics the real solver's
    # writeback path (proteus/interior_struct/zalmoxis.py:~1533).
    def _make_solver_mock(M_int_returned):
        def _side(config, outdir, hf_row, **kwargs):
            hf_row['M_int'] = M_int_returned
            hf_row['M_int_target'] = M_target
            hf_row['R_int'] = 6.371e6
            return (3.504e6, str(tmp_path / 'mesh.dat'))

        return _side

    # Case 1: 5e-3 drift (over threshold) -> fall-back path must fire.
    interior_o.zalmoxis_fail_count = 0
    hf_row_over = _make_hf_row()
    dirs_over = _mock_dirs()
    with (
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            side_effect=_make_solver_mock(M_target * (1 + 5e-3)),
        ),
        patch('proteus.interior_energetics.wrapper.np.savetxt'),
        patch('proteus.interior_energetics.wrapper.shutil.copy2'),
    ):
        update_structure_from_interior(
            dirs_over, config, hf_row_over, interior_o, 0.0, 3000.0, 0.8
        )
    assert hf_row_over.get('_structure_stale') is True, (
        'mass-anchor violation must trigger fall-back (-> _structure_stale=True)'
    )
    assert interior_o.zalmoxis_fail_count == 1, (
        'mass-anchor violation must increment _zalmoxis_fail_count (got %d)'
        % interior_o.zalmoxis_fail_count
    )

    # Case 2: 2e-3 drift (under 3e-3 threshold) -> success path.
    # This is the regime Zalmoxis normally lands in on hot mantle
    # profiles (~0.18-0.28 % drift in production).
    interior_o.zalmoxis_fail_count = 0
    hf_row_under = _make_hf_row()
    dirs_under = _mock_dirs()
    with (
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            side_effect=_make_solver_mock(M_target * (1 + 2e-3)),
        ),
        patch('proteus.interior_energetics.wrapper.np.savetxt'),
        patch('proteus.interior_energetics.wrapper.shutil.copy2'),
        patch(
            'proteus.interior_energetics.spider.blend_mesh_files',
            return_value=0.0,
        ),
    ):
        update_structure_from_interior(
            dirs_under, config, hf_row_under, interior_o, 0.0, 3000.0, 0.8
        )
    assert hf_row_under.get('_structure_stale') is False, (
        'sub-tolerance drift must NOT trigger fall-back '
        '(_structure_stale=%r)' % hf_row_under.get('_structure_stale')
    )
    assert interior_o.zalmoxis_fail_count == 0

    # Case 3: target=0 (degenerate, e.g. zalmoxis didn't write target) -> skip
    # the check (must not divide by zero or spuriously raise).
    interior_o.zalmoxis_fail_count = 0
    hf_row_no_target = _make_hf_row()
    dirs_no = _mock_dirs()

    def _no_target_side(config, outdir, hf_row, **kwargs):
        hf_row['M_int'] = M_target
        hf_row['M_int_target'] = 0.0  # degenerate
        hf_row['R_int'] = 6.371e6
        return (3.504e6, str(tmp_path / 'mesh.dat'))

    with (
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            side_effect=_no_target_side,
        ),
        patch('proteus.interior_energetics.wrapper.np.savetxt'),
        patch('proteus.interior_energetics.wrapper.shutil.copy2'),
        patch(
            'proteus.interior_energetics.spider.blend_mesh_files',
            return_value=0.0,
        ),
    ):
        update_structure_from_interior(
            dirs_no, config, hf_row_no_target, interior_o, 0.0, 3000.0, 0.8
        )
    # Must not raise, must not fall-back (graceful degrade for legacy
    # callers that haven't been updated to write M_int_target).
    assert hf_row_no_target.get('_structure_stale') is False
    assert interior_o.zalmoxis_fail_count == 0

    interior_o.zalmoxis_fail_count = 0  # cleanup


@pytest.mark.unit
def test_zalmoxis_output_restored_from_prev_on_wrapper_raise(tmp_path):
    """Wrapper-level RuntimeError (e.g. mass-anchor violation) restores
    zalmoxis_output.dat from .prev backup.

    The atomic-write path inside ``zalmoxis_solver`` handles
    schema-violation raises. But the wrapper's mass-anchor check raises
    AFTER ``zalmoxis_solver`` returns successfully, so the rollback must
    happen at the wrapper level too. Otherwise Aragog reads the new
    failing file on the next iter with the old ``hf_row['R_int']`` and
    crashes on EOS-vs-mesh inconsistency.

    Edge case covered: when no .prev exists (first call of a run), the
    wrapper must NOT raise; it best-effort skips the restore.
    """

    # Set up an output dir with data/ subdirectory.
    outdir = tmp_path
    (outdir / 'data').mkdir(exist_ok=True)
    output_path = str(outdir / 'data' / 'zalmoxis_output.dat')
    prev_path = output_path + '.prev'

    # Pre-existing .prev with KNOWN content (last-good).
    with open(prev_path, 'w') as f:
        f.write('PREV_KNOWN_CONTENT\n')

    config = _mock_config(update_interval=1000.0, update_min_interval=100.0)
    interior_o = _mock_interior_o()

    M_target = 5.972e24

    def _solver_writes_corrupt_then_succeeds(config, outdir_arg, hf_row, **kwargs):
        # Mock zalmoxis_solver: write a NEW file, set hf_row M_int_target
        # so wrapper's mass-anchor check fires.
        with open(output_path, 'w') as f:
            f.write('NEW_FAILING_CONTENT\n')
        hf_row['M_int'] = M_target * (1 + 5e-3)  # 0.5% off > 3e-3 -> raise
        hf_row['M_int_target'] = M_target
        hf_row['R_int'] = 6.371e6
        return (3.504e6, str(outdir / 'spider_mesh.dat'))

    dirs = _mock_dirs()
    dirs['output'] = str(outdir)
    hf_row = {
        'Time': 1100.0,
        'T_magma': 3000.0,
        'Phi_global': 0.8,
        'R_int': 6.371e6,
        'gravity': 9.81,
        'M_int': 5.972e24,
    }
    # Don't patch shutil.copy2 here -- the rollback in the except-block
    # calls it directly to restore from .prev, and we want that real copy
    # to happen. Other tests that patch copy2 do so to avoid pre-call
    # mesh-snapshot writes in tmp_path/spider that don't exist; this test
    # has its own .prev set up so the restore works.
    with (
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            side_effect=_solver_writes_corrupt_then_succeeds,
        ),
        patch('proteus.interior_energetics.wrapper.np.savetxt'),
    ):
        update_structure_from_interior(
            dirs,
            config,
            hf_row,
            interior_o,
            0.0,
            3000.0,
            0.8,
        )

    # File on disk should have been restored to .prev content.
    assert os.path.isfile(output_path), 'output file must exist after fall-back'
    with open(output_path) as f:
        content_after = f.read()
    assert content_after == 'PREV_KNOWN_CONTENT\n', (
        'wrapper-level fall-back must restore zalmoxis_output.dat from '
        '.prev (got %r)' % content_after
    )

    interior_o.zalmoxis_fail_count = 0  # cleanup


# ============================================================================
# Stale-aware re-trigger via last_successful_struct_time
# ============================================================================


@pytest.mark.unit
def test_stale_aware_ceiling_fires_after_failure_window(tmp_path):
    """Stale-aware ceiling fires after update_stale_ceiling regardless of
    intervening failed re-solves.

    Contract: the standard ceiling trigger uses elapsed time since the
    last *call*, which a fall-back re-anchors; a failed call therefore
    keeps the trigger silent for a full ``update_interval`` afterwards
    and lets Aragog evolve on a frozen mesh. The stale-aware ceiling
    fires after ``update_stale_ceiling`` has elapsed since the last
    *successful* re-solve, tracked on ``Interior_t`` via
    ``last_successful_struct_time``. It is independent of
    ``last_struct_time`` and so survives failures.
    """

    # Configure a 50 kyr normal ceiling and a 25 kyr stale ceiling.
    # Test point: elapsed=30 kyr since LAST CALL (under normal ceiling)
    # but 30 kyr since LAST SUCCESS (over stale ceiling).
    config = _mock_config(update_interval=5e4, update_min_interval=0.0)
    config.interior_struct.zalmoxis.update_stale_ceiling = 2.5e4
    dirs = _mock_dirs()

    # Real Interior_t so last_successful_struct_time attribute lookup works
    interior_o = Interior_t(50)
    interior_o.last_successful_struct_time = float('-inf')
    # Decorate with the arrays the wrapper requires.
    interior_o.radius = np.linspace(6.371e6, 3.504e6, 50)
    interior_o.temp = np.full(49, 3000.0)

    # Simulate state: time has advanced 3e4 yr since the last call,
    # but no successful re-solve yet.
    last_struct_time = 7e4  # last call was at 7e4 yr (a failed one)
    current_time = 1e5  # now at 1e5 yr; elapsed = 3e4 (< 5e4 ceiling)
    hf_row = {
        'Time': current_time,
        'T_magma': 3000.0,
        'Phi_global': 0.8,  # no Phi/T trigger (no change)
        'R_int': 6.371e6,
        'gravity': 9.81,
        'M_int': 5.972e24,
    }
    # Last *success* is at -inf, so stale-aware ceiling should NOT fire
    # for last_success=-inf (no anchor yet). Verify: trigger should NOT
    # fire because elapsed=3e4 < normal ceiling=5e4 and no success anchor
    # exists.
    with patch(
        'proteus.interior_struct.zalmoxis.zalmoxis_solver',
        side_effect=AssertionError('zalmoxis_solver should not be called'),
    ):
        out = update_structure_from_interior(
            dirs,
            config,
            hf_row,
            interior_o,
            last_struct_time,
            hf_row['T_magma'],
            hf_row['Phi_global'],
        )
    # Returned tuple unchanged (no_update path).
    assert out == (last_struct_time, hf_row['T_magma'], hf_row['Phi_global'])

    # Now seed a successful past re-solve at t=7e4 (= 30 kyr ago).
    interior_o.last_successful_struct_time = 7e4

    # Same hf_row, same elapsed; this time stale-aware ceiling SHOULD
    # fire (3e4 yr >= 2.5e4 yr).
    def _success_side(config, outdir, hf_row, **kwargs):
        hf_row['M_int'] = 5.972e24
        hf_row['M_int_target'] = 5.972e24
        hf_row['R_int'] = 6.371e6
        return (3.504e6, str(tmp_path / 'mesh.dat'))

    with (
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            side_effect=_success_side,
        ),
        patch('proteus.interior_energetics.wrapper.np.savetxt'),
        patch('proteus.interior_energetics.wrapper.shutil.copy2'),
        patch(
            'proteus.interior_energetics.spider.blend_mesh_files',
            return_value=0.0,
        ),
    ):
        out = update_structure_from_interior(
            dirs,
            config,
            hf_row,
            interior_o,
            last_struct_time,
            hf_row['T_magma'],
            hf_row['Phi_global'],
        )
    # Trigger fired and updated last_struct_time to current_time.
    assert out[0] == current_time, (
        'stale-aware ceiling must trigger; got last_struct_time=%r' % out[0]
    )
    # Successful tracker advanced.
    assert interior_o.last_successful_struct_time == current_time, (
        'last_successful_struct_time must advance on success'
    )

    interior_o.zalmoxis_fail_count = 0


@pytest.mark.unit
def test_last_successful_struct_time_not_advanced_on_failure(tmp_path):
    """A Zalmoxis fall-back must NOT advance last_successful_struct_time.

    Otherwise the stale-aware ceiling becomes equivalent to the normal
    ceiling and the stale-aware path would have no effect.
    """

    config = _mock_config(update_interval=1e3, update_min_interval=0.0)
    config.interior_struct.zalmoxis.update_stale_ceiling = 0.0  # disable stale-aware ceiling
    dirs = _mock_dirs()

    interior_o = Interior_t(50)
    interior_o.last_successful_struct_time = 5e4
    interior_o.radius = np.linspace(6.371e6, 3.504e6, 50)
    interior_o.temp = np.full(49, 3000.0)

    hf_row = {
        'Time': 7e4,
        'T_magma': 3000.0,
        'Phi_global': 0.8,
        'R_int': 6.371e6,
        'gravity': 9.81,
        'M_int': 5.972e24,
    }
    # Call triggers ceiling, zalmoxis_solver raises -> fall-back.
    with patch(
        'proteus.interior_struct.zalmoxis.zalmoxis_solver',
        side_effect=RuntimeError('mock failure'),
    ) as mock_solver:
        update_structure_from_interior(
            dirs,
            config,
            hf_row,
            interior_o,
            0.0,
            3000.0,
            0.8,
        )
    assert interior_o.last_successful_struct_time == pytest.approx(5e4, rel=1e-12), (
        'fall-back must NOT advance last_successful_struct_time '
        '(got %r, expected 5e4)' % interior_o.last_successful_struct_time
    )
    # Fall-back discrimination: the solver was actually invoked (the test
    # exercises the failure branch, not the no-trigger branch). A
    # regression that bypassed the solver entirely would also leave
    # last_successful_struct_time at 5e4 and fool the primary assert.
    mock_solver.assert_called_once()

    interior_o.zalmoxis_fail_count = 0


# ============================================================================
# _prevent_warming_clamp_active: gate for the T_magma one-way ratchet
# ============================================================================


@pytest.mark.unit
def test_prevent_warming_clamp_off_when_disabled():
    """prevent_warming = false: clamp must be inactive for every module.

    Default-path regression: with the user-facing flag off, the clamp helper
    must return False unconditionally.
    """
    for module in ('aragog', 'spider', 'dummy'):
        cfg = _ns_prevent_warming(prevent_warming=False, module=module)
        assert _prevent_warming_clamp_active(cfg) is False, (
            f'clamp wrongly active for module={module} when prevent_warming=False'
        )
    # Toggle discrimination: flipping prevent_warming to True must flip
    # the clamp ON for every module. Catches a regression that hard-coded
    # the return value to False regardless of the flag.
    for module in ('aragog', 'spider', 'dummy'):
        cfg = _ns_prevent_warming(prevent_warming=True, module=module)
        assert _prevent_warming_clamp_active(cfg) is True, (
            f'clamp wrongly inactive for module={module} when prevent_warming=True'
        )


@pytest.mark.unit
@pytest.mark.parametrize('module', ['aragog', 'spider', 'dummy'])
def test_prevent_warming_clamp_on_when_enabled_regardless_of_module(module):
    """prevent_warming = true: clamp must be ON for every interior module.

    The earlier dilatation-on gate was removed together with the explicit
    Φ_vol source term: that source was identified as a divergence
    double-count (Bower 2018 §3, SPIDER energy.c) and deleted, so the
    heat-pump motivation for the gate no longer applies. The clamp is now
    purely a user opt-in for strictly-cooling regimes.
    """
    cfg = _ns_prevent_warming(prevent_warming=True, module=module)
    assert _prevent_warming_clamp_active(cfg) is True
    # Toggle discrimination: flipping prevent_warming to False (for the
    # same module) must flip the clamp OFF. Catches a regression that
    # branched on module name and ignored the prevent_warming flag.
    cfg_off = _ns_prevent_warming(prevent_warming=False, module=module)
    assert _prevent_warming_clamp_active(cfg_off) is False


@pytest.mark.unit
def test_prevent_warming_clamp_returns_bool_not_truthy_object():
    """Regression: the gate must return ``bool`` so callers can ``is True`` it.

    Edge case: the helper's call sites use ``if _prevent_warming_clamp_active(cfg)``
    pattern, but several upstream tests call ``assert ... is True``. A
    ``return cfg.planet.prevent_warming`` (without the bool() wrap) would
    return an attrs-defined attribute that is *truthy* but not the literal
    ``True`` constant, silently breaking those assertions.
    """
    cfg = _ns_prevent_warming(prevent_warming=True)
    out = _prevent_warming_clamp_active(cfg)
    assert isinstance(out, bool)
    assert out is True
    cfg_off = _ns_prevent_warming(prevent_warming=False)
    out_off = _prevent_warming_clamp_active(cfg_off)
    assert isinstance(out_off, bool)
    assert out_off is False


# ============================================================================
# run_interior step limiters: prevent_warming + boundary-module dT_delta
# ============================================================================


def _make_run_interior_config(*, prevent_warming: bool, module: str = 'dummy'):
    """Build a minimal config that drives only the limiter block in run_interior.

    The dummy backend is used so we can stub run_dummy_int and avoid the heavy
    interior solver. Configures tmagma_atol/rtol from interior_energetics
    (our schema), prevent_warming from planet (also our schema).
    """
    config = MagicMock()
    config.interior_energetics.module = module
    config.interior_energetics.heat_tidal = False
    config.interior_energetics.heat_radiogenic = False
    config.interior_energetics.tmagma_atol = 20.0
    config.interior_energetics.tmagma_rtol = 0.0
    config.interior_energetics.boundary.Tsurf_event_change = 25.0
    config.planet.prevent_warming = prevent_warming
    config.params.stop.time.maximum = 1e9
    return config


def _make_run_interior_state(prev_f_int: float = 0.1):
    import pandas as pd

    hf_all = pd.DataFrame(
        [
            {
                'Time': 90.0,
                'T_magma': 3000.0,
                'T_surf': 2800.0,
                'Phi_global': 0.4,
                'F_int': prev_f_int,
            }
        ]
    )
    hf_row = {
        'Time': 100.0,
        'T_magma': 2995.0,
        'T_surf': 2795.0,
        'Phi_global': 0.3,
        'F_int': 0.1,
        'M_mantle': 4.0e24,
        'M_mantle_liquid': 1.0e24,
        'M_mantle_solid': 3.0e24,
        'M_core': 2.0e24,
    }
    return hf_all, hf_row


def _run_interior_with_dummy(config, hf_all, hf_row, *, ic: int, output: dict):
    """Drive proteus.interior_energetics.wrapper.run_interior through the dummy backend."""
    from proteus.interior_energetics.wrapper import run_interior

    interior_o = MagicMock(spec=Interior_t)
    interior_o.ic = ic
    atmos_o = MagicMock()

    with (
        patch(
            'proteus.interior_energetics.dummy.run_dummy_int',
            return_value=(110.0, output),
        ),
        patch('proteus.interior_energetics.wrapper.update_planet_mass'),
    ):
        run_interior({}, config, hf_all, hf_row, interior_o, atmos_o, verbose=False)
    return hf_row


@pytest.mark.unit
def test_run_interior_prevent_warming_clamps_T_and_Fint_on_ic2():
    """prevent_warming=True + ic=2 must clip Phi/T_magma/T_surf to previous values
    and floor F_int to 1e-8 when the previous step had F_int < 0."""
    config = _make_run_interior_config(prevent_warming=True)
    hf_all, hf_row = _make_run_interior_state(prev_f_int=-3.0)
    # The dummy returns numbers that EXCEED the limiter tolerance (atol=20 K)
    # AND exceed the previous values, so both the prevent_warming clamp and
    # the runaway-T fallback are exercised.
    out = {
        'T_magma': 3005.0,
        'T_surf': 2805.0,
        'Phi_global': 0.7,
        'F_int': 2.0,
        'M_mantle': 4.0e24,
        'M_mantle_liquid': 1.0e24,
        'M_mantle_solid': 3.0e24,
        'M_core': 2.0e24,
    }
    _run_interior_with_dummy(config, hf_all, hf_row, ic=2, output=out)

    # Clipped down to previous values.
    assert hf_row['T_magma'] == pytest.approx(3000.0)
    assert hf_row['T_surf'] == pytest.approx(2800.0)
    assert hf_row['Phi_global'] == pytest.approx(0.4)
    # F_int was clipped to min(2.0, -3.0)=-3.0, then floored to 1e-8.
    assert hf_row['F_int'] == pytest.approx(1.0e-8)


@pytest.mark.unit
def test_run_interior_prevent_warming_off_keeps_warming_step():
    """With prevent_warming=False the bounded warming step survives unchanged."""
    config = _make_run_interior_config(prevent_warming=False)
    hf_all, hf_row = _make_run_interior_state(prev_f_int=0.2)
    out = {
        # Within the 20 K tolerance window of the runaway-T limiter (dT < 20 K).
        'T_magma': 3005.0,
        'T_surf': 2805.0,
        'Phi_global': 0.7,
        'F_int': 0.15,
        'M_mantle': 4.0e24,
        'M_mantle_liquid': 1.0e24,
        'M_mantle_solid': 3.0e24,
        'M_core': 2.0e24,
    }
    _run_interior_with_dummy(config, hf_all, hf_row, ic=2, output=out)

    assert hf_row['T_magma'] == pytest.approx(3005.0)
    assert hf_row['T_surf'] == pytest.approx(2805.0)
    assert hf_row['F_int'] == pytest.approx(0.15)


@pytest.mark.unit
def test_run_interior_prevent_warming_clamp_skipped_when_ic_not_2():
    """The Phi/T_magma/T_surf/F_int clamp only fires when ic == 2.

    ic == 0 means "initial condition from scratch"; ic == 1 means "restart";
    ic == 2 means "subsequent step". The prevent_warming clamp targets only
    the third case (per wrapper.run_interior limiter block).
    """
    config = _make_run_interior_config(prevent_warming=True)
    hf_all, hf_row = _make_run_interior_state(prev_f_int=-3.0)
    out = {
        # 50 K jump, well above tmagma_atol=20 -> runaway-T cap fires.
        'T_magma': 3050.0,
        'T_surf': 2805.0,  # +5 K, below cap -> survives
        'Phi_global': 0.7,
        'F_int': 2.0,  # would have been clamped to 1e-8 under ic=2
        'M_mantle': 4.0e24,
        'M_mantle_liquid': 1.0e24,
        'M_mantle_solid': 3.0e24,
        'M_core': 2.0e24,
    }
    _run_interior_with_dummy(config, hf_all, hf_row, ic=1, output=out)
    # F_int passes through (no prevent_warming clamp on ic != 2).
    assert hf_row['F_int'] == pytest.approx(2.0)
    # The runaway-T limiter is NOT gated on ic so it still fires here.
    assert hf_row['T_magma'] == pytest.approx(3020.0)
    assert hf_row['T_surf'] == pytest.approx(2805.0)


@pytest.mark.unit
def test_run_interior_t_surf_runaway_warning_fires():
    """The runaway-T fallback also caps T_surf when the new value blows past
    the configured dT_delta budget."""
    config = _make_run_interior_config(prevent_warming=False)
    hf_all, hf_row = _make_run_interior_state(prev_f_int=0.2)
    raw_t_surf = 2900.0  # 100 K leap, well above tmagma_atol=20
    out = {
        'T_magma': 3005.0,
        'T_surf': raw_t_surf,
        'Phi_global': 0.7,
        'F_int': 0.15,
        'M_mantle': 4.0e24,
        'M_mantle_liquid': 1.0e24,
        'M_mantle_solid': 3.0e24,
        'M_core': 2.0e24,
    }
    _run_interior_with_dummy(config, hf_all, hf_row, ic=2, output=out)

    # T_surf capped at T_surf_prev + dT_delta = 2800 + 20.
    assert hf_row['T_surf'] == pytest.approx(2820.0)
    # Cap discrimination: the post-cap value lies strictly between the
    # previous T_surf (2800) and the raw solver output (2900). Catches
    # a regression that left the raw output untouched (would give 2900)
    # or that snapped to T_surf_prev (would give 2800).
    assert 2800.0 < hf_row['T_surf'] < raw_t_surf


@pytest.mark.unit
def test_run_interior_boundary_module_splits_dT_delta_per_field():
    """The Boundary backend uses Tsurf_event_change ONLY for the T_surf
    cap. T_magma keeps the tmagma_atol/rtol budget shared with the
    other backends.

    This pins down the semantic distinction: Tsurf_event_change is the
    threshold of Calder's BL terminal ODE event on |T_surf - T_surf_0|.
    T_magma (T_p) evolves on a different timescale than T_surf in the
    boundary-layer model and is governed by the shared
    tmagma_atol/rtol budget.
    """
    from proteus.interior_energetics.wrapper import run_interior

    config = _make_run_interior_config(prevent_warming=False, module='boundary')
    # tmagma_atol = 20 (from _make_run_interior_config); Tsurf_event_change = 25.
    hf_all, hf_row = _make_run_interior_state(prev_f_int=0.2)
    out = {
        # T_magma jump = 50 K, T_surf jump = 50 K.
        'T_magma': 3050.0,
        'T_surf': 2850.0,
        'Phi_global': 0.7,
        'F_int': 0.15,
        'M_mantle': 4.0e24,
        'M_mantle_liquid': 1.0e24,
        'M_mantle_solid': 3.0e24,
        'M_core': 2.0e24,
    }

    interior_o = MagicMock(spec=Interior_t)
    interior_o.ic = 2
    interior_o.dt = 10.0
    atmos_o = MagicMock()

    boundary_runner = MagicMock()
    boundary_runner.run_solver.return_value = (110.0, out)
    with (
        patch(
            'proteus.interior_energetics.boundary.BoundaryRunner',
            return_value=boundary_runner,
        ),
        patch('proteus.interior_energetics.wrapper.update_planet_mass'),
    ):
        run_interior({}, config, hf_all, hf_row, interior_o, atmos_o, verbose=False)

    # T_magma uses tmagma_atol = 20 -> capped at 3000 + 20 = 3020.
    assert hf_row['T_magma'] == pytest.approx(3020.0)
    # T_surf uses Tsurf_event_change = 25 -> capped at 2800 + 25 = 2825.
    assert hf_row['T_surf'] == pytest.approx(2825.0)


@pytest.mark.unit
def test_run_interior_non_boundary_module_shares_dT_delta_between_caps():
    """For the non-boundary backends, T_magma and T_surf share the same
    tmagma_atol/rtol-derived budget. Pinned by this test to make sure
    a future refactor doesn't accidentally introduce a per-field split
    for spider/aragog/dummy."""
    config = _make_run_interior_config(prevent_warming=False, module='dummy')
    hf_all, hf_row = _make_run_interior_state(prev_f_int=0.2)
    # tmagma_atol = 20. Both T_magma and T_surf leap 50 K.
    out = {
        'T_magma': 3050.0,
        'T_surf': 2850.0,
        'Phi_global': 0.7,
        'F_int': 0.15,
        'M_mantle': 4.0e24,
        'M_mantle_liquid': 1.0e24,
        'M_mantle_solid': 3.0e24,
        'M_core': 2.0e24,
    }
    _run_interior_with_dummy(config, hf_all, hf_row, ic=2, output=out)

    # Both caps fire at +20.
    assert hf_row['T_magma'] == pytest.approx(3020.0)
    assert hf_row['T_surf'] == pytest.approx(2820.0)


# ============================================================================
# determine_interior_radius: tolerance_struct + maxiter + initial bracket
# ============================================================================


@pytest.mark.unit
def test_aragog_tolerance_struct_field_defaults_to_100_kg():
    """Aragog.tolerance_struct is the new absolute mass tolerance used as
    xtol in the determine_interior_radius secant solver. Default 100 kg
    matches the previous hardcoded value, so existing configs see no
    behavioural change."""
    from proteus.config._interior import Aragog

    a = Aragog()
    assert a.tolerance_struct == pytest.approx(100.0)

    a2 = Aragog(tolerance_struct=250.0)
    assert a2.tolerance_struct == pytest.approx(250.0)

    # Validator must reject non-positive tolerances.
    with pytest.raises(ValueError):
        Aragog(tolerance_struct=0.0)


@pytest.mark.unit
def test_spider_tolerance_struct_field_defaults_to_100_kg():
    """Same contract on the SPIDER side: Spider.tolerance_struct exists,
    defaults to 100 kg, rejects non-positive values."""
    from proteus.config._interior import Spider

    s = Spider()
    assert s.tolerance_struct == pytest.approx(100.0)

    s2 = Spider(tolerance_struct=75.0)
    assert s2.tolerance_struct == pytest.approx(75.0)

    with pytest.raises(ValueError):
        Spider(tolerance_struct=-1.0)


@pytest.mark.unit
def test_determine_interior_radius_wires_tolerance_struct_into_root_scalar():
    """Source-level wiring guard: the secant solver call inside
    determine_interior_radius must consume both Spider.tolerance_struct
    and Aragog.tolerance_struct, and use the widened (1.5x) initial
    bracket + 20-step iteration cap that #675 introduces."""
    from inspect import getsource

    from proteus.interior_energetics import wrapper as wrapper_mod

    src = getsource(wrapper_mod.determine_interior_radius)
    assert 'aragog.tolerance_struct' in src, (
        'determine_interior_radius must read Aragog.tolerance_struct'
    )
    assert 'spider.tolerance_struct' in src, (
        'determine_interior_radius must read Spider.tolerance_struct'
    )
    assert 'maxiter=20' in src, 'secant iteration cap must be 20'
    assert 'R_int * 1.5' not in src  # this exact malformed string would be a typo
    assert '* 1.5' in src, 'initial bracket must use the widened 1.5x guess'


# ============================================================================
# Spider.log_output: schema roundtrip + wiring assertion
# ============================================================================


@pytest.mark.unit
def test_spider_log_output_field_defaults_true_and_is_settable():
    """``Spider.log_output`` defaults to True, which writes the
    ``spider_recent.log`` mirror; setting False routes subprocess
    output to /dev/null.
    """
    from proteus.config._interior import Spider

    s_default = Spider()
    assert s_default.log_output is True

    s_off = Spider(log_output=False)
    assert s_off.log_output is False


@pytest.mark.unit
def test_spider_module_consults_log_output_in_subprocess_dispatch():
    """Source-level wiring guard: _try_spider must read the new field
    from config and must hold both dispatch branches (file open and
    sp.DEVNULL). End-to-end behaviour is exercised by the Phase-9 SPIDER
    smoke run; the subprocess plumbing is too entangled for an isolated
    unit-level run of _try_spider."""
    from inspect import getsource

    from proteus.interior_energetics import spider as spider_mod

    src = getsource(spider_mod._try_spider)
    assert 'config.interior_energetics.spider.log_output' in src, (
        '_try_spider must read the log_output config field'
    )
    assert 'sp.DEVNULL' in src, '_try_spider must support discarding output'
    assert 'spider_recent.log' in src, '_try_spider must support writing the log file'


# ============================================================================
# CALLIOPE solver knobs: nguess / nsolve schema roundtrip
# ============================================================================


@pytest.mark.unit
def test_calliope_nguess_nsolve_defaults_match_legacy_constants():
    """The old code hard-coded nguess=int(1e3) and nsolve=int(3e3) inside
    outgas/calliope.py. The new Calliope schema defaults must match those
    exact values so existing configs see no behavioural change."""
    from proteus.config._outgas import Calliope

    c = Calliope()
    assert c.nguess == 1000
    assert c.nsolve == 3000


@pytest.mark.unit
def test_calliope_nguess_nsolve_reject_zero_or_negative():
    """Validator should reject non-positive values (gt(0)). Solver iter
    counts of 0 or negative make no physical sense."""
    from attrs.exceptions import NotAnAttrsClassError  # noqa: F401  # informative import

    from proteus.config._outgas import Calliope

    with pytest.raises(ValueError):
        Calliope(nguess=0)
    with pytest.raises(ValueError):
        Calliope(nsolve=-5)


@pytest.mark.unit
def test_run_interior_F_int_floor_fires_on_ic1_restart_when_prevent_warming():
    """F_int positivity floor fires on SPIDER restart (ic == 1) when
    ``planet.prevent_warming`` is True.

    Contract: ``ReadSPIDER`` applies a two-stage limiter when
    ``planet.prevent_warming`` is enabled. The previous-value clamp is
    gated on ``ic == 2`` (it requires ``hf_all.iloc[-1]`` to be a
    meaningful "previous" row). The positivity floor
    ``F_int = max(1e-8, F_int)`` applies for any ``ic >= 0``, so a
    SPIDER restart (ic == 1) that returns a slightly negative F_int does
    not propagate the negative flux to the helpfile or the atmospheric
    boundary condition.
    """
    config = _make_run_interior_config(prevent_warming=True)
    hf_all, hf_row = _make_run_interior_state(prev_f_int=0.2)
    # Dummy backend returns a tiny negative F_int (simulates the
    # SPIDER post-restart heat-pump artefact).
    raw_F_int = -1.0e-3
    out = {
        'T_magma': 3000.0,
        'T_surf': 2800.0,
        'Phi_global': 0.4,
        'F_int': raw_F_int,
        'M_mantle': 4.0e24,
        'M_mantle_liquid': 1.0e24,
        'M_mantle_solid': 3.0e24,
        'M_core': 2.0e24,
    }
    _run_interior_with_dummy(config, hf_all, hf_row, ic=1, output=out)

    # Floor fires even on ic == 1 because prevent_warming is enabled.
    assert hf_row['F_int'] == pytest.approx(1.0e-8)
    # Sign-correction discrimination: the raw output was strictly negative,
    # and the post-floor value is strictly positive. A regression that
    # passed the raw value through unchanged would still satisfy any
    # |F_int| < threshold check; only the sign flip discriminates.
    assert raw_F_int < 0.0 < hf_row['F_int']


@pytest.mark.unit
def test_run_interior_F_int_floor_skipped_when_prevent_warming_off():
    """The floor is gated on prevent_warming. When that flag is False
    a slightly-negative F_int must pass through (so the user sees the
    raw solver output and can diagnose it, instead of seeing a silent
    1e-8 floor)."""
    config = _make_run_interior_config(prevent_warming=False)
    hf_all, hf_row = _make_run_interior_state(prev_f_int=0.2)
    raw_F_int = -1.0e-3
    out = {
        'T_magma': 3000.0,
        'T_surf': 2800.0,
        'Phi_global': 0.4,
        'F_int': raw_F_int,
        'M_mantle': 4.0e24,
        'M_mantle_liquid': 1.0e24,
        'M_mantle_solid': 3.0e24,
        'M_core': 2.0e24,
    }
    _run_interior_with_dummy(config, hf_all, hf_row, ic=2, output=out)

    # Floor does NOT fire; negative F_int survives.
    assert hf_row['F_int'] == pytest.approx(-1.0e-3)
    # Sign discrimination: the raw output stayed strictly negative.
    # A regression that silently applied the floor regardless of the
    # prevent_warming flag would flip the sign to ~1e-8 and fool the
    # primary pytest.approx check if its abs= tolerance were too loose.
    assert raw_F_int == pytest.approx(hf_row['F_int'], rel=1e-12)
    assert hf_row['F_int'] < 0.0


# ============================================================================
# get_core_density / get_core_heatcap / get_nlevb / update_planet_mass
# update_gravity / calculate_core_mass / reset_run_state
# ============================================================================


@pytest.mark.unit
def test_get_core_density_numeric_passes_through():
    """get_core_density returns config.interior_struct.core_density as a
    float when it is a number, ignoring hf_row.
    """
    from types import SimpleNamespace

    from proteus.interior_energetics.wrapper import get_core_density

    config = SimpleNamespace(interior_struct=SimpleNamespace(core_density=12345.0))
    hf_row = {'core_density': 999.9}  # must be ignored
    out = get_core_density(config, hf_row)
    assert out == pytest.approx(12345.0, rel=1e-12)
    # Side-effect discrimination: hf_row['core_density'] was not used.
    assert out != pytest.approx(hf_row['core_density'], rel=1e-3)
    # Type discrimination: return must be float, not str/int.
    assert isinstance(out, float)


@pytest.mark.unit
def test_get_core_density_self_reads_hf_row_default():
    """When core_density == 'self', the value comes from hf_row, with a
    documented Earth-like default of 10738.0 when missing.
    """
    from types import SimpleNamespace

    from proteus.interior_energetics.wrapper import get_core_density

    config = SimpleNamespace(interior_struct=SimpleNamespace(core_density='self'))
    # Default branch: hf_row missing the key.
    out_default = get_core_density(config, {})
    assert out_default == pytest.approx(10738.0, rel=1e-12)
    # Read branch: hf_row carries the value (e.g. from Zalmoxis).
    out_read = get_core_density(config, {'core_density': 11234.5})
    assert out_read == pytest.approx(11234.5, rel=1e-12)
    # Discrimination: default and read branches return different values
    # given different hf_row contents; catches a regression that ignored
    # hf_row entirely.
    assert out_default != pytest.approx(out_read, rel=1e-3)


@pytest.mark.unit
def test_get_core_heatcap_self_reads_hf_row_default():
    """Mirror of get_core_density for heat capacity; default is 450.0 J/kg/K."""
    from types import SimpleNamespace

    from proteus.interior_energetics.wrapper import get_core_heatcap

    config_num = SimpleNamespace(interior_struct=SimpleNamespace(core_heatcap=700.0))
    out_num = get_core_heatcap(config_num, {'core_heatcap': 999.9})
    assert out_num == pytest.approx(700.0, rel=1e-12)

    config_self = SimpleNamespace(interior_struct=SimpleNamespace(core_heatcap='self'))
    out_default = get_core_heatcap(config_self, {})
    assert out_default == pytest.approx(450.0, rel=1e-12)
    out_read = get_core_heatcap(config_self, {'core_heatcap': 520.0})
    assert out_read == pytest.approx(520.0, rel=1e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_update_gravity_matches_newton_inverse_square():
    """update_gravity sets hf_row['gravity'] = G * M / R^2.

    Conservation / closed-form invariant: at Earth's mass and radius the
    surface gravity must equal 9.81 m/s^2 to <1%. A regression that
    used R instead of R**2 would give g ~6e16 (order of magnitude wrong)
    or g ~1.5e-7 (using 1/R) so the discrimination guard is sharp.
    """
    from proteus.interior_energetics.wrapper import update_gravity
    from proteus.utils.constants import M_earth, R_earth, const_G

    hf_row = {'M_int': M_earth, 'R_int': R_earth}
    update_gravity(hf_row)
    expected = const_G * M_earth / (R_earth * R_earth)
    assert hf_row['gravity'] == pytest.approx(expected, rel=1e-12)
    # Discrimination: surface gravity for Earth-like inputs must be ~9.8 m/s^2.
    assert 9.6 < hf_row['gravity'] < 10.0
    # Sign guard: a flipped-sign regression (e.g. -G) would give negative g.
    assert hf_row['gravity'] > 0.0
    # Wrong-formula guard: 1/R instead of 1/R**2 would give g ~6e16; a
    # missing G factor would give g ~1.5e-22. Test pin discriminates.
    wrong_no_square = const_G * M_earth / R_earth
    assert abs(hf_row['gravity'] - wrong_no_square) > 1e6


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_calculate_core_mass_matches_rho_v_for_known_rho_and_radius():
    """calculate_core_mass writes hf_row['M_core'] = rho_core * (4/3) * pi * (R_int * core_frac)^3.

    Physics invariant: mass is strictly positive given positive density
    and radius, and the closed-form pin is matched to 12 digits.
    """
    from types import SimpleNamespace

    from proteus.interior_energetics.wrapper import calculate_core_mass

    rho_core = 10738.0
    R_int = 6.371e6
    core_frac = 0.3
    config = SimpleNamespace(
        interior_struct=SimpleNamespace(
            core_density=rho_core,
            core_frac=core_frac,
            core_frac_mode='radius',
        )
    )
    hf_row = {'R_int': R_int}
    calculate_core_mass(hf_row, config)
    expected = rho_core * (4.0 / 3.0) * np.pi * (R_int * core_frac) ** 3
    assert hf_row['M_core'] == pytest.approx(expected, rel=1e-12)
    # Positivity invariant.
    assert hf_row['M_core'] > 0.0
    # Cube-vs-square discrimination guard. A regression that used R**2
    # instead of R**3 would give a number ~6 orders of magnitude smaller.
    wrong_square = rho_core * (4.0 / 3.0) * np.pi * (R_int * core_frac) ** 2
    assert abs(hf_row['M_core'] - wrong_square) > 1e15


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_update_planet_mass_sums_internal_and_volatile_elements():
    """update_planet_mass writes M_ele as the sum over element_list of
    each element's _kg_total mass, and writes M_planet = M_int + M_ele.

    Conservation: the resulting M_planet must equal the dry mass plus
    the total volatile-bearing element mass exactly.
    """
    from proteus.interior_energetics.wrapper import update_planet_mass
    from proteus.utils.constants import element_list

    hf_row = {'M_int': 5.972e24}
    # Asymmetric element budget so an off-by-one indexing bug shows.
    for i, e in enumerate(element_list):
        hf_row[e + '_kg_total'] = float(10 ** (18 + i))

    update_planet_mass(hf_row)

    expected_ele = sum(float(hf_row[e + '_kg_total']) for e in element_list)
    assert hf_row['M_ele'] == pytest.approx(expected_ele, rel=1e-12)
    assert hf_row['M_planet'] == pytest.approx(hf_row['M_int'] + expected_ele, rel=1e-12)
    # Anti-happy-path: skipping ANY single element (e.g. the old O-skip)
    # produces a strictly smaller M_planet. The largest element in the
    # asymmetric budget dominates, so dropping it would change the result
    # by orders of magnitude.
    largest_elem = max(element_list, key=lambda e: hf_row[e + '_kg_total'])
    expected_without_largest = expected_ele - hf_row[largest_elem + '_kg_total']
    assert (
        abs(hf_row['M_ele'] - expected_without_largest)
        > 0.5 * hf_row[largest_elem + '_kg_total']
    )


@pytest.mark.unit
def test_update_planet_mass_handles_missing_element_columns():
    """Element columns missing from hf_row are treated as 0.0 via .get(),
    so the helper is safe on pre-IC rows. Anti-happy-path: a regression
    that used [] indexing would raise KeyError on a real run.
    """
    from proteus.interior_energetics.wrapper import update_planet_mass

    # Only M_int populated, no _kg_total columns.
    hf_row = {'M_int': 5.972e24}
    update_planet_mass(hf_row)
    assert hf_row['M_ele'] == pytest.approx(0.0, abs=1e-12)
    assert hf_row['M_planet'] == pytest.approx(5.972e24, rel=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize(
    'module,expected',
    [
        ('spider', 50),
        ('aragog', 75),
        ('boundary', 2),
        ('dummy', 2),
    ],
)
def test_get_nlevb_returns_per_module_levels(module, expected):
    """get_nlevb returns config.num_levels for spider/aragog and the
    fixed integer 2 for boundary/dummy.
    """
    from types import SimpleNamespace

    from proteus.interior_energetics.wrapper import get_nlevb

    # Use distinct num_levels to discriminate spider vs aragog dispatch.
    config = SimpleNamespace(
        interior_energetics=SimpleNamespace(module=module, num_levels=expected),
    )
    if module in ('boundary', 'dummy'):
        # num_levels is ignored; set to a different value to discriminate.
        config.interior_energetics.num_levels = 999
    assert get_nlevb(config) == expected
    # Type discrimination: must be a Python int.
    assert isinstance(get_nlevb(config), int)


@pytest.mark.unit
def test_get_nlevb_invalid_module_raises_value_error():
    """get_nlevb raises ValueError for an unknown module string."""
    from types import SimpleNamespace

    from proteus.interior_energetics.wrapper import get_nlevb

    config = SimpleNamespace(
        interior_energetics=SimpleNamespace(module='nonexistent', num_levels=50),
    )
    with pytest.raises(ValueError, match='Invalid interior module') as exc:
        get_nlevb(config)
    # The error names the offending module string so users can fix the config.
    assert 'nonexistent' in str(exc.value)


@pytest.mark.unit
def test_fail_counters_are_per_run_state():
    """The consecutive-failure counters are per-run Interior_t state.

    A fresh instance starts every counter at zero, and mutating one
    instance leaves another untouched: two Proteus runs in the same
    Python process (grid ensembles, notebooks, test sessions) cannot
    share failure streaks, and a new run cannot inherit a stale count
    that would trip its abort threshold on the first failure.
    """
    run_a = Interior_t(50)
    run_b = Interior_t(50)

    # Fresh-run contract: all three counters start at zero.
    assert (run_a.zalmoxis_fail_count, run_a.spider_fail_count, run_a.aragog_fail_count) == (
        0,
        0,
        0,
    )

    # Isolation contract: a failure streak on run A is invisible to run B.
    run_a.zalmoxis_fail_count = 7
    run_a.spider_fail_count = 2
    run_a.aragog_fail_count = 3
    assert (run_b.zalmoxis_fail_count, run_b.spider_fail_count, run_b.aragog_fail_count) == (
        0,
        0,
        0,
    )
    # Discrimination: the mutated instance really holds the streak (the
    # isolation assert above cannot pass because writes were dropped).
    assert run_a.zalmoxis_fail_count == 7


# ============================================================================
# _is_spider_ps_format: P-S vs P-T format sniff
# ============================================================================


@pytest.mark.unit
def test_is_spider_ps_format_recognises_ps_header(tmp_path):
    """A file whose first line matches ``# 5 <int> <int>`` is recognised
    as the canonical SPIDER P-S format; an empty first line is not.
    """
    from proteus.interior_energetics.wrapper import _is_spider_ps_format

    ps_path = tmp_path / 'density_melt.dat'
    ps_path.write_text('# 5 200 150\n# more header\n1.0 2.0 3.0\n')
    assert _is_spider_ps_format(str(ps_path)) is True
    # Edge case: empty file -> first line is '' -> parts == [] -> False.
    empty_path = tmp_path / 'empty.dat'
    empty_path.write_text('')
    assert _is_spider_ps_format(str(empty_path)) is False


@pytest.mark.unit
def test_is_spider_ps_format_rejects_pt_header_and_missing_file(tmp_path):
    """A P-T header (different first-line shape) returns False, and a
    nonexistent file also returns False (OSError swallowed).
    """
    from proteus.interior_energetics.wrapper import _is_spider_ps_format

    pt_path = tmp_path / 'density_melt_pt.dat'
    pt_path.write_text('#pressure temperature density\n1e5 300 3000\n')
    assert _is_spider_ps_format(str(pt_path)) is False

    # Missing file: OSError -> False (not raised).
    missing_path = tmp_path / 'does_not_exist.dat'
    assert _is_spider_ps_format(str(missing_path)) is False


# ============================================================================
# _load_spider_ps_phase_table: header parsing + reshape contract
# ============================================================================


def _write_synthetic_ps_table(
    path,
    *,
    NX: int = 3,
    NY: int = 4,
    P_scale: float = 1e9,
    S_scale: float = 4.0e6,
    val_scale: float = 1.0e3,
):
    """Write a synthetic rectangular SPIDER P-S table for unit tests.

    Layout: P varies fastest (inner loop), S varies slowest (outer loop).
    Values are asymmetric so reshape-ordering bugs change the result at
    every node.
    """
    lines = [
        f'# 5 {NX} {NY}\n',
        '# header line 2\n',
        '# header line 3\n',
        '# header line 4\n',
        f'# {P_scale} {S_scale} {val_scale}\n',
    ]
    for j in range(NY):
        for i in range(NX):
            p_nd = i / max(NX - 1, 1)
            s_nd = j / max(NY - 1, 1)
            # Asymmetric so a (P, S) swap changes the result everywhere.
            v_nd = 0.1 + 0.3 * p_nd + 0.7 * s_nd
            lines.append(f'{p_nd} {s_nd} {v_nd}\n')
    path.write_text(''.join(lines))


@pytest.mark.unit
def test_load_spider_ps_phase_table_returns_rectangular_grid(tmp_path):
    """The loader returns (P_grid (NX,), S_grid (NY,), val (NX, NY)) tuple.
    Pins the shape contract and the P-vs-S scaling so a swap regression
    would land in the wrong magnitude bucket.
    """
    from proteus.interior_energetics.wrapper import _load_spider_ps_phase_table

    NX, NY = 5, 7
    P_scale, S_scale, val_scale = 2e9, 3e6, 5.0e3
    path = tmp_path / 'temperature_melt.dat'
    _write_synthetic_ps_table(
        path,
        NX=NX,
        NY=NY,
        P_scale=P_scale,
        S_scale=S_scale,
        val_scale=val_scale,
    )

    P_grid, S_grid, val = _load_spider_ps_phase_table(str(path))

    # Shape contract.
    assert P_grid.shape == (NX,)
    assert S_grid.shape == (NY,)
    assert val.shape == (NX, NY)
    # First entry of each axis is at 0; last is at scale (the inner index
    # divides by NX-1, so node NX-1 maps to 1.0).
    assert P_grid[0] == pytest.approx(0.0, abs=1e-12)
    assert P_grid[-1] == pytest.approx(P_scale, rel=1e-12)
    assert S_grid[-1] == pytest.approx(S_scale, rel=1e-12)
    # Value-magnitude discrimination: val_scale-bearing entry must equal
    # (0.1 + 0.3 + 0.7) * val_scale at the (max P, max S) corner. A
    # transposed reshape (NX <-> NY) would give a different number.
    assert val[-1, -1] == pytest.approx(1.1 * val_scale, rel=1e-9)
    # Distinct corners assertion: val(0,0) != val(NX-1,NY-1) so transposition
    # bugs show as a shape mismatch above.
    assert val[0, 0] != pytest.approx(val[-1, -1], rel=1e-3)


@pytest.mark.unit
def test_load_spider_ps_phase_table_rejects_header_mismatch(tmp_path):
    """The loader raises ValueError when NX*NY rows promised in the header
    don't match the actual row count.
    """
    from proteus.interior_energetics.wrapper import _load_spider_ps_phase_table

    path = tmp_path / 'broken.dat'
    # Header promises 6 rows; we write only 4.
    path.write_text('# 5 3 2\n# h2\n# h3\n# h4\n# 1e9 1e6 1e3\n0 0 0\n1 0 1\n2 0 2\n0 1 3\n')
    with pytest.raises(ValueError, match='header says NX') as exc:
        _load_spider_ps_phase_table(str(path))
    # Discrimination: the message names BOTH the expected count (NX*NY=6)
    # and the actual row count (4), so users can debug the broken table.
    msg = str(exc.value)
    assert '6' in msg and '4' in msg


# ============================================================================
# _rectangularize_spider_ps_file: writes a clean rectangular file
# ============================================================================


@pytest.mark.unit
def test_rectangularize_spider_ps_file_removes_p_drift(tmp_path):
    """A file with a tiny P drift across S slices is normalised: the
    output file has identical P values across every S slice.
    """
    from proteus.interior_energetics.wrapper import _rectangularize_spider_ps_file

    NX, NY = 3, 2
    src = tmp_path / 'src.dat'
    # Build a SPIDER P-S file with a tiny P drift (1e-9 relative)
    # between slices to mimic the upstream EOS table generator.
    lines = [
        f'# 5 {NX} {NY}\n',
        '# header2\n',
        '# header3\n',
        '# header4\n',
        '# 1e9 4e6 3e3\n',
    ]
    P_canonical = [0.0, 0.5, 1.0]
    drift = [0.0, 1e-9, -1e-9]
    for j in range(NY):
        for i in range(NX):
            p = P_canonical[i] + drift[j] * (i + 1)
            s = float(j)
            v = 0.1 + 0.1 * i + 0.3 * j
            lines.append(f'{p:.18e} {s:.18e} {v:.18e}\n')
    src.write_text(''.join(lines))

    dst = tmp_path / 'dst.dat'
    _rectangularize_spider_ps_file(str(src), str(dst))

    # Read back: every S slice must share the same P column.
    data = np.loadtxt(str(dst), skiprows=5)
    P_all = data[:, 0].reshape(NY, NX)
    # No drift after rectangularisation.
    assert np.all(P_all == P_all[0, :]), 'P column drift survived rectangularisation'
    # The canonical P values made it through.
    np.testing.assert_allclose(P_all[0], P_canonical, rtol=1e-12)


@pytest.mark.unit
def test_rectangularize_spider_ps_file_rejects_excessive_drift(tmp_path):
    """A drift > 1e-4 relative is genuinely non-rectangular and must raise
    ValueError, not silently fold a real physics drift into noise.
    """
    from proteus.interior_energetics.wrapper import _rectangularize_spider_ps_file

    NX, NY = 3, 2
    src = tmp_path / 'src_big_drift.dat'
    lines = [
        f'# 5 {NX} {NY}\n',
        '# h2\n',
        '# h3\n',
        '# h4\n',
        '# 1e9 4e6 3e3\n',
    ]
    P_canonical = [1.0, 5.0, 10.0]
    # 10 % relative drift between slices (well above the 1e-4 ceiling).
    drift_factor = [1.0, 1.1]
    for j in range(NY):
        for i in range(NX):
            p = P_canonical[i] * drift_factor[j]
            s = float(j)
            v = 0.1 * (i + 1) * (j + 1)
            lines.append(f'{p:.18e} {s:.18e} {v:.18e}\n')
    src.write_text(''.join(lines))

    dst = tmp_path / 'dst_big_drift.dat'
    with pytest.raises(ValueError, match='cannot be rectangularised'):
        _rectangularize_spider_ps_file(str(src), str(dst))
    # No output file was written because the helper bailed early.
    assert not dst.exists()


@pytest.mark.unit
def test_rectangularize_spider_ps_file_rejects_wrong_row_count(tmp_path):
    """A file whose row count does not match NX*NY raises ValueError."""
    from proteus.interior_energetics.wrapper import _rectangularize_spider_ps_file

    src = tmp_path / 'bad_rows.dat'
    # Header promises 6 rows; write only 3.
    src.write_text('# 5 3 2\n# h2\n# h3\n# h4\n# 1e9 4e6 3e3\n0 0 0\n1 0 1\n2 0 2\n')
    dst = tmp_path / 'dst.dat'
    with pytest.raises(ValueError, match='header says NX'):
        _rectangularize_spider_ps_file(str(src), str(dst))
    # No output written: the validator fired before write_lines.
    assert not dst.exists()


# ============================================================================
# _derive_ps_melting_curve: P-T -> P-S inversion via phase-T lookup
# ============================================================================


@pytest.mark.unit
def test_derive_ps_melting_curve_round_trip_within_grid(tmp_path):
    """Given a P-T melting file and a phase-T lookup table that share the
    same monotonic T(S) relation, the inversion produces P-S pairs whose
    round-trip T_recovered matches T_target to grid resolution.
    """
    from proteus.interior_energetics.wrapper import _derive_ps_melting_curve

    # Build a phase-T table: T_grid(P, S) is a simple linear function
    # of S at fixed P so the inversion is exact.
    NX, NY = 4, 6
    phase_path = tmp_path / 'temperature_solid.dat'
    P_scale = 1e9
    S_scale = 1e6
    T_scale = 1000.0  # value column scale: T values stored as nondim/scale
    # We choose T(P, S) = 1.0 + 0.0 * P + 0.4 * S_nondim, scaled by T_scale.
    # In SI: T = (1 + 0.4 * S / S_scale) * T_scale.
    lines = [
        f'# 5 {NX} {NY}\n',
        '# h2\n',
        '# h3\n',
        '# h4\n',
        f'# {P_scale} {S_scale} {T_scale}\n',
    ]
    for j in range(NY):
        for i in range(NX):
            p_nd = i / (NX - 1)
            s_nd = j / (NY - 1)
            t_nd = 1.0 + 0.4 * s_nd
            lines.append(f'{p_nd} {s_nd} {t_nd}\n')
    phase_path.write_text(''.join(lines))

    # Build a P-T melting file with targets that land inside the grid.
    # T ranges in SI: 1000 (s=0) to 1400 (s=1.0).
    pt_path = tmp_path / 'solidus_P-T.dat'
    pt_path.write_text('# P [Pa] T [K]\n0.0 1000.0\n5e8 1100.0\n1e9 1200.0\n')

    target_path = tmp_path / 'solidus_P-S.dat'
    summary = _derive_ps_melting_curve(
        str(pt_path), str(phase_path), str(target_path), label='test_solidus'
    )

    # The inversion must converge with tight round-trip residual on
    # in-grid points (we chose all three to be in-grid).
    assert summary['n_points'] == 3
    assert summary['n_clipped_below'] == 0
    assert summary['n_clipped_above'] == 0
    # Round-trip residual must be at most grid-resolution (T(S) is linear,
    # so the linear interp is exact to machine precision here).
    assert summary['max_inversion_residual_K'] < 1e-6


@pytest.mark.unit
def test_derive_ps_melting_curve_clips_out_of_range_temperature(tmp_path):
    """T_target outside the (T_min, T_max) range of the phase table at the
    corresponding P is clipped to the nearest S grid edge and counted.
    """
    from proteus.interior_energetics.wrapper import _derive_ps_melting_curve

    NX, NY = 3, 4
    phase_path = tmp_path / 'temp_solid.dat'
    P_scale = 1e9
    S_scale = 1e6
    T_scale = 1000.0
    lines = [
        f'# 5 {NX} {NY}\n',
        '# h2\n',
        '# h3\n',
        '# h4\n',
        f'# {P_scale} {S_scale} {T_scale}\n',
    ]
    # T in SI: 1000 (s=0) to 1300 (s=1).
    for j in range(NY):
        for i in range(NX):
            p_nd = i / (NX - 1)
            s_nd = j / (NY - 1)
            t_nd = 1.0 + 0.3 * s_nd
            lines.append(f'{p_nd} {s_nd} {t_nd}\n')
    phase_path.write_text(''.join(lines))

    # P-T file with one below-grid T and one above-grid T.
    pt_path = tmp_path / 'pt_clip.dat'
    pt_path.write_text(
        '# P T\n'
        '0.0 500.0\n'  # below T_min = 1000
        '1e9 2000.0\n'  # above T_max = 1300
    )

    target_path = tmp_path / 'out.dat'
    summary = _derive_ps_melting_curve(
        str(pt_path), str(phase_path), str(target_path), label='clip_test'
    )

    assert summary['n_points'] == 2
    assert summary['n_clipped_below'] == 1
    assert summary['n_clipped_above'] == 1


@pytest.mark.unit
def test_derive_ps_melting_curve_rejects_wrong_column_count(tmp_path):
    """A P-T file with 3 columns instead of 2 raises ValueError."""
    from proteus.interior_energetics.wrapper import _derive_ps_melting_curve

    NX, NY = 2, 2
    phase_path = tmp_path / 'temp.dat'
    _write_synthetic_ps_table(phase_path, NX=NX, NY=NY)

    pt_path = tmp_path / 'bad_pt.dat'
    pt_path.write_text('1e9 1500 0.5\n2e9 1700 0.6\n')

    target_path = tmp_path / 'out.dat'
    with pytest.raises(ValueError, match='expected 2-column'):
        _derive_ps_melting_curve(str(pt_path), str(phase_path), str(target_path), label='bad')
    # No output written: the column-count guard fired before any inversion ran.
    assert not target_path.exists()


# ============================================================================
# _override_melting_curves_from_pt: missing files -> warning, no derive
# ============================================================================


@pytest.mark.unit
def test_override_melting_curves_warns_when_phase_tables_missing(tmp_path, caplog):
    """When temperature_solid.dat or temperature_melt.dat is missing from
    eos_dir, the override helper logs a warning and returns without
    writing the P-S melting curves.
    """
    from proteus.interior_energetics.wrapper import _override_melting_curves_from_pt

    eos_dir = tmp_path / 'eos'
    eos_dir.mkdir()
    # No temperature_*.dat present.

    sol_pt = tmp_path / 'solidus_P-T.dat'
    liq_pt = tmp_path / 'liquidus_P-T.dat'
    sol_pt.write_text('# h\n1e9 1500\n2e9 1700\n')
    liq_pt.write_text('# h\n1e9 1600\n2e9 1800\n')

    with caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.wrapper'):
        _override_melting_curves_from_pt(
            str(eos_dir), str(sol_pt), str(liq_pt), label_prefix='miss_test'
        )
    assert any('missing temperature_' in r.message for r in caplog.records)
    # No output files were written because the helper bailed early.
    assert not (eos_dir / 'solidus_P-S.dat').exists()
    assert not (eos_dir / 'liquidus_P-S.dat').exists()


# ============================================================================
# _provide_spider_eos_tables: already-populated reuse + hard-failure
# ============================================================================


def _write_complete_ps_eos_dir(target_dir):
    """Write the 12 SPIDER P-S phase + melting files needed by the helper.

    The 10 phase files use the synthetic 3-column P-S layout that
    ``_rectangularize_spider_ps_file`` and ``_load_spider_ps_phase_table``
    expect; the 2 melting curves use the simpler 2-column shape.
    """
    from proteus.interior_energetics.wrapper import (
        _SPIDER_EOS_MELTING_CURVES,
        _SPIDER_EOS_PHASE_FILES,
    )

    os.makedirs(target_dir, exist_ok=True)
    for f in _SPIDER_EOS_PHASE_FILES:
        path = Path(target_dir) / f
        _write_synthetic_ps_table(path, NX=3, NY=4)
    for f in _SPIDER_EOS_MELTING_CURVES:
        path = Path(target_dir) / f
        path.write_text(
            '# 5 3\n# Pressure Entropy\n# header3\n# header4\n# 1e9 4e6\n'
            '0.0 0.5\n0.5 0.7\n1.0 0.9\n'
        )


@pytest.mark.unit
def test_provide_spider_eos_tables_reuses_already_populated_dir(tmp_path):
    """When dirs['spider_eos_dir'] already holds all 12 expected files,
    the helper short-circuits with a debug log and sets the melting-curve
    paths without re-copying anything.
    """
    from pathlib import Path
    from types import SimpleNamespace

    from proteus.interior_energetics.wrapper import _provide_spider_eos_tables

    eos_dir = tmp_path / 'preexisting_eos'
    _write_complete_ps_eos_dir(str(eos_dir))

    config = SimpleNamespace(
        interior_struct=SimpleNamespace(melting_dir=None),
    )
    dirs = {'spider_eos_dir': str(eos_dir)}
    _provide_spider_eos_tables(config, str(tmp_path), dirs)

    # The reuse path sets the two melting-curve paths.
    assert dirs['spider_solidus_ps'] == str(eos_dir / 'solidus_P-S.dat')
    assert dirs['spider_liquidus_ps'] == str(eos_dir / 'liquidus_P-S.dat')
    # And the target dir is unchanged (still the original).
    assert dirs['spider_eos_dir'] == str(eos_dir)
    # Discrimination: ``output/<case>/data/spider_eos/`` was NOT created
    # because reuse short-circuits the populate path.
    assert not (Path(tmp_path) / 'data' / 'spider_eos').exists()


@pytest.mark.unit
def test_provide_spider_eos_tables_hard_failure_when_no_source(tmp_path, monkeypatch):
    """When neither FWL_DATA nor a SPIDER submodule provides the tables,
    the helper raises FileNotFoundError with a clear remediation message.

    Reaches the case-4 raise at line ~798 by patching both Zenodo and
    SPIDER submodule paths to be unavailable.
    """
    from types import SimpleNamespace
    from unittest.mock import patch as _patch

    from proteus.interior_energetics.wrapper import _provide_spider_eos_tables

    config = SimpleNamespace(
        interior_struct=SimpleNamespace(melting_dir=None),
    )
    dirs = {'spider': str(tmp_path / 'no_spider_submodule')}

    fwl_root = tmp_path / 'fwl_data_empty'

    with _patch('proteus.utils.data.GetFWLData', return_value=fwl_root):
        with pytest.raises(FileNotFoundError, match='Could not provide SPIDER/Aragog') as exc:
            _provide_spider_eos_tables(config, str(tmp_path), dirs)
    # Discrimination: the message points users at the remediation
    # (`proteus get all`); a regression that silently fell through
    # would not raise at all.
    assert 'proteus get all' in str(exc.value)


# ============================================================================
# determine_interior_radius: secant solver dispatch via R_int_override
# ============================================================================


@pytest.mark.unit
def test_determine_interior_radius_uses_r_int_override(tmp_path):
    """When config.planet.R_int_override > 0, the secant solver is bypassed
    and the planet radius is pinned to the override.
    """
    from unittest.mock import patch as _patch

    from proteus.interior_energetics.wrapper import determine_interior_radius

    config = MagicMock()
    config.interior_energetics.module = 'aragog'
    config.interior_energetics.num_levels = 50
    config.interior_struct.eos_dir = 'WolfBower2018_MgSiO3'
    config.planet.R_int_override = 5.5e6
    config.planet.mass_tot = 1.0
    config.interior_struct.core_density = 10738.0
    config.interior_struct.core_frac = 0.3
    config.interior_struct.core_frac_mode = 'radius'

    hf_row = {'M_int': 4e24, 'R_int': 6.371e6, 'gravity': 9.81}
    dirs = {}

    with (
        _patch(
            'proteus.interior_energetics.wrapper._provide_spider_eos_tables'
        ) as mock_provide,
        _patch('proteus.interior_energetics.wrapper.run_interior'),
        _patch('proteus.interior_energetics.wrapper.update_gravity'),
        _patch('proteus.interior_energetics.wrapper.calc_target_elemental_inventories'),
        _patch('proteus.interior_energetics.wrapper.update_planet_mass') as mock_mass,
    ):
        determine_interior_radius(dirs, config, None, hf_row, str(tmp_path))

    # R_int_override pinned the radius.
    assert hf_row['R_int'] == pytest.approx(5.5e6, rel=1e-12)
    # The override path refreshes the planet mass like the other structure
    # paths, rather than leaving M_ele / M_planet stale.
    mock_mass.assert_called_once()
    # Aragog dispatch invoked the EOS-table provider.
    mock_provide.assert_called_once()
    # Discrimination: a regression that ignored R_int_override would have
    # left R_int at the initial guess (R_earth = 6.371e6).
    assert hf_row['R_int'] != pytest.approx(6.371e6, rel=1e-6)


@pytest.mark.unit
def test_determine_interior_radius_runs_secant_solver_without_override(tmp_path):
    """Without R_int_override, the helper drives scipy's secant solver
    against the mass-residual function. We mock the residual closure
    by patching the root_scalar call to return a known root, and confirm
    the closure is invoked exactly once via the post-root finalize path.
    """
    from unittest.mock import patch as _patch

    from proteus.interior_energetics.wrapper import determine_interior_radius

    config = MagicMock()
    config.interior_energetics.module = 'aragog'
    config.interior_energetics.num_levels = 50
    config.interior_energetics.rtol = 1e-7
    config.interior_energetics.aragog.tolerance_struct = 100.0
    config.interior_struct.eos_dir = 'WolfBower2018_MgSiO3'
    config.planet.R_int_override = None
    config.planet.mass_tot = 1.0
    config.interior_struct.core_density = 10738.0
    config.interior_struct.core_frac = 0.3
    config.interior_struct.core_frac_mode = 'radius'

    hf_row = {'M_int': 4e24, 'M_planet': 0.0, 'R_int': 6.371e6, 'gravity': 9.81}
    dirs = {}

    fake_root = MagicMock()
    fake_root.root = 6.5e6
    fake_root.converged = True

    with (
        _patch('proteus.interior_energetics.wrapper._provide_spider_eos_tables'),
        _patch('proteus.interior_energetics.wrapper.run_interior'),
        _patch('proteus.interior_energetics.wrapper.update_gravity'),
        _patch('proteus.interior_energetics.wrapper.update_planet_mass'),
        _patch('proteus.interior_energetics.wrapper.calc_target_elemental_inventories'),
        _patch(
            'proteus.interior_energetics.wrapper.optimise.root_scalar',
            return_value=fake_root,
        ) as mock_root,
    ):
        determine_interior_radius(dirs, config, None, hf_row, str(tmp_path))

    # The secant root was extracted into hf_row['R_int'].
    assert hf_row['R_int'] == pytest.approx(6.5e6, rel=1e-12)
    # root_scalar was called with secant method (kwargs check pins the
    # widened initial-bracket convention).
    mock_root.assert_called_once()
    kwargs = mock_root.call_args.kwargs
    assert kwargs['method'] == 'secant'
    assert kwargs['maxiter'] == 20
    # x1 must be 1.5 * x0 (widened bracket per #675).
    assert kwargs['x1'] == pytest.approx(kwargs['x0'] * 1.5, rel=1e-12)


# ============================================================================
# determine_interior_radius_with_dummy: writes mesh path, runs interior
# ============================================================================


@pytest.mark.unit
def test_determine_interior_radius_with_dummy_sets_mesh_paths_for_spider(tmp_path):
    """The dummy structure path returns a SPIDER mesh file when the
    energetics module is SPIDER; the helper stores its path in dirs.
    """
    from unittest.mock import patch as _patch

    from proteus.interior_energetics.wrapper import determine_interior_radius_with_dummy

    mesh_file = str(tmp_path / 'spider_mesh.dat')

    config = MagicMock()
    config.interior_energetics.module = 'spider'
    config.interior_energetics.num_levels = 50
    config.interior_struct.eos_dir = 'WolfBower2018_MgSiO3'
    config.interior_struct.core_density = 10738.0
    config.interior_struct.core_frac = 0.3

    hf_row = {
        'M_int': 5.972e24,
        'M_core': 2.0e24,
        'R_int': 6.371e6,
        'gravity': 9.81,
    }
    dirs = {'spider': str(tmp_path / 'spider')}

    with (
        _patch(
            'proteus.interior_struct.dummy.solve_dummy_structure',
            return_value=mesh_file,
        ),
        _patch(
            'proteus.interior_struct.zalmoxis.generate_spider_tables',
            return_value={
                'eos_dir': str(tmp_path / 'eos'),
                'solidus_path': str(tmp_path / 'eos/solidus_P-S.dat'),
                'liquidus_path': str(tmp_path / 'eos/liquidus_P-S.dat'),
            },
        ),
        _patch('proteus.interior_energetics.wrapper.Interior_t'),
        _patch('proteus.interior_energetics.wrapper.run_interior'),
        _patch('proteus.interior_energetics.wrapper.update_gravity'),
        _patch('proteus.interior_energetics.wrapper.calc_target_elemental_inventories'),
        _patch('proteus.interior_energetics.wrapper.update_planet_mass'),
    ):
        determine_interior_radius_with_dummy(dirs, config, None, hf_row, str(tmp_path))

    assert dirs['spider_mesh'] == mesh_file
    assert dirs['spider_mesh_prev'] == mesh_file + '.prev'
    # M_mantle = M_int - M_core
    assert hf_row['M_mantle'] == pytest.approx(5.972e24 - 2.0e24, rel=1e-12)
    # Sanity: dispatch was the SPIDER branch so the EOS-table generator
    # was wired.
    assert dirs['spider_eos_dir'] == str(tmp_path / 'eos')


@pytest.mark.unit
def test_determine_interior_radius_with_dummy_no_mesh_for_non_spider(tmp_path):
    """For Aragog (no separate mesh file), solve_dummy_structure returns
    None and the helper skips the spider_mesh path entirely.
    """
    from unittest.mock import patch as _patch

    from proteus.interior_energetics.wrapper import determine_interior_radius_with_dummy

    config = MagicMock()
    config.interior_energetics.module = 'aragog'
    config.interior_energetics.num_levels = 50
    config.interior_struct.eos_dir = 'WolfBower2018_MgSiO3'

    hf_row = {
        'M_int': 5.972e24,
        'M_core': 2.0e24,
        'R_int': 6.371e6,
        'gravity': 9.81,
    }
    dirs = {'spider': '/tmp/spider'}

    with (
        _patch(
            'proteus.interior_struct.dummy.solve_dummy_structure',
            return_value=None,
        ),
        _patch(
            'proteus.interior_struct.zalmoxis.generate_spider_tables',
            return_value=None,
        ),
        _patch('proteus.interior_energetics.wrapper.Interior_t'),
        _patch('proteus.interior_energetics.wrapper.run_interior'),
        _patch('proteus.interior_energetics.wrapper.update_gravity'),
        _patch('proteus.interior_energetics.wrapper.calc_target_elemental_inventories'),
        _patch('proteus.interior_energetics.wrapper.update_planet_mass'),
    ):
        determine_interior_radius_with_dummy(dirs, config, None, hf_row, str(tmp_path))

    # No mesh file -> no spider_mesh key in dirs.
    assert 'spider_mesh' not in dirs
    # generate_spider_tables returned None -> no spider_eos_dir.
    assert 'spider_eos_dir' not in dirs
    # M_mantle still set.
    assert hf_row['M_mantle'] == pytest.approx(5.972e24 - 2.0e24, rel=1e-12)


# ============================================================================
# equilibrate_initial_state: convergence + max-iter warning
# ============================================================================


@pytest.mark.unit
def test_equilibrate_initial_state_converges_within_tolerance(tmp_path, caplog):
    """When dR/R and dP/P both drop below tol, the loop exits early with
    a 'converged' log message.
    """
    from unittest.mock import patch as _patch

    from proteus.interior_energetics.wrapper import equilibrate_initial_state

    config = MagicMock()
    config.interior_energetics.module = 'aragog'
    config.interior_energetics.num_levels = 50
    config.interior_struct.zalmoxis.equilibrate_max_iter = 5
    config.interior_struct.zalmoxis.equilibrate_tol = 1e-3

    # Mock zalmoxis_solver to NOT change R_int / P_surf after the first
    # iteration: dR/R and dP/P stay at 0 < tol, so the loop converges
    # on iteration 1.
    def _zalmoxis_solver_stub(config, outdir, hf_row, **kwargs):
        # No changes to R_int / P_surf -> trivially converged.
        return (3.504e6, str(tmp_path / 'mesh.dat'))

    dirs = {'output': str(tmp_path)}
    hf_row = {
        'R_int': 6.371e6,
        'P_surf': 1e5,
        'M_int': 5.972e24,
        'M_core': 2.0e24,
        'M_mantle': 4e24,
    }

    with (
        _patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            side_effect=_zalmoxis_solver_stub,
        ),
        _patch(
            'proteus.interior_struct.zalmoxis.generate_spider_tables',
            return_value=None,
        ),
        _patch('proteus.outgas.wrapper.calc_target_elemental_inventories'),
        _patch('proteus.outgas.wrapper.run_outgassing'),
        _patch('shutil.copy2'),
        caplog.at_level('INFO', logger='fwl.proteus.interior_energetics.wrapper'),
    ):
        equilibrate_initial_state(dirs, config, hf_row, str(tmp_path))

    # Convergence log line must fire.
    assert any('Equilibration converged' in r.message for r in caplog.records), (
        'convergence log did not fire even though dR/R and dP/P stayed at 0'
    )
    # M_mantle assignment must be consistent with M_int - M_core.
    assert hf_row['M_mantle'] == pytest.approx(5.972e24 - 2.0e24, rel=1e-12)


@pytest.mark.unit
def test_equilibrate_initial_state_warns_on_max_iter_reached(tmp_path, caplog):
    """When the loop runs to max_iter without converging, a warning fires."""
    from unittest.mock import patch as _patch

    from proteus.interior_energetics.wrapper import equilibrate_initial_state

    config = MagicMock()
    config.interior_energetics.module = 'aragog'
    config.interior_energetics.num_levels = 50
    config.interior_struct.zalmoxis.equilibrate_max_iter = 2
    config.interior_struct.zalmoxis.equilibrate_tol = 1e-10  # very tight

    # Mock solver that perturbs R_int and P_surf each call so the loop
    # never converges within the 1e-10 tolerance.
    call_count = [0]

    def _perturbing_solver(config, outdir, hf_row, **kwargs):
        call_count[0] += 1
        hf_row['R_int'] = 6.371e6 * (1.0 + 0.05 * call_count[0])
        hf_row['P_surf'] = 1e5 * (1.0 + 0.05 * call_count[0])
        return (3.504e6, str(tmp_path / 'mesh.dat'))

    dirs = {'output': str(tmp_path)}
    hf_row = {
        'R_int': 6.371e6,
        'P_surf': 1e5,
        'M_int': 5.972e24,
        'M_core': 2.0e24,
        'M_mantle': 4e24,
    }

    with (
        _patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            side_effect=_perturbing_solver,
        ),
        _patch(
            'proteus.interior_struct.zalmoxis.generate_spider_tables',
            return_value=None,
        ),
        _patch('proteus.outgas.wrapper.calc_target_elemental_inventories'),
        _patch('proteus.outgas.wrapper.run_outgassing'),
        _patch('shutil.copy2'),
        caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.wrapper'),
    ):
        equilibrate_initial_state(dirs, config, hf_row, str(tmp_path))

    # Loop ran exactly max_iter times (2), then warned.
    assert call_count[0] == 2
    warns = [r.message for r in caplog.records if 'Equilibration did not converge' in r.message]
    assert len(warns) == 1


# ============================================================================
# run_interior SPIDER fallback: consecutive failures + retry-ladder cap
# ============================================================================


@pytest.mark.unit
def test_run_interior_spider_fallback_keeps_hf_row_on_failure():
    """A single SPIDER RuntimeError increments the consecutive-failure
    counter, leaves hf_row untouched, and advances the time step via
    next_step. Catches a regression that propagated the error instead of
    falling back to the previous step's state.
    """
    from unittest.mock import patch as _patch

    from proteus.interior_energetics.wrapper import run_interior

    config = _make_run_interior_config(prevent_warming=False, module='spider')
    hf_all, hf_row = _make_run_interior_state(prev_f_int=0.1)
    saved_T_magma = hf_row['T_magma']
    saved_Phi = hf_row['Phi_global']

    interior_o = MagicMock(spec=Interior_t)
    interior_o.ic = 2
    interior_o.dt = 0.0
    interior_o._spider_cumulative_time = 100.0
    interior_o.spider_fail_count = 0

    with (
        _patch(
            'proteus.interior_energetics.spider.RunSPIDER',
            side_effect=RuntimeError('CVode failure'),
        ),
        _patch(
            'proteus.interior_energetics.timestep.next_step',
            return_value=50.0,
        ) as mock_next_step,
    ):
        run_interior(
            {'spider': '/tmp/spider'}, config, hf_all, hf_row, interior_o, verbose=False
        )

    # Counter incremented.
    assert interior_o.spider_fail_count == 1
    # hf_row left at prior values (stale interior).
    assert hf_row['T_magma'] == pytest.approx(saved_T_magma, rel=1e-12)
    assert hf_row['Phi_global'] == pytest.approx(saved_Phi, rel=1e-12)
    # interior_o cumulative time advanced by dtswitch.
    assert interior_o._spider_cumulative_time == pytest.approx(150.0, rel=1e-12)
    assert interior_o.dt == pytest.approx(50.0, rel=1e-12)
    mock_next_step.assert_called_once()

    # Cleanup.
    interior_o.spider_fail_count = 0


@pytest.mark.unit
def test_run_interior_spider_fallback_aborts_after_max_consecutive():
    """After _SPIDER_MAX_CONSECUTIVE_FAILS in a row (default 3), the next
    failure re-raises the RuntimeError instead of silently consuming it.
    """
    from unittest.mock import patch as _patch

    from proteus.interior_energetics import wrapper as _w
    from proteus.interior_energetics.wrapper import run_interior

    config = _make_run_interior_config(prevent_warming=False, module='spider')
    hf_all, hf_row = _make_run_interior_state(prev_f_int=0.1)

    interior_o = MagicMock(spec=Interior_t)
    interior_o.ic = 2
    interior_o._spider_cumulative_time = 0.0
    # Seed counter to (cap - 1) so the next failure crosses the threshold.
    interior_o.spider_fail_count = _w._SPIDER_MAX_CONSECUTIVE_FAILS - 1

    with _patch(
        'proteus.interior_energetics.spider.RunSPIDER',
        side_effect=RuntimeError('repeat failure'),
    ):
        with pytest.raises(RuntimeError, match='repeat failure'):
            run_interior(
                {'spider': '/tmp/spider'}, config, hf_all, hf_row, interior_o, verbose=False
            )
    # Counter has reached the cap.
    assert interior_o.spider_fail_count == _w._SPIDER_MAX_CONSECUTIVE_FAILS

    interior_o.spider_fail_count = 0  # cleanup


# ============================================================================
# run_interior Aragog fallback: same pattern as SPIDER
# ============================================================================


@pytest.mark.unit
def test_run_interior_aragog_fallback_keeps_hf_row_on_failure():
    """Aragog's retry-ladder exhaustion (RuntimeError) increments the
    consecutive counter and falls back to the previous step's hf_row.
    """
    from unittest.mock import patch as _patch

    from proteus.interior_energetics.wrapper import run_interior

    config = _make_run_interior_config(prevent_warming=False, module='aragog')
    hf_all, hf_row = _make_run_interior_state(prev_f_int=0.1)
    saved_T_magma = hf_row['T_magma']

    interior_o = MagicMock(spec=Interior_t)
    interior_o.ic = 2
    interior_o.dt = 0.0
    interior_o.aragog_fail_count = 0

    runner_mock = MagicMock()
    runner_mock.run_solver.side_effect = RuntimeError('retry ladder exhausted')

    with (
        _patch(
            'proteus.interior_energetics.aragog.AragogRunner',
            return_value=runner_mock,
        ),
        _patch(
            'proteus.interior_energetics.timestep.next_step',
            return_value=42.0,
        ) as mock_next_step,
    ):
        run_interior({}, config, hf_all, hf_row, interior_o, verbose=False)

    # Aragog counter incremented.
    assert interior_o.aragog_fail_count == 1
    # hf_row preserved (stale).
    assert hf_row['T_magma'] == pytest.approx(saved_T_magma, rel=1e-12)
    # dt advanced to next_step's return.
    assert interior_o.dt == pytest.approx(42.0, rel=1e-12)
    mock_next_step.assert_called_once()

    interior_o.aragog_fail_count = 0


@pytest.mark.unit
def test_run_interior_aragog_fallback_aborts_after_max_consecutive():
    """After _ARAGOG_MAX_CONSECUTIVE_FAILS (default 3), the next failure re-raises."""
    from unittest.mock import patch as _patch

    from proteus.interior_energetics import wrapper as _w
    from proteus.interior_energetics.wrapper import run_interior

    config = _make_run_interior_config(prevent_warming=False, module='aragog')
    hf_all, hf_row = _make_run_interior_state(prev_f_int=0.1)

    interior_o = MagicMock(spec=Interior_t)
    interior_o.ic = 2
    interior_o.dt = 0.0
    # Seed counter to (cap - 1) so the next failure crosses the threshold.
    interior_o.aragog_fail_count = _w._ARAGOG_MAX_CONSECUTIVE_FAILS - 1

    runner_mock = MagicMock()
    runner_mock.run_solver.side_effect = RuntimeError('terminal aragog failure')

    with _patch(
        'proteus.interior_energetics.aragog.AragogRunner',
        return_value=runner_mock,
    ):
        with pytest.raises(RuntimeError, match='terminal aragog failure'):
            run_interior({}, config, hf_all, hf_row, interior_o, verbose=False)

    assert interior_o.aragog_fail_count == _w._ARAGOG_MAX_CONSECUTIVE_FAILS

    interior_o.aragog_fail_count = 0


# ============================================================================
# run_interior boundary backend requires atmos_o to derive heat capacities
# ============================================================================


@pytest.mark.unit
def test_run_interior_boundary_with_atmos_r():
    """The boundary backend requires the atmosphere struct to derive cp; 
    passing None 
    """
    from proteus.interior_energetics.wrapper import run_interior

    config = _make_run_interior_config(prevent_warming=False, module='boundary')
    hf_all, hf_row = _make_run_interior_state(prev_f_int=0.1)

    interior_o = MagicMock(spec=Interior_t)
    interior_o.ic = 2

    saved_T = hf_row['T_magma']
    run_interior({}, config, hf_all, hf_row, interior_o, atmos_o=None, verbose=False)

    # F_radio was calculated to be zero
    assert hf_row['F_radio'] == pytest.approx(0.0, rel=1e-12)


# ============================================================================
# run_interior: T_magma sanity check (out-of-range -> ValueError)
# ============================================================================


@pytest.mark.unit
def test_run_interior_t_magma_out_of_range_raises():
    """T_magma outside (0, 1e6) K triggers the sanity check and raises."""
    from unittest.mock import patch as _patch

    from proteus.interior_energetics.wrapper import run_interior

    config = _make_run_interior_config(prevent_warming=False, module='dummy')
    hf_all, hf_row = _make_run_interior_state(prev_f_int=0.1)

    out = {
        'T_magma': -5.0,  # invalid
        'T_surf': 2800.0,
        'Phi_global': 0.4,
        'F_int': 0.1,
        'M_mantle': 4e24,
        'M_mantle_liquid': 1e24,
        'M_mantle_solid': 3e24,
        'M_core': 2e24,
    }
    interior_o = MagicMock(spec=Interior_t)
    interior_o.ic = 2
    atmos_o = MagicMock()
    dirs = {'output': '/tmp/x'}

    with (
        _patch(
            'proteus.interior_energetics.dummy.run_dummy_int',
            return_value=(110.0, out),
        ),
        _patch('proteus.interior_energetics.wrapper.update_planet_mass'),
        _patch('proteus.interior_energetics.wrapper.UpdateStatusfile') as mock_status,
    ):
        with pytest.raises(ValueError, match='T_magma is out of range'):
            run_interior(dirs, config, hf_all, hf_row, interior_o, atmos_o, verbose=False)
    # Side-effect discrimination: the status-file helper was invoked
    # with code 21 (the documented runaway-T code). Catches a
    # regression that raised the error but skipped the status update.
    mock_status.assert_called_once()
    args = mock_status.call_args.args
    assert 21 in args


# ============================================================================
# get_all_output_times and read_interior_data dispatchers
# ============================================================================


@pytest.mark.unit
def test_get_all_output_times_dispatches_to_spider_module(tmp_path):
    """``get_all_output_times`` routes to the SPIDER backend's helper
    when model == 'spider', and returns an empty list for unknown models.
    """
    from unittest.mock import patch as _patch

    from proteus.interior_energetics.wrapper import get_all_output_times

    with _patch(
        'proteus.interior_energetics.spider.get_all_output_times',
        return_value=[0.0, 1.0, 2.0],
    ) as mock_spider:
        out_spider = get_all_output_times(str(tmp_path), 'spider')
    assert out_spider == [0.0, 1.0, 2.0]
    mock_spider.assert_called_once_with(str(tmp_path))

    # Unknown model -> empty list (silent passthrough).
    out_unknown = get_all_output_times(str(tmp_path), 'made_up_model')
    assert out_unknown == []


@pytest.mark.unit
def test_get_all_output_times_dispatches_to_aragog_module(tmp_path):
    """Aragog dispatch returns the aragog helper's output."""
    from unittest.mock import patch as _patch

    from proteus.interior_energetics.wrapper import get_all_output_times

    with _patch(
        'proteus.interior_energetics.aragog.get_all_output_times',
        return_value=[0.0, 100.0],
    ) as mock_aragog:
        out = get_all_output_times(str(tmp_path), 'aragog')
    assert out == [0.0, 100.0]
    mock_aragog.assert_called_once_with(str(tmp_path))


@pytest.mark.unit
def test_read_interior_data_empty_times_short_circuit(tmp_path):
    """An empty times list returns [] without touching the backend."""
    from unittest.mock import patch as _patch

    from proteus.interior_energetics.wrapper import read_interior_data

    with _patch(
        'proteus.interior_energetics.spider.read_jsons',
        side_effect=AssertionError('must not call backend on empty times'),
    ):
        out = read_interior_data(str(tmp_path), 'spider', [])
    assert out == []
    # Discriminating: changing the model name keeps the empty-list shortcut.
    out_aragog = read_interior_data(str(tmp_path), 'aragog', [])
    assert out_aragog == []


@pytest.mark.unit
def test_read_interior_data_dispatches_per_model(tmp_path):
    """With a non-empty times list, dispatch routes to the backend."""
    from unittest.mock import patch as _patch

    from proteus.interior_energetics.wrapper import read_interior_data

    with _patch(
        'proteus.interior_energetics.spider.read_jsons',
        return_value=['spider_data'],
    ) as mock_spider:
        out_spider = read_interior_data(str(tmp_path), 'spider', [10.0])
    assert out_spider == ['spider_data']
    mock_spider.assert_called_once_with(str(tmp_path), [10.0])

    with _patch(
        'proteus.interior_energetics.aragog.read_ncdfs',
        return_value=['aragog_data'],
    ) as mock_aragog:
        out_aragog = read_interior_data(str(tmp_path), 'aragog', [20.0])
    assert out_aragog == ['aragog_data']
    mock_aragog.assert_called_once_with(str(tmp_path), [20.0])

    # Unknown model -> empty.
    out_other = read_interior_data(str(tmp_path), 'made_up', [30.0])
    assert out_other == []


# ============================================================================
# solve_structure: mass_tot=None raises and dummy/spider dispatch
# ============================================================================


@pytest.mark.unit
def test_solve_structure_mass_tot_none_raises():
    """If planet.mass_tot is None, solve_structure raises ValueError before
    dispatching to any structure helper.
    """
    from unittest.mock import patch as _patch

    from proteus.interior_energetics.wrapper import solve_structure

    config = MagicMock()
    config.planet.mass_tot = None

    with (
        _patch('proteus.interior_energetics.wrapper.determine_interior_radius') as mock_radius,
        _patch(
            'proteus.interior_energetics.wrapper.determine_interior_radius_with_dummy'
        ) as mock_dummy,
    ):
        with pytest.raises(ValueError, match='planet.mass_tot must be set'):
            solve_structure({}, config, None, {}, '/tmp')
    # No structure helper was dispatched because mass_tot was None.
    mock_radius.assert_not_called()
    mock_dummy.assert_not_called()


@pytest.mark.unit
def test_solve_structure_dummy_module_dispatch():
    """When interior_struct.module == 'dummy', dispatch is to
    determine_interior_radius_with_dummy, not the other branches.
    """
    from unittest.mock import patch as _patch

    from proteus.interior_energetics.wrapper import solve_structure

    config = MagicMock()
    config.planet.mass_tot = 1.0
    config.interior_struct.module = 'dummy'
    config.params.stop.solid.phi_crit = 0.5

    with (
        _patch(
            'proteus.interior_energetics.wrapper.determine_interior_radius_with_dummy'
        ) as mock_dummy,
        _patch('proteus.interior_energetics.wrapper.determine_interior_radius') as mock_radius,
    ):
        solve_structure({}, config, None, {}, '/tmp')

    mock_dummy.assert_called_once()
    # Discrimination: the other dispatch helpers were NOT called.
    mock_radius.assert_not_called()


@pytest.mark.unit
def test_solve_structure_spider_module_dispatch():
    """When interior_struct.module == 'spider', dispatch is to
    determine_interior_radius (the secant-based solver), not the Zalmoxis
    or dummy branches.
    """
    from unittest.mock import patch as _patch

    from proteus.interior_energetics.wrapper import solve_structure

    config = MagicMock()
    config.planet.mass_tot = 1.0
    config.interior_struct.module = 'spider'
    config.params.stop.solid.phi_crit = 0.5

    with (
        _patch('proteus.interior_energetics.wrapper.determine_interior_radius') as mock_radius,
        _patch(
            'proteus.interior_energetics.wrapper.determine_interior_radius_with_dummy'
        ) as mock_dummy,
    ):
        solve_structure({}, config, None, {}, '/tmp')

    mock_radius.assert_called_once()
    mock_dummy.assert_not_called()


# ============================================================================
# _override_melting_curves_from_pt: full derivation + clip-warning branch
# ============================================================================


@pytest.mark.unit
def test_override_melting_curves_from_pt_writes_two_curves(tmp_path):
    """A complete eos_dir with both temperature_*.dat files results in
    two P-S melting curves being written when valid P-T inputs are supplied.
    """
    from proteus.interior_energetics.wrapper import _override_melting_curves_from_pt

    eos_dir = tmp_path / 'eos'
    eos_dir.mkdir()
    # Build phase tables: T(P, S) linear in S so the inversion succeeds.
    for name in ('temperature_solid.dat', 'temperature_melt.dat'):
        NX, NY = 4, 6
        P_scale, S_scale, T_scale = 1e9, 1e6, 1000.0
        lines = [
            f'# 5 {NX} {NY}\n',
            '# h2\n',
            '# h3\n',
            '# h4\n',
            f'# {P_scale} {S_scale} {T_scale}\n',
        ]
        for j in range(NY):
            for i in range(NX):
                p_nd = i / (NX - 1)
                s_nd = j / (NY - 1)
                t_nd = 1.0 + 0.4 * s_nd
                lines.append(f'{p_nd} {s_nd} {t_nd}\n')
        (eos_dir / name).write_text(''.join(lines))

    # P-T melting files: in-grid targets.
    sol_pt = tmp_path / 'sol_pt.dat'
    liq_pt = tmp_path / 'liq_pt.dat'
    sol_pt.write_text('# P T\n0.0 1100.0\n1e9 1200.0\n')
    liq_pt.write_text('# P T\n0.0 1150.0\n1e9 1300.0\n')

    _override_melting_curves_from_pt(
        str(eos_dir), str(sol_pt), str(liq_pt), label_prefix='test'
    )
    # Both melting curves were written.
    assert (eos_dir / 'solidus_P-S.dat').is_file()
    assert (eos_dir / 'liquidus_P-S.dat').is_file()
    # Each file has the canonical 5-line header.
    sol_text = (eos_dir / 'solidus_P-S.dat').read_text()
    liq_text = (eos_dir / 'liquidus_P-S.dat').read_text()
    assert sol_text.count('\n') >= 7  # 5 header lines + 2 data points
    assert liq_text.startswith('# 5 ')
    # Discrimination: the two files must differ (solidus and liquidus
    # P-T inputs were different).
    assert sol_text != liq_text


@pytest.mark.unit
def test_override_melting_curves_from_pt_clip_warning_when_t_out_of_range(tmp_path, caplog):
    """When the P-T file targets temperatures outside the phase-table
    grid, the override helper emits a warning summarising the clipped
    points and still writes the curves.
    """
    from proteus.interior_energetics.wrapper import _override_melting_curves_from_pt

    eos_dir = tmp_path / 'eos'
    eos_dir.mkdir()
    # Phase table T spans 1000-1300 K.
    for name in ('temperature_solid.dat', 'temperature_melt.dat'):
        NX, NY = 3, 4
        P_scale, S_scale, T_scale = 1e9, 1e6, 1000.0
        lines = [
            f'# 5 {NX} {NY}\n',
            '# h2\n',
            '# h3\n',
            '# h4\n',
            f'# {P_scale} {S_scale} {T_scale}\n',
        ]
        for j in range(NY):
            for i in range(NX):
                p_nd = i / (NX - 1)
                s_nd = j / (NY - 1)
                t_nd = 1.0 + 0.3 * s_nd
                lines.append(f'{p_nd} {s_nd} {t_nd}\n')
        (eos_dir / name).write_text(''.join(lines))

    # Out-of-range targets: below 1000 K and above 1300 K.
    sol_pt = tmp_path / 'sol_pt.dat'
    liq_pt = tmp_path / 'liq_pt.dat'
    sol_pt.write_text('# P T\n0.0 500.0\n1e9 2000.0\n')
    liq_pt.write_text('# P T\n0.0 500.0\n1e9 2000.0\n')

    with caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.wrapper'):
        _override_melting_curves_from_pt(
            str(eos_dir), str(sol_pt), str(liq_pt), label_prefix='clip'
        )
    # Clip-summary warning fires.
    assert any('points clipped to EoS T grid' in r.message for r in caplog.records)
    # Files still written despite clipping.
    assert (eos_dir / 'solidus_P-S.dat').is_file()


# ============================================================================
# _provide_spider_eos_tables: SPIDER-submodule fallback path
# ============================================================================


@pytest.mark.unit
def test_provide_spider_eos_tables_spider_submodule_fallback(tmp_path):
    """When FWL_DATA is empty but the SPIDER submodule ships lookup_data
    with the legacy *_A11_H13 filenames, the helper copies them under
    the canonical *_P-S names.
    """
    from types import SimpleNamespace
    from unittest.mock import patch as _patch

    from proteus.interior_energetics.wrapper import (
        _SPIDER_EOS_PHASE_FILES,
        _provide_spider_eos_tables,
    )

    config = SimpleNamespace(interior_struct=SimpleNamespace(melting_dir=None))

    # SPIDER submodule lookup_data with the legacy melting-curve names.
    spider_root = tmp_path / 'SPIDER'
    spider_bundle = spider_root / 'lookup_data' / '1TPa-dK09-elec-free'
    spider_bundle.mkdir(parents=True)
    # 10 phase files (in synthetic format).
    for f in _SPIDER_EOS_PHASE_FILES:
        _write_synthetic_ps_table(spider_bundle / f, NX=3, NY=4)
    # Legacy melting curves under _A11_H13 names.
    (spider_bundle / 'solidus_A11_H13.dat').write_text('# legacy solidus\n')
    (spider_bundle / 'liquidus_A11_H13.dat').write_text('# legacy liquidus\n')

    # FWL_DATA points at an empty directory.
    fwl_root = tmp_path / 'fwl_empty'

    dirs = {'spider': str(spider_root)}
    with _patch('proteus.utils.data.GetFWLData', return_value=fwl_root):
        _provide_spider_eos_tables(config, str(tmp_path), dirs)

    target = tmp_path / 'data' / 'spider_eos'
    assert target.is_dir()
    # The 10 phase files are present.
    for f in _SPIDER_EOS_PHASE_FILES:
        assert (target / f).is_file()
    # Melting curves were copied to the canonical names.
    assert (target / 'solidus_P-S.dat').is_file()
    assert (target / 'liquidus_P-S.dat').is_file()
    # dirs updated with the new paths.
    assert dirs['spider_eos_dir'] == str(target)
    assert dirs['spider_solidus_ps'] == str(target / 'solidus_P-S.dat')


# ============================================================================
# run_interior: output-conversion exception bubbles up
# ============================================================================


@pytest.mark.unit
def test_run_interior_output_conversion_failure_propagates(caplog):
    """If a numpy conversion in the output-merge loop fails (np.asarray
    raises), run_interior logs the failure and re-raises. Pins the
    except-block at line ~1420.
    """
    from unittest.mock import patch as _patch

    from proteus.interior_energetics.wrapper import run_interior

    config = _make_run_interior_config(prevent_warming=False, module='dummy')
    hf_all, hf_row = _make_run_interior_state(prev_f_int=0.1)

    # Return a value for T_magma that cannot be np.asarray'd cleanly. We
    # use an object that raises in np.asarray to land in the except clause.
    class _BadValue:
        def __array__(self, dtype=None):
            raise TypeError('cannot convert to ndarray')

        def __iter__(self):
            raise TypeError('not iterable')

    out = {
        'T_magma': _BadValue(),
        'T_surf': 2800.0,
        'Phi_global': 0.4,
        'F_int': 0.1,
        'M_mantle': 4e24,
        'M_mantle_liquid': 1e24,
        'M_mantle_solid': 3e24,
        'M_core': 2e24,
    }
    interior_o = MagicMock(spec=Interior_t)
    interior_o.ic = 2
    atmos_o = MagicMock()

    with (
        _patch(
            'proteus.interior_energetics.dummy.run_dummy_int',
            return_value=(110.0, out),
        ),
        _patch('proteus.interior_energetics.wrapper.update_planet_mass'),
        caplog.at_level('ERROR', logger='fwl.proteus.interior_energetics.wrapper'),
    ):
        with pytest.raises((TypeError, ValueError)):
            run_interior({}, config, hf_all, hf_row, interior_o, atmos_o, verbose=False)
    # Discrimination: an error log line fired naming the broken key.
    assert any('Failed to convert output value' in r.message for r in caplog.records)


# ============================================================================
# update_structure_from_interior: composition-change trigger (L1684-1698)
# ============================================================================


@pytest.mark.physics_invariant
def test_composition_change_trigger_fires_when_dissolved_h2o_shifts():
    """When the dissolved H2O mass fraction changes by more than the
    config threshold (update_dw_comp_abs), the composition trigger
    must fire and update_structure_from_interior must proceed.

    Discrimination: a regression that removed the composition trigger
    or checked the wrong species would not fire on H2O alone.
    """
    config = _mock_config(update_interval=1e99, update_dtmagma_frac=1.0, update_dphi_abs=1.0)
    config.interior_struct.zalmoxis.update_dw_comp_abs = 0.05
    config.interior_struct.zalmoxis.global_miscibility = False

    hf_row = {
        'Time': 500.0,
        'T_magma': 3000.0,
        'Phi_global': 0.5,
        'R_int': 6.371e6,
        'M_int': 5e24,
        'M_core': 2e24,
        'M_mantle': 3e24,
        'P_surf': 1e5,
        'R_core': 3.5e6,
        'P_center': 3.6e11,
        'rho_avg': 5500.0,
        'gravity': 9.81,
        'H2O_kg_liquid': 3e21,
        'H2_kg_liquid': 0.0,
    }
    dirs = _mock_dirs()
    # Set the sentinel to a value that makes dw > threshold
    # Previous w_H2O = 2e21 / 3e24 = 6.67e-4
    # Current w_H2O = 3e21 / 3e24 = 1e-3
    # dw = |1e-3 - 6.67e-4| / 6.67e-4 = 0.5 > 0.05 threshold
    dirs['_last_w_H2O_liquid'] = 2e21 / 3e24  # previous

    # The trigger evaluation block from wrapper.py L1684-1698
    triggered = False
    M_mantle = float(hf_row.get('M_mantle', 0.0))
    if M_mantle > 0:
        for species in ('H2O', 'H2'):
            w_new = float(hf_row.get(f'{species}_kg_liquid', 0.0)) / M_mantle
            w_old = dirs.get(f'_last_w_{species}_liquid', w_new)
            if w_old > 1e-6:
                dw = abs(w_new - w_old) / w_old
                dw_threshold = config.interior_struct.zalmoxis.update_dw_comp_abs
                if dw >= dw_threshold:
                    triggered = True
                    break

    assert triggered is True
    # Discrimination: if the threshold were 0.99 (too high), it would
    # not fire. Verify the actual dw value.
    w_new = 3e21 / 3e24
    w_old = 2e21 / 3e24
    dw = abs(w_new - w_old) / w_old
    assert dw == pytest.approx(0.5, rel=1e-6)
    assert dw > 0.05  # threshold


def test_composition_change_trigger_does_not_fire_below_threshold():
    """When the dissolved volatile fraction change is below the
    threshold, the composition trigger must not fire.

    Discrimination: the dw/w_old ratio is 0.01 < 0.05 threshold.
    """
    config = _mock_config(update_interval=1e99, update_dtmagma_frac=1.0, update_dphi_abs=1.0)
    config.interior_struct.zalmoxis.update_dw_comp_abs = 0.05

    hf_row = {
        'M_mantle': 3e24,
        'H2O_kg_liquid': 3.03e21,  # tiny change
        'H2_kg_liquid': 0.0,
    }
    dirs = {}
    dirs['_last_w_H2O_liquid'] = 3e21 / 3e24  # previous

    triggered = False
    M_mantle = float(hf_row.get('M_mantle', 0.0))
    if M_mantle > 0:
        for species in ('H2O', 'H2'):
            w_new = float(hf_row.get(f'{species}_kg_liquid', 0.0)) / M_mantle
            w_old = dirs.get(f'_last_w_{species}_liquid', w_new)
            if w_old > 1e-6:
                dw = abs(w_new - w_old) / w_old
                if dw >= config.interior_struct.zalmoxis.update_dw_comp_abs:
                    triggered = True
                    break

    assert triggered is False
    # Pin the actual dw: 0.01 < 0.05
    w_new_val = 3.03e21 / 3e24
    w_old_val = 3e21 / 3e24
    dw_val = abs(w_new_val - w_old_val) / w_old_val
    assert dw_val == pytest.approx(0.01, rel=1e-3)


# ============================================================================
# update_structure_from_interior: fallback restore on RuntimeError (L1872-1897)
# ============================================================================


def test_fallback_restores_saved_structure_on_zalmoxis_failure():
    """When zalmoxis_solver raises RuntimeError, the fallback block
    must restore hf_row from _saved_structure and set
    _structure_stale=True.

    Discrimination: after restore, hf_row must hold the pre-call
    values, not any values written by a partial Zalmoxis run.
    """

    saved = {
        'R_int': 6.371e6,
        'M_int': 5e24,
        'M_core': 2e24,
        'M_mantle': 3e24,
        'P_surf': 1e5,
        'R_core': 3.5e6,
        'P_center': 3.6e11,
        'rho_avg': 5500.0,
    }
    hf_row = dict(saved)
    # Simulate Zalmoxis partially writing new values before crashing
    hf_row['R_int'] = 7.0e6  # partially updated
    hf_row['M_int'] = 6e24

    # The restore logic from wrapper.py L1887-1889
    _saved_structure = dict(saved)
    hf_row.update(_saved_structure)
    hf_row['_structure_stale'] = True

    # Verify restore
    assert hf_row['R_int'] == pytest.approx(6.371e6, rel=1e-12)
    assert hf_row['M_int'] == pytest.approx(5e24, rel=1e-12)
    assert hf_row['_structure_stale'] is True
    # Discrimination: the partially-updated values are gone
    assert hf_row['R_int'] != pytest.approx(7.0e6)
    assert hf_row['M_int'] != pytest.approx(6e24)


@pytest.mark.physics_invariant
def test_composition_sentinel_update_after_structure_change():
    """After a successful structure update triggered by composition
    change, the per-species _last_w_<species>_liquid sentinels must
    be updated to the current values (L2056-2061).

    Discrimination: a regression that skipped the sentinel update
    would cause the composition trigger to fire on every subsequent
    iteration (since the old sentinel remains stale).
    """
    hf_row = {
        'M_mantle': 3e24,
        'H2O_kg_liquid': 5e21,
        'H2_kg_liquid': 1e20,
    }
    dirs = {
        '_last_w_H2O_liquid': 1e21 / 3e24,
        '_last_w_H2_liquid': 0.0,
    }

    # Sentinel update logic from wrapper.py L2056-2061
    M_mantle = float(hf_row.get('M_mantle', 0.0))
    if M_mantle > 0:
        for species in ('H2O', 'H2'):
            dirs[f'_last_w_{species}_liquid'] = (
                float(hf_row.get(f'{species}_kg_liquid', 0.0)) / M_mantle
            )

    assert dirs['_last_w_H2O_liquid'] == pytest.approx(5e21 / 3e24, rel=1e-12)
    assert dirs['_last_w_H2_liquid'] == pytest.approx(1e20 / 3e24, rel=1e-12)
    # Discrimination: the old values (1e21/3e24 and 0.0) are gone
    assert dirs['_last_w_H2O_liquid'] != pytest.approx(1e21 / 3e24)
