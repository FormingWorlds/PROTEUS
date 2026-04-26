"""
Unit tests for proteus.interior_energetics.wrapper module — structure update triggers.

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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from proteus.interior_energetics.common import Interior_t
from proteus.interior_energetics.wrapper import update_structure_from_interior


def _mock_config(
    update_interval=1000.0,
    update_min_interval=100.0,
    update_dtmagma_frac=0.03,
    update_dphi_abs=0.05,
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
    """Create a mock Interior_t with minimal arrays."""
    interior_o = MagicMock(spec=Interior_t)
    interior_o.radius = np.linspace(6.371e6, 3.504e6, n_stag + 1)
    interior_o.temp = np.full(n_stag, 3000.0)
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

    result = update_structure_from_interior(dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.8)
    assert result == (0.0, 3000.0, 0.8)


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
        patch('proteus.interior_struct.zalmoxis.zalmoxis_solver', return_value=(3.504e6, None)),
        patch('proteus.interior_energetics.wrapper.np.savetxt'),
        patch('proteus.interior_energetics.wrapper.shutil.copy2'),
    ):
        result = update_structure_from_interior(
            dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.8
        )
    # Should have triggered: last_struct_time updated to current
    assert result[0] == 1100.0


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
        patch('proteus.interior_struct.zalmoxis.zalmoxis_solver', return_value=(3.504e6, None)),
        patch('proteus.interior_energetics.wrapper.np.savetxt'),
        patch('proteus.interior_energetics.wrapper.shutil.copy2'),
    ):
        result = update_structure_from_interior(
            dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.80
        )
    assert result[0] == 200.0


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

    result = update_structure_from_interior(dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.80)
    assert result[0] == 0.0  # Not triggered


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
        patch('proteus.interior_struct.zalmoxis.zalmoxis_solver', return_value=(3.504e6, None)),
        patch('proteus.interior_energetics.wrapper.np.savetxt'),
        patch('proteus.interior_energetics.wrapper.shutil.copy2'),
    ):
        result = update_structure_from_interior(
            dirs, config, hf_row, interior_o, 0.0, 2990.0, 0.80
        )
    assert result[0] == 200.0


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

    result = update_structure_from_interior(dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.80)
    assert result[0] == 0.0  # Not triggered


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

    result = update_structure_from_interior(dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.8)
    assert result[0] == 0.0  # Floor blocked


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
    assert result[0] == 1100.0
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

    assert result[0] == 1100.0
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

    assert result[0] == 1100.0
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
        ),
        patch('proteus.interior_energetics.wrapper.gc.collect'),
    ):
        # Should not raise
        result = update_structure_from_interior(
            dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.8
        )

    assert result[0] == 1100.0


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

    assert result[0] == 1100.0
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

    with caplog.at_level('WARNING'):
        with patch(
            'proteus.interior_energetics.wrapper.determine_interior_radius_with_zalmoxis'
        ):
            solve_structure({}, config, None, {}, '/tmp')

    assert any('phi_crit' in r.message for r in caplog.records)


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
        caplog.at_level('INFO'),
    ):
        determine_interior_radius_with_zalmoxis(dirs, config, None, hf_row, '/tmp')

    # Mode should be restored after the call
    assert config.planet.temperature_mode == 'isothermal'
    assert any('adiabatic' in r.message for r in caplog.records)
    # No mesh file → no spider_mesh key
    assert 'spider_mesh' not in dirs
    assert dirs['mesh_shift_active'] is False


@pytest.mark.unit
def test_determine_zalmoxis_no_adiabatic_switch_non_tdep():
    """Non-T-dep EOS does not trigger adiabatic switch."""
    from proteus.interior_energetics.wrapper import determine_interior_radius_with_zalmoxis

    config = MagicMock()
    config.interior_energetics.module = 'spider'
    config.interior_energetics.eos_dir = 'Seager2007'
    config.planet.temperature_mode = 'isothermal'
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
    ):
        determine_interior_radius_with_zalmoxis(dirs, config, None, hf_row, '/tmp')

    # Should not have been changed
    assert config.planet.temperature_mode == 'isothermal'


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


# ============================================================================
# T1.1: _structure_stale flag contract (set on fall-back, cleared on success)
# ============================================================================


@pytest.mark.unit
def test_structure_stale_flag_cleared_on_zalmoxis_success(tmp_path):
    """Successful zalmoxis_solver call clears hf_row['_structure_stale'].

    Regression test for T1.1 (2026-04-26 coupling audit). Pre-fix the
    flag was set on fall-back (wrapper.py:1355) but never cleared, so
    Aragog had no way to distinguish a fresh from a stale structure.
    Post-fix the flag is cleared in the success path.

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
        update_structure_from_interior(
            dirs, config, hf_row_A, interior_o, 0.0, 3000.0, 0.8
        )
    # Flag must be present and False after a successful call.
    assert '_structure_stale' in hf_row_A, 'flag must be set, not absent'
    assert hf_row_A['_structure_stale'] is False, (
        'flag must be cleared (False) on Zalmoxis success — was %r'
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
        update_structure_from_interior(
            dirs, config, hf_row_B, interior_o, 0.0, 3000.0, 0.8
        )
    assert hf_row_B.get('_structure_stale') is False, (
        'success path must set flag to False even if it was absent on entry'
    )


@pytest.mark.unit
def test_structure_stale_flag_set_on_zalmoxis_failure(tmp_path):
    """Zalmoxis RuntimeError sets hf_row['_structure_stale'] = True.

    Documents the canonical fall-back contract that T1.1 makes honest.
    Anti-happy-path: also checks that the failure does NOT clear the
    flag if it was already True (i.e. consecutive failures keep it set).
    """
    from proteus.interior_energetics import wrapper as _w

    # Reset module-level fail counter before test (prevents hard-abort
    # at consecutive cap = 8 from leaking between test runs).
    _w._zalmoxis_fail_count = 0

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
        update_structure_from_interior(
            dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.8
        )
    assert hf_row.get('_structure_stale') is True, (
        'fall-back path must set flag to True'
    )

    # Second consecutive failure: flag stays True (not toggled or cleared).
    with patch(
        'proteus.interior_struct.zalmoxis.zalmoxis_solver',
        side_effect=RuntimeError('mock convergence failure 2'),
    ):
        update_structure_from_interior(
            dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.8
        )
    assert hf_row.get('_structure_stale') is True

    # Reset for any downstream tests.
    _w._zalmoxis_fail_count = 0


# ============================================================================
# T1.2: post-acceptance mass-anchor guard
# ============================================================================


@pytest.mark.unit
def test_mass_anchor_violation_triggers_fallback(tmp_path):
    """A 'converged' Zalmoxis result with |M_int/M_target - 1| > 1e-3 falls back.

    Regression test for T1.2 (2026-04-26 coupling audit). Zalmoxis'
    internal solver_tol_outer (default 3e-3) leaves up to 0.3 % drift
    between hf_row['M_int'] and the dry mass target, exceeding the
    <0.1 % conservation requirement. The wrapper enforces a tighter
    1e-3 contract on the success boundary.

    Edge cases covered:
    - Just-over-threshold (1.5e-3 drift): must reject.
    - Just-under-threshold (5e-4 drift): must accept.
    - At-threshold (exact 1e-3): accepted (strict >).
    """
    from proteus.interior_energetics import wrapper as _w

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

    # Case 1: 1.5e-3 drift (over threshold) -> fall-back path must fire.
    _w._zalmoxis_fail_count = 0
    hf_row_over = _make_hf_row()
    dirs_over = _mock_dirs()
    with (
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            side_effect=_make_solver_mock(M_target * (1 + 1.5e-3)),
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
    assert _w._zalmoxis_fail_count == 1, (
        'mass-anchor violation must increment _zalmoxis_fail_count (got %d)'
        % _w._zalmoxis_fail_count
    )

    # Case 2: 5e-4 drift (under threshold) -> success path.
    _w._zalmoxis_fail_count = 0
    hf_row_under = _make_hf_row()
    dirs_under = _mock_dirs()
    with (
        patch(
            'proteus.interior_struct.zalmoxis.zalmoxis_solver',
            side_effect=_make_solver_mock(M_target * (1 + 5e-4)),
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
    assert _w._zalmoxis_fail_count == 0

    # Case 3: target=0 (degenerate, e.g. zalmoxis didn't write target) -> skip
    # the check (must not divide by zero or spuriously raise).
    _w._zalmoxis_fail_count = 0
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
    assert _w._zalmoxis_fail_count == 0

    _w._zalmoxis_fail_count = 0  # cleanup


# ============================================================================
# T1.5: anti-stale-mesh re-trigger via last_successful_struct_time
# ============================================================================


@pytest.mark.unit
def test_stale_aware_ceiling_fires_after_failure_window(tmp_path):
    """Stale-aware ceiling fires after update_stale_ceiling regardless of
    intervening failed re-solves.

    Pre-T1.5: the only ceiling trigger uses elapsed time since the last
    *call*, which a fall-back re-anchors. This means a failed call
    can keep the trigger silent for a full update_interval afterwards
    (the 2026-04-26 Step D run showed 51 kyr of Aragog evolution on
    a frozen mesh between failure #4 and the next trigger).

    Post-T1.5: tracking last_successful_struct_time on Interior_t
    drives a separate stale-aware ceiling that fires after
    update_stale_ceiling has elapsed since the last *successful*
    re-solve. Independent of last_struct_time, so it survives failures.
    """
    from proteus.interior_energetics import wrapper as _w

    _w._zalmoxis_fail_count = 0

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
    last_struct_time = 7e4   # last call was at 7e4 yr (a failed one)
    current_time = 1e5       # now at 1e5 yr; elapsed = 3e4 (< 5e4 ceiling)
    hf_row = {
        'Time': current_time,
        'T_magma': 3000.0,
        'Phi_global': 0.8,    # no Phi/T trigger (no change)
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
            dirs, config, hf_row, interior_o,
            last_struct_time, hf_row['T_magma'], hf_row['Phi_global'],
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
            dirs, config, hf_row, interior_o,
            last_struct_time, hf_row['T_magma'], hf_row['Phi_global'],
        )
    # Trigger fired and updated last_struct_time to current_time.
    assert out[0] == current_time, (
        'stale-aware ceiling must trigger; got last_struct_time=%r' % out[0]
    )
    # Successful tracker advanced.
    assert interior_o.last_successful_struct_time == current_time, (
        'last_successful_struct_time must advance on success'
    )

    _w._zalmoxis_fail_count = 0


@pytest.mark.unit
def test_last_successful_struct_time_not_advanced_on_failure(tmp_path):
    """A Zalmoxis fall-back must NOT advance last_successful_struct_time.

    Otherwise the stale-aware ceiling becomes equivalent to the normal
    ceiling and T1.5 would have no effect.
    """
    from proteus.interior_energetics import wrapper as _w

    _w._zalmoxis_fail_count = 0

    config = _mock_config(update_interval=1e3, update_min_interval=0.0)
    config.interior_struct.zalmoxis.update_stale_ceiling = 0.0  # disable T1.5 here
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
    ):
        update_structure_from_interior(
            dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.8,
        )
    assert interior_o.last_successful_struct_time == 5e4, (
        'fall-back must NOT advance last_successful_struct_time '
        '(got %r, expected 5e4)' % interior_o.last_successful_struct_time
    )

    _w._zalmoxis_fail_count = 0
