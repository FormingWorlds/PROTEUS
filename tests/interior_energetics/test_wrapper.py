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
    config.interior_struct.update_interval = update_interval
    config.interior_struct.update_min_interval = update_min_interval
    config.interior_struct.update_dtmagma_frac = update_dtmagma_frac
    config.interior_struct.update_dphi_abs = update_dphi_abs
    config.interior_struct.mesh_max_shift = mesh_max_shift
    config.interior_struct.mesh_convergence_interval = mesh_convergence_interval
    config.interior_struct.zalmoxis.temperature_mode = 'isothermal'
    config.interior_struct.zalmoxis.temperature_profile_file = None
    config.interior_struct.zalmoxis.num_levels = num_levels
    config.interior_energetics.module = 'spider'
    config.interior_energetics.spider.num_levels = num_levels
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
    config.planet.planet_mass_tot = 1.0
    config.interior_struct.module = 'zalmoxis'
    config.params.stop.solid.phi_crit = 0.005

    with caplog.at_level('WARNING'):
        with patch('proteus.interior_energetics.wrapper.determine_interior_radius_with_zalmoxis'):
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
    config.interior_energetics.spider.num_levels = 50
    config.interior_struct.zalmoxis.temperature_mode = 'isothermal'
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
    config.interior_energetics.spider.num_levels = 50
    config.interior_struct.zalmoxis.temperature_mode = 'isothermal'
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
    assert config.interior_struct.zalmoxis.temperature_mode == 'isothermal'
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
    config.interior_energetics.spider.num_levels = 50
    config.interior_struct.zalmoxis.temperature_mode = 'isothermal'
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
    assert config.interior_struct.zalmoxis.temperature_mode == 'isothermal'


@pytest.mark.unit
def test_solve_structure_invalid_module():
    """solve_structure raises ValueError for unknown struct.module."""
    from proteus.interior_energetics.wrapper import solve_structure

    config = MagicMock()
    config.planet.planet_mass_tot = 1.0
    config.interior_struct.module = 'nonexistent_module'
    config.params.stop.solid.phi_crit = 0.5

    with pytest.raises(ValueError, match='Invalid structure interior module'):
        solve_structure({}, config, None, {}, '/tmp')
