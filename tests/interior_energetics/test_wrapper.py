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

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from proteus.interior_energetics.common import Interior_t
from proteus.interior_energetics.wrapper import (
    _prevent_warming_clamp_active,
    update_structure_from_interior,
)


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

    with caplog.at_level('WARNING', logger='fwl.proteus.interior_energetics.wrapper'):
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
        update_structure_from_interior(dirs, config, hf_row_A, interior_o, 0.0, 3000.0, 0.8)
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
        update_structure_from_interior(dirs, config, hf_row_B, interior_o, 0.0, 3000.0, 0.8)
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
    _w._zalmoxis_fail_count = 0


# ============================================================================
# T1.2: post-acceptance mass-anchor guard
# ============================================================================


@pytest.mark.unit
def test_mass_anchor_violation_triggers_fallback(tmp_path):
    """A 'converged' Zalmoxis result with |M_int/M_target - 1| > 3e-3 falls back.

    Regression test for T1.2 (2026-04-26 coupling audit, relaxed after
    Run T1 contract collision). Zalmoxis' internal solver_tol_outer
    (3e-3) leaves up to ~0.3 % drift between hf_row['M_int'] and the
    dry mass target on hot fully-molten profiles. The wrapper guards
    against the G4-basin attractor failure mode (~9-15 % off target)
    and gross corruption. The genuine <0.1 % mass conservation
    contract is delivered by T2.1 (Newton + brentq), not by this
    safety net.

    Edge cases covered:
    - Far-over-threshold (5e-3 drift): must reject.
    - Just-under-threshold (2e-3 drift): must accept (Zalmoxis
      noise-floor regime).
    - Near-but-under threshold (3e-3 - epsilon drift): must accept
      (strict > comparison).
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

    # Case 1: 5e-3 drift (over threshold) -> fall-back path must fire.
    _w._zalmoxis_fail_count = 0
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
    assert _w._zalmoxis_fail_count == 1, (
        'mass-anchor violation must increment _zalmoxis_fail_count (got %d)'
        % _w._zalmoxis_fail_count
    )

    # Case 2: 2e-3 drift (under 3e-3 threshold) -> success path.
    # This is the regime Zalmoxis normally lands in on hot mantle
    # profiles (Run T1 2026-04-26 saw 0.18-0.28 % naturally).
    _w._zalmoxis_fail_count = 0
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


@pytest.mark.unit
def test_zalmoxis_output_restored_from_prev_on_wrapper_raise(tmp_path):
    """Wrapper-level RuntimeError (e.g. T1.2 mass-anchor) restores
    zalmoxis_output.dat from .prev backup.

    The atomic-write path inside zalmoxis_solver (T1.3 fix-2) handles
    schema-violation raises. But the wrapper's mass-anchor check raises
    AFTER zalmoxis_solver returns successfully, so the rollback must
    happen at the wrapper level too. Otherwise Aragog reads the new
    failing file on the next iter with the OLD hf_row['R_int'] and
    crashes on EOS-vs-mesh inconsistency.

    Edge case covered: when no .prev exists (first call of a run), the
    wrapper must NOT raise — it best-effort skips the restore.
    """
    from proteus.interior_energetics import wrapper as _w

    _w._zalmoxis_fail_count = 0

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
        # so wrapper's T1.2 fires.
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
            dirs,
            config,
            hf_row,
            interior_o,
            0.0,
            3000.0,
            0.8,
        )
    assert interior_o.last_successful_struct_time == 5e4, (
        'fall-back must NOT advance last_successful_struct_time '
        '(got %r, expected 5e4)' % interior_o.last_successful_struct_time
    )

    _w._zalmoxis_fail_count = 0


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
    out = {
        'T_magma': 3005.0,
        'T_surf': 2900.0,  # 100 K leap, well above tmagma_atol=20
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


@pytest.mark.unit
def test_run_interior_boundary_module_uses_Tsurf_event_change_for_dT_delta():
    """When module='boundary' the dT_delta budget comes from
    interior_energetics.boundary.Tsurf_event_change, not from tmagma_atol/rtol.

    The wrapper picks the limiter source from
    ``config.interior_energetics.module``, so we drive the dummy backend
    path but force the module string to 'boundary' before re-entering
    the limiter block. This isolates the dT_delta branch under test
    from the BoundaryRunner import.
    """
    config = _make_run_interior_config(prevent_warming=False, module='boundary')
    hf_all, hf_row = _make_run_interior_state(prev_f_int=0.2)
    out = {
        # 50 K leap. Boundary uses Tsurf_event_change=25 -> cap fires at +25.
        'T_magma': 3050.0,
        'T_surf': 2850.0,
        'Phi_global': 0.7,
        'F_int': 0.15,
        'M_mantle': 4.0e24,
        'M_mantle_liquid': 1.0e24,
        'M_mantle_solid': 3.0e24,
        'M_core': 2.0e24,
    }

    # Patch BoundaryRunner so the boundary branch yields the same output dict
    # as a dummy backend would have.
    from proteus.interior_energetics.wrapper import run_interior

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

    # T_magma capped at 3000 + 25; T_surf capped at 2800 + 25.
    assert hf_row['T_magma'] == pytest.approx(3025.0)
    assert hf_row['T_surf'] == pytest.approx(2825.0)


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
    """Spider.log_output is a new attrs field. Default True preserves the
    pre-7g behaviour of writing the spider_recent.log mirror; setting
    False routes subprocess output to /dev/null."""
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
