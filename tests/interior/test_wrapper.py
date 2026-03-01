"""
Unit tests for proteus.interior.wrapper module — structure update triggers.

Tests the dynamic trigger logic in update_structure_from_interior() that
decides when to re-run Zalmoxis with SPIDER's evolved temperature profile.
Also tests solve_structure() dispatch and phi_crit warning.

Testing standards and documentation:
- docs/test_infrastructure.md: Test infrastructure overview
- docs/test_categorization.md: Test marker definitions
- docs/test_building.md: Best practices for test construction

Functions tested:
- update_structure_from_interior(): Trigger logic (ceiling, dT, dPhi, early exit)
- solve_structure(): phi_crit warning, dispatch to Zalmoxis
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from proteus.interior.common import Interior_t
from proteus.interior.wrapper import update_structure_from_interior


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
    config.struct.update_interval = update_interval
    config.struct.update_min_interval = update_min_interval
    config.struct.update_dtmagma_frac = update_dtmagma_frac
    config.struct.update_dphi_abs = update_dphi_abs
    config.struct.mesh_max_shift = mesh_max_shift
    config.struct.mesh_convergence_interval = mesh_convergence_interval
    config.struct.zalmoxis.temperature_mode = 'isothermal'
    config.struct.zalmoxis.temperature_profile_file = None
    config.struct.zalmoxis.num_levels = num_levels
    config.interior.module = 'spider'
    config.interior.spider.num_levels = num_levels
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
        patch('proteus.interior.zalmoxis.zalmoxis_solver', return_value=(3.504e6, None)),
        patch('proteus.interior.wrapper.np.savetxt'),
        patch('proteus.interior.wrapper.shutil.copy2'),
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
        patch('proteus.interior.zalmoxis.zalmoxis_solver', return_value=(3.504e6, None)),
        patch('proteus.interior.wrapper.np.savetxt'),
        patch('proteus.interior.wrapper.shutil.copy2'),
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
        patch('proteus.interior.zalmoxis.zalmoxis_solver', return_value=(3.504e6, None)),
        patch('proteus.interior.wrapper.np.savetxt'),
        patch('proteus.interior.wrapper.shutil.copy2'),
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
        patch('proteus.interior.zalmoxis.zalmoxis_solver', return_value=(3.504e6, None)),
        patch('proteus.interior.wrapper.np.savetxt'),
        patch('proteus.interior.wrapper.shutil.copy2'),
    ):
        update_structure_from_interior(dirs, config, hf_row, interior_o, 0.0, 3000.0, 0.8)
    # No mesh file returned → convergence state reset
    assert dirs['mesh_shift_active'] is False
    assert dirs['mesh_convergence_steps'] == 0
