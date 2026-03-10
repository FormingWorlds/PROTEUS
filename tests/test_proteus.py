"""
Unit tests for proteus.proteus module — Zalmoxis mesh restoration on resume.

Tests the resume code path in Proteus.start() that restores the Zalmoxis
mesh file path when resuming a SPIDER interior simulation.

Testing standards and documentation:
- docs/test_infrastructure.md: Test infrastructure overview
- docs/test_categorization.md: Test marker definitions
- docs/test_building.md: Best practices for test construction

Functions tested:
- Proteus.start(): Resume path restoring spider_mesh and spider_mesh_prev
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _make_proteus_instance(tmp_path, *, struct_module='zalmoxis', interior_module='spider'):
    """Build a Proteus object with mocked config and directories."""
    from proteus.proteus import Proteus

    config = MagicMock()
    config.struct.module = struct_module
    config.struct.update_interval = 0
    config.interior.module = interior_module
    config.interior.spider.num_levels = 50
    config.interior.eos_dir = 'WolfBower2018_MgSiO3'
    config.orbit.module = None
    # Attributes used during start() setup
    config.params.out.logging = 'WARNING'
    config.params.stop.iters.minimum = 10
    config.params.stop.iters.maximum = 1000
    config.atmos_clim.albedo_from_file = False

    directories = {
        'output': str(tmp_path),
        'output/data': str(tmp_path / 'data'),
        'spider': '/nonexistent/spider',
    }

    with (
        patch('proteus.proteus.read_config_object', return_value=config),
        patch('proteus.utils.coupler.set_directories', return_value=directories),
    ):
        p = Proteus(config_path='dummy.toml')

    return p


# All the lazy imports inside start() that must be mocked to reach the resume path.
_START_PATCHES = [
    'proteus.atmos_chem.wrapper.run_chemistry',
    'proteus.atmos_clim.run_atmosphere',
    'proteus.atmos_clim.common.Albedo_t',
    'proteus.atmos_clim.common.Atmos_t',
    'proteus.escape.wrapper.run_escape',
    'proteus.interior.wrapper.run_interior',
    'proteus.interior.wrapper.solve_structure',
    'proteus.interior.wrapper.update_planet_mass',
    'proteus.observe.wrapper.run_observe',
    'proteus.orbit.wrapper.run_orbit',
    'proteus.outgas.wrapper.calc_target_elemental_inventories',
    'proteus.outgas.wrapper.run_desiccated',
    'proteus.outgas.wrapper.run_outgassing',
    'proteus.star.wrapper.get_new_spectrum',
    'proteus.star.wrapper.scale_spectrum_to_toa',
    'proteus.star.wrapper.update_stellar_mass',
    'proteus.star.wrapper.update_stellar_quantities',
    'proteus.star.wrapper.write_spectrum',
    'proteus.utils.coupler.CreateHelpfileFromDict',
    'proteus.utils.coupler.CreateLockFile',
    'proteus.utils.coupler.ExtendHelpfile',
    'proteus.utils.coupler.PrintCurrentState',
    'proteus.utils.coupler.UpdatePlots',
    'proteus.utils.coupler.WriteHelpfileToCSV',
    'proteus.utils.coupler.print_citation',
    'proteus.utils.coupler.print_header',
    'proteus.utils.coupler.print_module_configuration',
    'proteus.utils.coupler.print_stoptime',
    'proteus.utils.coupler.print_system_configuration',
    'proteus.utils.coupler.remove_excess_files',
    'proteus.utils.coupler.validate_module_versions',
    'proteus.utils.coupler.UpdateStatusfile',
    'proteus.utils.data.download_sufficient_data',
    'proteus.utils.terminate.print_termination_criteria',
]


class _StopAfterMeshRestore(Exception):
    """Sentinel exception to stop start() after the mesh restoration block."""


def _resume_with_patches(p, hf_df):
    """Call p.start(resume=True) with all start() imports mocked.

    Uses ExitStack to avoid Python's nested-block limit.
    Stops at init_star (after mesh restoration).
    """
    with ExitStack() as stack:
        for target in _START_PATCHES:
            stack.enter_context(patch(target))

        stack.enter_context(patch('proteus.interior.wrapper.get_nlevb', return_value=50))
        stack.enter_context(
            patch('proteus.utils.coupler.ReadHelpfileFromCSV', return_value=hf_df)
        )
        stack.enter_context(
            patch('proteus.outgas.wrapper.check_desiccation', return_value=False)
        )
        stack.enter_context(patch('proteus.utils.coupler.ZeroHelpfileRow', return_value={}))

        # Interior_t mock
        mock_interior_t = stack.enter_context(patch('proteus.interior.common.Interior_t'))
        mock_int = MagicMock()
        mock_int.ic = 1
        mock_interior_t.return_value = mock_int

        # Stop execution at init_star (runs after mesh restoration)
        stack.enter_context(
            patch(
                'proteus.star.wrapper.init_star',
                side_effect=_StopAfterMeshRestore,
            )
        )
        stack.enter_context(
            patch(
                'proteus.orbit.wrapper.init_orbit',
                side_effect=_StopAfterMeshRestore,
            )
        )

        with pytest.raises(_StopAfterMeshRestore):
            p.start(resume=True, offline=True)


def _make_hf_df():
    """Minimal helpfile DataFrame for resume tests (>init_loops+1 rows)."""
    return pd.DataFrame(
        {
            'Time': [0.0, 100.0, 200.0, 300.0, 400.0],
            'R_int': [6.371e6] * 5,
            'gravity': [9.81] * 5,
            'T_magma': [3000.0, 2800.0, 2600.0, 2400.0, 2200.0],
            'T_eqm': [255.0] * 5,
            'F_atm': [100.0] * 5,
        }
    )


@pytest.mark.unit
def test_proteus_resume_restores_zalmoxis_mesh(tmp_path):
    """start(resume=True) restores spider_mesh path from output directory."""
    p = _make_proteus_instance(tmp_path)

    data_dir = tmp_path / 'data'
    data_dir.mkdir(exist_ok=True)
    mesh_file = data_dir / 'spider_mesh.dat'
    mesh_file.write_text('# 3 2\n6.371e6 0.0 3500.0 -9.81\n')
    prev_file = data_dir / 'spider_mesh.dat.prev'
    prev_file.write_text('# 3 2\n6.371e6 0.0 3500.0 -9.81\n')

    _resume_with_patches(p, _make_hf_df())

    assert p.directories.get('spider_mesh') == str(mesh_file)
    assert p.directories.get('spider_mesh_prev') == str(prev_file)
    assert p.directories.get('mesh_shift_active') is False
    assert p.directories.get('mesh_convergence_steps') == 0


@pytest.mark.unit
def test_proteus_resume_no_mesh_file(tmp_path):
    """start(resume=True) skips mesh restoration when mesh file absent."""
    p = _make_proteus_instance(tmp_path)
    (tmp_path / 'data').mkdir(exist_ok=True)

    _resume_with_patches(p, _make_hf_df())

    assert 'spider_mesh' not in p.directories


@pytest.mark.unit
def test_proteus_resume_mesh_no_prev(tmp_path):
    """start(resume=True) restores mesh but skips .prev when absent."""
    p = _make_proteus_instance(tmp_path)

    data_dir = tmp_path / 'data'
    data_dir.mkdir(exist_ok=True)
    mesh_file = data_dir / 'spider_mesh.dat'
    mesh_file.write_text('# 3 2\n6.371e6 0.0 3500.0 -9.81\n')

    _resume_with_patches(p, _make_hf_df())

    assert p.directories.get('spider_mesh') == str(mesh_file)
    assert 'spider_mesh_prev' not in p.directories
    assert p.directories.get('mesh_shift_active') is False
