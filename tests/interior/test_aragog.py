"""
Unit tests for proteus.interior.aragog module — Zalmoxis integration paths.

Tests the Zalmoxis-specific branches in AragogRunner.setup_solver() that set
inner_radius from zalmoxis_solver and configure temperature-dependent initial
conditions.

Testing standards and documentation:
- docs/test_infrastructure.md: Test infrastructure overview
- docs/test_categorization.md: Test marker definitions
- docs/test_building.md: Best practices for test construction

Functions tested:
- AragogRunner.setup_solver(): Zalmoxis branches for inner_radius, EOS fallback
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_aragog_config(*, struct_module='self', mantle_eos='Seager2007:silicate'):
    """Create a mock config for AragogRunner.setup_solver tests."""
    config = MagicMock()
    config.struct.module = struct_module
    config.struct.corefrac = 0.55
    config.struct.zalmoxis.mantle_eos = mantle_eos
    config.struct.core_density = 12500.0
    config.struct.core_heatcap = 880.0
    config.interior.num_levels = 20
    config.interior.aragog.bulk_modulus = 200e9
    config.interior.aragog.mass_coordinates = False
    config.interior.trans_conduction = True
    config.interior.trans_convection = True
    config.interior.trans_grav_sep = False
    config.interior.aragog.mixing = True
    config.interior.aragog.dilatation = False
    config.interior.heat_radiogenic = False
    config.interior.heat_tidal = False
    config.interior.aragog.initial_condition = 1
    config.interior.aragog.init_file = 'dummy.txt'
    config.interior.aragog.tsurf_init = 4000.0
    config.interior.aragog.basal_temperature = 5000.0
    config.interior.num_tolerance = 1e-4
    config.interior.aragog.tsurf_poststep_change = 100.0
    config.interior.aragog.event_triggering = True
    config.interior.aragog.inner_boundary_condition = 1
    config.interior.aragog.inner_boundary_value = 5000.0
    config.interior.params.out.logging = 'WARNING'
    config.struct.eos_dir = 'WolfBower2018_MgSiO3'
    config.struct.melting_dir = 'Wolf_Bower+2018'
    return config


@pytest.mark.unit
def test_setup_solver_zalmoxis_inner_radius(tmp_path):
    """setup_solver uses zalmoxis_solver for inner_radius when struct.module='zalmoxis'."""
    from proteus.interior.aragog import AragogRunner

    outdir = str(tmp_path)
    config = _make_aragog_config(struct_module='zalmoxis')

    hf_row = {
        'R_int': 6.371e6,
        'gravity': 9.81,
        'T_magma': 3000.0,
        'T_eqm': 255.0,
        'F_atm': 100.0,
    }
    interior_o = MagicMock()
    interior_o.tides = np.zeros(20)

    # Create EOS dir
    eos_dir = (
        tmp_path / 'interior_lookup_tables' / 'EOS' / 'dynamic' / 'WolfBower2018_MgSiO3' / 'P-T'
    )
    eos_dir.mkdir(parents=True)
    (eos_dir / 'heat_capacity_melt.dat').write_text('dummy')
    mc_dir = tmp_path / 'interior_lookup_tables' / 'Melting_curves'
    mc_dir.mkdir(parents=True)

    with (
        patch(
            'proteus.interior.zalmoxis.zalmoxis_solver',
            return_value=(3.48e6, None),
        ) as mock_zal,
        patch('proteus.interior.aragog.FWL_DATA_DIR', tmp_path),
        patch('proteus.interior.aragog.Parameters'),
        patch('proteus.interior.aragog.Solver'),
    ):
        AragogRunner.setup_solver(config, hf_row, interior_o, outdir)

    mock_zal.assert_called_once()


@pytest.mark.unit
def test_setup_solver_zalmoxis_wolfbower_temp(tmp_path):
    """setup_solver uses Zalmoxis T-profile for WolfBower2018 EOS (initial_condition=2)."""
    from proteus.interior.aragog import AragogRunner

    outdir = str(tmp_path)
    config = _make_aragog_config(struct_module='zalmoxis', mantle_eos='WolfBower2018:MgSiO3')

    hf_row = {
        'R_int': 6.371e6,
        'gravity': 9.81,
        'T_magma': 3000.0,
        'T_eqm': 255.0,
        'F_atm': 100.0,
    }
    interior_o = MagicMock()
    interior_o.tides = np.zeros(20)

    eos_dir = (
        tmp_path / 'interior_lookup_tables' / 'EOS' / 'dynamic' / 'WolfBower2018_MgSiO3' / 'P-T'
    )
    eos_dir.mkdir(parents=True)
    (eos_dir / 'heat_capacity_melt.dat').write_text('dummy')
    mc_dir = tmp_path / 'interior_lookup_tables' / 'Melting_curves'
    mc_dir.mkdir(parents=True)

    with (
        patch(
            'proteus.interior.zalmoxis.zalmoxis_solver',
            return_value=(3.48e6, None),
        ),
        patch('proteus.interior.aragog.FWL_DATA_DIR', tmp_path),
        patch('proteus.interior.aragog.Parameters'),
        patch('proteus.interior.aragog.Solver'),
        patch('proteus.interior.aragog._InitialConditionParameters') as mock_ic,
    ):
        AragogRunner.setup_solver(config, hf_row, interior_o, outdir)

    # WolfBower2018 should set initial_condition=2 with zalmoxis_output_temp.txt
    assert mock_ic.called
    call_kwargs = mock_ic.call_args[1]
    assert call_kwargs['initial_condition'] == 2
    assert 'zalmoxis_output_temp.txt' in call_kwargs['init_file']


@pytest.mark.unit
def test_setup_solver_eos_fallback(tmp_path):
    """setup_solver falls back to legacy EOS path when unified path is missing."""
    from proteus.interior.aragog import AragogRunner

    outdir = str(tmp_path)
    config = _make_aragog_config(struct_module='self')

    hf_row = {
        'R_int': 6.371e6,
        'gravity': 9.81,
        'T_magma': 3000.0,
        'T_eqm': 255.0,
        'F_atm': 100.0,
    }
    interior_o = MagicMock()
    interior_o.tides = np.zeros(20)

    # Only create legacy path, NOT unified path
    legacy_dir = (
        tmp_path
        / 'interior_lookup_tables'
        / '1TPa-dK09-elec-free'
        / 'MgSiO3_Wolf_Bower_2018_1TPa'
    )
    legacy_dir.mkdir(parents=True)
    (legacy_dir / 'heat_capacity_melt.dat').write_text('dummy')
    mc_dir = tmp_path / 'interior_lookup_tables' / 'Melting_curves'
    mc_dir.mkdir(parents=True)

    with (
        patch('proteus.interior.aragog.FWL_DATA_DIR', tmp_path),
        patch('proteus.interior.aragog.Parameters'),
        patch('proteus.interior.aragog.Solver') as mock_solver,
    ):
        AragogRunner.setup_solver(config, hf_row, interior_o, outdir)

    assert mock_solver.called


@pytest.mark.unit
def test_setup_solver_eos_not_found(tmp_path):
    """setup_solver raises FileNotFoundError when EOS data is missing."""
    from proteus.interior.aragog import AragogRunner

    outdir = str(tmp_path)
    config = _make_aragog_config(struct_module='self')
    config.struct.eos_dir = 'NonexistentEOS'

    hf_row = {
        'R_int': 6.371e6,
        'gravity': 9.81,
        'T_magma': 3000.0,
        'T_eqm': 255.0,
        'F_atm': 100.0,
    }
    interior_o = MagicMock()
    interior_o.tides = np.zeros(20)

    with (
        patch('proteus.interior.aragog.FWL_DATA_DIR', tmp_path),
        pytest.raises(FileNotFoundError, match='Aragog lookup data not found'),
    ):
        AragogRunner.setup_solver(config, hf_row, interior_o, outdir)
