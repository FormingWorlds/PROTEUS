"""
Unit tests for proteus.interior_energetics.common module — Interior_t lookup table loading.

Tests the _load_rho_melt() method which loads SPIDER's P-S density_melt
table with FWL_DATA fallback logic.

Testing standards and documentation:
- docs/test_infrastructure.md: Test infrastructure overview
- docs/test_categorization.md: Test marker definitions
- docs/test_building.md: Best practices for test construction

Functions tested:
- Interior_t._load_rho_melt(): Load density_melt.dat with path fallback
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from proteus.interior_energetics.common import Interior_t


def _make_density_melt_file(filepath, nP=3, nS=4):
    """Create a minimal density_melt.dat file for testing.

    Format: 5-line header with dimensions and scale factors,
    followed by nS*nP lines of (P_nondim, S_nondim, value_nondim).
    """
    head = 5
    P_scale = 1e9
    S_scale = 2600.0
    val_scale = 3000.0
    with open(filepath, 'w') as f:
        f.write(f'# {head} {nP} {nS}\n')
        f.write('# units: Pa J/(kg K) kg/m^3\n')
        f.write('# SPIDER P-S density table\n')
        f.write('# test data\n')
        f.write(f'# {P_scale} {S_scale} {val_scale}\n')
        for j in range(nS):
            for i in range(nP):
                P_nd = float(i) / max(nP - 1, 1)
                S_nd = float(j) / max(nS - 1, 1)
                rho_nd = 1.0 + 0.1 * P_nd + 0.05 * S_nd
                f.write(f'{P_nd} {S_nd} {rho_nd}\n')


@pytest.mark.unit
def test_load_rho_melt_local_path(tmp_path):
    """Loads density_melt.dat from local SPIDER lookup_data path."""
    spider_dir = str(tmp_path / 'spider')
    eos_subdir = os.path.join(spider_dir, 'lookup_data', '1TPa-dK09-elec-free')
    os.makedirs(eos_subdir)
    _make_density_melt_file(os.path.join(eos_subdir, 'density_melt.dat'), nP=3, nS=4)

    interior_o = Interior_t(50)
    # Set FWL_DATA to nonexistent path so fallback triggers
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('FWL_DATA', str(tmp_path / 'nonexistent'))
        interior_o._load_rho_melt(spider_dir, 'WolfBower2018_MgSiO3')

    assert interior_o.lookup_rho_melt is not None
    assert interior_o.lookup_rho_melt.shape == (4, 3, 3)


@pytest.mark.unit
def test_load_rho_melt_fwl_data_path(tmp_path):
    """Loads density_melt.dat from FWL_DATA when available."""
    fwl_data = str(tmp_path / 'fwl_data')
    eos_path = os.path.join(
        fwl_data,
        'interior_lookup_tables',
        'EOS',
        'dynamic',
        'WolfBower2018_MgSiO3',
        'P-S',
    )
    os.makedirs(eos_path)
    _make_density_melt_file(os.path.join(eos_path, 'density_melt.dat'), nP=5, nS=6)

    interior_o = Interior_t(50)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('FWL_DATA', fwl_data)
        interior_o._load_rho_melt('/nonexistent/spider', 'WolfBower2018_MgSiO3')

    assert interior_o.lookup_rho_melt is not None
    assert interior_o.lookup_rho_melt.shape == (6, 5, 3)


@pytest.mark.unit
def test_load_rho_melt_both_missing(tmp_path):
    """Missing density_melt.dat from both paths logs warning, sets None."""
    interior_o = Interior_t(50)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('FWL_DATA', str(tmp_path / 'nonexistent'))
        interior_o._load_rho_melt(str(tmp_path / 'also_nonexistent'), 'SomeEOS')

    # lookup_rho_melt is not set (remains default from __init__)
    assert not hasattr(interior_o, 'lookup_rho_melt') or interior_o.lookup_rho_melt is None


@pytest.mark.unit
def test_load_rho_melt_scaling(tmp_path):
    """Verify that loaded data is correctly scaled by the header scale factors."""
    spider_dir = str(tmp_path / 'spider')
    eos_subdir = os.path.join(spider_dir, 'lookup_data', '1TPa-dK09-elec-free')
    os.makedirs(eos_subdir)
    filepath = os.path.join(eos_subdir, 'density_melt.dat')

    nP, nS = 2, 2
    P_scale, S_scale, val_scale = 1e9, 2600.0, 3000.0
    with open(filepath, 'w') as f:
        f.write(f'# 5 {nP} {nS}\n')
        f.write('# units\n')
        f.write('# info\n')
        f.write('# info\n')
        f.write(f'# {P_scale} {S_scale} {val_scale}\n')
        # Write known values
        f.write('0.1 0.2 0.5\n')
        f.write('0.3 0.2 0.6\n')
        f.write('0.1 0.8 0.7\n')
        f.write('0.3 0.8 0.9\n')

    interior_o = Interior_t(50)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('FWL_DATA', str(tmp_path / 'nonexistent'))
        interior_o._load_rho_melt(spider_dir, 'WolfBower2018_MgSiO3')

    # First entry: P=0.1*1e9, S=0.2*2600, rho=0.5*3000
    np.testing.assert_allclose(interior_o.lookup_rho_melt[0, 0, 0], 0.1 * P_scale, rtol=1e-10)
    np.testing.assert_allclose(interior_o.lookup_rho_melt[0, 0, 1], 0.2 * S_scale, rtol=1e-10)
    np.testing.assert_allclose(interior_o.lookup_rho_melt[0, 0, 2], 0.5 * val_scale, rtol=1e-10)
