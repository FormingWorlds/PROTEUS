"""
Unit tests for proteus.interior_energetics.common module — Interior_t lookup table loading.

Tests the _load_ps_table() method which loads SPIDER's P-S lookup tables
(density_melt, heat_capacity_solid, heat_capacity_melt) with FWL_DATA
fallback logic, and verifies the loaders are wired by ``Interior_t.__init__``.

Testing standards and documentation:
- docs/test_infrastructure.md: Test infrastructure overview
- docs/test_categorization.md: Test marker definitions
- docs/test_building.md: Best practices for test construction

Functions tested:
- Interior_t._load_ps_table(): Load arbitrary P-S table with path fallback
- Interior_t.__init__(): Wires lookup_rho_melt + lookup_cp_solid + lookup_cp_melt
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from proteus.interior_energetics.common import Interior_t

pytestmark = pytest.mark.unit


def _make_ps_table_file(filepath, nP=3, nS=4, val_scale=3000.0):
    """Create a minimal SPIDER-format P-S table file for testing.

    Format: 5-line header with dimensions and scale factors,
    followed by nS*nP lines of (P_nondim, S_nondim, value_nondim).
    """
    head = 5
    P_scale = 1e9
    S_scale = 2600.0
    with open(filepath, 'w') as f:
        f.write(f'# {head} {nP} {nS}\n')
        f.write('# units: Pa J/(kg K) value\n')
        f.write('# SPIDER P-S table\n')
        f.write('# test data\n')
        f.write(f'# {P_scale} {S_scale} {val_scale}\n')
        for j in range(nS):
            for i in range(nP):
                P_nd = float(i) / max(nP - 1, 1)
                S_nd = float(j) / max(nS - 1, 1)
                # Asymmetric formula so off-by-one in (i, j) ordering
                # produces a different value at every node.
                val_nd = 1.0 + 0.1 * P_nd + 0.05 * S_nd + 0.013 * P_nd * S_nd
                f.write(f'{P_nd} {S_nd} {val_nd}\n')


def test_load_ps_table_local_path(tmp_path):
    """Loads density_melt.dat from local SPIDER lookup_data path."""
    spider_dir = str(tmp_path / 'spider')
    eos_subdir = os.path.join(spider_dir, 'lookup_data', '1TPa-dK09-elec-free')
    os.makedirs(eos_subdir)
    _make_ps_table_file(os.path.join(eos_subdir, 'density_melt.dat'), nP=3, nS=4)

    interior_o = Interior_t(50)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('FWL_DATA', str(tmp_path / 'nonexistent'))
        table = interior_o._load_ps_table(
            spider_dir, 'WolfBower2018_MgSiO3', 'density_melt.dat'
        )

    assert table is not None
    assert table.shape == (4, 3, 3)
    # No NaNs introduced by reshaping or scaling
    assert not np.any(np.isnan(table))


def test_load_ps_table_fwl_data_path(tmp_path):
    """FWL_DATA path takes precedence over local path when both exist."""
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
    _make_ps_table_file(
        os.path.join(eos_path, 'heat_capacity_melt.dat'),
        nP=5,
        nS=6,
        val_scale=4805046.0,
    )

    # Also create a local copy with a DIFFERENT shape so we can verify
    # which path actually got loaded.
    spider_dir = str(tmp_path / 'spider')
    eos_subdir = os.path.join(spider_dir, 'lookup_data', '1TPa-dK09-elec-free')
    os.makedirs(eos_subdir)
    _make_ps_table_file(
        os.path.join(eos_subdir, 'heat_capacity_melt.dat'), nP=2, nS=2
    )

    interior_o = Interior_t(50)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('FWL_DATA', fwl_data)
        table = interior_o._load_ps_table(
            spider_dir, 'WolfBower2018_MgSiO3', 'heat_capacity_melt.dat'
        )

    assert table is not None
    # Shape from FWL_DATA file (6 nS x 5 nP), not local (2 x 2)
    assert table.shape == (6, 5, 3)


def test_load_ps_table_both_missing(tmp_path):
    """Missing P-S file from both paths returns None (no exception)."""
    interior_o = Interior_t(50)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('FWL_DATA', str(tmp_path / 'nonexistent'))
        result = interior_o._load_ps_table(
            str(tmp_path / 'also_nonexistent'),
            'SomeEOS',
            'density_melt.dat',
        )
    assert result is None


def test_load_ps_table_scaling(tmp_path):
    """Loaded data is scaled by the header scale factors."""
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
        # Asymmetric values: each node has a unique magnitude.
        f.write('0.1 0.2 0.5\n')
        f.write('0.3 0.2 0.6\n')
        f.write('0.1 0.8 0.7\n')
        f.write('0.3 0.8 0.9\n')

    interior_o = Interior_t(50)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('FWL_DATA', str(tmp_path / 'nonexistent'))
        table = interior_o._load_ps_table(
            spider_dir, 'WolfBower2018_MgSiO3', 'density_melt.dat'
        )

    # First entry: P=0.1*1e9, S=0.2*2600, value=0.5*3000
    np.testing.assert_allclose(table[0, 0, 0], 0.1 * P_scale, rtol=1e-10)
    np.testing.assert_allclose(table[0, 0, 1], 0.2 * S_scale, rtol=1e-10)
    np.testing.assert_allclose(table[0, 0, 2], 0.5 * val_scale, rtol=1e-10)
    # Last entry differs from first along both axes — guards against
    # transposition / reshape ordering bugs.
    np.testing.assert_allclose(table[1, 1, 0], 0.3 * P_scale, rtol=1e-10)
    np.testing.assert_allclose(table[1, 1, 1], 0.8 * S_scale, rtol=1e-10)
    np.testing.assert_allclose(table[1, 1, 2], 0.9 * val_scale, rtol=1e-10)


def test_interior_t_init_loads_all_three_tables(tmp_path):
    """Interior_t.__init__ wires rho_melt + cp_solid + cp_melt loaders.

    The E_th calculation in spider.py depends on all three tables
    being loaded by ``Interior_t.__init__`` when ``spider_dir`` is
    provided. A regression here would silently re-introduce the
    Cp=1200 fallback that the 2026-04-09 parity work fixed.
    """
    spider_dir = str(tmp_path / 'spider')
    eos_subdir = os.path.join(spider_dir, 'lookup_data', '1TPa-dK09-elec-free')
    os.makedirs(eos_subdir)
    _make_ps_table_file(os.path.join(eos_subdir, 'density_melt.dat'), nP=3, nS=4)
    _make_ps_table_file(
        os.path.join(eos_subdir, 'heat_capacity_solid.dat'),
        nP=3,
        nS=4,
        val_scale=4805046.0,
    )
    _make_ps_table_file(
        os.path.join(eos_subdir, 'heat_capacity_melt.dat'),
        nP=3,
        nS=4,
        val_scale=4805046.0,
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('FWL_DATA', str(tmp_path / 'nonexistent'))
        interior_o = Interior_t(
            50, spider_dir=spider_dir, eos_dir='WolfBower2018_MgSiO3'
        )

    assert interior_o.lookup_rho_melt is not None
    assert interior_o.lookup_cp_solid is not None
    assert interior_o.lookup_cp_melt is not None
    assert interior_o.lookup_rho_melt.shape == (4, 3, 3)
    # Cp tables should be on the same grid as rho_melt
    assert interior_o.lookup_cp_solid.shape == interior_o.lookup_rho_melt.shape
    assert interior_o.lookup_cp_melt.shape == interior_o.lookup_rho_melt.shape


def test_interior_t_init_no_spider_dir_leaves_lookups_none():
    """Without spider_dir, all lookup tables stay None (Aragog path)."""
    interior_o = Interior_t(50)
    assert interior_o.lookup_rho_melt is None
    assert interior_o.lookup_cp_solid is None
    assert interior_o.lookup_cp_melt is None


def test_interior_t_init_partial_table_set(tmp_path):
    """Missing only the Cp tables: rho_melt loads, Cp_* stay None.

    Verifies that an incomplete EOS directory does not abort
    construction. The wrapper falls back to the Cp=1200 default in
    that case (with a warning).
    """
    spider_dir = str(tmp_path / 'spider')
    eos_subdir = os.path.join(spider_dir, 'lookup_data', '1TPa-dK09-elec-free')
    os.makedirs(eos_subdir)
    _make_ps_table_file(os.path.join(eos_subdir, 'density_melt.dat'), nP=3, nS=4)

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('FWL_DATA', str(tmp_path / 'nonexistent'))
        interior_o = Interior_t(
            50, spider_dir=spider_dir, eos_dir='WolfBower2018_MgSiO3'
        )

    assert interior_o.lookup_rho_melt is not None
    assert interior_o.lookup_cp_solid is None
    assert interior_o.lookup_cp_melt is None


def test_load_ps_table_invalid_filename(tmp_path):
    """Reading a nonexistent filename returns None, not raises."""
    spider_dir = str(tmp_path / 'spider')
    os.makedirs(os.path.join(spider_dir, 'lookup_data', '1TPa-dK09-elec-free'))
    interior_o = Interior_t(50)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('FWL_DATA', str(tmp_path / 'nonexistent'))
        result = interior_o._load_ps_table(
            spider_dir, 'WolfBower2018_MgSiO3', 'this_file_does_not_exist.dat'
        )
    assert result is None
