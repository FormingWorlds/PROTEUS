"""
Unit tests for proteus.atmos_chem.dummy module.

Parameterized atmospheric chemistry without a kinetics solver. Generates
vertical mixing ratio profiles for ~17 species with altitude-dependent
structure (cold trap, photolysis layer, well-mixed background).

Physics tested:
- Well-mixed parent species below photolysis level
- H2O cold trap via Clausius-Clapeyron
- Photolysis product increase toward TOA
- VMR normalization (sum to 1 at each level)
- CSV output format matches VULCAN convention
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from proteus.atmos_chem.dummy import _ALL_SPECIES, _build_profiles, run_dummy_chem


def _make_hf_row(
    T_magma=3000.0,
    P_surf=100.0,
    gravity=9.81,
    H2O_vmr=0.9,
    CO2_vmr=0.08,
    N2_vmr=0.01,
    H2_vmr=0.005,
    CH4_vmr=0.005,
):
    """Build a minimal hf_row for dummy chemistry tests."""
    hf_row = {
        'T_magma': T_magma,
        'T_surf': T_magma,
        'P_surf': P_surf,
        'gravity': gravity,
        'atm_kg_per_mol': 0.018,
        'Time': 1e6,
        'H2O_vmr': H2O_vmr,
        'CO2_vmr': CO2_vmr,
        'N2_vmr': N2_vmr,
        'H2_vmr': H2_vmr,
        'CO_vmr': 0.0,
        'CH4_vmr': CH4_vmr,
        'SO2_vmr': 0.0,
        'S2_vmr': 0.0,
        'H2S_vmr': 0.0,
        'NH3_vmr': 0.0,
        'O2_vmr': 0.0,
    }
    return hf_row


def _make_config():
    config = MagicMock()
    config.atmos_clim.p_top = 1e-6
    return config


@pytest.mark.unit
def test_dummy_chem_profile_shape():
    """Output has correct number of levels and species."""
    hf_row = _make_hf_row()
    config = _make_config()
    result = _build_profiles(hf_row, config, num_levels=50)

    assert result['tmp'].shape == (50,)
    assert result['p'].shape == (50,)
    assert result['z'].shape == (50,)
    assert result['Kzz'].shape == (50,)
    for sp in _ALL_SPECIES:
        assert sp in result, f'Missing species {sp}'
        assert result[sp].shape == (50,)


@pytest.mark.unit
def test_dummy_chem_vmr_normalization():
    """VMRs sum to 1.0 at every level."""
    hf_row = _make_hf_row()
    config = _make_config()
    result = _build_profiles(hf_row, config, num_levels=50)

    total = np.zeros(50)
    for sp in _ALL_SPECIES:
        total += result[sp]

    np.testing.assert_allclose(total, 1.0, atol=1e-10)


@pytest.mark.unit
def test_dummy_chem_h2o_cold_trap():
    """H2O VMR decreases above cold trap (lower T -> lower saturation)."""
    hf_row = _make_hf_row(T_magma=2000.0, P_surf=100.0, H2O_vmr=0.9)
    config = _make_config()
    result = _build_profiles(hf_row, config, num_levels=50)

    h2o = result['H2O']
    # Surface VMR should be close to input
    assert h2o[-1] > 0.5
    # TOA VMR should be much lower (cold trap)
    assert h2o[0] < h2o[-1]


@pytest.mark.unit
def test_dummy_chem_photolysis_products():
    """Photolysis products (O, OH, H) increase toward TOA."""
    hf_row = _make_hf_row()
    config = _make_config()
    result = _build_profiles(hf_row, config, num_levels=50)

    for product in ('O', 'OH', 'H'):
        vmr = result[product]
        # Should be higher at TOA than at surface
        assert vmr[0] > vmr[-1], f'{product} should increase toward TOA'


@pytest.mark.unit
def test_dummy_chem_pressure_ordering():
    """Pressure increases from TOA (index 0) to surface (index -1)."""
    hf_row = _make_hf_row()
    config = _make_config()
    result = _build_profiles(hf_row, config, num_levels=50)

    p = result['p']
    assert p[0] < p[-1], 'TOA pressure should be less than surface'
    assert np.all(np.diff(p) > 0), 'Pressure must increase monotonically'


@pytest.mark.unit
def test_dummy_chem_altitude_ordering():
    """Altitude decreases from TOA (index 0) to surface (index -1)."""
    hf_row = _make_hf_row()
    config = _make_config()
    result = _build_profiles(hf_row, config, num_levels=50)

    z = result['z']
    assert z[0] > z[-1], 'TOA altitude should be greater than surface'


@pytest.mark.unit
def test_dummy_chem_no_negative_vmrs():
    """No species should have negative VMR at any level."""
    hf_row = _make_hf_row()
    config = _make_config()
    result = _build_profiles(hf_row, config, num_levels=50)

    for sp in _ALL_SPECIES:
        assert np.all(result[sp] >= 0), f'{sp} has negative VMR'


@pytest.mark.unit
def test_dummy_chem_zero_atmosphere():
    """Zero surface VMRs should produce zero profiles without crashing."""
    hf_row = _make_hf_row(H2O_vmr=0, CO2_vmr=0, N2_vmr=0, H2_vmr=0, CH4_vmr=0)
    config = _make_config()
    result = _build_profiles(hf_row, config, num_levels=30)

    for sp in _ALL_SPECIES:
        assert np.all(result[sp] == 0.0), f'{sp} should be zero'


@pytest.mark.unit
def test_dummy_chem_writes_csv(tmp_path):
    """run_dummy_chem writes a CSV file in VULCAN-compatible format."""
    hf_row = _make_hf_row()
    config = _make_config()
    dirs = {'output': str(tmp_path)}

    success = run_dummy_chem(dirs, config, hf_row)
    assert success is True

    csv_path = tmp_path / 'offchem' / 'dummy.csv'
    assert csv_path.exists()

    # Read and check structure
    import pandas as pd

    df = pd.read_csv(csv_path, delimiter=r'\s+')
    assert 'tmp' in df.columns
    assert 'p' in df.columns
    assert 'H2O' in df.columns
    assert 'O' in df.columns
    assert len(df) == 50


@pytest.mark.unit
def test_dummy_chem_online_mode(tmp_path):
    """Online mode writes per-snapshot CSV files."""
    hf_row = _make_hf_row()
    hf_row['Time'] = 5000.0
    config = _make_config()
    dirs = {'output': str(tmp_path)}

    success = run_dummy_chem(dirs, config, hf_row, online=True)
    assert success is True

    csv_path = tmp_path / 'offchem' / 'dummy_5000.csv'
    assert csv_path.exists()
