"""
Unit tests for proteus.atmos_chem.vulcan module

Tests for run_vulcan_offline() and run_vulcan_online() functions which interface
with the VULCAN atmospheric chemistry solver.

Physics tested:
- Module guard: VULCAN requires AGNI atmosphere module
- Network selection: CHO, NCHO, SNCHO chemical networks
- Per-snapshot file naming in online mode (vulcan_{year}.pkl/.csv)
- One-time network compilation optimization in online mode (_made attribute)
- Output file parsing and CSV writing

Related documentation:
- docs/test_infrastructure.md: Testing standards and structure
- docs/test_categorization.md: Test classification criteria
- docs/test_building.md: Best practices for test implementation

Mocking strategy:
- Mock the entire `vulcan` package (not available in CI)
- Mock file I/O (np.loadtxt, glob, pickle, read_atmosphere_data)
- Mock os.makedirs, shutil.rmtree to avoid filesystem side effects
"""

from __future__ import annotations

import importlib
import pickle
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock the third-party `vulcan` package (not available in CI)
mock_vulcan_package = MagicMock()
mock_vulcan_package.__file__ = '/mock/vulcan/__init__.py'
mock_vulcan_package.__version__ = '0.0.0-test'
mock_vulcan_package.paths.THERMO_DIR = '/mock/vulcan/thermo'
sys.modules['vulcan'] = mock_vulcan_package

# Ensure we import the REAL proteus.atmos_chem.vulcan module, not a mock
# (test_atmos_chem.py may have placed a MagicMock in sys.modules)
_saved_mock = sys.modules.pop('proteus.atmos_chem.vulcan', None)
import proteus.atmos_chem.vulcan as _vulcan_mod  # noqa: E402

importlib.reload(_vulcan_mod)  # Force re-import with the real vulcan mock

from proteus.atmos_chem.vulcan import (  # noqa: E402
    run_vulcan_offline,
    run_vulcan_online,
)


def _make_mock_config(*, network='CHO', photo_on=True, atmos_module='agni'):
    """Helper to create a realistic mock config for VULCAN tests.

    Parameters set to physically valid defaults:
    - network='CHO': Carbon-Hydrogen-Oxygen chemistry (simplest network)
    - photo_on=True: Photochemistry enabled
    - atmos_module='agni': Required AGNI atmosphere module
    """
    config = MagicMock()
    config.atmos_clim.module = atmos_module
    config.atmos_chem.vulcan.clip_fl = 1e-20
    config.atmos_chem.vulcan.clip_vmr = 1e-10
    config.atmos_chem.vulcan.make_funs = True
    config.atmos_chem.vulcan.ini_mix = 'profile'
    config.atmos_chem.vulcan.fix_surf = False
    config.atmos_chem.vulcan.network = network
    config.atmos_chem.vulcan.save_frames = False
    config.atmos_chem.vulcan.yconv_cri = 0.05
    config.atmos_chem.vulcan.slope_cri = 1e-4
    config.atmos_chem.photo_on = photo_on
    config.atmos_chem.moldiff_on = True
    config.atmos_chem.updraft_const = 0.0
    config.atmos_chem.Kzz_on = True
    config.atmos_chem.Kzz_const = None
    config.orbit.zenith_angle = 48.0
    config.orbit.s0_factor = 0.5
    return config


def _make_mock_hf_row(*, year=1000.0):
    """Helper to create a realistic helpfile row.

    Physical values represent a young rocky planet:
    - T=1000yr: Early magma ocean phase
    - R_star=6.96e8 m: Solar radius
    - separation=1.496e11 m: 1 AU
    """
    hf_row = {
        'Time': year,
        'R_star': 6.96e8,
        'separation': 1.496e11,
        'R_int': 6.371e6,
        'gravity': 9.81,
    }
    # Add element mass entries for all elements in element_list
    from proteus.utils.constants import element_list, vol_list

    for e in element_list:
        hf_row[e + '_kg_atm'] = 1e15

    # Add VMR entries for all volatiles
    for gas in vol_list:
        hf_row[gas + '_vmr'] = 1e-4
    return hf_row


def _make_mock_atmos():
    """Helper to create mock atmosphere data from read_atmosphere_data.

    Physical structure: 5-level atmosphere from surface (1e5 Pa) to TOA (1 Pa).
    Temperature decreases with altitude (adiabatic-like profile).
    """
    nlevels = 5
    atmos = {
        'p': np.logspace(5, 0, nlevels),
        'pl': np.logspace(5, 0, nlevels),
        'tmpl': np.linspace(500.0, 200.0, nlevels),
        'Kzz': np.full(nlevels, 1e6),
    }
    # Add VMR profiles (nlevels-1 values; code appends one extra level)
    for gas in ['H2O', 'CO2', 'H2', 'CO', 'N2']:
        atmos[gas + '_vmr'] = np.full(nlevels - 1, 1e-4)
    return atmos


def _make_mock_vulcan_result():
    """Helper to create mock VULCAN pickle result.

    Structure matches real VULCAN output: atm dict + variable dict with
    species names and mixing ratio matrix.
    """
    nz = 5
    nsp = 3
    return {
        'atm': {
            'Tco': np.linspace(500, 200, nz),
            'pco': np.logspace(6, 1, nz),
            'zco': np.linspace(0, 1e7, nz + 1),
            'Kzz': np.full(nz - 1, 1e6),
        },
        'variable': {
            'species': ['H2', 'H2O', 'CO2'],
            'ymix': np.random.default_rng(42).random((nz, nsp)),
        },
    }


@pytest.mark.unit
def test_run_vulcan_offline_non_agni():
    """
    Test run_vulcan_offline returns False when atmosphere module is not AGNI.

    Physics: VULCAN chemistry requires AGNI radiative-convective model for
    atmospheric TP profiles. Other atmosphere modules are incompatible.
    """
    config = _make_mock_config(atmos_module='janus')
    dirs = {'output': '/tmp/test', 'output/offchem': '/tmp/test/offchem'}
    hf_row = _make_mock_hf_row()

    result = run_vulcan_offline(dirs, config, hf_row)

    assert result is False


@pytest.mark.unit
def test_run_vulcan_online_non_agni():
    """
    Test run_vulcan_online returns False when atmosphere module is not AGNI.

    Physics: Same constraint as offline — VULCAN needs AGNI TP profiles.
    """
    config = _make_mock_config(atmos_module='janus')
    dirs = {'output': '/tmp/test', 'output/offchem': '/tmp/test/offchem'}
    hf_row = _make_mock_hf_row()

    result = run_vulcan_online(dirs, config, hf_row)

    assert result is False


@pytest.mark.unit
@patch('proteus.atmos_chem.vulcan.read_atmosphere_data')
@patch('proteus.atmos_chem.vulcan.glob.glob')
@patch('proteus.atmos_chem.vulcan.np.loadtxt')
@patch('proteus.atmos_chem.vulcan.np.savetxt')
@patch('proteus.atmos_chem.vulcan.os.makedirs')
def test_run_vulcan_online_per_snapshot_filenames(
    mock_makedirs,
    mock_savetxt,
    mock_loadtxt,
    mock_glob,
    mock_read_atmos,
    tmp_path,
):
    """
    Test online mode generates per-snapshot filenames (vulcan_{year}.pkl/.csv).

    Physics: Online mode writes separate output files per snapshot so each
    timestep's chemistry result is preserved, unlike offline mode which
    overwrites a single file.
    """
    config = _make_mock_config()
    year = 5000.0
    hf_row = _make_mock_hf_row(year=year)

    offchem_dir = tmp_path / 'offchem'
    offchem_dir.mkdir(parents=True, exist_ok=True)
    dirs = {
        'output': str(tmp_path),
        'output/offchem': str(offchem_dir),
    }

    # Mock atmosphere data
    mock_read_atmos.return_value = [_make_mock_atmos()]

    # Mock stellar flux file discovery
    sflux_file = tmp_path / 'data' / '5000.sflux'
    (tmp_path / 'data').mkdir(parents=True, exist_ok=True)
    mock_glob.return_value = [str(sflux_file)]

    # Mock stellar spectrum data (2 rows: wavelength, flux)
    # Shape Nx2: rows are data points (wavelength, flux), .T gives 2xN
    mock_loadtxt.return_value = np.array([[200.0, 1e-5], [300.0, 1e-4], [400.0, 1e-3]])

    # Mock VULCAN Config and solver
    mock_vulcan_package.Config.return_value = MagicMock()
    mock_vulcan_package.main.return_value = None
    mock_vulcan_package.make_all.return_value = None

    # Create the expected per-snapshot result pickle
    result_file = offchem_dir / f'vulcan_{int(year)}.pkl'
    vulcan_result = _make_mock_vulcan_result()
    with open(result_file, 'wb') as f:
        pickle.dump(vulcan_result, f)

    # Reset the _made attribute if set by previous tests
    if hasattr(run_vulcan_online, '_made'):
        del run_vulcan_online._made

    result = run_vulcan_online(dirs, config, hf_row)

    assert result is True

    # Verify per-snapshot CSV was written
    csv_calls = [call for call in mock_savetxt.call_args_list if 'vulcan_5000.csv' in str(call)]
    assert len(csv_calls) == 1


@pytest.mark.unit
@patch('proteus.atmos_chem.vulcan.read_atmosphere_data')
@patch('proteus.atmos_chem.vulcan.glob.glob')
@patch('proteus.atmos_chem.vulcan.np.loadtxt')
@patch('proteus.atmos_chem.vulcan.np.savetxt')
@patch('proteus.atmos_chem.vulcan.os.makedirs')
def test_run_vulcan_online_make_funs_once(
    mock_makedirs,
    mock_savetxt,
    mock_loadtxt,
    mock_glob,
    mock_read_atmos,
    tmp_path,
):
    """
    Test online mode only compiles chemical network once (_made attribute).

    Physics: VULCAN's make_chem_funs is expensive (compiles reaction network
    into numerical functions). In online mode this only needs to happen once
    since the network doesn't change between snapshots.
    """
    config = _make_mock_config()
    config.atmos_chem.vulcan.make_funs = True
    hf_row = _make_mock_hf_row(year=100.0)

    offchem_dir = tmp_path / 'offchem'
    offchem_dir.mkdir(parents=True, exist_ok=True)
    dirs = {
        'output': str(tmp_path),
        'output/offchem': str(offchem_dir),
    }

    mock_read_atmos.return_value = [_make_mock_atmos()]

    sflux_file = tmp_path / 'data' / '100.sflux'
    (tmp_path / 'data').mkdir(parents=True, exist_ok=True)
    mock_glob.return_value = [str(sflux_file)]
    # Shape Nx2: rows are data points (wavelength, flux), .T gives 2xN
    mock_loadtxt.return_value = np.array([[200.0, 1e-5], [300.0, 1e-4], [400.0, 1e-3]])

    mock_vulcan_package.Config.return_value = MagicMock()
    mock_vulcan_package.main.return_value = None
    mock_vulcan_package.make_all.reset_mock()

    # Create result pickle for first call
    result_file = offchem_dir / 'vulcan_100.pkl'
    vulcan_result = _make_mock_vulcan_result()
    with open(result_file, 'wb') as f:
        pickle.dump(vulcan_result, f)

    # Reset _made to simulate first call
    if hasattr(run_vulcan_online, '_made'):
        del run_vulcan_online._made

    # First call — should compile network
    run_vulcan_online(dirs, config, hf_row)
    assert mock_vulcan_package.make_all.call_count == 1
    assert hasattr(run_vulcan_online, '_made')

    # Second call — should NOT recompile
    mock_vulcan_package.make_all.reset_mock()

    hf_row2 = _make_mock_hf_row(year=200.0)
    result_file2 = offchem_dir / 'vulcan_200.pkl'
    with open(result_file2, 'wb') as f:
        pickle.dump(vulcan_result, f)

    run_vulcan_online(dirs, config, hf_row2)
    mock_vulcan_package.make_all.assert_not_called()

    # Cleanup _made for other tests
    if hasattr(run_vulcan_online, '_made'):
        del run_vulcan_online._made


@pytest.mark.unit
@patch('proteus.atmos_chem.vulcan.read_atmosphere_data')
@patch('proteus.atmos_chem.vulcan.glob.glob')
@patch('proteus.atmos_chem.vulcan.np.loadtxt')
@patch('proteus.atmos_chem.vulcan.np.savetxt')
@patch('proteus.atmos_chem.vulcan.os.makedirs')
def test_run_vulcan_online_missing_result_file(
    mock_makedirs,
    mock_savetxt,
    mock_loadtxt,
    mock_glob,
    mock_read_atmos,
    tmp_path,
):
    """
    Test online mode returns False when VULCAN output file is missing.

    Physics: VULCAN solver may fail to converge or crash, producing no
    output pickle. The function should detect this and return False.
    """
    config = _make_mock_config()
    config.atmos_chem.vulcan.make_funs = False
    hf_row = _make_mock_hf_row(year=300.0)

    offchem_dir = tmp_path / 'offchem'
    offchem_dir.mkdir(parents=True, exist_ok=True)
    dirs = {
        'output': str(tmp_path),
        'output/offchem': str(offchem_dir),
    }

    mock_read_atmos.return_value = [_make_mock_atmos()]
    mock_glob.return_value = [str(tmp_path / 'data' / '300.sflux')]
    (tmp_path / 'data').mkdir(parents=True, exist_ok=True)
    # Shape Nx2: rows are data points (wavelength, flux), .T gives 2xN
    mock_loadtxt.return_value = np.array([[200.0, 1e-5], [300.0, 1e-4], [400.0, 1e-3]])

    mock_vulcan_package.Config.return_value = MagicMock()
    mock_vulcan_package.main.return_value = None

    if hasattr(run_vulcan_online, '_made'):
        del run_vulcan_online._made

    # Do NOT create the result pickle — simulates VULCAN failure
    result = run_vulcan_online(dirs, config, hf_row)

    assert result is False


@pytest.mark.unit
@patch('proteus.atmos_chem.vulcan.read_atmosphere_data')
@patch('proteus.atmos_chem.vulcan.glob.glob')
@patch('proteus.atmos_chem.vulcan.np.loadtxt')
@patch('proteus.atmos_chem.vulcan.np.savetxt')
@patch('proteus.atmos_chem.vulcan.shutil.rmtree')
@patch('proteus.atmos_chem.vulcan.os.makedirs')
def test_run_vulcan_offline_unrecognised_network(
    mock_makedirs,
    mock_rmtree,
    mock_savetxt,
    mock_loadtxt,
    mock_glob,
    mock_read_atmos,
    tmp_path,
):
    """
    Test offline mode returns False for unrecognised chemical network.

    Physics: Only CHO, NCHO, and SNCHO networks are implemented. An invalid
    network name should be caught before attempting to run VULCAN.
    """
    config = _make_mock_config(network='INVALID')
    hf_row = _make_mock_hf_row()

    offchem_dir = tmp_path / 'offchem'
    offchem_dir.mkdir(parents=True, exist_ok=True)
    dirs = {
        'output': str(tmp_path),
        'output/offchem': str(offchem_dir),
    }

    mock_read_atmos.return_value = [_make_mock_atmos()]
    mock_glob.return_value = [str(tmp_path / 'data' / '1000.sflux')]
    (tmp_path / 'data').mkdir(parents=True, exist_ok=True)
    # Shape Nx2: rows are data points (wavelength, flux), .T gives 2xN
    mock_loadtxt.return_value = np.array([[200.0, 1e-5], [300.0, 1e-4], [400.0, 1e-3]])

    mock_vulcan_package.Config.return_value = MagicMock()

    result = run_vulcan_offline(dirs, config, hf_row)

    assert result is False


@pytest.mark.unit
@pytest.mark.parametrize(
    'network,expected_elements',
    [
        ('CHO', ['H', 'O', 'C']),
        ('NCHO', ['H', 'O', 'C', 'N']),
        ('SNCHO', ['H', 'O', 'C', 'N', 'S']),
    ],
)
@patch('proteus.atmos_chem.vulcan.read_atmosphere_data')
@patch('proteus.atmos_chem.vulcan.glob.glob')
@patch('proteus.atmos_chem.vulcan.np.loadtxt')
@patch('proteus.atmos_chem.vulcan.np.savetxt')
@patch('proteus.atmos_chem.vulcan.os.makedirs')
def test_run_vulcan_online_network_selection(
    mock_makedirs,
    mock_savetxt,
    mock_loadtxt,
    mock_glob,
    mock_read_atmos,
    network,
    expected_elements,
    tmp_path,
):
    """
    Test online mode correctly selects chemical network and element lists.

    Physics: Different networks include different chemical elements and species.
    CHO = Carbon-Hydrogen-Oxygen only; NCHO adds Nitrogen; SNCHO adds Sulfur.
    The element list determines which reactions are included in the solver.
    """
    config = _make_mock_config(network=network)
    config.atmos_chem.vulcan.make_funs = False
    hf_row = _make_mock_hf_row(year=400.0)

    offchem_dir = tmp_path / 'offchem'
    offchem_dir.mkdir(parents=True, exist_ok=True)
    dirs = {
        'output': str(tmp_path),
        'output/offchem': str(offchem_dir),
    }

    mock_read_atmos.return_value = [_make_mock_atmos()]
    mock_glob.return_value = [str(tmp_path / 'data' / '400.sflux')]
    (tmp_path / 'data').mkdir(parents=True, exist_ok=True)
    # Shape Nx2: rows are data points (wavelength, flux), .T gives 2xN
    mock_loadtxt.return_value = np.array([[200.0, 1e-5], [300.0, 1e-4], [400.0, 1e-3]])

    mock_vcfg = MagicMock()
    mock_vulcan_package.Config.return_value = mock_vcfg
    mock_vulcan_package.main.return_value = None

    # Create result pickle
    result_file = offchem_dir / 'vulcan_400.pkl'
    vulcan_result = _make_mock_vulcan_result()
    with open(result_file, 'wb') as f:
        pickle.dump(vulcan_result, f)

    if hasattr(run_vulcan_online, '_made'):
        del run_vulcan_online._made

    result = run_vulcan_online(dirs, config, hf_row)

    assert result is True
    # Verify the atom_list was set correctly for this network
    mock_vcfg.atom_list = expected_elements


@pytest.mark.unit
@patch('proteus.atmos_chem.vulcan.read_atmosphere_data')
@patch('proteus.atmos_chem.vulcan.glob.glob')
@patch('proteus.atmos_chem.vulcan.np.loadtxt')
@patch('proteus.atmos_chem.vulcan.np.savetxt')
@patch('proteus.atmos_chem.vulcan.os.makedirs')
def test_run_vulcan_online_uses_exist_ok(
    mock_makedirs,
    mock_savetxt,
    mock_loadtxt,
    mock_glob,
    mock_read_atmos,
    tmp_path,
):
    """
    Test online mode uses exist_ok=True for directory creation.

    Physics: In online mode, the offchem directory is created once and reused
    across snapshots, unlike offline mode which clears it each time.
    """
    config = _make_mock_config()
    config.atmos_chem.vulcan.make_funs = False
    hf_row = _make_mock_hf_row(year=600.0)

    offchem_dir = tmp_path / 'offchem'
    offchem_dir.mkdir(parents=True, exist_ok=True)
    dirs = {
        'output': str(tmp_path),
        'output/offchem': str(offchem_dir),
    }

    mock_read_atmos.return_value = [_make_mock_atmos()]
    mock_glob.return_value = [str(tmp_path / 'data' / '600.sflux')]
    (tmp_path / 'data').mkdir(parents=True, exist_ok=True)
    # Shape Nx2: rows are data points (wavelength, flux), .T gives 2xN
    mock_loadtxt.return_value = np.array([[200.0, 1e-5], [300.0, 1e-4], [400.0, 1e-3]])

    mock_vulcan_package.Config.return_value = MagicMock()
    mock_vulcan_package.main.return_value = None

    result_file = offchem_dir / 'vulcan_600.pkl'
    vulcan_result = _make_mock_vulcan_result()
    with open(result_file, 'wb') as f:
        pickle.dump(vulcan_result, f)

    if hasattr(run_vulcan_online, '_made'):
        del run_vulcan_online._made

    run_vulcan_online(dirs, config, hf_row)

    # Verify makedirs was called with exist_ok=True
    mock_makedirs.assert_called_with(str(offchem_dir), exist_ok=True)
