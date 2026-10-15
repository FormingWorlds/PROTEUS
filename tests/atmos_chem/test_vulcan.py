"""
Unit tests for proteus.atmos_chem.vulcan module

Tests for the run_vulcan() function which interfaces with the VULCAN
atmospheric chemistry solver. The function supports both offline mode
(single post-processing run) and online mode (per-snapshot during simulation).

Physics tested:
- Module guard: VULCAN requires AGNI atmosphere module
- Network selection: CHO, NCHO, SNCHO chemical networks
- Per-snapshot file naming in online mode (vulcan_{year}.pkl/.csv)
- One-time network compilation optimization in online mode (_made attribute)
- Output file parsing and CSV writing
- Directory handling differences between offline (wipe) and online (exist_ok)
- Stellar input closure: star.dat and orbit_radius share the flux-weighted
  distance a (1 - e^2)^(1/4), so VULCAN's own dilution returns the stored
  top-of-atmosphere spectrum on an eccentric orbit
- Config contract: PROTEUS sets every field that `vulcan.Config.__init__`
  defines in the installed VULCAN, and no field VULCAN does not define
- Boundary files: bot_BC_flux_file and top_BC_flux_file are written in every
  run with the surface VMRs and zero fluxes; with a fixed surface the VMRs
  VULCAN reads from bot_BC_flux_file equal the use_fix_sp_bot dict the solver
  applies, and with a free surface no flag makes VULCAN read either file
- Config pass-through: each [atmos_chem.vulcan] solver option reaches the
  VULCAN Config unchanged, and the AGNI options of the RCE solver follow
  [atmos_clim] and [atmos_clim.agni]
- Network filter: a gas with an element outside the network is not fixed at
  the surface, while the background gas is still chosen from all gases
- Condensation: every gas of the network that VULCAN can condense is
  included, with the configured radius and a condensed-phase density
- NetCDF output: VULCAN's NetCDF is moved to data/<time>_chem.nc

Related documentation:
- docs/How-to/testing.md: Running, writing, and marking tests; coverage and CI
- docs/Explanations/test_framework.md: Test tiers, physics invariants, and quality rules

Mocking strategy:
- Mock the entire `vulcan` package (not available in CI); the config
  contract test reads the installed VULCAN's config.py source instead
- Mock file I/O (np.loadtxt, glob, pickle, read_atmosphere_data)
- Mock os.makedirs, shutil.rmtree to avoid filesystem side effects
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import pickle
import sys
from pathlib import Path
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
    network_condensation,
    run_vulcan,
)
from proteus.utils.constants import CHEM_CONDENSATE_DENSITY  # noqa: E402
from proteus.utils.helper import format_subyear_time  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


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
    config.atmos_chem.vulcan.yconv_min = 0.5
    config.atmos_chem.vulcan.st_factor = 0.5
    config.atmos_chem.vulcan.conv_step = 100
    config.atmos_chem.vulcan.use_vm_mol = False
    config.atmos_chem.vulcan.use_vz = True
    config.atmos_chem.vulcan.use_condense = False
    config.atmos_chem.vulcan.use_settling = False
    config.atmos_chem.vulcan.r_p = 5e-3
    config.atmos_chem.vulcan.runtime = 1e22
    config.atmos_chem.vulcan.dttry = 1e-8
    config.atmos_chem.vulcan.dt_min = 1e-10
    config.atmos_chem.vulcan.atol = 0.1
    config.atmos_chem.vulcan.mtol = 1e-22
    config.atmos_chem.vulcan.mtol_conv = 1e-20
    config.atmos_chem.vulcan.flux_atol = 1.0
    config.atmos_chem.vulcan.rtol = 0.9
    config.atmos_chem.vulcan.post_conden_rtol = 0.1
    config.atmos_clim.rayleigh = True
    config.atmos_clim.surf_greyalbedo = 0.1
    config.atmos_clim.num_levels = 50
    config.atmos_clim.agni.solution_atol = 0.5
    config.atmos_clim.agni.solution_rtol = 0.15
    config.atmos_clim.agni.solve_energy = True
    config.atmos_chem.photo_on = photo_on
    config.atmos_chem.moldiff_on = True
    config.atmos_chem.updraft_const = 0.0
    config.atmos_chem.Kzz_on = True
    config.atmos_chem.Kzz_const = None
    config.orbit.zenith_angle = 48.0
    config.orbit.s0_factor = 0.5
    config.params.out.logging = 'INFO'
    config.params.out.plot_fmt = 'png'
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
        'semimajorax': 1.496e11,
        'eccentricity': 0.0,
        'R_int': 6.371e6,
        'gravity': 9.81,
        'T_surf': 1500.0,
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
    Test run_vulcan returns False in offline mode when atmosphere module is not AGNI.

    Physics: VULCAN chemistry requires AGNI radiative-convective model for
    atmospheric TP profiles. Other atmosphere modules are incompatible.
    """
    config = _make_mock_config(atmos_module='janus')
    dirs = {'output': '/tmp/test', 'output/offchem': '/tmp/test/offchem'}
    hf_row = _make_mock_hf_row()

    mock_vulcan_package.run_vulcan.reset_mock()
    result = run_vulcan(dirs, config, hf_row)

    assert result is False
    # Discrimination: the guard must short-circuit BEFORE any VULCAN solver
    # call. A regression that returned False from a later code path would
    # still satisfy the False return but would have dispatched into the
    # mocked vulcan package. Assert vulcan.run_vulcan was not called.
    mock_vulcan_package.run_vulcan.assert_not_called()


@pytest.mark.unit
def test_run_vulcan_online_non_agni():
    """
    Test run_vulcan returns False in online mode when atmosphere module is not AGNI.

    Physics: same constraint as offline. VULCAN needs AGNI TP profiles.
    """
    config = _make_mock_config(atmos_module='janus')
    dirs = {'output': '/tmp/test', 'output/offchem': '/tmp/test/offchem'}
    hf_row = _make_mock_hf_row()

    mock_vulcan_package.run_vulcan.reset_mock()
    result = run_vulcan(dirs, config, hf_row, online=True)

    assert result is False
    # Discrimination: parallel arm to the offline guard. The guard must
    # short-circuit before any VULCAN solver call regardless of mode. A
    # regression that conditioned the guard on `not online` would let the
    # online path through and would invoke vulcan.run_vulcan on a janus config.
    mock_vulcan_package.run_vulcan.assert_not_called()


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
    mock_vulcan_package.run_vulcan.return_value = None
    mock_vulcan_package.make_chem_funs.make_all.return_value = None

    # Create the expected per-snapshot result pickle
    result_file = offchem_dir / f'vulcan_{int(year)}.pkl'
    vulcan_result = _make_mock_vulcan_result()
    with open(result_file, 'wb') as f:
        pickle.dump(vulcan_result, f)

    # Reset the _made attribute if set by previous tests
    if hasattr(run_vulcan, '_made'):
        del run_vulcan._made

    result = run_vulcan(dirs, config, hf_row, online=True)

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
    mock_vulcan_package.run_vulcan.return_value = None
    mock_vulcan_package.make_chem_funs.make_all.reset_mock()

    # Create result pickle for first call
    result_file = offchem_dir / 'vulcan_100.pkl'
    vulcan_result = _make_mock_vulcan_result()
    with open(result_file, 'wb') as f:
        pickle.dump(vulcan_result, f)

    # Reset _made to simulate first call
    if hasattr(run_vulcan, '_made'):
        del run_vulcan._made

    # First call: should compile network
    run_vulcan(dirs, config, hf_row, online=True)
    assert mock_vulcan_package.make_chem_funs.make_all.call_count == 1
    assert hasattr(run_vulcan, '_made')

    # Second call: should NOT recompile
    mock_vulcan_package.make_chem_funs.make_all.reset_mock()

    hf_row2 = _make_mock_hf_row(year=200.0)
    result_file2 = offchem_dir / 'vulcan_200.pkl'
    with open(result_file2, 'wb') as f:
        pickle.dump(vulcan_result, f)

    run_vulcan(dirs, config, hf_row2, online=True)
    mock_vulcan_package.make_chem_funs.make_all.assert_not_called()

    # Cleanup _made for other tests
    if hasattr(run_vulcan, '_made'):
        del run_vulcan._made


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
    mock_vulcan_package.run_vulcan.return_value = None

    if hasattr(run_vulcan, '_made'):
        del run_vulcan._made

    # Do NOT create the result pickle, simulates VULCAN failure
    result = run_vulcan(dirs, config, hf_row, online=True)

    assert result is False
    # Discrimination: the missing pickle is at the per-snapshot path
    # `vulcan_300.pkl`. A regression that swallowed the missing-result
    # path (e.g. returned True with empty output) would not leave the
    # disk in this state. Pin: the expected pickle still must not exist
    # after run_vulcan returns (it was never created).
    expected_pkl = offchem_dir / 'vulcan_300.pkl'
    assert not expected_pkl.exists()


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
    mock_vulcan_package.run_vulcan.reset_mock()

    result = run_vulcan(dirs, config, hf_row)

    assert result is False
    # Discrimination: the unrecognised-network branch must short-circuit
    # BEFORE the solver invocation. A regression that fell through to a
    # default network (e.g. silently coerced 'INVALID' to 'CHO') would
    # have invoked vulcan.run_vulcan and would not return False.
    mock_vulcan_package.run_vulcan.assert_not_called()


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
    mock_vulcan_package.run_vulcan.return_value = None

    # Create result pickle
    result_file = offchem_dir / 'vulcan_400.pkl'
    vulcan_result = _make_mock_vulcan_result()
    with open(result_file, 'wb') as f:
        pickle.dump(vulcan_result, f)

    if hasattr(run_vulcan, '_made'):
        del run_vulcan._made

    result = run_vulcan(dirs, config, hf_row, online=True)

    assert result is True
    # Discrimination: the production code sets `vcfg.atom_list = ele_arr`
    # from the network branch in atmos_chem/vulcan.py. The mock captures
    # the assigned value; pin the exact element list per network so a
    # regression that selected the wrong network branch (or that lost the
    # 'C' element from CHO) is caught here. The original assignment-style
    # check (`mock_vcfg.atom_list = expected_elements`) was a no-op write.
    assert mock_vcfg.atom_list == expected_elements


@pytest.mark.unit
@patch('proteus.atmos_chem.vulcan.read_atmosphere_data')
@patch('proteus.atmos_chem.vulcan.glob.glob')
@patch('proteus.atmos_chem.vulcan.np.loadtxt')
@patch('proteus.atmos_chem.vulcan.np.savetxt')
@patch('proteus.atmos_chem.vulcan.os.makedirs')
def test_run_vulcan_boundary_pressures_use_level_edges(
    mock_makedirs,
    mock_savetxt,
    mock_loadtxt,
    mock_glob,
    mock_read_atmos,
    tmp_path,
):
    """vcfg.P_b and vcfg.P_t read from level edges (atmos['pl']), not centres.

    Physics: ``atmos['p']`` are cell-centre pressures (length nz); ``atmos['pl']``
    are level-edge pressures (length nz+1) that bracket every cell. The
    bottom (surface) and top (TOA) boundary pressures sit on the edges,
    not the centres; using ``p`` truncates the column by a half-cell at
    each end.

    The mock atmosphere here puts ``pl`` one order of magnitude wider
    than ``p`` at each end so the formula difference is unambiguous:
    correct fix uses ``pl[-1] * 10 = 1e7`` barye; the old bug would
    produce ``p[-1] * 10 = 1e6`` barye.
    """
    config = _make_mock_config()
    config.atmos_chem.vulcan.make_funs = False
    hf_row = _make_mock_hf_row(year=700.0)

    offchem_dir = tmp_path / 'offchem'
    offchem_dir.mkdir(parents=True, exist_ok=True)
    dirs = {
        'output': str(tmp_path),
        'output/offchem': str(offchem_dir),
    }

    # p (cell centres) and pl (level edges) have a distinct max and a
    # distinct min so the formula change is unambiguous. Lengths match
    # the lengths the production code already assumes for tmpl/Kzz/p
    # in this test harness; the physically-correct fix only touches
    # which array P_b and P_t are computed from, not which one drives
    # the savetxt rows.
    nlevels = 5
    atmos = {
        'p': np.logspace(5, 1, nlevels),  # 1e5..1e1 Pa (centres)
        'pl': np.logspace(6, 0, nlevels),  # 1e6..1e0 Pa (edges, wider)
        'tmpl': np.linspace(500.0, 200.0, nlevels),
        'Kzz': np.full(nlevels, 1e6),
    }
    for gas in ['H2O', 'CO2', 'H2', 'CO', 'N2']:
        atmos[gas + '_vmr'] = np.full(nlevels - 1, 1e-4)
    mock_read_atmos.return_value = [atmos]

    mock_glob.return_value = [str(tmp_path / 'data' / '700.sflux')]
    (tmp_path / 'data').mkdir(parents=True, exist_ok=True)
    mock_loadtxt.return_value = np.array([[200.0, 1e-5], [300.0, 1e-4], [400.0, 1e-3]])

    mock_vcfg = MagicMock()
    mock_vulcan_package.Config.return_value = mock_vcfg
    mock_vulcan_package.run_vulcan.return_value = None

    result_file = offchem_dir / 'vulcan_700.pkl'
    vulcan_result = _make_mock_vulcan_result()
    with open(result_file, 'wb') as f:
        pickle.dump(vulcan_result, f)

    if hasattr(run_vulcan, '_made'):
        del run_vulcan._made

    run_vulcan(dirs, config, hf_row, online=True)

    # Expected: P_b = max(pl) * 10 = 1e6 * 10 = 1e7 barye;
    #           P_t = min(pl) * 10 = 1e0 * 10 = 1e1 barye.
    # Old bug would have produced 1e6 and 1e2 respectively.
    assert mock_vcfg.P_b == pytest.approx(1.0e7)
    assert mock_vcfg.P_t == pytest.approx(1.0e1)
    # Sanity: not the buggy values from atmos['p'].
    assert mock_vcfg.P_b != pytest.approx(1.0e6)
    assert mock_vcfg.P_t != pytest.approx(1.0e2)


@pytest.mark.unit
@pytest.mark.physics_invariant
@patch('proteus.atmos_chem.vulcan.read_atmosphere_data')
@patch('proteus.atmos_chem.vulcan.glob.glob')
@patch('proteus.atmos_chem.vulcan.np.loadtxt')
@patch('proteus.atmos_chem.vulcan.np.savetxt')
@patch('proteus.atmos_chem.vulcan.os.makedirs')
def test_run_vulcan_star_file_and_orbit_radius_share_the_flux_weighted_distance(
    mock_makedirs,
    mock_savetxt,
    mock_loadtxt,
    mock_glob,
    mock_read_atmos,
    tmp_path,
):
    """On an eccentric orbit star.dat holds the stellar-surface spectrum and
    orbit_radius the distance VULCAN dilutes it by, both at the flux-weighted
    distance a (1 - e^2)^(1/4) the stored spectrum was written at.

    Invariant: VULCAN's top-of-atmosphere flux, star.dat times
    (R_star / orbit_radius)^2, equals the stored spectrum. At e = 0.8 the
    time-averaged separation a (1 + e^2 / 2) is 2.9 times further in flux
    terms, so a star.dat undone with it would be 2.9 times too bright.
    """
    from proteus.utils.constants import AU, R_sun

    config = _make_mock_config()
    config.atmos_chem.vulcan.make_funs = False
    hf_row = _make_mock_hf_row(year=700.0)
    hf_row['semimajorax'] = 0.029 * AU
    hf_row['eccentricity'] = 0.8
    hf_row['separation'] = 0.029 * AU * (1.0 + 0.5 * 0.8**2)
    d_flux = 0.029 * AU * (1.0 - 0.8**2) ** 0.25

    offchem_dir = tmp_path / 'offchem'
    offchem_dir.mkdir(parents=True, exist_ok=True)
    dirs = {'output': str(tmp_path), 'output/offchem': str(offchem_dir)}
    mock_read_atmos.return_value = [_make_mock_atmos()]
    mock_glob.return_value = [str(tmp_path / 'data' / '700.sflux')]
    toa_flux = np.array([1e-5, 1e-4, 1e-3])
    mock_loadtxt.return_value = np.array([[200.0, 300.0, 400.0], toa_flux]).T

    mock_vcfg = MagicMock()
    mock_vulcan_package.Config.return_value = mock_vcfg
    mock_vulcan_package.run_vulcan.return_value = None
    with open(offchem_dir / 'vulcan_700.pkl', 'wb') as f:
        pickle.dump(_make_mock_vulcan_result(), f)
    if hasattr(run_vulcan, '_made'):
        del run_vulcan._made

    run_vulcan(dirs, config, hf_row, online=True)

    star_call = [c for c in mock_savetxt.call_args_list if str(c.args[0]).endswith('star.dat')]
    assert len(star_call) == 1
    star_fl = star_call[0].args[1][:, 1]
    np.testing.assert_allclose(star_fl, toa_flux * (d_flux / hf_row['R_star']) ** 2, rtol=1e-12)
    assert mock_vcfg.orbit_radius == pytest.approx(d_flux / AU, rel=1e-12)
    # Closure: VULCAN's own dilution returns the stored top-of-atmosphere flux.
    r_star_m = mock_vcfg.r_star * R_sun
    np.testing.assert_allclose(
        star_fl * (r_star_m / (mock_vcfg.orbit_radius * AU)) ** 2, toa_flux
    )
    # Guard: undoing the separation would make star.dat 2.9 times brighter.
    wrong = toa_flux * (hf_row['separation'] / hf_row['R_star']) ** 2
    np.testing.assert_allclose(wrong / star_fl, 2.90, rtol=0.01)


@pytest.mark.unit
@patch('proteus.atmos_chem.vulcan.read_atmosphere_data')
@patch('proteus.atmos_chem.vulcan.glob.glob')
@patch('proteus.atmos_chem.vulcan.np.loadtxt')
@patch('proteus.atmos_chem.vulcan.np.savetxt')
@patch('proteus.atmos_chem.vulcan.shutil.rmtree')
@patch('proteus.atmos_chem.vulcan.os.makedirs')
def test_run_vulcan_online_uses_exist_ok(
    mock_makedirs,
    mock_rmtree,
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
    mock_vulcan_package.run_vulcan.return_value = None

    result_file = offchem_dir / 'vulcan_600.pkl'
    vulcan_result = _make_mock_vulcan_result()
    with open(result_file, 'wb') as f:
        pickle.dump(vulcan_result, f)

    if hasattr(run_vulcan, '_made'):
        del run_vulcan._made

    run_vulcan(dirs, config, hf_row, online=True)

    # Verify makedirs was called with exist_ok=True
    mock_makedirs.assert_called_with(str(offchem_dir), exist_ok=True)
    # Discrimination: the online branch must NOT call shutil.rmtree (that
    # is the offline path's wipe-and-recreate behaviour). A regression
    # that took the offline branch in online mode would still satisfy the
    # `exist_ok=True` makedirs call but would also have wiped the directory.
    mock_rmtree.assert_not_called()


@pytest.mark.unit
@patch('proteus.atmos_chem.vulcan.find_latest_atmosphere_time')
@patch('proteus.atmos_chem.vulcan.read_atmosphere_data')
@patch('proteus.atmos_chem.vulcan.shutil.rmtree')
@patch('proteus.atmos_chem.vulcan.os.makedirs')
def test_run_vulcan_no_atmosphere_files_returns_false(
    mock_makedirs, mock_rmtree, mock_read_atmos, mock_latest, tmp_path
):
    """
    With no atmosphere NetCDF available, run_vulcan degrades gracefully.

    Anti-crash contract: the exact-time read returns None and no snapshot
    exists on disk (find_latest returns None), so the function returns False
    rather than raising, and never dispatches into the VULCAN solver.
    """
    config = _make_mock_config()
    hf_row = _make_mock_hf_row(year=1000.0)
    dirs = {'output': str(tmp_path), 'output/offchem': str(tmp_path / 'offchem')}

    mock_read_atmos.return_value = None
    mock_latest.return_value = None
    mock_vulcan_package.run_vulcan.reset_mock()

    result = run_vulcan(dirs, config, hf_row)

    assert result is False
    # The fallback was attempted (exact-time read failed) ...
    mock_latest.assert_called_once()
    # ... and the solver was never reached.
    mock_vulcan_package.run_vulcan.assert_not_called()


@pytest.mark.unit
@patch('proteus.atmos_chem.vulcan.find_latest_atmosphere_time')
@patch('proteus.atmos_chem.vulcan.read_atmosphere_data')
@patch('proteus.atmos_chem.vulcan.shutil.rmtree')
@patch('proteus.atmos_chem.vulcan.os.makedirs')
def test_run_vulcan_falls_back_to_latest_available_time(
    mock_makedirs, mock_rmtree, mock_read_atmos, mock_latest, tmp_path
):
    """
    A missing exact-time file makes run_vulcan re-read at the latest available
    snapshot time before giving up.

    Discrimination: the exact time is 1000 but the latest available is 42; the
    second read must request [42.0] (not [1000.0]), proving the fallback drives
    the snapshot selection. Here the fallback read also returns None so the run
    still exits False without a TypeError (guarding the None[0] crash), while
    demonstrating the selection.
    """
    config = _make_mock_config()
    hf_row = _make_mock_hf_row(year=1000.0)
    dirs = {'output': str(tmp_path), 'output/offchem': str(tmp_path / 'offchem')}

    mock_read_atmos.return_value = None  # both the exact and fallback reads fail
    mock_latest.return_value = 42.0
    mock_vulcan_package.run_vulcan.reset_mock()

    result = run_vulcan(dirs, config, hf_row)

    assert result is False
    # Two reads: the exact time, then the fallback time.
    assert mock_read_atmos.call_count == 2
    first_times = mock_read_atmos.call_args_list[0].args[1]
    second_times = mock_read_atmos.call_args_list[1].args[1]
    assert first_times == [1000.0]
    assert second_times == [42.0]
    mock_vulcan_package.run_vulcan.assert_not_called()


class _RecordingConfig:
    """Stand-in for `vulcan.Config` that records which fields are assigned.

    It has no defaults, so a field PROTEUS reads without setting raises
    AttributeError, as a missing field would.
    """

    def __init__(self):
        object.__setattr__(self, 'assigned', set())

    def __setattr__(self, name, value):
        self.assigned.add(name)
        object.__setattr__(self, name, value)


def _installed_vulcan_config_fields() -> set[str]:
    """Names of the fields `Config.__init__` sets in the installed VULCAN.

    Parsed from config.py, because this module replaces `vulcan` in
    sys.modules with a mock. Skips when VULCAN is not installed.
    """
    saved = sys.modules.pop('vulcan', None)
    try:
        spec = importlib.util.find_spec('vulcan')
    finally:
        if saved is not None:
            sys.modules['vulcan'] = saved
    if spec is None or not spec.submodule_search_locations:
        pytest.skip('VULCAN is not installed')

    tree = ast.parse(Path(spec.submodule_search_locations[0], 'config.py').read_text())
    config_cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Config'
    )
    init = next(
        n for n in config_cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__'
    )
    fields = set()
    for node in ast.walk(init):
        targets = node.targets if isinstance(node, ast.Assign) else []
        if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == 'self'
            ):
                fields.add(target.attr)
    return fields


def _run_online_with_recording_config(tmp_path, config, hf_row):
    """Run run_vulcan in online mode with mocked I/O; return the result and config."""
    year = int(hf_row['Time'])
    offchem_dir = tmp_path / 'offchem'
    offchem_dir.mkdir(parents=True, exist_ok=True)
    dirs = {'output': str(tmp_path), 'output/offchem': str(offchem_dir)}
    with open(offchem_dir / f'vulcan_{year}.pkl', 'wb') as f:
        pickle.dump(_make_mock_vulcan_result(), f)

    vcfg = _RecordingConfig()
    mock_vulcan_package.Config.return_value = vcfg
    mock_vulcan_package.run_vulcan.reset_mock()
    if hasattr(run_vulcan, '_made'):
        del run_vulcan._made

    with (
        patch('proteus.atmos_chem.vulcan.read_atmosphere_data') as mock_read_atmos,
        patch('proteus.atmos_chem.vulcan.glob.glob') as mock_glob,
        patch('proteus.atmos_chem.vulcan.np.loadtxt') as mock_loadtxt,
        patch('proteus.atmos_chem.vulcan.np.savetxt'),
    ):
        mock_read_atmos.return_value = [_make_mock_atmos()]
        mock_glob.return_value = [str(tmp_path / 'data' / f'{year}.sflux')]
        mock_loadtxt.return_value = np.array([[200.0, 1e-5], [300.0, 1e-4], [400.0, 1e-3]])
        result = run_vulcan(dirs, config, hf_row, online=True)
    return result, vcfg


@pytest.mark.unit
def test_run_vulcan_sets_every_field_of_the_installed_vulcan_config(tmp_path):
    """PROTEUS sets every field of the installed VULCAN Config, and only those.

    Contract: a field PROTEUS leaves unset runs at VULCAN's own default (for
    example its HD 189733b boundary files), and a field VULCAN does not
    define (a misspelling such as save_types for save_type) is silently
    ignored. Both show only in a coupled run. Fails when VULCAN adds or
    renames a Config field until the wrapper is updated to match.
    """
    fields = _installed_vulcan_config_fields()
    # Guard against a parser that finds nothing, which would pass vacuously.
    assert {'network', 'out_name', 'atom_list'} <= fields

    result, vcfg = _run_online_with_recording_config(
        tmp_path, _make_mock_config(), _make_mock_hf_row(year=800.0)
    )

    assert result is True
    mock_vulcan_package.run_vulcan.assert_called_once_with(vcfg, True)
    assert sorted(fields - vcfg.assigned) == [], 'VULCAN Config fields PROTEUS does not set'
    # VULCAN reads these two with condensation on, but its Config does not define them.
    assert sorted(vcfg.assigned - fields) == ['use_ini_cold_trap', 'use_relax'], (
        'fields PROTEUS sets that VULCAN lacks'
    )
    # PROTEUS parses the pickle output, so it must be requested.
    assert 'pickle' in vcfg.save_type


@pytest.mark.unit
@pytest.mark.parametrize('fix_surf', [True, False], ids=['fixed_surface', 'free_surface'])
def test_run_vulcan_boundary_files_hold_surface_mixing_ratios(tmp_path, fix_surf):
    """Both boundary files list the surface VMRs above clip_vmr with zero flux
    and deposition velocity, and VULCAN reads them only where PROTEUS fixes them.

    VULCAN reads bot_BC_flux_file (species in column 1, fixed VMR in column 4)
    when use_fix_sp_bot is non-empty, so with a fixed surface the file must
    agree with that dict. With a free surface the dict is empty and the flux
    flags are off, so neither file is read. Asymmetric VMRs (0.6, 0.3) catch a
    swapped species; N2 below clip_vmr must be absent.
    """
    config = _make_mock_config()
    config.atmos_chem.vulcan.make_funs = False
    config.atmos_chem.vulcan.fix_surf = fix_surf
    hf_row = _make_mock_hf_row(year=900.0)
    hf_row['H2O_vmr'] = 0.6
    hf_row['CO2_vmr'] = 0.3
    hf_row['N2_vmr'] = 1e-12  # below clip_vmr = 1e-10

    result, vcfg = _run_online_with_recording_config(tmp_path, config, hf_row)
    assert result is True

    def _parse(fpath):
        # Parse as VULCAN's BC_flux does: skip comments and blank lines.
        rows = [
            line.split()
            for line in Path(fpath).read_text().splitlines()
            if line.strip() and not line.startswith('#')
        ]
        return {row[0]: tuple(float(v) for v in row[1:4]) for row in rows}

    bot = _parse(vcfg.bot_BC_flux_file)
    top = _parse(vcfg.top_BC_flux_file)
    assert top == bot
    assert all(flux == 0.0 and vdep == 0.0 for flux, vdep, _ in bot.values())

    bot_vmr = {sp: vals[2] for sp, vals in bot.items()}
    assert bot_vmr == pytest.approx(vcfg.const_mix, rel=1e-8)
    assert bot_vmr['H2O'] == pytest.approx(0.6, rel=1e-8)
    assert bot_vmr['CO2'] == pytest.approx(0.3, rel=1e-8)
    assert 'N2' not in bot_vmr
    assert all(0.0 < vmr <= 1.0 for vmr in bot_vmr.values())

    # Neither flux boundary is active, so VULCAN reads the files only for use_fix_sp_bot.
    assert vcfg.use_botflux is False
    assert vcfg.use_topflux is False
    if fix_surf:
        assert bot_vmr == pytest.approx(vcfg.use_fix_sp_bot, rel=1e-8)
    else:
        assert vcfg.use_fix_sp_bot == {}


@pytest.mark.unit
def test_run_vulcan_passes_solver_options_and_agni_settings_to_vulcan_config(
    tmp_path, monkeypatch
):
    """Each [atmos_chem.vulcan] solver option reaches the VULCAN Config, the
    AGNI settings of VULCAN's RCE solver follow the PROTEUS AGNI config, and
    the plot format follows params.out.plot_fmt.

    Every option gets a distinct non-default value, so a swapped pair (atol
    with rtol, mtol with mtol_conv, agni_atol with agni_rtol) or a value left
    at its hard-coded default fails. dt_max follows runtime by a factor 1e-4.
    """
    config = _make_mock_config()
    config.atmos_chem.vulcan.make_funs = False
    _write_network(tmp_path, monkeypatch, 'CHO_photo_network.txt', '')
    opts = {
        'use_vz': False,
        'use_condense': True,
        'use_settling': True,
        'st_factor': 0.3,
        'conv_step': 250,
        'yconv_min': 0.07,
        'runtime': 3.0e18,
        'dttry': 2.0e-7,
        'dt_min': 4.0e-11,
        'atol': 0.02,
        'mtol': 3.0e-23,
        'mtol_conv': 5.0e-21,
        'flux_atol': 0.6,
        'rtol': 0.35,
        'post_conden_rtol': 0.08,
    }
    for key, val in opts.items():
        setattr(config.atmos_chem.vulcan, key, val)
    config.params.out.plot_fmt = 'pdf'
    config.atmos_clim.rayleigh = False
    config.atmos_clim.surf_greyalbedo = 0.27
    config.atmos_clim.num_levels = 73
    config.atmos_clim.agni.solution_atol = 0.4
    config.atmos_clim.agni.solution_rtol = 0.12
    config.atmos_clim.agni.solve_energy = True

    result, vcfg = _run_online_with_recording_config(
        tmp_path, config, _make_mock_hf_row(year=700.0)
    )

    assert result is True
    for key, val in opts.items():
        assert getattr(vcfg, key) == val, key
    assert vcfg.plot_fmt == 'pdf'
    assert vcfg.use_rayleigh is False
    assert vcfg.surf_albedo == pytest.approx(0.27, rel=1e-12)
    assert vcfg.agni_nlev == 73
    assert vcfg.agni_atol == pytest.approx(0.4, rel=1e-12)
    assert vcfg.agni_rtol == pytest.approx(0.12, rel=1e-12)
    assert vcfg.solve_rce is True
    assert vcfg.Tsurf_guess == pytest.approx(1500.0, rel=1e-12)  # K, from hf_row
    assert vcfg.dt_max == pytest.approx(3.0e14, rel=1e-12)
    # The RCE solver stays off, so the AGNI settings do not change the run.
    assert vcfg.agni_call_frq == 0


# Network file in VULCAN's layout: condensation reactions sit between the
# special and the photodissociation sections, which both hold 'A -> B' lines too.
_NETWORK_WITH_CONDENSATION = """# 2-body reactions
1 [ OH + H2 -> H2O + H                 ]   3.57E-16    1.52      1740.0

# special cases
# id	# Reactions
1027 [  OH + CH3 + M -> CH3OH + M         ]

# reverse stops
# condensation
1029 [  S8 -> S8_l_s                      ]
1031 [  S4 -> S4_l_s                      ]
1033 [  H2O -> H2O_l_s                    ]

# photo disscoiation (no reversals!)
# id	# Reactions                                     sp		br_index
1035 [ H2O -> H + OH                      ]       		H2O		1
"""


def _write_network(tmp_path, monkeypatch, name, text):
    """Write a network file into a stand-in VULCAN thermo folder and point the mock at it."""
    thermo = tmp_path / 'thermo'
    thermo.mkdir(exist_ok=True)
    (thermo / name).write_text(text)
    monkeypatch.setattr(mock_vulcan_package.paths, 'THERMO_DIR', str(thermo))
    return thermo / name


@pytest.mark.unit
def test_network_condensation_reads_only_the_condensation_section(tmp_path):
    """Only the reactions under '# condensation' are returned, keyed by the gas
    that condenses; the photodissociation H2O -> H + OH below it and the
    reactions above it are not."""
    fpath = tmp_path / 'network.txt'
    fpath.write_text(_NETWORK_WITH_CONDENSATION)

    pairs = network_condensation(str(fpath))

    assert pairs == {'S8': 'S8_l_s', 'S4': 'S4_l_s', 'H2O': 'H2O_l_s'}
    # A file without the section has no condensation.
    fpath.write_text(_NETWORK_WITH_CONDENSATION.replace('# condensation', '# other'))
    assert network_condensation(str(fpath)) == {}


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_run_vulcan_condenses_every_supported_gas_of_the_network(tmp_path, monkeypatch):
    """With use_condense, VULCAN condenses each gas of the network that it has
    saturation data for (S8, H2O here, not S4), as particles of the configured
    radius and a condensed-phase density.

    Invariant: every particle density lies between ammonia ice (0.8 g cm-3) and
    graphite (2.3 g cm-3); a density in kg m-3 would be 1000 times larger.
    """
    config = _make_mock_config(network='SNCHO')
    config.atmos_chem.vulcan.make_funs = False
    config.atmos_chem.vulcan.use_condense = True
    config.atmos_chem.vulcan.use_settling = True
    config.atmos_chem.vulcan.r_p = 2.5e-4  # cm
    _write_network(tmp_path, monkeypatch, 'SNCHO_photo_network.txt', _NETWORK_WITH_CONDENSATION)

    result, vcfg = _run_online_with_recording_config(
        tmp_path, config, _make_mock_hf_row(year=650.0)
    )

    assert result is True
    assert vcfg.condense_sp == ['S8', 'H2O']
    assert vcfg.non_gas_sp == ['S8_l_s', 'H2O_l_s']
    assert vcfg.r_p == pytest.approx({'S8_l_s': 2.5e-4, 'H2O_l_s': 2.5e-4}, rel=1e-12)
    assert vcfg.rho_p == pytest.approx({'S8_l_s': 2.07, 'H2O_l_s': 1.0}, rel=1e-12)
    assert all(0.8 <= CHEM_CONDENSATE_DENSITY[gas] <= 2.3 for gas in CHEM_CONDENSATE_DENSITY)
    # VULCAN reads both without defining them; relaxation and the cold trap stay off.
    assert vcfg.use_relax == []
    assert vcfg.use_ini_cold_trap is False


@pytest.mark.unit
@pytest.mark.parametrize(
    'use_condense,network,text',
    [
        (False, 'SNCHO', _NETWORK_WITH_CONDENSATION),
        (True, 'CHO', '# 2-body reactions\n1 [ OH + H2 -> H2O + H ] 3.57E-16 1.52 1740.0\n'),
    ],
    ids=['condensation_off', 'network_without_condensation'],
)
def test_run_vulcan_without_condensation_sets_no_condensates(
    tmp_path, monkeypatch, use_condense, network, text
):
    """With condensation off, or a network that has no condensation reaction,
    no gas condenses and no particle is defined; the second case warns."""
    config = _make_mock_config(network=network)
    config.atmos_chem.vulcan.make_funs = False
    config.atmos_chem.vulcan.use_condense = use_condense
    _write_network(tmp_path, monkeypatch, f'{network}_photo_network.txt', text)

    with patch('proteus.atmos_chem.vulcan.log') as mock_log:
        result, vcfg = _run_online_with_recording_config(
            tmp_path, config, _make_mock_hf_row(year=660.0)
        )

    assert result is True
    assert (vcfg.condense_sp, vcfg.non_gas_sp, vcfg.r_p, vcfg.rho_p) == ([], [], {}, {})
    assert mock_log.warning.called is use_condense


@pytest.mark.unit
@pytest.mark.parametrize(
    'network,kept',
    [
        ('CHO', set()),
        ('NCHO', {'N2', 'NH3'}),
        ('SNCHO', {'N2', 'NH3', 'S2', 'SO2', 'H2S'}),
    ],
)
def test_run_vulcan_fixes_only_gases_whose_elements_are_in_the_network(tmp_path, network, kept):
    """A gas with an element outside the network is neither in const_mix nor
    fixed at the surface nor written to the boundary file, because VULCAN
    stops on a species the network does not hold. N and S gases sit well above
    clip_vmr, so only the network decides whether they are kept."""
    config = _make_mock_config(network=network)
    config.atmos_chem.vulcan.make_funs = False
    config.atmos_chem.vulcan.fix_surf = True
    hf_row = _make_mock_hf_row(year=670.0)
    candidates = {'N2', 'NH3', 'S2', 'SO2', 'H2S'}
    for gas in candidates:
        hf_row[gas + '_vmr'] = 1e-3  # 1e7 times clip_vmr

    result, vcfg = _run_online_with_recording_config(tmp_path, config, hf_row)

    assert result is True
    assert set(vcfg.const_mix) & candidates == kept
    assert set(vcfg.use_fix_sp_bot) & candidates == kept
    rows = Path(vcfg.bot_BC_flux_file).read_text().splitlines()
    in_file = {row.split()[0] for row in rows if row.strip() and not row.startswith('#')}
    assert in_file & candidates == kept
    # The CHO gases are kept in every network.
    assert {'H2O', 'CO2', 'H2', 'CO', 'CH4', 'O2'} <= set(vcfg.const_mix)


@pytest.mark.unit
def test_run_vulcan_background_gas_counts_gases_outside_the_network(tmp_path):
    """An N2-dominated atmosphere has N2 as background gas also with the CHO
    network, which drops N2 itself: VULCAN takes the molecular-diffusion law
    from atm_base. Choosing among network gases only would give CO2."""
    config = _make_mock_config(network='CHO')
    config.atmos_chem.vulcan.make_funs = False
    hf_row = _make_mock_hf_row(year=680.0)
    hf_row['N2_vmr'] = 0.7
    hf_row['CO2_vmr'] = 0.2
    hf_row['H2_vmr'] = 0.01

    result, vcfg = _run_online_with_recording_config(tmp_path, config, hf_row)

    assert result is True
    assert vcfg.atm_base == 'N2'
    assert 'N2' not in vcfg.const_mix


@pytest.mark.unit
def test_run_vulcan_online_moves_netcdf_to_data_folder(tmp_path):
    """In online mode VULCAN's per-snapshot vulcan_<int year>.nc moves to
    data/<year with sub-year precision>_chem.nc and leaves the offchem folder.
    The year 1234.5 tells the sub-year name from the integer one."""
    config = _make_mock_config()
    config.atmos_chem.vulcan.make_funs = False
    hf_row = _make_mock_hf_row(year=1234.5)
    (tmp_path / 'data').mkdir()
    src = tmp_path / 'offchem' / 'vulcan_1234.nc'
    src.parent.mkdir()
    src.write_bytes(b'netcdf-test')

    result, _ = _run_online_with_recording_config(tmp_path, config, hf_row)

    assert result is True
    dst = tmp_path / 'data' / '1234p500_chem.nc'
    assert format_subyear_time(1234.5) + '_chem.nc' == dst.name
    assert dst.read_bytes() == b'netcdf-test'
    assert not src.exists()
    assert not (tmp_path / 'data' / '1234_chem.nc').exists()


@pytest.mark.unit
def test_run_vulcan_offline_moves_netcdf_to_data_folder(tmp_path):
    """In offline mode the offchem folder is wiped before VULCAN runs, and the
    vulcan.nc that VULCAN writes moves to data/<year>_chem.nc."""
    config = _make_mock_config()
    config.atmos_chem.vulcan.make_funs = False
    hf_row = _make_mock_hf_row(year=2000.0)
    (tmp_path / 'data').mkdir()
    offchem = tmp_path / 'offchem'
    dirs = {'output': str(tmp_path), 'output/offchem': str(offchem)}

    def _fake_vulcan(*_args):
        # VULCAN writes both outputs into the freshly created offchem folder.
        (offchem / 'vulcan.nc').write_bytes(b'netcdf-offline')
        with open(offchem / 'vulcan.pkl', 'wb') as f:
            pickle.dump(_make_mock_vulcan_result(), f)

    mock_vulcan_package.Config.return_value = _RecordingConfig()
    mock_vulcan_package.run_vulcan.side_effect = _fake_vulcan
    try:
        with (
            patch('proteus.atmos_chem.vulcan.read_atmosphere_data') as mock_read_atmos,
            patch('proteus.atmos_chem.vulcan.glob.glob') as mock_glob,
            patch('proteus.atmos_chem.vulcan.np.loadtxt') as mock_loadtxt,
            patch('proteus.atmos_chem.vulcan.np.savetxt'),
        ):
            mock_read_atmos.return_value = [_make_mock_atmos()]
            mock_glob.return_value = [str(tmp_path / 'data' / '2000.sflux')]
            mock_loadtxt.return_value = np.array([[200.0, 1e-5], [300.0, 1e-4], [400.0, 1e-3]])
            result = run_vulcan(dirs, config, hf_row)
    finally:
        mock_vulcan_package.run_vulcan.side_effect = None

    assert result is True
    assert (tmp_path / 'data' / '2000p000_chem.nc').read_bytes() == b'netcdf-offline'
    assert not (offchem / 'vulcan.nc').exists()
