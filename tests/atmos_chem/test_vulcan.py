"""
Unit tests for proteus.atmos_chem.vulcan

Tests run_vulcan(), which runs the VULCAN chemical kinetics solver on one
atmosphere snapshot after the PROTEUS simulation, and its file helpers.

Contracts and invariants tested:
- Module guard: VULCAN requires the AGNI atmosphere module
- Network selection: CHO, NCHO, SNCHO chemical networks and element lists
- Output: VULCAN writes only NetCDF (save_type ['netcdf'], no pickle); the
  file moves to data/<time>_chem.nc and PROTEUS writes
  offchem/vulcan_<time>.csv from it, with the time in the sub-year form
  (12.32 yr is 12p320)
- CSV values: pressure dyn cm-2 to Pa, height cm to m, Kzz on the interior
  interfaces, rows from the top of the column down, condensates left out
- Network compilation: with make_funs, VULCAN compiles the network unless an
  earlier run of the sequence has (network_compiled), and never without it
- NetCDF reader: VULCAN's values come back in CGS, bottom up, with the
  filled Kzz ends as NaN
- Initial composition: vmrs.dat is on VULCAN's own grid, from the bottom of
  the column up, interpolated in log pressure; the mixing ratios on each row
  keep their sum; ini_mix = 'previous' starts from the latest earlier
  data/<time>_chem.nc and falls back to the climate profile
- Stellar input closure: star.dat and orbit_radius share the flux-weighted
  distance a (1 - e^2)^(1/4)
- Config contract: PROTEUS sets every field that `vulcan.Config.__init__`
  defines in the installed VULCAN, and no field VULCAN does not define
- Boundary files hold the surface VMRs with zero fluxes, and VULCAN reads
  them only with a fixed surface
- Config pass-through of the solver and AGNI options
- Network filter and background gas
- Condensation of every gas of the network that VULCAN can condense

Mocking strategy:
- The `vulcan` package is a MagicMock (not available in CI); its solver is
  replaced by a function that writes a NetCDF file in VULCAN's schema
  (tests/helpers/_vulcan_netcdf.py). The config contract test reads the
  installed VULCAN's config.py source instead
- read_atmosphere_data is mocked with an AGNI-ordered profile (top of the
  column first); every other file is real and written under tmp_path
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tests.helpers._vulcan_netcdf import vulcan_layers, write_vulcan_netcdf

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
    _find_previous_chem,
    _write_ini_table,
    network_condensation,
    network_settings,
    read_vulcan_netcdf,
    run_vulcan,
)
from proteus.utils.constants import CHEM_CONDENSATE_DENSITY  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.fixture(autouse=True)
def _vulcan_mock_in_sys_modules():
    """run_vulcan imports `vulcan` when called, so this file's mock is put in
    sys.modules for each test, whatever another test file left there."""
    with patch.dict(sys.modules, {'vulcan': mock_vulcan_package}):
        yield


# Species of the mocked VULCAN result; H2O_l_s is a condensate and stays out of the CSV
_RESULT_SPECIES = ['H2', 'H2O', 'CO2', 'H2O_l_s']


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
    # Add VMR entries for all volatiles
    from proteus.utils.constants import vol_list

    for gas in vol_list:
        hf_row[gas + '_vmr'] = 1e-4
    return hf_row


def _make_mock_atmos(*, ncell=5, p_top=1.0, p_bot=1.0e5, vmrs=None):
    """Mock AGNI snapshot as read_atmosphere_data returns it, top of the column first.

    `pl` holds the ncell+1 level edges [Pa], increasing downwards as in AGNI,
    and `p` the ncell layer centres between them, on which the `<gas>_vmr`
    arrays are defined. `tmpl` [K] and `Kzz` [m2 s-1] are on the edges.
    """
    pl = np.logspace(np.log10(p_top), np.log10(p_bot), ncell + 1)
    atmos = {
        'p': np.sqrt(pl[1:] * pl[:-1]),
        'pl': pl,
        'tmpl': np.linspace(200.0, 500.0, ncell + 1),
        'Kzz': np.full(ncell + 1, 1e2),
    }
    if vmrs is None:
        vmrs = {gas: np.full(ncell, 1e-4) for gas in ['H2O', 'CO2', 'H2', 'CO', 'N2']}
    for gas, vmr in vmrs.items():
        atmos[gas + '_vmr'] = np.asarray(vmr, dtype=float)
    return atmos


def _result_x_gas(nz: int) -> np.ndarray:
    """Mixing ratios of `_RESULT_SPECIES` on nz layers, distinct per column, rows sum to 1."""
    h2o = np.linspace(0.6, 0.2, nz)
    co2 = np.linspace(0.1, 0.3, nz)
    cond = np.full(nz, 0.05)
    return np.column_stack([1.0 - h2o - co2 - cond, h2o, co2, cond])


def _netcdf_solver(layers=None):
    """Stand-in for `vulcan.run_vulcan` that writes a result NetCDF where VULCAN would."""
    layers = layers or vulcan_layers(nz=6)

    def _solve(vcfg, make_network):
        fpath = Path(vcfg.output_dir) / (vcfg.out_name + '.nc')
        write_vulcan_netcdf(
            str(fpath), layers, _RESULT_SPECIES, _result_x_gas(len(layers['p']))
        )

    return _solve


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


def _write_sflux(tmp_path, year, flux=(1e-5, 1e-4, 1e-3)):
    """Write a stellar spectrum file data/<year>.sflux [nm, erg cm-2 s-1 nm-1]."""
    data = tmp_path / 'data'
    data.mkdir(exist_ok=True)
    rows = '\n'.join(f'{wl:.1f} {fl:.6e}' for wl, fl in zip((200.0, 300.0, 400.0), flux))
    (data / f'{int(year)}.sflux').write_text('# WL(nm) Flux\n' + rows + '\n')


def _run(tmp_path, config, hf_row, *, network_compiled=False, atmos=None, solver='netcdf'):
    """Run run_vulcan on mocked atmosphere input; return the result, config and dirs.

    `solver='netcdf'` makes the mocked VULCAN write a result NetCDF, None makes
    it write nothing, and a callable is used as the solver itself.
    """
    _write_sflux(tmp_path, hf_row['Time'])
    dirs = {'output': str(tmp_path), 'output/offchem': str(tmp_path / 'offchem')}
    vcfg = _RecordingConfig()
    mock_vulcan_package.Config.return_value = vcfg
    mock_vulcan_package.run_vulcan.reset_mock()
    mock_vulcan_package.make_chem_funs.make_all.reset_mock()
    if solver == 'netcdf':
        solver = _netcdf_solver()
    mock_vulcan_package.run_vulcan.side_effect = solver
    try:
        with patch('proteus.atmos_chem.vulcan.read_atmosphere_data') as mock_read_atmos:
            mock_read_atmos.return_value = [atmos if atmos is not None else _make_mock_atmos()]
            result = run_vulcan(dirs, config, hf_row, network_compiled=network_compiled)
    finally:
        mock_vulcan_package.run_vulcan.side_effect = None
    return result, vcfg, dirs


def _read_table(fpath):
    """Read a VULCAN `table` initial-composition file as VULCAN does."""
    return np.genfromtxt(fpath, names=True, dtype=None, skip_header=1)


@pytest.mark.unit
def test_run_vulcan_refuses_non_agni_atmosphere(tmp_path):
    """VULCAN needs the AGNI temperature-pressure profiles, so with JANUS
    run_vulcan returns False before calling the solver or writing a file."""
    config = _make_mock_config(atmos_module='janus')
    dirs = {'output': str(tmp_path), 'output/offchem': str(tmp_path / 'offchem')}
    mock_vulcan_package.run_vulcan.reset_mock()

    result = run_vulcan(dirs, config, _make_mock_hf_row())

    assert result is False
    mock_vulcan_package.run_vulcan.assert_not_called()
    assert not (tmp_path / 'offchem').exists()


@pytest.mark.unit
def test_run_vulcan_unrecognised_network_returns_false_before_solver(tmp_path):
    """Only CHO, NCHO and SNCHO are implemented; another name returns False
    before the solver runs, so no result file appears."""
    result, _, _ = _run(tmp_path, _make_mock_config(network='INVALID'), _make_mock_hf_row())

    assert result is False
    mock_vulcan_package.run_vulcan.assert_not_called()
    assert not list(tmp_path.rglob('*_chem.nc'))


@pytest.mark.unit
@pytest.mark.parametrize(
    'network,expected_elements',
    [
        ('CHO', ['H', 'O', 'C']),
        ('NCHO', ['H', 'O', 'C', 'N']),
        ('SNCHO', ['H', 'O', 'C', 'N', 'S']),
    ],
)
def test_run_vulcan_network_selection(tmp_path, network, expected_elements):
    """Each network has its own element list: CHO, NCHO adds nitrogen, SNCHO
    adds sulfur. The network file follows the network and photochemistry."""
    config = _make_mock_config(network=network)
    config.atmos_chem.vulcan.make_funs = False

    result, vcfg, _ = _run(tmp_path, config, _make_mock_hf_row(year=400.0))

    assert result is True
    assert vcfg.atom_list == expected_elements
    assert Path(vcfg.network).name == f'{network}_photo_network.txt'


@pytest.mark.unit
@pytest.mark.parametrize(
    'network,photo_on,expected_file',
    [
        ('CHO', True, 'CHO_photo_network.txt'),
        ('CHO', False, 'CHO_thermo_network.txt'),
        ('NCHO', True, 'NCHO_photo_network.txt'),
        ('NCHO', False, 'NCHO_thermo_network.txt'),
        ('SNCHO', True, 'SNCHO_photo_network.txt'),
        ('SNCHO', False, 'SNCHO_photo_network.txt'),
    ],
    ids=[
        'CHO_photo',
        'CHO_thermo',
        'NCHO_photo',
        'NCHO_thermo',
        'SNCHO_photo',
        'SNCHO_has_no_thermo_network',
    ],
)
def test_network_settings_selects_the_file_and_species_of_each_network(
    network, photo_on, expected_file
):
    """Each network maps to its own file, photochemical or thermochemical, and
    SNCHO has only the photochemical one. Every species to plot or to scatter
    is made of elements of the network, which VULCAN requires."""
    from proteus.utils.helper import mol_to_ele

    network_file, ele_arr, plt_arr, sct_arr = network_settings(
        _make_mock_config(network=network, photo_on=photo_on)
    )

    assert network_file == expected_file
    assert set(ele_arr) == set(network)  # the network name spells its elements
    for sp in plt_arr + sct_arr:
        assert set(mol_to_ele(sp)) <= set(ele_arr), sp


@pytest.mark.unit
def test_network_settings_refuses_an_unknown_network():
    """An unrecognised network gives None and logs an error, rather than
    falling back to a default network."""
    with patch('proteus.atmos_chem.vulcan.log') as mock_log:
        settings = network_settings(_make_mock_config(network='INVALID'))

    assert settings is None
    mock_log.error.assert_called_once()
    assert 'INVALID' in mock_log.error.call_args.args[0]


@pytest.mark.unit
def test_run_vulcan_reads_layer_centre_pressures_of_the_snapshot(tmp_path):
    """The atmosphere snapshot at hf_row['Time'] is read once, with the
    layer-centre pressures 'p' among the extra keys, and there is no fallback
    to another snapshot."""
    with patch('proteus.atmos_chem.vulcan.read_atmosphere_data') as mock_read_atmos:
        mock_read_atmos.return_value = None
        dirs = {'output': str(tmp_path), 'output/offchem': str(tmp_path / 'offchem')}
        mock_vulcan_package.run_vulcan.reset_mock()
        result = run_vulcan(dirs, _make_mock_config(), _make_mock_hf_row(year=1000.0))

    assert result is False
    mock_read_atmos.assert_called_once_with(
        str(tmp_path), [1000.0], extra_keys=['p', 'pl', 'tmpl', 'x_gas', 'Kzz']
    )
    mock_vulcan_package.run_vulcan.assert_not_called()


@pytest.mark.unit
def test_run_vulcan_boundary_pressures_use_level_edges(tmp_path):
    """vcfg.P_b and vcfg.P_t come from the level edges 'pl', not the centres.

    The edges bracket every cell, so the bottom and top of VULCAN's column sit
    on them; with 'p' the column is cut by half a cell at each end. With edges
    1 Pa to 1e6 Pa on 6 cells the centres reach only 10**5.5 Pa and 10**0.5 Pa.
    """
    config = _make_mock_config()
    config.atmos_chem.vulcan.make_funs = False
    atmos = _make_mock_atmos(ncell=6, p_top=1.0, p_bot=1.0e6)

    result, vcfg, _ = _run(tmp_path, config, _make_mock_hf_row(year=700.0), atmos=atmos)

    assert result is True
    assert vcfg.P_b == pytest.approx(1.0e7, rel=1e-12)  # 1e6 Pa in dyn cm-2
    assert vcfg.P_t == pytest.approx(1.0e1, rel=1e-12)
    assert vcfg.P_b != pytest.approx(10.0 * np.amax(atmos['p']), rel=1e-3)
    assert vcfg.nz == 7  # one VULCAN layer per AGNI level edge


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_run_vulcan_star_file_and_orbit_radius_share_the_flux_weighted_distance(tmp_path):
    """On an eccentric orbit star.dat holds the stellar-surface spectrum and
    orbit_radius the distance VULCAN dilutes it by, both at the flux-weighted
    distance a (1 - e^2)^(1/4) the stored spectrum was written at.

    Invariant: VULCAN's top-of-atmosphere flux, star.dat times
    (R_star / orbit_radius)^2, equals the stored spectrum. At e = 0.8 the
    time-averaged separation a (1 + e^2 / 2) is 2.9 times further in flux
    terms, so a star.dat undone with it would be 2.9 times too bright.
    star.dat stores 5 significant digits, so the tolerance is 1e-4.
    """
    from proteus.utils.constants import AU, R_sun

    config = _make_mock_config()
    config.atmos_chem.vulcan.make_funs = False
    hf_row = _make_mock_hf_row(year=700.0)
    hf_row['semimajorax'] = 0.029 * AU
    hf_row['eccentricity'] = 0.8
    hf_row['separation'] = 0.029 * AU * (1.0 + 0.5 * 0.8**2)
    d_flux = 0.029 * AU * (1.0 - 0.8**2) ** 0.25
    toa_flux = np.array([1e-5, 1e-4, 1e-3])

    result, vcfg, _ = _run(tmp_path, config, hf_row)

    assert result is True
    star_fl = np.loadtxt(tmp_path / 'offchem' / 'star.dat')[:, 1]
    np.testing.assert_allclose(star_fl, toa_flux * (d_flux / hf_row['R_star']) ** 2, rtol=1e-4)
    assert vcfg.orbit_radius == pytest.approx(d_flux / AU, rel=1e-12)
    # Closure: VULCAN's own dilution returns the stored top-of-atmosphere flux.
    r_star_m = vcfg.r_star * R_sun
    np.testing.assert_allclose(
        star_fl * (r_star_m / (vcfg.orbit_radius * AU)) ** 2, toa_flux, rtol=1e-4
    )
    # Guard: undoing the separation would make star.dat 2.9 times brighter.
    wrong = toa_flux * (hf_row['separation'] / hf_row['R_star']) ** 2
    np.testing.assert_allclose(wrong / star_fl, 2.90, rtol=0.01)


@pytest.mark.unit
def test_run_vulcan_writes_only_netcdf_and_no_pickle(tmp_path):
    """VULCAN is asked for NetCDF only, under the sub-year stem of the
    snapshot time; its file moves to data/<time>_chem.nc and no pickle is
    written or left anywhere in the output."""
    result, vcfg, _ = _run(tmp_path, _make_mock_config(), _make_mock_hf_row(year=700.0))

    assert result is True
    assert vcfg.save_type == ['netcdf']
    assert vcfg.out_name == 'vulcan_700p000'
    assert (tmp_path / 'data' / '700p000_chem.nc').is_file()
    assert not (tmp_path / 'offchem' / 'vulcan_700p000.nc').exists()
    assert not list(tmp_path.rglob('*.pkl'))


@pytest.mark.unit
@pytest.mark.parametrize(
    'make_funs,network_compiled,expected',
    [(True, False, True), (True, True, False), (False, False, False), (False, True, False)],
    ids=['first_run', 'network_already_compiled', 'make_funs_off', 'make_funs_off_compiled'],
)
def test_run_vulcan_compiles_the_network_only_when_needed(
    tmp_path, make_funs, network_compiled, expected
):
    """VULCAN's own solver call compiles the network when make_funs is on and
    no earlier run of the sequence compiled it, and never with make_funs off.
    PROTEUS never compiles it a second time through make_chem_funs."""
    config = _make_mock_config()
    config.atmos_chem.vulcan.make_funs = make_funs

    result, vcfg, _ = _run(
        tmp_path, config, _make_mock_hf_row(year=300.0), network_compiled=network_compiled
    )

    assert result is True
    mock_vulcan_package.run_vulcan.assert_called_once_with(vcfg, expected)
    mock_vulcan_package.make_chem_funs.make_all.assert_not_called()


@pytest.mark.unit
def test_run_vulcan_names_files_with_the_sub_year_time(tmp_path):
    """At t = 12.32 yr the result is data/12p320_chem.nc and
    offchem/vulcan_12p320.csv; the whole-year names of the old online mode
    (vulcan_12.*) are not written."""
    result, vcfg, _ = _run(tmp_path, _make_mock_config(), _make_mock_hf_row(year=12.32))

    assert result is True
    assert vcfg.out_name == 'vulcan_12p320'
    assert (tmp_path / 'data' / '12p320_chem.nc').is_file()
    assert (tmp_path / 'offchem' / 'vulcan_12p320.csv').is_file()
    assert not (tmp_path / 'offchem' / 'vulcan_12.csv').exists()
    assert not (tmp_path / 'data' / '12_chem.nc').exists()


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_run_vulcan_csv_converts_the_netcdf_result_to_si_top_down(tmp_path):
    """The CSV holds the VULCAN NetCDF result from the top of the column down:
    pressure [Pa] = dyn cm-2 / 10, height [m] = bottom interface of each layer
    in cm / 100, Kzz [cm2 s-1] on the interior interfaces with the lowest one
    repeated, and the gas mixing ratios without the condensate H2O_l_s.

    Discrimination: a /1e5 (bar) conversion is 1e4 times smaller, and a
    missing flip puts the 1e8 dyn cm-2 bottom layer in row 0.
    """
    layers = vulcan_layers(P_b=1.0e8, P_t=1.0e2, nz=6)

    result, _, _ = _run(
        tmp_path,
        _make_mock_config(),
        _make_mock_hf_row(year=900.0),
        solver=_netcdf_solver(layers),
    )

    assert result is True
    csv = pd.read_csv(tmp_path / 'offchem' / 'vulcan_900p000.csv', delimiter=r'\s+')
    assert list(csv.columns) == ['tmp', 'p', 'z', 'Kzz', 'H2', 'H2O', 'CO2']
    np.testing.assert_allclose(csv['p'], layers['p'][::-1] / 10.0, rtol=1e-8)
    np.testing.assert_allclose(csv['z'], layers['zl'][:-1][::-1] / 100.0, rtol=1e-8)
    kzz = np.insert(layers['Kzz'], 0, layers['Kzz'][0])
    np.testing.assert_allclose(csv['Kzz'], kzz[::-1], rtol=1e-8)
    np.testing.assert_allclose(csv['tmp'], layers['tmp'][::-1], rtol=1e-8)
    np.testing.assert_allclose(csv['H2O'], _result_x_gas(6)[::-1, 1], rtol=1e-8)
    # Top-down: pressure increases down the rows, and the top layer comes first.
    assert np.all(np.diff(csv['p']) > 0)
    assert csv['p'].iloc[0] == pytest.approx(1.0e1, rel=1e-8)  # 1e2 dyn cm-2
    assert csv['p'].iloc[-1] > 1.0e6  # the bottom 1e8 dyn cm-2 is 1e7 Pa, not 1e3 bar


@pytest.mark.unit
def test_run_vulcan_without_netcdf_output_returns_false(tmp_path):
    """When VULCAN writes no NetCDF (a failed run), run_vulcan returns False
    and writes neither data/<time>_chem.nc nor the CSV."""
    result, _, _ = _run(
        tmp_path, _make_mock_config(), _make_mock_hf_row(year=300.0), solver=None
    )

    assert result is False
    assert not (tmp_path / 'data' / '300p000_chem.nc').exists()
    assert not (tmp_path / 'offchem' / 'vulcan_300p000.csv').exists()


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_run_vulcan_profile_table_starts_at_the_bottom_of_the_column(tmp_path):
    """With ini_mix = 'profile', vmrs.dat starts at the bottom of the column,
    where VULCAN puts its first layer: row 0 holds the surface mixing ratios.

    The AGNI profile is top-first with H2O rising from 0.1 at the top to 0.9 at
    the surface, so a table written in AGNI order would put 0.1 in row 0.
    Invariant: H2O + CO2 = 1 on every source level, so on every row too.
    """
    ncell = 6
    h2o = np.linspace(0.1, 0.9, ncell)  # top of the column first, as in AGNI
    atmos = _make_mock_atmos(ncell=ncell, vmrs={'H2O': h2o, 'CO2': 1.0 - h2o})

    result, vcfg, _ = _run(
        tmp_path, _make_mock_config(), _make_mock_hf_row(year=500.0), atmos=atmos
    )

    assert result is True
    assert vcfg.ini_mix == 'table'
    table = _read_table(vcfg.vul_ini)
    assert len(table) == vcfg.nz
    assert table['Pressure'][0] == pytest.approx(vcfg.P_b, rel=1e-12)
    assert np.all(np.diff(table['Pressure']) < 0)  # bottom up
    assert table['H2O'][0] == pytest.approx(h2o[-1], rel=1e-12)
    assert table['H2O'][-1] == pytest.approx(h2o[0], rel=1e-12)
    assert np.all(np.diff(table['H2O']) <= 0)
    np.testing.assert_allclose(table['H2O'] + table['CO2'], 1.0, rtol=1e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_write_ini_table_interpolates_in_log_pressure(tmp_path):
    """Mixing ratios are interpolated linearly in log10 P onto the target grid,
    held at the end values outside the source range, and keep their sum.

    Source (given top-down to check the sort): 0.2, 0.5, 0.8 at 1e2, 1e4,
    1e6. At 1e5 log interpolation gives 0.65, linear in P gives 0.527; at 1e3
    0.35 against 0.2136. Targets 1e7 and 1e1 lie outside the source range.
    """
    fpath = tmp_path / 'vmrs.dat'
    p_src = np.array([1e2, 1e4, 1e6])
    a = np.array([0.2, 0.5, 0.8])
    pco = np.array([1e7, 1e5, 1e3, 1e1])

    _write_ini_table(str(fpath), pco, p_src, {'H2O': a, 'CO2': 1.0 - a})

    table = _read_table(fpath)
    np.testing.assert_allclose(table['Pressure'], pco, rtol=1e-12)
    np.testing.assert_allclose(table['H2O'], [0.8, 0.65, 0.35, 0.2], rtol=1e-12)
    assert abs(table['H2O'][1] - (0.5 + 0.3 * (1e5 - 1e4) / (1e6 - 1e4))) > 0.1
    np.testing.assert_allclose(table['H2O'] + table['CO2'], 1.0, rtol=1e-12)
    assert np.all((table['H2O'] >= 0.0) & (table['H2O'] <= 1.0))


@pytest.mark.unit
def test_find_previous_chem_takes_the_latest_strictly_earlier_result(tmp_path):
    """The previous result is the latest data/<time>_chem.nc before the given
    time: a result at the same time or later is never used, and with none
    earlier there is no previous result."""
    for name in ('5p000_chem.nc', '12p320_chem.nc', '20p000_chem.nc', '30p000_chem.nc'):
        (tmp_path / name).write_bytes(b'')
    (tmp_path / '15p000_atm.nc').write_bytes(b'')  # an atmosphere snapshot is not a result

    assert _find_previous_chem(str(tmp_path), 20.0) == str(tmp_path / '12p320_chem.nc')
    assert _find_previous_chem(str(tmp_path), 100.0) == str(tmp_path / '30p000_chem.nc')
    assert _find_previous_chem(str(tmp_path), 5.0) is None
    assert _find_previous_chem(str(tmp_path / 'missing'), 50.0) is None


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_run_vulcan_previous_starts_from_the_latest_earlier_result(tmp_path):
    """With ini_mix = 'previous', vmrs.dat holds the mixing ratios of the latest
    earlier result, interpolated in log pressure onto the new VULCAN grid.

    The earlier result at 10 yr has 6 layers from 1e8 to 1e2 dyn cm-2; the new
    grid has 8 layers from 1e7 to 1e1. Results at 20 yr (the current time) and
    50 yr carry H2O = 0.99 and must not be used. Invariant: the rows of the
    earlier result sum to 1, so the interpolated rows do too.
    """
    data = tmp_path / 'data'
    data.mkdir()
    prev = vulcan_layers(P_b=1.0e8, P_t=1.0e2, nz=6)
    write_vulcan_netcdf(str(data / '10p000_chem.nc'), prev, _RESULT_SPECIES, _result_x_gas(6))
    marker = np.tile([0.0, 0.99, 0.01, 0.0], (6, 1))
    for name in ('20p000_chem.nc', '50p000_chem.nc'):
        write_vulcan_netcdf(str(data / name), prev, _RESULT_SPECIES, marker)
    config = _make_mock_config()
    config.atmos_chem.vulcan.ini_mix = 'previous'
    config.atmos_chem.vulcan.fix_surf = True
    atmos = _make_mock_atmos(ncell=7, p_top=1.0, p_bot=1.0e6)

    result, vcfg, _ = _run(tmp_path, config, _make_mock_hf_row(year=20.0), atmos=atmos)

    assert result is True
    assert vcfg.ini_mix == 'table'
    table = _read_table(vcfg.vul_ini)
    assert list(table.dtype.names) == ['Pressure', *_RESULT_SPECIES]
    pco = np.logspace(7.0, 1.0, 8)
    np.testing.assert_allclose(table['Pressure'], pco, rtol=1e-12)
    expected_h2o = np.interp(
        np.log10(pco), np.log10(prev['p'][::-1]), _result_x_gas(6)[::-1, 1]
    )
    np.testing.assert_allclose(table['H2O'], expected_h2o, rtol=1e-12)
    assert np.amax(table['H2O']) < 0.99
    rows = sum(table[sp] for sp in _RESULT_SPECIES)
    np.testing.assert_allclose(rows, 1.0, rtol=1e-12)


@pytest.mark.unit
def test_run_vulcan_previous_without_earlier_result_uses_the_profile(tmp_path):
    """With ini_mix = 'previous' and no earlier result, vmrs.dat holds the
    climate profile of the snapshot, as with ini_mix = 'profile'."""
    ncell = 5
    h2o = np.linspace(0.3, 0.7, ncell)
    atmos = _make_mock_atmos(ncell=ncell, vmrs={'H2O': h2o, 'CO2': 1.0 - h2o})
    config = _make_mock_config()
    config.atmos_chem.vulcan.ini_mix = 'previous'
    config.atmos_chem.vulcan.fix_surf = True

    result, vcfg, _ = _run(tmp_path, config, _make_mock_hf_row(year=10.0), atmos=atmos)

    assert result is True
    assert vcfg.ini_mix == 'table'
    table = _read_table(vcfg.vul_ini)
    assert list(table.dtype.names) == ['Pressure', 'H2O', 'CO2']
    assert table['H2O'][0] == pytest.approx(0.7, rel=1e-12)
    assert table['H2O'][-1] == pytest.approx(0.3, rel=1e-12)


@pytest.mark.unit
def test_run_vulcan_outgas_starts_from_constant_surface_mixing_ratios(tmp_path):
    """With ini_mix = 'outgas' VULCAN starts from the surface mixing ratios at
    every level (const_mix), which hold the outgassed VMRs above clip_vmr."""
    config = _make_mock_config()
    config.atmos_chem.vulcan.ini_mix = 'outgas'
    hf_row = _make_mock_hf_row(year=10.0)
    hf_row['H2O_vmr'] = 0.6
    hf_row['CO2_vmr'] = 0.3

    result, vcfg, _ = _run(tmp_path, config, hf_row)

    assert result is True
    assert vcfg.ini_mix == 'const_mix'
    assert vcfg.const_mix['H2O'] == pytest.approx(0.6, rel=1e-12)
    assert vcfg.const_mix['CO2'] == pytest.approx(0.3, rel=1e-12)


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

    result, vcfg, _ = _run(tmp_path, _make_mock_config(), _make_mock_hf_row(year=800.0))

    assert result is True
    mock_vulcan_package.run_vulcan.assert_called_once_with(vcfg, True)
    assert sorted(fields - vcfg.assigned) == [], 'VULCAN Config fields PROTEUS does not set'
    # VULCAN reads these two with condensation on, but its Config does not define them.
    assert sorted(vcfg.assigned - fields) == ['use_ini_cold_trap', 'use_relax'], (
        'fields PROTEUS sets that VULCAN lacks'
    )
    # PROTEUS reads the NetCDF output only.
    assert vcfg.save_type == ['netcdf']


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

    result, vcfg, _ = _run(tmp_path, config, hf_row)
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

    result, vcfg, _ = _run(tmp_path, config, _make_mock_hf_row(year=700.0))

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

    result, vcfg, _ = _run(tmp_path, config, _make_mock_hf_row(year=650.0))

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
        result, vcfg, _ = _run(tmp_path, config, _make_mock_hf_row(year=660.0))

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

    result, vcfg, _ = _run(tmp_path, config, hf_row)

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

    result, vcfg, _ = _run(tmp_path, config, hf_row)

    assert result is True
    assert vcfg.atm_base == 'N2'
    assert 'N2' not in vcfg.const_mix


# -----------------------------------------------------------------------
# VULCAN NetCDF reader
# -----------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_read_vulcan_netcdf_returns_cgs_columns_bottom_up(tmp_path):
    """The reader returns VULCAN's arrays unchanged in CGS, bottom up: layer
    pressures decrease with height and stay positive, Kzz is NaN on the
    bottom and top interfaces that VULCAN fills, and x_gas keeps one column
    per species in the order of the species names.

    An asymmetric composition (0.6, 0.3, 0.1) catches a transposed x_gas.
    """
    layers = vulcan_layers(P_b=1.0e8, P_t=1.0e2, nz=6)
    species = ['H2O', 'CO2', 'H2O_l_s']
    x_gas = np.tile([0.6, 0.3, 0.1], (6, 1))
    fpath = str(tmp_path / '12p320_chem.nc')
    write_vulcan_netcdf(fpath, layers, species, x_gas)

    out = read_vulcan_netcdf(fpath)

    assert out['species'] == species
    assert out['x_gas'].shape == (6, 3)
    np.testing.assert_allclose(out['x_gas'][:, 0], 0.6, rtol=1e-12)
    np.testing.assert_allclose(out['x_gas'].sum(axis=1), 1.0, rtol=1e-12)
    # dyn cm-2 as stored: a Pa conversion would give 1e7 at the bottom
    np.testing.assert_allclose(out['p'], layers['p'], rtol=1e-12)
    assert out['p'][0] == pytest.approx(1.0e8, rel=1e-12)
    assert np.all(out['p'] > 0.0) and np.all(np.diff(out['p']) < 0.0)
    np.testing.assert_allclose(out['tmp'], layers['tmp'], rtol=1e-12)
    np.testing.assert_allclose(out['zl'], layers['zl'], rtol=1e-12)
    assert np.isnan(out['Kzz'][0]) and np.isnan(out['Kzz'][-1])
    np.testing.assert_allclose(out['Kzz'][1:-1], layers['Kzz'], rtol=1e-12)


def test_read_vulcan_netcdf_missing_file_raises(tmp_path):
    """A missing NetCDF raises rather than returning an empty result, and the
    reader creates nothing in its place."""
    fpath = tmp_path / 'missing_chem.nc'
    with pytest.raises(FileNotFoundError):
        read_vulcan_netcdf(str(fpath))
    assert not fpath.exists()
