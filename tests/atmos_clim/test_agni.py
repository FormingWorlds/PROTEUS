"""
Unit tests for proteus.atmos_clim.agni module.

This module tests the AGNI atmosphere interface including:
- Aerosol discovery (_determine_aerosols)
- Condensate species determination (_determine_condensates)
- AGNI atmosphere initialization (init_agni_atmos)

See also:
- docs/How-to/test_infrastructure.md
- docs/How-to/test_categorization.md
- docs/How-to/test_building.md
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import proteus.atmos_clim.agni as agni_mod
from proteus.atmos_clim.agni import _determine_aerosols, _determine_condensates, init_agni_atmos

pytestmark = pytest.mark.unit


@pytest.mark.unit
@patch('proteus.atmos_clim.agni.os.listdir')
@patch('proteus.atmos_clim.agni.os.path.isdir')
def test_determine_aerosols_success(mock_isdir, mock_listdir):
    """
    Test aerosol discovery when scattering data directory exists.

    Physical scenario: Scattering data for aerosols (e.g., sulfate, silicate)
    is available in FWL_DATA/scattering/scattering/*.mon files.
    """
    mock_isdir.return_value = True
    mock_listdir.return_value = [
        'Sulfate.mon',
        'Silicate.mon',
        'Haze.mon',
        'other_file.txt',  # Should be ignored
        'readme.md',  # Should be ignored
    ]

    dirs = {'fwl': '/fake/fwl/path'}
    aerosols = _determine_aerosols(dirs)

    # Verify correct aerosols found and sorted
    assert len(aerosols) == 3
    assert aerosols == ['Haze', 'Silicate', 'Sulfate']  # alphabetically sorted

    # Verify correct directory was checked
    mock_isdir.assert_called_once_with('/fake/fwl/path/scattering/scattering')


@pytest.mark.unit
@patch('proteus.atmos_clim.agni.os.path.isdir')
def test_determine_aerosols_missing_directory(mock_isdir):
    """
    Test aerosol discovery when scattering directory doesn't exist.

    Physical scenario: FWL_DATA not properly downloaded or scattering
    data not installed. Should return empty list and warn.
    """
    mock_isdir.return_value = False

    dirs = {'fwl': '/nonexistent/path'}
    aerosols = _determine_aerosols(dirs)

    # Should return empty list without crashing
    assert aerosols == []
    mock_isdir.assert_called_once()


@pytest.mark.unit
@patch('proteus.atmos_clim.agni.os.listdir')
@patch('proteus.atmos_clim.agni.os.path.isdir')
def test_determine_aerosols_empty_directory(mock_isdir, mock_listdir):
    """
    Test aerosol discovery when directory exists but has no .mon files.

    Physical scenario: Scattering directory present but empty or only
    contains non-aerosol files.
    """
    mock_isdir.return_value = True
    mock_listdir.return_value = ['readme.txt', 'config.yaml']

    dirs = {'fwl': '/path/to/fwl'}
    aerosols = _determine_aerosols(dirs)

    # Should return empty list
    assert aerosols == []


@pytest.mark.unit
@patch('proteus.atmos_clim.agni.os.listdir')
@patch('proteus.atmos_clim.agni.os.path.isdir')
def test_determine_aerosols_single_species(mock_isdir, mock_listdir):
    """
    Test aerosol discovery with only one aerosol type.

    Physical scenario: Limited scattering data with only one aerosol species
    available (e.g., only sulfate aerosols).
    """
    mock_isdir.return_value = True
    mock_listdir.return_value = ['Sulfate.mon']

    dirs = {'fwl': '/path/to/fwl'}
    aerosols = _determine_aerosols(dirs)

    assert len(aerosols) == 1
    assert aerosols == ['Sulfate']


@pytest.mark.unit
def test_determine_condensates():
    """
    Test condensate species determination from volatile list.

    Physical scenario: Given a list of volatile species, filter out
    those that are always dry (H2, N2, CO) to get condensable
    species like H2O, CO2, He, CH4, etc.
    """
    # Test with mixed list of dry and condensable species
    vol_list = ['H2O', 'CO2', 'N2', 'CH4', 'He', 'H2', 'CO']
    condensates = _determine_condensates(vol_list)

    # N2, H2, CO should be filtered out (always dry)
    assert 'H2O' in condensates
    assert 'CO2' in condensates
    assert 'CH4' in condensates
    assert 'He' in condensates  # He is condensable in AGNI
    assert 'N2' not in condensates
    assert 'H2' not in condensates
    assert 'CO' not in condensates


@pytest.mark.unit
def test_determine_condensates_all_dry():
    """
    Test condensate determination with only dry species.

    Physical scenario: Hydrogen-nitrogen-CO dominated atmosphere with no
    condensable species.
    """
    vol_list = ['H2', 'N2', 'CO']
    condensates = _determine_condensates(vol_list)

    # Should return empty list
    assert condensates == []


@pytest.mark.unit
def test_determine_condensates_all_condensable():
    """
    Test condensate determination with all condensable species.

    Physical scenario: Rocky planet atmosphere with water, CO2, and other
    condensable volatiles but no hydrogen/helium.
    """
    vol_list = ['H2O', 'CO2', 'NH3', 'CH4', 'SO2']
    condensates = _determine_condensates(vol_list)

    # All should remain
    assert len(condensates) == len(vol_list)
    assert set(condensates) == set(vol_list)


@pytest.mark.unit
def test_determine_condensates_empty_list():
    """
    Test condensate determination with empty volatile list.

    Physical scenario: Edge case where no volatiles are specified.
    """
    condensates = _determine_condensates([])
    assert condensates == []


class _FakeAtmosphere:
    def __init__(self):
        self.transparent = False


class _FakeAGNI:
    def __init__(self):
        self.last_setup_args = None
        self.last_setup_kwargs = None
        self.last_allocate_input_star = None
        self.atmosphere = SimpleNamespace(
            Atmos_t=lambda: _FakeAtmosphere(),
            setup_b=self._setup_b,
            allocate_b=self._allocate_b,
        )
        # setpt routines: record-only stubs
        self.setpt = SimpleNamespace(
            fromncdf_b=lambda *_a, **_k: None,
            loglinear_b=lambda *_a, **_k: None,
            isothermal_b=lambda *_a, **_k: None,
            dry_adiabat_b=lambda *_a, **_k: None,
            analytic_b=lambda *_a, **_k: None,
            stratosphere_b=lambda *_a, **_k: None,
        )

    def _setup_b(self, atmos, *args, **kwargs):
        self.last_setup_args = args
        self.last_setup_kwargs = kwargs
        return True

    def _allocate_b(self, atmos, input_star, **kwargs):
        self.last_allocate_input_star = input_star
        return True


def _build_greygas_config():
    """Build a config object that triggers the grey-gas dispatch."""
    return SimpleNamespace(
        atmos_clim=SimpleNamespace(
            aerosols_enabled=False,
            cloud_enabled=False,
            rayleigh=False,
            surf_greyalbedo=0.3,
            surface_d=0.0,
            surface_k=0.0,
            tmp_minimum=50.0,
            num_levels=40,
            p_top=1e-5,
            overlap_method='ro',
            agni=SimpleNamespace(
                spectral_file='greygas',
                verbosity=2,
                oceans=False,
                rainout=False,
                chemistry='none',
                surf_material='greybody',
                surf_roughness=0.0,
                surf_windspeed=0.0,
                phs_timescale=1.0,
                evap_efficiency=1.0,
                fastchem_floor=1e-30,
                fastchem_maxiter_chem=1,
                fastchem_maxiter_solv=1,
                fastchem_xtol_chem=1e-6,
                fastchem_xtol_elem=1e-6,
                real_gas=False,
                mlt_criterion='a',
                ini_profile='isothermal',
                grey_opacity_lw=0.1,
                grey_opacity_sw=0.2,
                check_safe_gas=False,
            ),
        ),
        orbit=SimpleNamespace(s0_factor=1.0, zenith_angle=48.0),
        params=SimpleNamespace(out=SimpleNamespace(logging='INFO')),
    )


@pytest.mark.unit
def test_init_agni_atmos_greygas_bypasses_spectral_copy(monkeypatch, tmp_path):
    """Greygas path should not call get_spfile_path or pass a stellar spectrum.

    When spectral_file='greygas' is set, AGNI uses the grey-gas RT scheme and
    does not need a SOCRATES spectral file or stellar flux to be copied into
    the runtime directory.
    """
    fake_agni = _FakeAGNI()
    fake_jl = SimpleNamespace(AGNI=fake_agni, Dict=dict, Char=str)

    output_dir = tmp_path / 'out'
    data_dir = output_dir / 'data'
    data_dir.mkdir(parents=True)
    (data_dir / '100.sflux').write_text('sflux', encoding='utf-8')

    dirs = {'output': str(output_dir), 'agni': '/fake/agni', 'fwl': '/fake/fwl'}
    config = _build_greygas_config()
    hf_row = {
        'F_ins': 1000.0,
        'albedo_pl': 0.2,
        'T_surf': 900.0,
        'gravity': 9.8,
        'R_int': 6.4e6,
        'P_surf': 1.0,
    }

    monkeypatch.setattr(agni_mod, 'jl', fake_jl)
    monkeypatch.setattr(agni_mod, 'convert', lambda _typ, value: value)
    monkeypatch.setattr(agni_mod, '_construct_voldict', lambda *_a, **_k: {'H2O': 1.0})
    monkeypatch.setattr(agni_mod, 'sync_log_files', lambda *_a, **_k: None)
    monkeypatch.setattr(
        agni_mod,
        'get_spfile_path',
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError('get_spfile_path should not be called for greygas')
        ),
    )

    atmos = init_agni_atmos(dirs, config, hf_row)

    assert atmos is not None

    # setup_b positional args: [dirs['agni'], dirs['output'], input_sf, ...]
    assert fake_agni.last_setup_args[2] == 'greygas'

    # Empty stellar path prevents AGNI from modifying/copying runtime spectral assets.
    assert fake_agni.last_allocate_input_star == ''

    # grey_opacity_lw/sw should be forwarded as the Greek-named AGNI kwargs.
    assert fake_agni.last_setup_kwargs['κ_grey_lw'] == pytest.approx(0.1)
    assert fake_agni.last_setup_kwargs['κ_grey_sw'] == pytest.approx(0.2)


@pytest.mark.unit
def test_init_agni_atmos_greygas_does_not_glob_sflux(monkeypatch, tmp_path):
    """Regression: in grey-gas mode, init_agni_atmos must not require any
    *.sflux file to exist. Before this fix, an unconditional
    `glob.glob('*.sflux'); sorted(...)[-1]` at the top of the function
    would crash with IndexError on a fresh output dir before reaching the
    grey-gas dispatch.
    """
    fake_agni = _FakeAGNI()
    fake_jl = SimpleNamespace(AGNI=fake_agni, Dict=dict, Char=str)

    output_dir = tmp_path / 'out'
    data_dir = output_dir / 'data'
    data_dir.mkdir(parents=True)
    # NOTE: no *.sflux file written. Pre-fix this would have crashed.

    dirs = {'output': str(output_dir), 'agni': '/fake/agni', 'fwl': '/fake/fwl'}
    config = _build_greygas_config()
    hf_row = {
        'F_ins': 1000.0,
        'albedo_pl': 0.2,
        'T_surf': 900.0,
        'gravity': 9.8,
        'R_int': 6.4e6,
        'P_surf': 1.0,
    }

    monkeypatch.setattr(agni_mod, 'jl', fake_jl)
    monkeypatch.setattr(agni_mod, 'convert', lambda _typ, value: value)
    monkeypatch.setattr(agni_mod, '_construct_voldict', lambda *_a, **_k: {'H2O': 1.0})
    monkeypatch.setattr(agni_mod, 'sync_log_files', lambda *_a, **_k: None)
    monkeypatch.setattr(
        agni_mod,
        'get_spfile_path',
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError('get_spfile_path should not be called for greygas')
        ),
    )

    # Must not raise.
    atmos = init_agni_atmos(dirs, config, hf_row)

    assert atmos is not None
    assert fake_agni.last_setup_args[2] == 'greygas'
    assert fake_agni.last_allocate_input_star == ''


@pytest.mark.unit
def test_init_agni_atmos_non_greygas_no_sflux_raises_filenotfound(monkeypatch, tmp_path):
    """When AGNI needs a fresh spectral file (no runtime.sf, no
    user-provided path), it must have a stellar spectrum to seed from.
    A missing *.sflux in that branch should raise FileNotFoundError
    instead of IndexError, so the caller sees a clear diagnostic."""
    fake_agni = _FakeAGNI()
    fake_jl = SimpleNamespace(AGNI=fake_agni, Dict=dict, Char=str)

    output_dir = tmp_path / 'out'
    data_dir = output_dir / 'data'
    data_dir.mkdir(parents=True)
    # No *.sflux; no runtime.sf either.

    dirs = {'output': str(output_dir), 'agni': '/fake/agni', 'fwl': '/fake/fwl'}
    # Use the same scaffold as greygas test but flip spectral_file to None
    # so the function takes the "AGNI copy from FWL_DATA" branch.
    config = _build_greygas_config()
    config.atmos_clim.agni.spectral_file = None
    hf_row = {
        'F_ins': 1000.0,
        'albedo_pl': 0.2,
        'T_surf': 900.0,
        'gravity': 9.8,
        'R_int': 6.4e6,
        'P_surf': 1.0,
    }

    monkeypatch.setattr(agni_mod, 'jl', fake_jl)
    monkeypatch.setattr(agni_mod, 'convert', lambda _typ, value: value)
    monkeypatch.setattr(agni_mod, '_construct_voldict', lambda *_a, **_k: {'H2O': 1.0})
    monkeypatch.setattr(agni_mod, 'sync_log_files', lambda *_a, **_k: None)
    monkeypatch.setattr(agni_mod, 'UpdateStatusfile', lambda *_a, **_k: None)
    monkeypatch.setattr(agni_mod, 'get_spfile_path', lambda *_a, **_k: '/fake/spfile')

    with pytest.raises(FileNotFoundError, match='No stellar spectrum'):
        init_agni_atmos(dirs, config, hf_row)
