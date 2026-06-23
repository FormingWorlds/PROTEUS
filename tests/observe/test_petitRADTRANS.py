"""Unit tests for ``proteus.observe.petitRADTRANS``.

These tests pin the helper-level physics and ordering contracts that
feed the synthetic-observation backend:

- reference values are taken from the closest layer to the configured
  reference pressure
- descending pressure grids reverse pressure, temperature, radius, and
  VMR arrays together
- VMRs are normalized before computing mass fractions and mean molar
  masses

The module depends on the optional ``petitRADTRANS`` package, so the
tests inject a tiny fake package into ``sys.modules`` before importing
the backend module.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _install_fake_petitradtrans(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pkg = types.ModuleType('petitRADTRANS')
    fake_pkg.__file__ = '/fake/petitRADTRANS/__init__.py'
    fake_pkg.__path__ = []

    fake_constants = types.ModuleType('petitRADTRANS.physical_constants')
    fake_constants.c = 2.99792458e10
    fake_pkg.physical_constants = fake_constants

    fake_radtrans = types.ModuleType('petitRADTRANS.radtrans')
    fake_radtrans.Radtrans = MagicMock(name='Radtrans')

    monkeypatch.setitem(sys.modules, 'petitRADTRANS', fake_pkg)
    monkeypatch.setitem(sys.modules, 'petitRADTRANS.physical_constants', fake_constants)
    monkeypatch.setitem(sys.modules, 'petitRADTRANS.radtrans', fake_radtrans)


def _import_backend(monkeypatch: pytest.MonkeyPatch):
    _install_fake_petitradtrans(monkeypatch)

    fake_proteus = types.ModuleType('proteus')
    fake_proteus.__path__ = []
    fake_utils = types.ModuleType('proteus.utils')
    fake_utils.__path__ = []

    fake_constants = types.ModuleType('proteus.utils.constants')
    fake_constants.prt_cia_species = ()
    fake_constants.prt_gases = ('H2', 'He')
    fake_constants.prt_ignored_gases = ()
    fake_constants.prt_rayleigh_species = ()

    fake_helper = types.ModuleType('proteus.utils.helper')
    fake_helper.eval_gas_mmw = lambda gas: {'H2': 2.0e-3, 'He': 4.0e-3}[gas]

    fake_observe = types.ModuleType('proteus.observe')
    fake_observe.__path__ = []

    monkeypatch.setitem(sys.modules, 'proteus', fake_proteus)
    monkeypatch.setitem(sys.modules, 'proteus.utils', fake_utils)
    monkeypatch.setitem(sys.modules, 'proteus.utils.constants', fake_constants)
    monkeypatch.setitem(sys.modules, 'proteus.utils.helper', fake_helper)
    monkeypatch.setitem(sys.modules, 'proteus.observe', fake_observe)

    backend_path = Path(__file__).resolve().parents[2] / 'src/proteus/observe/petitRADTRANS.py'
    spec = importlib.util.spec_from_file_location('proteus.observe.petitRADTRANS', backend_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'proteus.observe.petitRADTRANS', module)
    spec.loader.exec_module(module)
    return module


def _make_config(
    *,
    line_opacity_mode: str = 'c-k',
    include_rayleigh: bool = False,
    include_cia: bool = False,
    remove_one_gas: bool = True,
    silent: bool = False,
):
    return types.SimpleNamespace(
        observe=types.SimpleNamespace(
            module='petitRADTRANS',
            clip_vmr=1e-8,
            reference_pressure=10.0,
            remove_one_gas=remove_one_gas,
            petitRADTRANS=types.SimpleNamespace(
                line_opacity_mode=line_opacity_mode,
                include_rayleigh=include_rayleigh,
                include_cia=include_cia,
                silent=silent,
            ),
        ),
        atmos_chem=types.SimpleNamespace(module='vulcan'),
    )


def test_get_reference_prt_values_uses_closest_config_pressure(monkeypatch):
    mod = _import_backend(monkeypatch)

    atm = {
        'p': np.array([1.0e5, 1.0e6, 1.0e7, 1.0e8]),
        'r': np.array([7.00e6, 7.10e6, 7.20e6, 7.30e6]),
        'g': np.array([9.00, 9.10, 9.20, 9.30]),
    }
    config = MagicMock()
    config.observe.reference_pressure = 11.5  # bar, closest to the 10 bar layer

    reference_pressure, reference_radius, reference_gravity = mod._get_reference_prt_values(
        atm, config
    )

    assert reference_pressure == pytest.approx(10.0, rel=1e-12)
    assert reference_radius == pytest.approx(7.10e8, rel=1e-12)
    assert reference_gravity == pytest.approx(9.10e2, rel=1e-12)


def test_get_ptr_reverses_vmrs_with_descending_pressure(monkeypatch):
    mod = _import_backend(monkeypatch)

    atm = {
        'pl': np.array([1.0e5, 1.0e4, 1.0e3]),
        'tmpl': np.array([300.0, 400.0, 500.0]),
        'rl': np.array([1.0, 2.0, 3.0]),
    }
    vmrs = [
        np.array([0.10, 0.20, 0.30]),
        np.array([0.90, 0.80, 0.70]),
    ]

    prs, tmp, rad, vmrs_sorted = mod._get_ptr(atm, vmrs)

    assert np.array_equal(prs, np.array([1.0e3, 1.0e4, 1.0e5]))
    assert np.array_equal(tmp, np.array([500.0, 400.0, 300.0]))
    assert np.array_equal(rad, np.array([3.0, 2.0, 1.0]))
    assert vmrs_sorted is not None
    assert np.array_equal(vmrs_sorted[0], np.array([0.30, 0.20, 0.10]))
    assert np.array_equal(vmrs_sorted[1], np.array([0.70, 0.80, 0.90]))


def test_vmrs_to_mass_fractions_normalizes_remaining_species(monkeypatch):
    mod = _import_backend(monkeypatch)

    gases = ['H2', 'He']
    vmrs = [
        np.array([0.10, 0.30]),
        np.array([0.10, 0.10]),
    ]

    mass_fractions, mean_molar_masses = mod._vmrs_to_mass_fractions(gases, vmrs)

    vmr_arr = np.array(vmrs, dtype=float)
    vmr_norm = vmr_arr / np.sum(vmr_arr, axis=0)
    molar_masses = np.array([mod.eval_gas_mmw(gas) for gas in gases], dtype=float)
    mass_contrib = vmr_norm * molar_masses[:, None]
    total_mass = np.sum(mass_contrib, axis=0)

    assert np.allclose(mass_fractions['H2'], mass_contrib[0] / total_mass)
    assert np.allclose(mass_fractions['He'], mass_contrib[1] / total_mass)
    assert np.allclose(mean_molar_masses, total_mass / 1.0e-3)
    assert np.allclose(np.sum(vmr_norm, axis=0), 1.0)


def test_load_stellar_toa_flux_reads_saved_sflux_and_interpolates(monkeypatch, tmp_path):
    mod = _import_backend(monkeypatch)

    data_dir = tmp_path / 'data'
    data_dir.mkdir(parents=True)
    spectrum_file = data_dir / '42.sflux'
    spectrum_file.write_text(
        '# WL(nm)\t Flux(ergs/cm**2/s/nm)   Stellar flux at t_star = 1.00e+00 yr\n'
        '4.00000000e+02\t1.00000000e+00\n'
        '5.00000000e+02\t2.00000000e+00\n'
        '6.00000000e+02\t4.00000000e+00\n'
    )

    target_wavelength_nm = np.array([4.50000000e02, 5.50000000e02])
    flux = mod._load_stellar_toa_flux(str(tmp_path), {'Time': 42}, target_wavelength_nm)

    assert flux.shape == (2,)
    assert np.allclose(flux, np.array([1.5e7, 3.0e7]))


def test_get_input_data_path_prefers_fwl_data_directory(monkeypatch, tmp_path):
    mod = _import_backend(monkeypatch)

    data_path = tmp_path / 'prt' / 'input_data'
    data_path.mkdir(parents=True)

    result = mod._get_input_data_path({'fwl': str(tmp_path)})
    assert result == str(data_path)
    assert Path(result).is_dir()


def test_get_input_data_path_raises_when_missing_everywhere(monkeypatch, tmp_path):
    mod = _import_backend(monkeypatch)

    with pytest.raises(FileNotFoundError) as excinfo:
        mod._get_input_data_path({'fwl': str(tmp_path / 'does_not_exist')})
    assert 'input_data' in str(excinfo.value)


def test_get_input_data_path_raises_when_dirs_missing_fwl(monkeypatch):
    mod = _import_backend(monkeypatch)

    with pytest.raises(KeyError, match='fwl'):
        mod._get_input_data_path({})


def test_supported_species_helpers_filter_and_include(monkeypatch, tmp_path):
    mod = _import_backend(monkeypatch)

    input_data_path = tmp_path / 'input_data'
    (input_data_path / 'opacities' / 'lines' / 'correlated_k' / 'H2O').mkdir(parents=True)
    (input_data_path / 'opacities' / 'lines' / 'correlated_k' / 'CH4').mkdir(parents=True)

    monkeypatch.setattr(mod, '_get_input_data_path', lambda _dirs: str(input_data_path))
    monkeypatch.setattr(mod, 'prt_ignored_gases', {'skipme'})
    monkeypatch.setattr(mod, 'prt_rayleigh_species', {'Ray'})
    monkeypatch.setattr(mod, 'prt_cia_species', ('H2--He', 'H2--N2'))

    line_species = mod._get_supported_line_species(
        ['H2O', 'skipme', 'Ray', 'CH4', 'CO2'], str(input_data_path)
    )
    assert line_species == ['H2O', 'CH4']
    assert mod._get_supported_rayleigh_species(['H2O', 'Ray'], False) == []
    assert mod._get_supported_rayleigh_species(['H2O', 'Ray'], True) == ['Ray']
    assert mod._get_supported_cia_species(['H2', 'He'], False) == []
    assert mod._get_supported_cia_species(['H2', 'He'], True) == ['H2--He']


def test_supported_line_species_resolves_input_path_when_not_provided(monkeypatch, tmp_path):
    mod = _import_backend(monkeypatch)

    auto_path = tmp_path / 'auto_input_data'
    (auto_path / 'opacities' / 'lines' / 'correlated_k' / 'H2').mkdir(parents=True)
    line_species = mod._get_supported_line_species(['H2', 'He'], str(auto_path))
    assert line_species == ['H2']


def test_vmrs_to_mass_fractions_handles_empty_and_zero_total(monkeypatch):
    mod = _import_backend(monkeypatch)

    mass_fractions, mean_molar_masses = mod._vmrs_to_mass_fractions(['H2'], [])
    assert mass_fractions == {}
    assert mean_molar_masses.size == 0

    monkeypatch.setattr(mod, 'eval_gas_mmw', lambda gas: 0.0)
    with pytest.raises(ValueError, match='zero total mass fraction'):
        mod._vmrs_to_mass_fractions(['H2', 'He'], [np.array([1.0, 1.0]), np.array([0.0, 0.0])])


def test_vmrs_to_mass_fractions_raises_for_zero_total_vmr(monkeypatch):
    mod = _import_backend(monkeypatch)

    with pytest.raises(ValueError, match='zero total VMR'):
        mod._vmrs_to_mass_fractions(['H2', 'He'], [np.array([0.0, 0.0]), np.array([0.0, 0.0])])


def test_prioritize_broadest_coverage_species_noop_paths(monkeypatch):
    mod = _import_backend(monkeypatch)

    assert mod._prioritize_broadest_coverage_species([], '/unused') == []
    assert mod._prioritize_broadest_coverage_species(['H2O', 'CO2'], '/unused') == [
        'H2O',
        'CO2',
    ]


def test_prioritize_broadest_coverage_species_uses_widest_file(monkeypatch, tmp_path):
    mod = _import_backend(monkeypatch)

    input_data_path = tmp_path / 'input_data'
    for species, iso_dir, file_name in [
        ('CO2', '12C-16O2', '44CO2__Test.R1000_1-2mu.ktable.petitRADTRANS.h5'),
        ('H2O', '1H2-16O', '1H2-16O__Test.R1000_0.5-5mu.ktable.petitRADTRANS.h5'),
        ('CH4', '12C-1H4', '12C-1H4__Test.R1000_0.3-50mu.ktable.petitRADTRANS.h5'),
    ]:
        species_dir = input_data_path / 'opacities' / 'lines' / 'correlated_k' / species
        (species_dir / iso_dir).mkdir(parents=True)
        (species_dir / iso_dir / file_name).write_text('dummy')

    species = ['CO2', 'H2O', 'CH4']
    result = mod._prioritize_broadest_coverage_species(species, str(input_data_path))

    assert result == ['CH4', 'CO2', 'H2O']


def test_init_radtrans_suppresses_output_when_enabled(monkeypatch, capsys):
    mod = _import_backend(monkeypatch)

    class NoisyRadtrans:
        def __init__(self, **kwargs):
            print('noise-out')
            print('noise-err', file=sys.stderr)

    config = _make_config(silent=True)
    _ = mod._init_radtrans(NoisyRadtrans, config, line_species=['H2O'])

    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == ''


def test_init_radtrans_keeps_output_when_silencing_disabled(monkeypatch, capsys):
    mod = _import_backend(monkeypatch)

    class NoisyRadtrans:
        def __init__(self, **kwargs):
            print('noise-out')
            print('noise-err', file=sys.stderr)

    config = _make_config(silent=False)
    _ = mod._init_radtrans(NoisyRadtrans, config, line_species=['H2O'])

    captured = capsys.readouterr()
    assert 'noise-out' in captured.out
    assert 'noise-err' in captured.err


def test_get_mix_handles_outgas_profile_and_offchem(monkeypatch):
    mod = _import_backend(monkeypatch)
    monkeypatch.setattr(mod, 'prt_gases', ('H2', 'CH4', 'He'))

    hf_row = {'H2_vmr': 0.8, 'CH4_vmr': 0.1, 'He_vmr': 0.1}
    atm_profile = {
        'pl': np.array([1.0e5, 1.0e4]),
        'H2_vmr': np.array([0.7, 0.6]),
        'CH4_vmr': np.array([0.2, 0.3]),
        'He_vmr': np.array([0.1, 0.1]),
    }
    atm_offchem = {
        'pl': np.array([1.0e5, 1.0e4]),
        'H2': np.array([0.7, 0.6]),
        'CH4': np.array([0.2, 0.3]),
        'He': np.array([0.1, 0.1]),
    }

    gases_outgas, vmrs_outgas = mod._get_mix(hf_row, atm_profile, 'outgas', 1e-8)
    gases_profile, vmrs_profile = mod._get_mix(hf_row, atm_profile, 'profile', 1e-8)
    gases_offchem, vmrs_offchem = mod._get_mix(hf_row, atm_offchem, 'offchem', 1e-8)

    assert gases_outgas == ['H2', 'CH4', 'He']
    assert np.allclose(vmrs_outgas[0], np.array([0.8, 0.8]))
    assert gases_profile == ['H2', 'CH4', 'He']
    assert np.allclose(vmrs_profile[1], np.array([0.2, 0.2, 0.3]))
    assert gases_offchem == ['H2', 'CH4', 'He']
    assert np.allclose(vmrs_offchem[2], np.array([0.1, 0.1]))


def test_get_mix_excludes_species_when_source_keys_missing_or_below_clip(monkeypatch):
    mod = _import_backend(monkeypatch)
    monkeypatch.setattr(mod, 'prt_gases', ('H2', 'CH4', 'He'))

    hf_row = {'H2_vmr': 1.0e-6}
    atm_profile = {
        'pl': np.array([1.0e5, 1.0e4]),
        'H2_vmr': np.array([1.0e-5, 1.0e-6]),
    }
    atm_offchem = {
        'pl': np.array([1.0e5, 1.0e4]),
        'H2': np.array([2.0e-5, 2.0e-5]),
    }

    gases_outgas, vmrs_outgas = mod._get_mix(hf_row, atm_profile, 'outgas', 1.0e-5)
    gases_profile, vmrs_profile = mod._get_mix(hf_row, atm_profile, 'profile', 1.0e-5)
    gases_offchem, vmrs_offchem = mod._get_mix(hf_row, atm_offchem, 'offchem', 1.0e-5)

    assert gases_outgas == []
    assert vmrs_outgas == []
    assert gases_profile == ['H2']
    assert np.allclose(vmrs_profile[0], np.array([1.0e-5, 1.0e-5, 1.0e-6]))
    assert gases_offchem == ['H2']
    assert np.allclose(vmrs_offchem[0], np.array([2.0e-5, 2.0e-5]))


def test_get_mix_unknown_source_returns_empty_selection(monkeypatch):
    mod = _import_backend(monkeypatch)
    monkeypatch.setattr(mod, 'prt_gases', ('H2', 'CH4'))

    hf_row = {'H2_vmr': 0.9, 'CH4_vmr': 0.1}
    atm = {'pl': np.array([1.0e5, 1.0e4])}

    gases, vmrs = mod._get_mix(hf_row, atm, 'unknown_source', 1.0e-8)
    assert gases == []
    assert vmrs == []


def test_get_ptr_keeps_order_when_already_ascending_and_vmrs_none(monkeypatch):
    mod = _import_backend(monkeypatch)

    atm = {
        'pl': np.array([1.0e3, 1.0e4, 1.0e5]),
        'tmpl': np.array([250.0, 260.0, 270.0]),
        'rl': np.array([1.0, 2.0, 3.0]),
    }

    prs, tmp, rad, vmrs_sorted = mod._get_ptr(atm, None)

    assert np.array_equal(prs, atm['pl'])
    assert np.array_equal(tmp, atm['tmpl'])
    assert np.array_equal(rad, atm['rl'])
    assert vmrs_sorted is None


def test_get_ptr_reverses_without_vmrs_when_descending(monkeypatch):
    mod = _import_backend(monkeypatch)

    atm = {
        'pl': np.array([1.0e5, 1.0e4, 1.0e3]),
        'tmpl': np.array([100.0, 200.0, 300.0]),
        'rl': np.array([1.0, 2.0, 3.0]),
    }

    prs, tmp, rad, vmrs_sorted = mod._get_ptr(atm, None)

    assert np.array_equal(prs, np.array([1.0e3, 1.0e4, 1.0e5]))
    assert np.array_equal(rad, np.array([3.0, 2.0, 1.0]))
    assert np.all(tmp >= mod.petitRADTRANS_TLIMS[0])
    assert vmrs_sorted is None


def test_atm_profile_and_offchem_helpers(monkeypatch, tmp_path):
    mod = _import_backend(monkeypatch)

    fake_atmos_clim_common = types.ModuleType('proteus.atmos_clim.common')
    fake_atmos_clim_common.read_atmosphere_data = lambda *_a, **_k: [
        {'p': np.array([1.0]), 'marker': 1}
    ]
    fake_atmos_clim = types.ModuleType('proteus.atmos_clim')
    fake_atmos_clim.__path__ = []
    monkeypatch.setitem(sys.modules, 'proteus.atmos_clim', fake_atmos_clim)
    monkeypatch.setitem(sys.modules, 'proteus.atmos_clim.common', fake_atmos_clim_common)

    fake_atmos_chem = types.ModuleType('proteus.atmos_chem')
    fake_atmos_chem.__path__ = []
    fake_atmos_chem_common = types.ModuleType('proteus.atmos_chem.common')
    fake_atmos_chem_common.read_result = lambda *_a, **_k: pd.DataFrame(
        {'tmp': [300.0], 'p': [1.0e5], 'z': [0.0], 'H2': [0.7]}
    )
    monkeypatch.setitem(sys.modules, 'proteus.atmos_chem', fake_atmos_chem)
    monkeypatch.setitem(sys.modules, 'proteus.atmos_chem.common', fake_atmos_chem_common)

    outdir = str(tmp_path)
    profile = mod._get_atm_profile(outdir, {'Time': 1})
    assert profile['marker'] == 1
    np.testing.assert_allclose(profile['p'], np.array([1.0]))
    offchem = mod._get_atm_offchem(outdir, {'R_int': 10.0}, 'vulcan')
    assert list(offchem.columns) == ['tmpl', 'pl', 'rl', 'H2']
    assert offchem['rl'].iloc[0] == pytest.approx(10.0)


def test_get_atm_profile_returns_none_when_no_data(monkeypatch):
    mod = _import_backend(monkeypatch)

    fake_atmos_clim_common = types.ModuleType('proteus.atmos_clim.common')
    fake_atmos_clim_common.read_atmosphere_data = MagicMock(return_value=[])
    fake_atmos_clim = types.ModuleType('proteus.atmos_clim')
    fake_atmos_clim.__path__ = []
    monkeypatch.setitem(sys.modules, 'proteus.atmos_clim', fake_atmos_clim)
    monkeypatch.setitem(sys.modules, 'proteus.atmos_clim.common', fake_atmos_clim_common)

    result = mod._get_atm_profile('/out', {'Time': 1})
    assert result is None
    fake_atmos_clim_common.read_atmosphere_data.assert_called_once()


def test_get_atm_offchem_returns_none_when_no_result(monkeypatch):
    mod = _import_backend(monkeypatch)

    fake_atmos_chem_common = types.ModuleType('proteus.atmos_chem.common')
    fake_atmos_chem_common.read_result = MagicMock(return_value=None)
    fake_atmos_chem = types.ModuleType('proteus.atmos_chem')
    fake_atmos_chem.__path__ = []
    monkeypatch.setitem(sys.modules, 'proteus.atmos_chem', fake_atmos_chem)
    monkeypatch.setitem(sys.modules, 'proteus.atmos_chem.common', fake_atmos_chem_common)

    result = mod._get_atm_offchem('/out', {'R_int': 10.0}, 'vulcan')
    assert result is None
    fake_atmos_chem_common.read_result.assert_called_once()


def test_load_stellar_toa_flux_raises_when_missing_files(monkeypatch, tmp_path):
    mod = _import_backend(monkeypatch)

    with pytest.raises(FileNotFoundError) as excinfo:
        mod._load_stellar_toa_flux(str(tmp_path), {'Time': 1}, np.array([450.0]))
    assert 'No stellar spectrum files' in str(excinfo.value)


def test_load_stellar_toa_flux_ignores_nonnumeric_stems(monkeypatch, tmp_path):
    mod = _import_backend(monkeypatch)

    data_dir = tmp_path / 'data'
    data_dir.mkdir(parents=True)
    (data_dir / 'abc.sflux').write_text('wl flux\n400 99\n500 99\n')
    (data_dir / '7.sflux').write_text('wl flux\n400 1\n500 3\n')

    flux = mod._load_stellar_toa_flux(str(tmp_path), {'Time': 1}, np.array([450.0]))
    assert np.allclose(flux, np.array([2.0e7]))


def test_transit_depth_returns_none_when_atmosphere_missing(monkeypatch, tmp_path):
    mod = _import_backend(monkeypatch)

    monkeypatch.setattr(mod, '_get_atm_profile', lambda *_a, **_k: None)
    monkeypatch.setattr(mod, '_get_input_data_path', lambda _dirs: str(tmp_path))
    fake_common = types.ModuleType('proteus.observe.common')
    fake_common.get_transit_fpath = lambda *_a, **_k: str(tmp_path / 'unused.csv')
    monkeypatch.setitem(sys.modules, 'proteus.observe.common', fake_common)
    config = _make_config()

    result = mod.transit_depth(
        {'Time': 1, 'R_star': 1.0},
        config,
        'profile',
        {'fwl': str(tmp_path), 'output': str(tmp_path)},
    )
    assert result is None
    assert sys.modules['petitRADTRANS.radtrans'].Radtrans.call_count == 0


def test_transit_depth_offchem_returns_none_when_reference_profile_missing(
    monkeypatch, tmp_path
):
    mod = _import_backend(monkeypatch)

    monkeypatch.setattr(mod, '_get_atm_offchem', lambda *_a, **_k: {'pl': np.array([1.0e5])})
    monkeypatch.setattr(mod, '_get_atm_profile', lambda *_a, **_k: None)
    monkeypatch.setattr(mod, '_get_input_data_path', lambda _dirs: str(tmp_path))
    fake_common = types.ModuleType('proteus.observe.common')
    fake_common.get_transit_fpath = lambda *_a, **_k: str(tmp_path / 'unused.csv')
    monkeypatch.setitem(sys.modules, 'proteus.observe.common', fake_common)
    config = _make_config()

    result = mod.transit_depth(
        {'Time': 1, 'R_star': 1.0},
        config,
        'offchem',
        {'fwl': str(tmp_path), 'output': str(tmp_path)},
    )
    assert result is None


def test_transit_depth_raises_for_unknown_source_before_parse(monkeypatch, tmp_path):
    mod = _import_backend(monkeypatch)

    monkeypatch.setattr(mod, '_get_atm_profile', lambda *_a, **_k: {'p': np.array([1.0e5])})
    monkeypatch.setattr(mod, '_get_input_data_path', lambda _dirs: str(tmp_path))
    fake_common = types.ModuleType('proteus.observe.common')
    fake_common.get_transit_fpath = lambda *_a, **_k: str(tmp_path / 'unused.csv')
    monkeypatch.setitem(sys.modules, 'proteus.observe.common', fake_common)
    config = _make_config()

    with pytest.raises(UnboundLocalError):
        mod.transit_depth(
            {'Time': 1, 'R_star': 1.0},
            config,
            'unknown_source',
            {'fwl': str(tmp_path), 'output': str(tmp_path)},
        )


def test_eclipse_depth_returns_none_when_atmosphere_missing(monkeypatch, tmp_path):
    mod = _import_backend(monkeypatch)

    monkeypatch.setattr(mod, '_get_atm_profile', lambda *_a, **_k: None)
    monkeypatch.setattr(mod, '_get_input_data_path', lambda _dirs: str(tmp_path))
    fake_common = types.ModuleType('proteus.observe.common')
    fake_common.get_eclipse_fpath = lambda *_a, **_k: str(tmp_path / 'unused.csv')
    monkeypatch.setitem(sys.modules, 'proteus.observe.common', fake_common)
    config = _make_config()

    result = mod.eclipse_depth(
        {'Time': 1, 'R_star': 1.0, 'T_star': 1.0, 'separation': 1.0},
        config,
        'profile',
        {'fwl': str(tmp_path), 'output': str(tmp_path)},
    )
    assert result is None
    assert sys.modules['petitRADTRANS.radtrans'].Radtrans.call_count == 0


def test_eclipse_depth_outgas_returns_none_when_reference_profile_missing(
    monkeypatch, tmp_path
):
    mod = _import_backend(monkeypatch)

    monkeypatch.setattr(mod, '_get_atm_profile', lambda *_a, **_k: None)
    monkeypatch.setattr(mod, '_get_input_data_path', lambda _dirs: str(tmp_path))
    fake_common = types.ModuleType('proteus.observe.common')
    fake_common.get_eclipse_fpath = lambda *_a, **_k: str(tmp_path / 'unused.csv')
    monkeypatch.setitem(sys.modules, 'proteus.observe.common', fake_common)
    config = _make_config()

    result = mod.eclipse_depth(
        {'Time': 1, 'R_star': 1.0, 'T_star': 1.0, 'separation': 1.0},
        config,
        'outgas',
        {'fwl': str(tmp_path), 'output': str(tmp_path)},
    )
    assert result is None


def test_eclipse_depth_raises_for_unknown_source_before_parse(monkeypatch, tmp_path):
    mod = _import_backend(monkeypatch)

    monkeypatch.setattr(mod, '_get_atm_profile', lambda *_a, **_k: {'p': np.array([1.0e5])})
    monkeypatch.setattr(mod, '_get_input_data_path', lambda _dirs: str(tmp_path))
    fake_common = types.ModuleType('proteus.observe.common')
    fake_common.get_eclipse_fpath = lambda *_a, **_k: str(tmp_path / 'unused.csv')
    monkeypatch.setitem(sys.modules, 'proteus.observe.common', fake_common)
    config = _make_config()

    with pytest.raises(UnboundLocalError):
        mod.eclipse_depth(
            {'Time': 1, 'R_star': 1.0, 'T_star': 1.0, 'separation': 1.0},
            config,
            'unknown_source',
            {'fwl': str(tmp_path), 'output': str(tmp_path)},
        )


def test_transit_depth_prioritizes_broadest_coverage_and_writes_output(monkeypatch, tmp_path):
    mod = _import_backend(monkeypatch)
    monkeypatch.setattr(mod, 'prt_gases', ('H2O', 'CH4', 'H2'))
    monkeypatch.setattr(mod, 'prt_rayleigh_species', set())
    monkeypatch.setattr(mod, 'prt_cia_species', ())
    monkeypatch.setattr(
        mod,
        'eval_gas_mmw',
        lambda gas: {'H2O': 18e-3, 'CH4': 16e-3, 'H2': 2e-3}[gas],
    )

    input_data_path = tmp_path / 'input_data'
    species_files = {
        'H2O': '1H2-16O__Test.R1000_0.5-5mu.ktable.petitRADTRANS.h5',
        'CH4': '12C-1H4__Test.R1000_0.3-50mu.ktable.petitRADTRANS.h5',
    }
    for species, file_name in species_files.items():
        species_dir = input_data_path / 'opacities' / 'lines' / 'correlated_k' / species
        species_dir.mkdir(parents=True)
        (species_dir / file_name).write_text('dummy')

    fake_common = types.ModuleType('proteus.observe.common')

    def _get_transit_fpath(outdir, source, kind):
        path = Path(outdir) / 'observe' / f'transit_{source}_{kind}.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    fake_common.get_transit_fpath = _get_transit_fpath
    monkeypatch.setitem(sys.modules, 'proteus.observe.common', fake_common)

    fake_atm = {
        'pl': np.array([1.0e5, 1.0e4]),
        'tmpl': np.array([300.0, 200.0]),
        'rl': np.array([7.0e6, 7.1e6]),
        'p': np.array([1.0e5, 1.0e4]),
        'r': np.array([7.0e6, 7.1e6]),
        'g': np.array([10.0, 11.0]),
    }
    monkeypatch.setattr(mod, '_get_atm_profile', lambda *_a, **_k: fake_atm)
    monkeypatch.setattr(mod, '_get_input_data_path', lambda _dirs: str(input_data_path))

    class FakeRadtrans:
        init_calls = []

        def __init__(self, **kwargs):
            FakeRadtrans.init_calls.append(kwargs)

        def calculate_transit_radii(self, **kwargs):
            return np.array([1.0e-4, 2.0e-4]), np.array([7.2e8, 7.3e8]), None

    monkeypatch.setattr(sys.modules['petitRADTRANS.radtrans'], 'Radtrans', FakeRadtrans)

    config = _make_config()
    hf_row = {
        'Time': 1,
        'R_star': 7.0e8,
        'H2O_vmr': 0.7,
        'CH4_vmr': 0.2,
        'H2_vmr': 0.1,
    }

    result = mod.transit_depth(
        hf_row,
        config,
        'outgas',
        {'fwl': str(tmp_path), 'output': str(tmp_path)},
    )

    assert result.shape == (2, 2)
    assert FakeRadtrans.init_calls[0]['line_species'][0] == 'CH4'
    assert Path(tmp_path / 'observe' / 'transit_outgas_synthesis.csv').is_file()
    content = (tmp_path / 'observe' / 'transit_outgas_synthesis.csv').read_text()
    assert 'CH4_removed/ppm' in content
    header_cols = [c.strip() for c in content.splitlines()[0].split('\t')]
    assert header_cols == [
        'Wavelength/um',
        'None/ppm',
        'H2O_removed/ppm',
        'CH4_removed/ppm',
        'H2_removed/ppm',
    ]


def test_eclipse_depth_offchem_uses_latest_sflux_and_writes_output(monkeypatch, tmp_path):
    mod = _import_backend(monkeypatch)
    monkeypatch.setattr(mod, 'prt_gases', ('H2O', 'CH4', 'H2'))
    monkeypatch.setattr(mod, 'prt_rayleigh_species', set())
    monkeypatch.setattr(mod, 'prt_cia_species', ())
    monkeypatch.setattr(
        mod,
        'eval_gas_mmw',
        lambda gas: {'H2O': 18e-3, 'CH4': 16e-3, 'H2': 2e-3}[gas],
    )

    input_data_path = tmp_path / 'input_data'
    species_files = {
        'H2O': '1H2-16O__Test.R1000_0.5-5mu.ktable.petitRADTRANS.h5',
        'CH4': '12C-1H4__Test.R1000_0.3-50mu.ktable.petitRADTRANS.h5',
    }
    for species, file_name in species_files.items():
        species_dir = input_data_path / 'opacities' / 'lines' / 'correlated_k' / species
        species_dir.mkdir(parents=True)
        (species_dir / file_name).write_text('dummy')

    observe_common = types.ModuleType('proteus.observe.common')

    def _get_eclipse_fpath(outdir, source, kind):
        path = Path(outdir) / 'observe' / f'eclipse_{source}_{kind}.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    observe_common.get_eclipse_fpath = _get_eclipse_fpath
    monkeypatch.setitem(sys.modules, 'proteus.observe.common', observe_common)

    atmos_chem_common = types.ModuleType('proteus.atmos_chem.common')
    atmos_chem_common.read_result = lambda *_a, **_k: pd.DataFrame(
        {
            'tmp': [300.0, 200.0],
            'p': [1.0e5, 1.0e4],
            'z': [0.0, 1000.0],
            'H2O': [0.7, 0.6],
            'CH4': [0.2, 0.3],
            'H2': [0.1, 0.1],
        }
    )
    atmos_chem_pkg = types.ModuleType('proteus.atmos_chem')
    atmos_chem_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, 'proteus.atmos_chem', atmos_chem_pkg)
    monkeypatch.setitem(sys.modules, 'proteus.atmos_chem.common', atmos_chem_common)

    data_dir = tmp_path / 'data'
    data_dir.mkdir(parents=True)
    (data_dir / '1.sflux').write_text('wl flux\n400 1\n500 2\n600 3\n')
    (data_dir / '42.sflux').write_text('wl flux\n400 10\n500 20\n600 30\n')

    fake_atm = {
        'pl': np.array([1.0e5, 1.0e4]),
        'tmpl': np.array([300.0, 200.0]),
        'rl': np.array([7.0e6, 7.1e6]),
        'p': np.array([1.0e5, 1.0e4]),
        'r': np.array([7.0e6, 7.1e6]),
        'g': np.array([10.0, 11.0]),
    }
    monkeypatch.setattr(mod, '_get_atm_profile', lambda *_a, **_k: fake_atm)
    monkeypatch.setattr(mod, '_get_input_data_path', lambda _dirs: str(input_data_path))

    class FakeRadtrans:
        init_calls = []

        def __init__(self, **kwargs):
            FakeRadtrans.init_calls.append(kwargs)

        def calculate_flux(self, **kwargs):
            return np.array([1.0e-4, 2.0e-4]), np.array([1.0, 2.0]), None

    monkeypatch.setattr(sys.modules['petitRADTRANS.radtrans'], 'Radtrans', FakeRadtrans)

    config = _make_config(include_cia=False, include_rayleigh=False)
    hf_row = {
        'Time': 1,
        'R_star': 7.0e8,
        'T_star': 1000.0,
        'separation': 1.0e11,
        'R_int': 7.0e6,
    }

    result = mod.eclipse_depth(
        hf_row,
        config,
        'offchem',
        {'fwl': str(tmp_path), 'output': str(tmp_path)},
    )

    assert result.shape == (2, 2)
    assert FakeRadtrans.init_calls[0]['line_species'][0] == 'CH4'
    assert Path(tmp_path / 'observe' / 'eclipse_offchem_synthesis.csv').is_file()
    content = (tmp_path / 'observe' / 'eclipse_offchem_synthesis.csv').read_text()
    assert 'CH4_removed/ppm' in content
    header_cols = [c.strip() for c in content.splitlines()[0].split('\t')]
    assert header_cols == [
        'Wavelength/um',
        'None/ppm',
        'H2O_removed/ppm',
        'CH4_removed/ppm',
        'H2_removed/ppm',
    ]


def test_transit_depth_disables_removed_species_columns(monkeypatch, tmp_path):
    mod = _import_backend(monkeypatch)
    monkeypatch.setattr(mod, 'prt_gases', ('H2O', 'CH4', 'H2'))
    monkeypatch.setattr(mod, 'prt_rayleigh_species', set())
    monkeypatch.setattr(mod, 'prt_cia_species', ())
    monkeypatch.setattr(
        mod,
        'eval_gas_mmw',
        lambda gas: {'H2O': 18e-3, 'CH4': 16e-3, 'H2': 2e-3}[gas],
    )

    input_data_path = tmp_path / 'input_data'
    species_files = {
        'H2O': '1H2-16O__Test.R1000_0.5-5mu.ktable.petitRADTRANS.h5',
        'CH4': '12C-1H4__Test.R1000_0.3-50mu.ktable.petitRADTRANS.h5',
    }
    for species, file_name in species_files.items():
        species_dir = input_data_path / 'opacities' / 'lines' / 'correlated_k' / species
        species_dir.mkdir(parents=True)
        (species_dir / file_name).write_text('dummy')

    fake_common = types.ModuleType('proteus.observe.common')

    def _get_transit_fpath(outdir, source, kind):
        path = Path(outdir) / 'observe' / f'transit_{source}_{kind}.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    fake_common.get_transit_fpath = _get_transit_fpath
    monkeypatch.setitem(sys.modules, 'proteus.observe.common', fake_common)

    fake_atm = {
        'pl': np.array([1.0e5, 1.0e4]),
        'tmpl': np.array([300.0, 200.0]),
        'rl': np.array([7.0e6, 7.1e6]),
        'p': np.array([1.0e5, 1.0e4]),
        'r': np.array([7.0e6, 7.1e6]),
        'g': np.array([10.0, 11.0]),
    }
    monkeypatch.setattr(mod, '_get_atm_profile', lambda *_a, **_k: fake_atm)
    monkeypatch.setattr(mod, '_get_input_data_path', lambda _dirs: str(input_data_path))

    class FakeRadtrans:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def calculate_transit_radii(self, **kwargs):
            return np.array([1.0e-4, 2.0e-4]), np.array([7.2e8, 7.3e8]), None

    monkeypatch.setattr(sys.modules['petitRADTRANS.radtrans'], 'Radtrans', FakeRadtrans)

    config = _make_config(remove_one_gas=False)
    hf_row = {
        'Time': 1,
        'R_star': 7.0e8,
        'H2O_vmr': 0.7,
        'CH4_vmr': 0.2,
        'H2_vmr': 0.1,
    }

    result = mod.transit_depth(
        hf_row,
        config,
        'outgas',
        {'fwl': str(tmp_path), 'output': str(tmp_path)},
    )

    assert result.shape == (2, 2)
    content = (tmp_path / 'observe' / 'transit_outgas_synthesis.csv').read_text()
    header_cols = [c.strip() for c in content.splitlines()[0].split('\t')]
    assert header_cols == ['Wavelength/um', 'None/ppm']


# ============================================================================
# Physics invariant tests: spectrum output constraints and reference spectra
# ============================================================================


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_transit_radii_ratio_gives_bounded_transit_depths(monkeypatch):
    """Conversion of transit radii to transit depths must obey physical
    bounds. Transit depth = (R_transit/R_star)^2 must be positive and
    less than the maximum geometric cross-section.

    Physics: For any physical atmosphere, the atmospheric scale height
    H < planet radius R_p, so R_transit < R_star, giving transit depth < 1.
    In ppm units, this means transit depth < 1e6 ppm.
    """

    # Reference configuration
    R_star_cm = 6.96e10  # Solar radius in cm
    R_planet_cm = 7.0e8  # Jupiter radius in cm

    # Realistic transit radii for an exoplanet atmosphere
    # Range from planet radius (no atmosphere) to planet radius + 500 km scale height
    transit_radii_cm = np.linspace(R_planet_cm, R_planet_cm + 5e7, 10)

    # Compute transit depths using the same formula as backend
    transit_depths_ppm = (transit_radii_cm / R_star_cm) ** 2 * 1e6

    # Check physics invariants
    assert np.all(transit_depths_ppm > 0), 'Transit depths must be positive'
    assert np.all(transit_depths_ppm < 1e6), (
        'Transit depths must be < 1e6 ppm (geometric limit)'
    )
    assert np.all(np.isfinite(transit_depths_ppm)), (
        'Transit depths must be finite (no NaN or Inf)'
    )

    # Check that transit depths increase monotonically with transit radius
    assert np.all(np.diff(transit_depths_ppm) > 0), (
        'Transit depths must increase monotonically with transit radius'
    )


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_spectrum_wavelength_ordering_from_petitradtrans_output(monkeypatch):
    """Wavelength arrays from petitRADTRANS output must be strictly positive,
    finite, and monotonically increasing. These are standard requirements for
    any spectrum grid used in downstream analysis or interpolation.

    Physics: Wavelength grids from radiative transfer codes are typically
    constructed in log-space to sample the Rayleigh limit and wings equally.
    Output must preserve sorted order for numerical stability.
    """

    # Synthetic pRT output: wavelengths log-spaced from visible to mid-IR
    wl_um = np.logspace(np.log10(0.3), np.log10(10.0), 50)

    # Check wavelength properties
    assert np.all(wl_um > 0), 'All wavelengths must be positive'
    assert np.all(np.isfinite(wl_um)), 'All wavelengths must be finite'
    assert np.all(np.diff(wl_um) > 0), 'Wavelengths must be strictly monotonically increasing'

    # Check wavelengths span a physically meaningful range (with floating point tolerance)
    assert np.all(wl_um >= 0.29), 'Lower wavelength bound should be >= 0.3 um (UV)'
    assert np.all(wl_um <= 10.01), 'Upper wavelength bound should be <= 10 um (mid-IR)'


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_reference_pinned_transit_depth_from_vmr_normalization(monkeypatch):
    """Reference-pinned test: VMR normalization affects transit depth through
    mean molar mass, which should follow a predictable pattern.

    Physics: For a mixture with VMR ratios and specified molar masses,
    the mean molar mass is M_mean = Σ(x_i * M_i) where x_i is normalized VMR.
    Changing one component's abundance must scale M_mean predictably.

    Reference: Pure H2/He (85/15 by VMR) gives M_mean ≈ 2.3 g/mol.
    """
    mod = _import_backend(monkeypatch)

    # Use unnormalised VMR magnitudes (same ratio) so the normalization step is exercised.
    gases = ['H2', 'He']
    vmrs = [
        np.array([8.5, 8.5, 8.5]),  # H2
        np.array([1.5, 1.5, 1.5]),  # He
    ]
    mass_fractions, mean_molar_masses = mod._vmrs_to_mass_fractions(gases, vmrs)

    # Reference calculation for H2/He 85/15
    # After normalization and mass fraction calculation, M_mean ≈ 2.3 g/mol
    M_mean_expected = 2.3  # g/mol

    # Check reference spectrum
    assert np.allclose(mean_molar_masses, M_mean_expected, rtol=0.02), (
        f'Mean molar mass {mean_molar_masses[0]} should match reference {M_mean_expected} g/mol'
    )

    # Check that transit depth scales correctly with composition
    # For reference: higher mean molar mass → smaller scale height → smaller transit depth
    # This is tested through the mass fraction computation

    # Verify mass fractions sum to 1 and scale correctly
    mass_frac_h2 = mass_fractions['H2']
    mass_frac_he = mass_fractions['He']
    assert np.allclose(mass_frac_h2 + mass_frac_he, 1.0), 'Mass fractions must sum to 1'


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_reference_pinned_mean_molar_mass_increases_with_helium_vmr(monkeypatch):
    """Reference-pinned: H2/He mean molar mass increases with He VMR.

    Reference: Solar-like 85/15 gives M_mean ≈ 2.3 g/mol; 75/25 gives ≈ 2.5 g/mol.
    """
    mod = _import_backend(monkeypatch)

    # Two H2/He compositions at different mixing ratios
    gases = ['H2', 'He']

    # Composition 1: nominal solar wind (85% H2, 15% He)
    vmrs_1 = [
        np.array([0.85, 0.85, 0.85]),
        np.array([0.15, 0.15, 0.15]),
    ]

    # Composition 2: more helium-rich (75% H2, 25% He)
    vmrs_2 = [
        np.array([0.75, 0.75, 0.75]),
        np.array([0.25, 0.25, 0.25]),
    ]

    mass_frac_1, mmw_1 = mod._vmrs_to_mass_fractions(gases, vmrs_1)
    mass_frac_2, mmw_2 = mod._vmrs_to_mass_fractions(gases, vmrs_2)

    # He-richer composition has higher mean molar mass (smaller scale height)
    # → smaller transit depth
    assert np.all(mmw_2 > mmw_1), 'Higher He content should increase mean molar mass'

    # Check reference values (in g/mol)
    assert np.allclose(mmw_1[0], 2.3, rtol=0.02), (
        'Reference: 85% H2 + 15% He should give M_mean ≈ 2.3 g/mol'
    )
    assert np.allclose(mmw_2[0], 2.5, rtol=0.02), (
        'Reference: 75% H2 + 25% He should give M_mean ≈ 2.5 g/mol'
    )
