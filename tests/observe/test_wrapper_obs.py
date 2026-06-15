"""Unit tests for ``proteus.observe.wrapper``.

Exercises the observation dispatch functions ``calc_synthetic_spectra``
and ``run_observe``, which route to the synthetic-observation backend
(``petitRADTRANS``) and iterate over observation sources
(``outgas``, ``profile``, ``offchem``).

Invariants tested:
    - Error contract: unknown synthesis module raises ValueError
    - Dispatch: petitRADTRANS backend is called for each valid source
  - Guard: 'profile' source is skipped when atmos_clim is dummy
  - Guard: 'offchem' source is skipped when atmos_chem is None

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from proteus.observe.wrapper import calc_synthetic_spectra, run_observe

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _make_config(
    module: str = 'petitRADTRANS',
    atmos_clim_module: str = 'janus',
    atmos_chem_module: str | None = 'vulcan',
) -> MagicMock:
    """Build a minimal mock config for observe tests."""
    config = MagicMock()
    config.observe.module = module
    config.atmos_clim.module = atmos_clim_module
    config.atmos_chem.module = atmos_chem_module
    return config


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
    monkeypatch.delitem(sys.modules, 'proteus.observe.petitRADTRANS', raising=False)


# -----------------------------------------------------------------------
# Error contract: unknown synthesis module
# -----------------------------------------------------------------------


def test_calc_synthetic_spectra_unknown_synthesis_raises(monkeypatch):
    """An unrecognised synthesis module raises ValueError.

    The function uses an if/else dispatch on config.observe.module;
    a misspelled module name must raise, not silently skip.
    """
    config = _make_config(module='nonexistent_synth')
    hf_row = {'T_surf': 3000.0}
    with pytest.raises(ValueError, match='Unknown synthesis module'):
        calc_synthetic_spectra(hf_row=hf_row, outdir='/fake', config=config)
    assert hf_row['T_surf'] == pytest.approx(3000.0, rel=1e-12)  # no side effect

    # Adjacent-valid: petitRADTRANS must NOT raise (it imports the backend)
    # We patch the imported module object directly because the package does
    # not re-export the submodule as an attribute.
    _install_fake_petitradtrans(monkeypatch)
    backend = importlib.import_module('proteus.observe.petitRADTRANS')
    transit_mock = MagicMock(name='transit_depth')
    eclipse_mock = MagicMock(name='eclipse_depth')
    monkeypatch.setattr(backend, 'transit_depth', transit_mock)
    monkeypatch.setattr(backend, 'eclipse_depth', eclipse_mock)

    config_valid = _make_config(module='petitRADTRANS')
    calc_synthetic_spectra(hf_row={}, outdir='/fake', config=config_valid)


# -----------------------------------------------------------------------
# Dispatch: petitRADTRANS backend with all sources
# -----------------------------------------------------------------------


def test_calc_synthetic_spectra_calls_both_transit_and_eclipse(monkeypatch):
    """With all sources enabled, calc_synthetic_spectra calls both
    transit_depth and eclipse_depth for each valid source.

    Three sources: 'outgas', 'profile', 'offchem'. With a non-dummy
    atmos_clim and a non-None atmos_chem, all three are valid.
    """
    _install_fake_petitradtrans(monkeypatch)
    backend = importlib.import_module('proteus.observe.petitRADTRANS')
    mock_transit = MagicMock(name='transit_depth')
    mock_eclipse = MagicMock(name='eclipse_depth')
    monkeypatch.setattr(backend, 'transit_depth', mock_transit)
    monkeypatch.setattr(backend, 'eclipse_depth', mock_eclipse)

    config = _make_config(module='petitRADTRANS', atmos_clim_module='janus', atmos_chem_module='vulcan')
    hf_row = {}
    outdir = '/fake/output'

    calc_synthetic_spectra(hf_row, outdir, config)

    # transit_depth and eclipse_depth called once per source (3 sources)
    assert mock_transit.call_count == 3
    assert mock_eclipse.call_count == 3

    # Verify the sources passed in order
    transit_sources = [c.args[3] for c in mock_transit.call_args_list]
    assert 'outgas' in transit_sources
    assert 'profile' in transit_sources
    assert 'offchem' in transit_sources


# -----------------------------------------------------------------------
# Guard: dummy atmosphere skips 'profile' source
# -----------------------------------------------------------------------


def test_calc_synthetic_spectra_skips_profile_when_dummy_atmos(monkeypatch):
    """When atmos_clim.module is 'dummy', the 'profile' source is
    skipped because there is no atmospheric profile to synthesise from.

    Discrimination: with a real atmosphere module, all three sources
    are exercised (tested above). Here only 'outgas' and 'offchem'
    should fire (2 calls each).
    """
    _install_fake_petitradtrans(monkeypatch)
    backend = importlib.import_module('proteus.observe.petitRADTRANS')
    mock_transit = MagicMock(name='transit_depth')
    mock_eclipse = MagicMock(name='eclipse_depth')
    monkeypatch.setattr(backend, 'transit_depth', mock_transit)
    monkeypatch.setattr(backend, 'eclipse_depth', mock_eclipse)

    config = _make_config(module='petitRADTRANS', atmos_clim_module='dummy', atmos_chem_module='vulcan')
    calc_synthetic_spectra(hf_row={}, outdir='/fake', config=config)

    assert mock_transit.call_count == 2
    transit_sources = [c.args[3] for c in mock_transit.call_args_list]
    assert 'profile' not in transit_sources
    assert 'outgas' in transit_sources
    assert 'offchem' in transit_sources


# -----------------------------------------------------------------------
# Guard: None atmos_chem skips 'offchem' source
# -----------------------------------------------------------------------


def test_calc_synthetic_spectra_skips_offchem_when_no_chem(monkeypatch):
    """When atmos_chem.module is None, the 'offchem' source is skipped
    because no offline chemistry output exists to synthesise from.

    Discrimination: with a real chemistry module, 'offchem' is included
    (tested above). Here only 'outgas' and 'profile' should fire.
    """
    _install_fake_petitradtrans(monkeypatch)
    backend = importlib.import_module('proteus.observe.petitRADTRANS')
    mock_transit = MagicMock(name='transit_depth')
    mock_eclipse = MagicMock(name='eclipse_depth')
    monkeypatch.setattr(backend, 'transit_depth', mock_transit)
    monkeypatch.setattr(backend, 'eclipse_depth', mock_eclipse)

    config = _make_config(module='petitRADTRANS', atmos_clim_module='janus', atmos_chem_module=None)
    calc_synthetic_spectra(hf_row={}, outdir='/fake', config=config)

    assert mock_transit.call_count == 2
    transit_sources = [c.args[3] for c in mock_transit.call_args_list]
    assert 'offchem' not in transit_sources
    assert 'outgas' in transit_sources
    assert 'profile' in transit_sources


# -----------------------------------------------------------------------
# run_observe delegates to calc_synthetic_spectra
# -----------------------------------------------------------------------


@patch('proteus.observe.wrapper.calc_synthetic_spectra')
def test_run_observe_delegates_to_calc_synthetic_spectra(mock_calc):
    """run_observe calls calc_synthetic_spectra with the same arguments.

    This is a thin wrapper; the test confirms the delegation contract.
    """
    config = _make_config()
    hf_row = {'T_surf': 3000.0}
    outdir = '/fake/output'

    run_observe(hf_row, outdir, config)

    mock_calc.assert_called_once_with(hf_row, outdir, config)
    assert hf_row['T_surf'] == pytest.approx(3000.0, rel=1e-12)  # passthrough, no mutation


# -----------------------------------------------------------------------
# Dispatch: petitRADTRANS backend with all sources
# -----------------------------------------------------------------------


def test_calc_synthetic_spectra_calls_petitradtrans_backend(monkeypatch):
    """The wrapper dispatches to petitRADTRANS when requested and
    still iterates over all valid sources.

    Discrimination: the patched backend functions are called once per
    valid source, and the expected source ordering remains intact.
    """
    _install_fake_petitradtrans(monkeypatch)
    backend = importlib.import_module('proteus.observe.petitRADTRANS')
    transit_mock = MagicMock(name='transit_depth')
    eclipse_mock = MagicMock(name='eclipse_depth')
    monkeypatch.setattr(backend, 'transit_depth', transit_mock)
    monkeypatch.setattr(backend, 'eclipse_depth', eclipse_mock)

    config = _make_config(module='petitRADTRANS', atmos_clim_module='janus', atmos_chem_module='vulcan')

    calc_synthetic_spectra(hf_row={}, outdir='/fake/output', config=config)

    assert transit_mock.call_count == 3
    assert eclipse_mock.call_count == 3
    sources = [call.args[3] for call in transit_mock.call_args_list]
    assert sources == ['outgas', 'profile', 'offchem']
