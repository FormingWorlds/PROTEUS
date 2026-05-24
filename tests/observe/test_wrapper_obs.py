"""Unit tests for ``proteus.observe.wrapper``.

Exercises the observation dispatch functions ``calc_synthetic_spectra``
and ``run_observe``, which route to the correct synthesis backend
(PLATON) and iterate over observation sources (outgas, profile, offchem).

Invariants tested:
  - Error contract: unknown synthesis module raises ValueError
  - Dispatch: PLATON backend is called for each valid source
  - Guard: 'profile' source is skipped when atmos_clim is dummy
  - Guard: 'offchem' source is skipped when atmos_chem is None

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from proteus.observe.wrapper import calc_synthetic_spectra, run_observe

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _make_config(
    synthesis: str = 'platon',
    atmos_clim_module: str = 'janus',
    atmos_chem_module: str | None = 'vulcan',
) -> MagicMock:
    """Build a minimal mock config for observe tests."""
    config = MagicMock()
    config.observe.synthesis = synthesis
    config.atmos_clim.module = atmos_clim_module
    config.atmos_chem.module = atmos_chem_module
    return config


# -----------------------------------------------------------------------
# Error contract: unknown synthesis module
# -----------------------------------------------------------------------


def test_calc_synthetic_spectra_unknown_synthesis_raises():
    """An unrecognised synthesis module raises ValueError.

    The function uses an if/else dispatch on config.observe.synthesis;
    a misspelled module name must raise, not silently skip.
    """
    config = _make_config(synthesis='nonexistent_synth')
    hf_row = {'T_surf': 3000.0}
    with pytest.raises(ValueError, match='Unknown synthesis module'):
        calc_synthetic_spectra(hf_row=hf_row, outdir='/fake', config=config)
    assert hf_row['T_surf'] == pytest.approx(3000.0, rel=1e-12)  # no side effect

    # Adjacent-valid: 'platon' must NOT raise (it imports the backend)
    # We mock the import to avoid requiring PLATON to be installed
    with (
        patch('proteus.observe.platon.transit_depth'),
        patch('proteus.observe.platon.eclipse_depth'),
    ):
        config_valid = _make_config(synthesis='platon')
        calc_synthetic_spectra(hf_row={}, outdir='/fake', config=config_valid)


# -----------------------------------------------------------------------
# Dispatch: PLATON backend with all sources
# -----------------------------------------------------------------------


@patch('proteus.observe.platon.eclipse_depth')
@patch('proteus.observe.platon.transit_depth')
def test_calc_synthetic_spectra_calls_both_transit_and_eclipse(mock_transit, mock_eclipse):
    """With all sources enabled, calc_synthetic_spectra calls both
    transit_depth and eclipse_depth for each valid source.

    Three sources: 'outgas', 'profile', 'offchem'. With a non-dummy
    atmos_clim and a non-None atmos_chem, all three are valid.
    """
    config = _make_config(
        synthesis='platon',
        atmos_clim_module='janus',
        atmos_chem_module='vulcan',
    )
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


@patch('proteus.observe.platon.eclipse_depth')
@patch('proteus.observe.platon.transit_depth')
def test_calc_synthetic_spectra_skips_profile_when_dummy_atmos(mock_transit, mock_eclipse):
    """When atmos_clim.module is 'dummy', the 'profile' source is
    skipped because there is no atmospheric profile to synthesise from.

    Discrimination: with a real atmosphere module, all three sources
    are exercised (tested above). Here only 'outgas' and 'offchem'
    should fire (2 calls each).
    """
    config = _make_config(
        synthesis='platon',
        atmos_clim_module='dummy',
        atmos_chem_module='vulcan',
    )
    calc_synthetic_spectra(hf_row={}, outdir='/fake', config=config)

    assert mock_transit.call_count == 2
    transit_sources = [c.args[3] for c in mock_transit.call_args_list]
    assert 'profile' not in transit_sources
    assert 'outgas' in transit_sources
    assert 'offchem' in transit_sources


# -----------------------------------------------------------------------
# Guard: None atmos_chem skips 'offchem' source
# -----------------------------------------------------------------------


@patch('proteus.observe.platon.eclipse_depth')
@patch('proteus.observe.platon.transit_depth')
def test_calc_synthetic_spectra_skips_offchem_when_no_chem(mock_transit, mock_eclipse):
    """When atmos_chem.module is None, the 'offchem' source is skipped
    because no offline chemistry output exists to synthesise from.

    Discrimination: with a real chemistry module, 'offchem' is included
    (tested above). Here only 'outgas' and 'profile' should fire.
    """
    config = _make_config(
        synthesis='platon',
        atmos_clim_module='janus',
        atmos_chem_module=None,
    )
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
