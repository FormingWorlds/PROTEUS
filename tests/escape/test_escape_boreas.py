from __future__ import annotations

import logging
from pathlib import Path

import pytest

from proteus.config import read_config_object
from proteus.escape.boreas import BOREAS_GASES, _set_boreas_params
from proteus.utils.constants import AU, M_earth, R_earth, element_list

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


log = logging.getLogger(__name__)


@pytest.fixture(scope='session', autouse=True)
def _skip_if_no_boreas():
    pytest.importorskip('boreas')


def _make_minimal_hf_row() -> dict:
    """Construct `hf_row` with inventories for escape checks."""
    hf_row: dict = {}

    # Orbital/stellar derived quantities
    hf_row['semimajorax'] = 0.05 * AU  # meters

    # Atmospheric state used by escape
    hf_row['T_obs'] = 400.0  # K
    hf_row['F_xuv'] = 10.0  # W m^-2
    hf_row['R_obs'] = 1.0 * R_earth  # m
    hf_row['M_planet'] = 1.0 * M_earth  # kg

    # Supply VMRs at the XUV level for all BOREAS-supported gases
    for g in BOREAS_GASES:
        hf_row[f'{g}_vmr_xuv'] = 0.0
    hf_row['H2_vmr_xuv'] = 0.5
    hf_row['CO_vmr_xuv'] = 0.2
    hf_row['H2S_vmr_xuv'] = 0.2
    hf_row['N2_vmr_xuv'] = 0.1

    # Provide bulk/atmospheric elemental inventories used by calc_unfract_fluxes
    # Oxygen is skipped by calc_unfract_fluxes; others are used to form ratios
    # Values are arbitrary but positive and distinct
    for e in element_list:
        hf_row[f'{e}_kg_atm'] = 0.0
        hf_row[f'{e}_kg_total'] = 0.0
    masses_atm = {
        'H': 1.0e17,
        'C': 2.0e17,
        'N': 3.0e17,
        'S': 4.0e17,
    }
    for e, m in masses_atm.items():
        hf_row[f'{e}_kg_atm'] = m
    masses_bulk = {
        'H': 5.0e20,
        'C': 2.5e20,
        'N': 1.0e20,
        'S': 0.5e20,
    }
    for e, m in masses_bulk.items():
        hf_row[f'{e}_kg_total'] = m

    return hf_row


@pytest.fixture(scope='module')
def boreas_config() -> object:
    """Load the demo BOREAS TOML into a validated Config object."""
    config_path = Path('input/dummy.toml')
    cfg = read_config_object(config_path)
    # Ensure module selection
    cfg.escape.module = 'boreas'
    return cfg


def test_boreas_params(boreas_config):
    """``_set_boreas_params`` populates the BOREAS parameter object from
    ``hf_row``, with FXUV converted from W/m^2 to mW/m^2 (factor 1e3) and
    albedo defaulting to 0.0 when not set in the config.
    """
    hf_row = _make_minimal_hf_row()

    # make parameters object
    params = _set_boreas_params(boreas_config, hf_row)

    # check it is setup property
    assert params.FXUV == hf_row['F_xuv'] * 1e3
    assert params.albedo == 0.0


# ---------------------------------------------------------------------------
# RunBOREAS: unit conversions, error handling, fractionation
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_run_boreas_unit_conversions_and_fractionation(boreas_config, monkeypatch):
    """RunBOREAS converts BOREAS CGS outputs to SI and stores them in
    hf_row. The conversion factors are the physics contract:

    - Mdot: g/s -> kg/s (factor 1e-3)
    - cs: cm/s -> m/s (factor 1e-2)
    - RXUV: cm -> m (factor 1e-2)
    - per-element flux: atoms/cm^2/s -> kg/m^2/s via atomic mass
      then global rate via 4 pi R_xuv^2.

    The test mocks the BOREAS library to return known CGS values and
    verifies the SI outputs land at the right order of magnitude.
    """
    from unittest.mock import MagicMock

    import boreas as boreas_lib

    from proteus.escape.boreas import run_boreas

    hf_row = _make_minimal_hf_row()

    # Mock MassLoss + Fractionation
    mock_ml = MagicMock()
    mock_ml.compute_mass_loss_parameters.return_value = [
        {'regime': 'EL', 'RXUV': 7e8, 'Mdot': 1e10, 'cs': 1e6}  # CGS
    ]
    mock_frac = MagicMock()
    mock_frac.execute.return_value = [
        {
            'regime': 'EL',
            'Mdot': 1e10,  # g/s
            'cs': 1e6,  # cm/s
            'RXUV': 7e8,  # cm
            'phi_H_num': 1e12,  # atoms/cm^2/s
            'phi_O_num': 5e10,
            'phi_C_num': 0.0,
            'phi_N_num': 0.0,
            'phi_S_num': 0.0,
            'x_O': 0.05,
            'x_C': 0.0,
            'x_N': 0.0,
            'x_S': 0.0,
        }
    ]

    monkeypatch.setattr(boreas_lib, 'MassLoss', lambda params: mock_ml)
    monkeypatch.setattr(boreas_lib, 'Fractionation', lambda params: mock_frac)

    # Enable fractionation so the per-element branch is exercised
    boreas_config.escape.boreas.fractionate = True

    dirs = {'output': '/tmp'}
    run_boreas(boreas_config, hf_row, dirs)

    # Bulk conversions: g/s -> kg/s, cm/s -> m/s, cm -> m
    assert hf_row['esc_rate_total'] == pytest.approx(1e10 * 1e-3, rel=1e-12)
    assert hf_row['cs_xuv'] == pytest.approx(1e6 * 1e-2, rel=1e-12)
    assert hf_row['R_xuv'] == pytest.approx(7e8 * 1e-2, rel=1e-12)
    # Discrimination guard: a regression that forgot the 1e-3 factor
    # would land at 1e10 (not 1e7); a regression that used 1e-6 would
    # land at 1e4. The three-order-of-magnitude gap is well above any
    # tolerance.
    assert hf_row['esc_rate_total'] > 1e5
    assert hf_row['esc_rate_total'] < 1e9
    # Per-element escape rates: H must be nonzero
    assert hf_row['esc_rate_H'] > 0
    # Elements with zero flux get zero rate
    assert hf_row['esc_rate_C'] == pytest.approx(0.0, abs=1e-30)


@pytest.mark.unit
def test_run_boreas_raises_on_solver_failure(boreas_config, monkeypatch):
    """When the BOREAS mass-loss solver raises internally, RunBOREAS
    must catch the exception, update the statusfile to code 28, and
    reraise as RuntimeError. This pins the error-propagation contract
    at L131-134.
    """
    from unittest.mock import MagicMock

    import boreas as boreas_lib

    from proteus.escape.boreas import run_boreas

    hf_row = _make_minimal_hf_row()

    mock_ml = MagicMock()
    mock_ml.compute_mass_loss_parameters.side_effect = ValueError('solver diverged')
    monkeypatch.setattr(boreas_lib, 'MassLoss', lambda params: mock_ml)

    dirs = {'output': '/tmp'}
    with pytest.raises(RuntimeError, match='Encountered problem when running BOREAS'):
        run_boreas(boreas_config, hf_row, dirs)
