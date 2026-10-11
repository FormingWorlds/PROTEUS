from __future__ import annotations

import logging
import math
from types import SimpleNamespace

import pytest

import proteus.accretion.wrapper as wrapper
from proteus.accretion.atmloss import (
    _ATMLOSS_THIN_ATM_WARN,
    _as_float,
    _format_roche_flag,
    _log_zephyrus_loss,
)
from proteus.accretion.common import ImpactEvent

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _mock_event() -> ImpactEvent:
    return ImpactEvent(
        time=1000.0,
        M_target_before=5.972e24,
        M_impactor=1.0e23,
        M_merged_after=6.072e24,
        v_impact=1.2e4,
        v_esc=1.12e4,
        impact_parameter=0.3,
        R_target_before=6.371e6,
        R_impactor=1.5e6,
        rho_target=5515.0,
        rho_impactor=5515.0,
        a_before=1.496e11,
        a_after=1.496e11,
        e_before=0.01,
        e_after=0.01,
        id_target=1,
        id_impactor=2,
    )


@pytest.mark.unit
def test_as_float_conversions():
    """Verify _as_float converts numbers and maps invalid types to NaN.

    Verifies clause: float, int, and numeric strings convert to float; None,
    non-numeric strings, and complex objects return NaN without raising.
    """
    assert _as_float(3.14) == pytest.approx(3.14)
    assert _as_float(42) == pytest.approx(42.0)
    assert _as_float('1.25e3') == pytest.approx(1250.0)
    assert math.isnan(_as_float(None))
    assert math.isnan(_as_float('not_a_number'))
    assert math.isnan(_as_float({'a': 1}))


@pytest.mark.unit
def test_format_roche_flag_neutral_and_extrapolated():
    """Format flags for Roche et al. (2026) law with neutral and extrapolated modes.

    Verifies clause: X_FF_zero_energy omits the extrapolation note when has_range_flag
    is False and includes it when has_range_flag is True; v_sub_escape formats held loss;
    fitted parameters format bounds and any active clamp; unknown flags return bare name.
    """
    fitted_range = {'gamma': (0.1, 0.5), 'f_atm': (0.01, 0.2)}

    # Zero-energy far-field flag: neutral vs extrapolated note
    neutral_msg = _format_roche_flag(
        'X_FF_zero_energy',
        {'X_FF_zero_energy': 0.0014},
        fitted_range,
        has_range_flag=False,
    )
    assert neutral_msg == 'far-field loss of 0.0014 without impact energy'

    extrap_msg = _format_roche_flag(
        'X_FF_zero_energy',
        {'X_FF_zero_energy': 0.0801},
        fitted_range,
        has_range_flag=True,
    )
    assert (
        extrap_msg
        == 'far-field loss of 0.0801 without impact energy (log10 f_atm extrapolation)'
    )

    # Sub-escape velocity flag
    sub_esc_msg = _format_roche_flag(
        'v_sub_escape',
        {'v_ratio': 0.85},
        fitted_range,
    )
    assert sub_esc_msg == 'v_sub_escape: near-field loss held at v_c = v_esc (v_ratio = 0.85)'

    # Clamped parameter
    clamped_msg = _format_roche_flag(
        'gamma',
        {'gamma': 0.55, 'clamped': {'gamma': 0.5}},
        fitted_range,
    )
    assert clamped_msg == 'gamma = 0.55 (fitted 0.1 to 0.5), evaluated at 0.5'

    # Unknown flag fallback
    unknown_msg = _format_roche_flag('custom_future_flag', {}, fitted_range)
    assert unknown_msg == 'custom_future_flag'


@pytest.mark.unit
def test_log_zephyrus_loss_neutral_header(caplog):
    """Log Roche et al. (2026) warning with neutral header when only zero-energy flag is set.

    Verifies clause: when no parameter is out of range or clamped, X_FF_zero_energy emits
    a warning with the neutral header 'Roche et al. (2026) law:' and omits both
    'outside its fitted range' and '; the loss fraction is extrapolated'.
    """
    event = _mock_event()
    mock_result = SimpleNamespace(
        fraction=0.45,
        diagnostics={
            'v_ratio': 1.5,
            'gamma': 0.4,
            'X_NF': 0.4,
            'X_FF': 0.05,
            'X_FF_zero_energy': 0.0014,
        },
        flags=('X_FF_zero_energy',),
    )
    fitted_range = {'gamma': (0.1, 0.5), 'f_atm': (0.01, 0.2)}

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.accretion.wrapper'):
        _log_zephyrus_loss('roche2026', event, mock_result, 0.012, fitted_range)

    records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(records) == 1
    msg = records[0].getMessage()
    assert 'Roche et al. (2026) law: far-field loss of 0.0014 without impact energy' in msg
    assert 'outside its fitted range' not in msg
    assert 'extrapolated' not in msg


@pytest.mark.unit
def test_log_zephyrus_loss_v_sub_escape_extrapolated(caplog):
    """Log warning for v_sub_escape marks loss fraction as extrapolated.

    Verifies clause: sub-escape velocity emits 'outside its fitted range' in the header
    and appends '; the loss fraction is extrapolated' in the warning tail.
    """
    event = _mock_event()
    mock_result = SimpleNamespace(
        fraction=0.15,
        diagnostics={
            'v_ratio': 0.85,
            'gamma': 0.2,
            'X_NF': 0.1,
            'X_FF': 0.05,
        },
        flags=('v_sub_escape',),
    )
    fitted_range = {'gamma': (0.1, 0.5), 'f_atm': (0.01, 0.2)}

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.accretion.wrapper'):
        _log_zephyrus_loss('roche2026', event, mock_result, 0.02, fitted_range)

    records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(records) == 1
    msg = records[0].getMessage()
    assert 'Roche et al. (2026) law outside its fitted range:' in msg
    assert 'v_sub_escape: near-field loss held at v_c = v_esc (v_ratio = 0.85)' in msg
    assert msg.endswith('; the loss fraction is extrapolated')


@pytest.mark.unit
def test_log_zephyrus_loss_kegerreis_thin_regime(caplog):
    """Log Kegerreis et al. (2020) thin-atmosphere warning above threshold.

    Verifies clause: when f_atm > _ATMLOSS_THIN_ATM_WARN, Kegerreis logs warning
    about thin-atmosphere regime; at or below threshold it logs no warning.
    """
    event = _mock_event()
    mock_result = SimpleNamespace(
        fraction=0.3,
        diagnostics={'v_ratio': 1.2, 'gamma': 0.2},
        flags=(),
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.accretion.wrapper'):
        _log_zephyrus_loss(
            'kegerreis2020', event, mock_result, _ATMLOSS_THIN_ATM_WARN + 0.01, {}
        )
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.accretion.wrapper'):
        _log_zephyrus_loss(
            'kegerreis2020', event, mock_result, _ATMLOSS_THIN_ATM_WARN - 0.01, {}
        )
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 0


@pytest.mark.unit
def test_zephyrus_loss_fraction_monkeypatch_target(monkeypatch):
    """Monkeypatching _zephyrus_loss_fraction on wrapper intercepts loss calls.

    Verifies clause: _impact_loss_fraction in wrapper invokes _zephyrus_loss_fraction
    from wrapper namespace; monkeypatching the helper to raise causes failure.
    """

    def _exploding_zephyrus(*args, **kwargs):
        raise RuntimeError('canary monkeypatch intercepted')

    monkeypatch.setattr(wrapper, '_zephyrus_loss_fraction', _exploding_zephyrus)
    assert wrapper._zephyrus_loss_fraction is _exploding_zephyrus
    cfg = SimpleNamespace(accretion=SimpleNamespace(atmloss_module='zephyrus'))

    with pytest.raises(RuntimeError) as exc_info:
        wrapper._impact_loss_fraction(cfg, {}, _mock_event())
    assert 'canary monkeypatch intercepted' in str(exc_info.value)
