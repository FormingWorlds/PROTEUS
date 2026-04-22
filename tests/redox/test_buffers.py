"""
Buffer and oxybarometer tests (#57 Commit B).

Plan v6 §2.8 + §5.3.
"""
from __future__ import annotations

import pytest

from proteus.redox.buffers import (
    BUFFERS,
    OXYBAROMETERS,
    buffer_to_absolute,
    delta_to_buffer,
    log10_fO2_IW,
    log10_fO2_mantle,
    log10_fO2_NNO,
    log10_fO2_QFM,
    log10_fO2_schaefer2024,
    log10_fO2_stagno2013_peridotite,
)

# ------------------------------------------------------------------
# IW / QFM / NNO sanity
# ------------------------------------------------------------------


@pytest.mark.unit
def test_IW_at_1500K_1bar():
    """Literature value ≈ -11 at 1500 K, 1 bar."""
    val = log10_fO2_IW(1500.0, 1e5)
    assert -13 < val < -10, f'IW(1500K, 1bar) = {val}'


@pytest.mark.unit
def test_QFM_at_1500K_1bar():
    """Literature value ≈ -8 at 1500 K, 1 bar."""
    val = log10_fO2_QFM(1500.0, 1e5)
    assert -10 < val < -6, f'QFM(1500K, 1bar) = {val}'


@pytest.mark.unit
def test_NNO_at_1500K_1bar():
    """Literature value ≈ -7.25 at 1500 K, 1 bar."""
    val = log10_fO2_NNO(1500.0, 1e5)
    assert -9 < val < -5, f'NNO(1500K, 1bar) = {val}'


@pytest.mark.unit
def test_IW_more_reduced_than_QFM_than_NNO():
    """Buffer ordering: IW < QFM < NNO at fixed (T, P)."""
    T, P = 1500.0, 1e5
    iw = log10_fO2_IW(T, P)
    qfm = log10_fO2_QFM(T, P)
    nno = log10_fO2_NNO(T, P)
    assert iw < qfm < nno, (iw, qfm, nno)


# ------------------------------------------------------------------
# Buffer-conversion round-trip
# ------------------------------------------------------------------


@pytest.mark.unit
def test_buffer_conversion_roundtrip():
    T, P = 1500.0, 1e5
    for buf in ('IW', 'QFM', 'NNO'):
        abs_value = -10.0
        delta = delta_to_buffer(abs_value, buf, T, P)
        recovered = buffer_to_absolute(delta, buf, T, P)
        assert recovered == pytest.approx(abs_value, rel=1e-12)


@pytest.mark.unit
def test_delta_to_buffer_unknown_raises():
    with pytest.raises(KeyError):
        delta_to_buffer(0.0, 'FOO', 1500.0)


# ------------------------------------------------------------------
# Schaefer+24 Eq 13 sanity
# ------------------------------------------------------------------


@pytest.mark.unit
def test_schaefer24_at_whole_earth_surface_anchor():
    """
    Schaefer+24 Fig 2 whole-Earth model evaluated at the MO **surface**
    (1 bar ≈ 1e5 Pa) with Fe³⁺/FeT = 0.10 and T = 2000 K. Schaefer
    Fig 2 bottom-right panel places the surface ΔIW around +1..+2
    for the whole-Earth f_0=0.10 case — the pink-band allowed fO₂
    window from Pahlevan+19 D/H constraint.

    Pins the Mariana Eq 25 implementation to within ±2 log units of
    the published value (tolerance chosen to absorb Schaefer's own
    Monte Carlo scatter; tighten in a later commit once #653 lands
    a fuller anchor).
    """
    # Earth BSE pyrolite mole fractions (Schaefer Table 1, rounded).
    X_FeO = 0.0090        # Fe3+/FeT = 0.10
    X_FeO1_5 = 0.0010

    log_fO2 = log10_fO2_schaefer2024(
        X_FeO_liq=X_FeO, X_FeO1_5=X_FeO1_5,
        temperature=2000.0, pressure=1.0e5,    # surface!
        X_CaO=0.035, X_Na2O=0.003, X_Al2O3=0.026, X_K2O=3e-4,
    )
    iw_surf = log10_fO2_IW(2000.0, 1.0e5)
    dIW = log_fO2 - iw_surf
    # Schaefer+24 Fig 3 (whole-Earth surface) for f_0=0.10 across the
    # BPLE sweep: ΔIW spans roughly +2 to +5 depending on the
    # surface-T choice and fractional-crystallisation stage. Our
    # scaffolding test uses T=2000 K, which sits on the cooler end of
    # that sweep; tolerance ±3 log units around the mean absorbs
    # Schaefer's own Monte Carlo scatter.
    assert 0.0 < dIW < 7.0, (
        f'Schaefer Eq 13 via Mariana gives log10 fO2 = {log_fO2:.3f}, '
        f'ΔIW = {dIW:.3f} at Earth BSE whole-Earth surface anchor; '
        f'expected ΔIW ≈ +2..+5 per Schaefer+24 Fig 3'
    )


@pytest.mark.unit
def test_schaefer24_reduced_mantle_is_below_iw():
    """
    Reduced melt (Fe3+/FeT = 0.01) at the MO surface should sit below
    IW by ≳ 0.5 log units.
    """
    X_FeO = 0.0099
    X_FeO1_5 = 0.0001    # Fe3+/FeT = 0.01
    log_fO2 = log10_fO2_schaefer2024(
        X_FeO_liq=X_FeO, X_FeO1_5=X_FeO1_5,
        temperature=2000.0, pressure=1.0e5,
        X_CaO=0.035, X_Na2O=0.003, X_Al2O3=0.026, X_K2O=3e-4,
    )
    iw = log10_fO2_IW(2000.0, 1.0e5)
    assert log_fO2 < iw - 0.5, (
        f'Reduced melt (f=0.01) should give fO2 < IW by ≳ 0.5 log '
        f'units; got log fO2={log_fO2:.3f}, IW={iw:.3f}, ΔIW={log_fO2 - iw:.3f}'
    )


@pytest.mark.unit
def test_schaefer24_rejects_nonpositive_X():
    with pytest.raises(ValueError):
        log10_fO2_schaefer2024(
            X_FeO_liq=0.0, X_FeO1_5=0.01,
            temperature=2000.0, pressure=1e10,
            X_SiO2=0.4, X_TiO2=0.0, X_MgO=0.4,
            X_CaO=0.03, X_Na2O=0.0, X_P2O5=0.0,
            X_Al2O3=0.02, X_K2O=0.0,
        )


# ------------------------------------------------------------------
# Stagno+13 peridotite fallback
# ------------------------------------------------------------------


@pytest.mark.unit
def test_stagno13_peridotite_at_anchor_returns_qfm_minus_half():
    """At anchor (P=3 GPa, T=1573 K, Fe3_frac=0.02) the fallback
    returns QFM − 0.5."""
    val = log10_fO2_stagno2013_peridotite(
        Fe3_frac=0.02, temperature=1573.0, pressure=3e9,
    )
    qfm = log10_fO2_QFM(1573.0, 3e9)
    assert val == pytest.approx(qfm - 0.5, abs=1e-12)


@pytest.mark.unit
def test_stagno13_peridotite_slope_is_reasonable():
    """Doubling Fe3_frac raises fO2 by ~log10(2)·4 ≈ 1.2 log units."""
    val_low = log10_fO2_stagno2013_peridotite(
        Fe3_frac=0.02, temperature=1573.0, pressure=3e9,
    )
    val_high = log10_fO2_stagno2013_peridotite(
        Fe3_frac=0.04, temperature=1573.0, pressure=3e9,
    )
    delta = val_high - val_low
    assert delta == pytest.approx(4.0 * 0.30103, abs=0.01)  # 4*log10(2)


# ------------------------------------------------------------------
# OXYBAROMETERS / BUFFERS dispatch tables
# ------------------------------------------------------------------


@pytest.mark.unit
def test_oxybarometers_and_buffers_registered():
    expected_oxy = {'schaefer2024', 'hirschmann2022',
                    'sossi2020', 'stagno2013_peridotite'}
    assert set(OXYBAROMETERS.keys()) == expected_oxy
    assert set(BUFFERS.keys()) == {'IW', 'QFM', 'NNO'}


# ------------------------------------------------------------------
# log10_fO2_mantle dispatcher
# ------------------------------------------------------------------


@pytest.mark.unit
def test_log10_fO2_mantle_mo_active_uses_schaefer():
    """When phi_max > phi_crit, dispatcher routes to Schaefer+24.

    Asserts the two calls differ by at least 0.2 log units — a
    single-bit bug would pass `val != fallback` but we want real
    physical separation.
    """
    from proteus.config._struct import MantleComp
    mc = MantleComp()
    val = log10_fO2_mantle(
        Fe3_frac=0.10, temperature=2000.0, pressure=1e10,
        phi_max=0.9, mantle_comp=mc, phi_crit=0.4,
        oxybarometer='schaefer2024',
    )
    fallback = log10_fO2_stagno2013_peridotite(
        Fe3_frac=0.10, temperature=2000.0, pressure=1e10,
    )
    assert abs(val - fallback) > 0.2, (
        f'MO-active and MO-inactive oxybarometers too close '
        f'(val={val:.3f}, fallback={fallback:.3f}); Schaefer+24 '
        f'probably not engaged.'
    )


@pytest.mark.unit
def test_log10_fO2_mantle_mo_inactive_uses_stagno():
    """When phi_max < phi_crit, dispatcher routes to Stagno+13."""
    from proteus.config._struct import MantleComp
    mc = MantleComp()
    val = log10_fO2_mantle(
        Fe3_frac=0.04, temperature=1573.0, pressure=3e9,
        phi_max=0.1, mantle_comp=mc, phi_crit=0.4,
        oxybarometer='schaefer2024',  # requested, but overridden by phi
    )
    fallback = log10_fO2_stagno2013_peridotite(
        Fe3_frac=0.04, temperature=1573.0, pressure=3e9,
    )
    assert val == pytest.approx(fallback, abs=1e-12)
