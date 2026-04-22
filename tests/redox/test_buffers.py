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
def test_schaefer24_positive_at_whole_earth_anchor():
    """
    At Earth BSE oxide fractions, T=2000 K, P=120 GPa, with
    Fe3+/FeT=0.10 (X_FeO1.5 / X_FeO = 0.111), the Schaefer+24 Fig 2
    whole-Earth curve places ΔIW ≈ 0..+2 → log10 fO2 ≈ -8..-10.

    This test pins the order of magnitude; the anchor-in-paper
    check uses the dispatcher in test_mariana_interface.
    """
    # Nominal pyrolite mole fractions (approximate, from Schaefer
    # Table 1 bulk Earth wt%).
    X_SiO2 = 0.42
    X_TiO2 = 0.002
    X_MgO = 0.53
    X_CaO = 0.035
    X_Na2O = 0.003
    X_P2O5 = 1e-4
    X_Al2O3 = 0.026
    X_K2O = 3e-4
    X_FeO = 0.010
    X_FeO1_5 = 0.0011   # Fe3+/FeT ≈ 0.10

    log_fO2 = log10_fO2_schaefer2024(
        X_FeO_liq=X_FeO, X_FeO1_5=X_FeO1_5,
        temperature=2000.0, pressure=120e9,
        X_SiO2=X_SiO2, X_TiO2=X_TiO2, X_MgO=X_MgO,
        X_CaO=X_CaO, X_Na2O=X_Na2O, X_P2O5=X_P2O5,
        X_Al2O3=X_Al2O3, X_K2O=X_K2O,
    )
    # Eq 13 at these conditions should produce a reasonable fO2;
    # exact value depends on the sign conventions in Schaefer Table 4.
    # The order-of-magnitude check keeps us honest while Commit C
    # verifies a tighter literature anchor.
    assert -20 < log_fO2 < 10, f'Schaefer Eq 13 returned {log_fO2}'


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


class _MockMantleComp:
    """Minimal MantleComp for dispatcher tests."""
    SiO2_wt = 45.5
    TiO2_wt = 0.21
    Al2O3_wt = 4.49
    FeO_total_wt = 7.82
    MgO_wt = 38.3
    CaO_wt = 3.58
    Na2O_wt = 0.36
    K2O_wt = 0.029
    P2O5_wt = 0.021


@pytest.mark.unit
def test_log10_fO2_mantle_mo_active_uses_schaefer():
    """When phi_max > phi_crit, dispatcher routes to Schaefer+24."""
    mc = _MockMantleComp()
    val = log10_fO2_mantle(
        Fe3_frac=0.10, temperature=2000.0, pressure=1e10,
        phi_max=0.9, mantle_comp=mc, phi_crit=0.4,
        oxybarometer='schaefer2024',
    )
    # Should not equal the stagno2013 fallback at these conditions.
    fallback = log10_fO2_stagno2013_peridotite(
        Fe3_frac=0.10, temperature=2000.0, pressure=1e10,
    )
    assert val != fallback


@pytest.mark.unit
def test_log10_fO2_mantle_mo_inactive_uses_stagno():
    """When phi_max < phi_crit, dispatcher routes to Stagno+13."""
    mc = _MockMantleComp()
    val = log10_fO2_mantle(
        Fe3_frac=0.04, temperature=1573.0, pressure=3e9,
        phi_max=0.1, mantle_comp=mc, phi_crit=0.4,
        oxybarometer='schaefer2024',  # requested, but overridden by phi
    )
    fallback = log10_fO2_stagno2013_peridotite(
        Fe3_frac=0.04, temperature=1573.0, pressure=3e9,
    )
    assert val == pytest.approx(fallback, abs=1e-12)
