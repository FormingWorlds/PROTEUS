"""Unit tests for ``proteus.interior_energetics.radiogenic``.

Invariants checked:
  - heat production follows ``C * 1e-6 * X * h * 2**((t0 - t) / t_half)`` and
    halves after one half-life;
  - the K, U and Th inputs at Solar System formation in
    ``input/planets/mercury.toml`` give the heating history of the present-day
    Mercury values of Jaggi et al. (2021, Appendix A), because one set is the
    other decayed back in time;
  - isotopes with a zero element total or a zero ratio are left out.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from proteus.interior_energetics.radiogenic import (
    radio_t0_years,
    radiogenic_heating,
    radionuclide_inventory,
)
from proteus.utils.constants import radnuc_data

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_INPUT = Path(__file__).resolve().parents[2] / 'input'
_RADIO_KEYS = (
    'radio_tref',
    'radio_Al',
    'radio_Al26_abund',
    'radio_Fe',
    'radio_Fe60_abund',
    'radio_K',
    'radio_K40_abund',
    'radio_U',
    'radio_U235_abund',
    'radio_U238_abund',
    'radio_Th',
    'radio_Th232_abund',
)


def _config_from_toml(path: Path, age_ini: float | None = None) -> SimpleNamespace:
    """The radionuclide inputs and star.age_ini [Gyr] of a config file."""
    with open(path, 'rb') as f:
        data = tomllib.load(f)
    ie = {k: data['interior_energetics'][k] for k in _RADIO_KEYS}
    age = data['star']['age_ini'] if age_ini is None else age_ini
    return SimpleNamespace(
        interior_energetics=SimpleNamespace(**ie), star=SimpleNamespace(age_ini=age)
    )


def _config(**radio) -> SimpleNamespace:
    """A config with every isotope off except the ones given."""
    ie = {k: 0.0 for k in _RADIO_KEYS}
    ie.update(radio)
    return SimpleNamespace(
        interior_energetics=SimpleNamespace(**ie), star=SimpleNamespace(age_ini=0.1)
    )


def _kut(inventory):
    """The long-lived isotopes of an inventory."""
    return [e for e in inventory if e[0] in ('k40', 'th232', 'u235', 'u238')]


def test_inventory_skips_isotopes_with_zero_element_or_zero_ratio():
    """An isotope reaches the interior only when element total and ratio are both > 0;
    uranium gives two entries from one element total."""
    cfg = _config(
        radio_Al=2.35e4,  # element present, ratio zero: skipped
        radio_Fe60_abund=1e-8,  # ratio set, element zero: skipped
        radio_K=310.0,
        radio_K40_abund=1.1668e-4,
        radio_U=0.031,
        radio_U235_abund=0.0072045,
        radio_U238_abund=0.9927955,
    )
    inv = radionuclide_inventory(cfg)
    assert [e[0] for e in inv] == ['k40', 'u235', 'u238']
    assert inv[0][1] == pytest.approx(310.0, rel=1e-12)
    assert inv[0][2] == pytest.approx(1.1668e-4, rel=1e-12)
    assert inv[1][1] == pytest.approx(inv[2][1], rel=1e-12)  # one U total for both
    # All isotopes off: empty list, and zero heating rather than an error
    assert radionuclide_inventory(_config()) == []
    assert radiogenic_heating([], 0.0, 1e9) == pytest.approx(0.0, abs=1e-30)


def test_radio_t0_is_negative_for_inputs_before_the_model_start():
    """Inputs at formation (radio_tref = 0) sit age_ini before model time zero [yr]."""
    cfg = _config(radio_tref=0.0)
    cfg.star.age_ini = 0.010  # Gyr
    assert radio_t0_years(cfg) == pytest.approx(-1.0e7, rel=1e-12)
    cfg.interior_energetics.radio_tref = 4.567
    assert radio_t0_years(cfg) == pytest.approx(4.557e9, rel=1e-12)


@pytest.mark.physics_invariant
def test_heating_matches_closed_form_and_halves_after_one_half_life():
    """26Al at the canonical ratio in 2.35 wt% Al: H(t0) = C * 1e-6 * X * h, and H
    halves after one half-life and falls by 2**-3 after three."""
    inv = [('al26', 2.35e4, 5.25e-5)]
    h0 = 2.35e4 * 1e-6 * 5.25e-5 * radnuc_data['al26']['heatprod']
    t_half = radnuc_data['al26']['halflife']
    t0 = 2.0e6  # yr, off zero so a dropped t0 shows
    times = np.array([t0, t0 + t_half, t0 + 3 * t_half])
    h = radiogenic_heating(inv, t0, times)
    np.testing.assert_allclose(h, h0 * np.array([1.0, 0.5, 0.125]), rtol=1e-12)
    assert 1e-7 < h[0] < 1e-6  # about 4.4e-7 W/kg; a missing 1e-6 gives 0.44
    assert h[1] < h[0]  # decay, not growth (sign of the exponent)
    # Scalar time gives a float equal to the array element
    assert radiogenic_heating(inv, t0, t0 + t_half) == pytest.approx(h[1], rel=1e-12)


@pytest.mark.physics_invariant
def test_mercury_formation_inputs_reproduce_present_day_heating():
    """K, U and Th given at Solar System formation (mercury.toml) give the heating
    history of the present-day Mercury mantle of Jaggi et al. (2021, Appendix A:
    K 403 ppmw, U 28 ppbw, Th 49 ppbw, 40K/K 1.17e-4, 235U/U 0.007, 238U/U 0.993)
    at every stellar age. The 5 significant digits of the config values set the
    5e-5 tolerance."""
    merc = _config_from_toml(_INPUT / 'planets' / 'mercury.toml', age_ini=0.0)
    pres = _config_from_toml(_INPUT / 'all_options.toml', age_ini=0.0)
    pres_ie = pres.interior_energetics
    pres_ie.radio_K, pres_ie.radio_U, pres_ie.radio_Th = 403.0, 0.028, 0.049  # ppmw
    pres_ie.radio_K40_abund = 1.17e-4  # kg/kg
    pres_ie.radio_U235_abund, pres_ie.radio_U238_abund = 0.007, 0.993
    ages = np.array([0.0, 1.0e7, 1.0e9, 4.567e9])  # stellar age [yr]
    h_merc = radiogenic_heating(_kut(radionuclide_inventory(merc)), radio_t0_years(merc), ages)
    h_pres = radiogenic_heating(_kut(radionuclide_inventory(pres)), radio_t0_years(pres), ages)
    np.testing.assert_allclose(h_merc, h_pres, rtol=5e-5)
    # Today's value is near 5.4e-12 W/kg, at formation about 6.3 times larger
    assert h_pres[-1] == pytest.approx(5.40e-12, rel=1e-2)
    assert h_pres[0] / h_pres[-1] == pytest.approx(6.32, rel=1e-2)

    # Wrong set-up 1: formation ratios with present-day element totals (U, Th not grown)
    ie = merc.interior_energetics
    wrong = _config_from_toml(_INPUT / 'planets' / 'mercury.toml', age_ini=0.0)
    wrong.interior_energetics.radio_U = pres_ie.radio_U
    wrong.interior_energetics.radio_Th = pres_ie.radio_Th
    h_wrong = radiogenic_heating(_kut(radionuclide_inventory(wrong)), 0.0, ages)
    assert np.all(np.abs(h_wrong / h_pres - 1.0) > 0.05)
    # Wrong set-up 2: present-day values read as formation values (no back-scaling)
    pres_ie.radio_tref = 0.0
    h_wrong2 = radiogenic_heating(_kut(radionuclide_inventory(pres)), 0.0, ages)
    # 6.3 times too low at formation, still 0.6 of the right value today
    assert np.all(np.abs(h_wrong2 / h_pres - 1.0) > 0.3)
    # The formation 235U/U + 238U/U still sums to 1
    assert ie.radio_U235_abund + ie.radio_U238_abund == pytest.approx(1.0, abs=1e-9)
