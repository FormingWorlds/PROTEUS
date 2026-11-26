"""Unit tests for ``proteus.interior_energetics.radiogenic``.

Invariants checked:
  - heat production follows ``C * 1e-6 * X * h * 2**((t0 - t) / t_half)`` and
    halves after one half-life;
  - atom ratios become mass ratios through the isotope and element molar
    masses, and the config defaults are the Ruedas (2017) atom ratios by mass;
  - the K, U and Th inputs at Solar System formation in
    ``input/planets/mercury.toml`` give the heating history of the present-day
    Mercury values of Jaggi et al. (2021, Appendix A), because one set is the
    other decayed back in time;
  - heating at the model start above ``RADIO_HEATING_MAX`` is clipped to it and
    still decays with the isotope half-lives;
  - isotopes with a zero element total or a zero ratio are left out.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from proteus.config._interior import Interior
from proteus.interior_energetics.radiogenic import (
    limit_radiogenic_heating,
    mass_ratio_from_atom_ratio,
    radio_t0_years,
    radiogenic_heating,
    radiogenic_inputs,
    radionuclide_inventory,
)
from proteus.utils.constants import RADIO_HEATING_MAX, element_mmw, radnuc_data

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


def _uranium_mass_ratios(r235: float, r238: float) -> tuple[float, float]:
    """235U/U and 238U/U by mass from atom ratios; U holds only these two isotopes."""
    m_u = r235 * radnuc_data['u235']['mass'] + r238 * radnuc_data['u238']['mass']  # kg/mol
    return (
        mass_ratio_from_atom_ratio('u235', r235, m_u),
        mass_ratio_from_atom_ratio('u238', r238, m_u),
    )


def test_inventory_skips_isotopes_with_zero_element_or_zero_ratio():
    """An isotope reaches the interior only when element total and ratio are both > 0;
    uranium gives two entries from one element total, and each ratio is its own field."""
    cfg = _config(
        radio_Al=2.35e4,  # element present, ratio zero: skipped
        radio_Fe60_abund=1e-8,  # ratio set, element zero: skipped
        radio_K=310.0,
        radio_K40_abund=1.1926e-4,
        radio_Th=0.124,
        radio_Th232_abund=0.5,  # off 1, so a hard-coded Th ratio shows
        radio_U=0.031,
        radio_U235_abund=0.0071141,
        radio_U238_abund=0.9928859,
    )
    inv = radionuclide_inventory(cfg)
    assert [e[0] for e in inv] == ['k40', 'th232', 'u235', 'u238']
    assert inv[0][1] == pytest.approx(310.0, rel=1e-12)
    assert inv[0][2] == pytest.approx(1.1926e-4, rel=1e-12)
    assert inv[1][2] == pytest.approx(0.5, rel=1e-12)
    assert inv[2][1] == pytest.approx(inv[3][1], rel=1e-12)  # one U total for both
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


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_heating_matches_closed_form_and_halves_after_one_half_life():
    """Analytical limit: 26Al in 2.35 wt% Al at the canonical 26Al/Al gives
    H(t0) = C * 1e-6 * X * h, and the decay law halves H after one half-life and
    divides it by 8 after three."""
    inv = [('al26', 2.35e4, 5.0563e-5)]
    h0 = 2.35e4 * 1e-6 * 5.0563e-5 * radnuc_data['al26']['heatprod']
    t_half = radnuc_data['al26']['halflife']
    t0 = 2.0e6  # yr, off zero so a dropped t0 shows
    n_half = np.array([0.0, 1.0, 3.0])
    h = radiogenic_heating(inv, t0, t0 + n_half * t_half)
    np.testing.assert_allclose(h, h0 * 2.0**-n_half, rtol=1e-12)
    assert 1e-7 < h[0] < 1e-6  # about 4.3e-7 W/kg; a missing 1e-6 gives 0.43
    assert h[1] < h[0]  # decay, not growth (sign of the exponent)
    # exp(-t / t_half) without ln 2 would leave 0.37 after one half-life, not 0.5
    assert abs(h[1] / h[0] - np.exp(-1.0)) > 0.1
    # Scalar time gives a float equal to the array element
    assert radiogenic_heating(inv, t0, t0 + t_half) == pytest.approx(h[1], rel=1e-12)


@pytest.mark.physics_invariant
def test_atom_ratios_become_mass_ratios_through_the_molar_masses():
    """The canonical 26Al/27Al = 5.25e-5 (atoms) is 5.0563e-5 by mass, 3.7 % lower,
    because 26Al is lighter than the mean Al atom; the config defaults are the
    present-day atom ratios of Ruedas (2017) by mass, and 235U/U + 238U/U = 1."""
    r_al = 5.25e-5 / (1.0 + 5.25e-5)  # 26Al per Al atom
    x_al = mass_ratio_from_atom_ratio('al26', r_al, element_mmw['Al'])
    assert x_al == pytest.approx(5.0563e-5, rel=1e-4)
    assert abs(x_al / 5.25e-5 - 1.0) > 0.03  # the atom ratio used as a mass ratio
    # 40K is heavier than the mean K atom, so its mass ratio is the larger one
    x_k = mass_ratio_from_atom_ratio('k40', 1.1668e-4, element_mmw['K'])
    assert x_k > 1.1668e-4
    defaults = Interior(module='aragog')
    assert defaults.radio_K40_abund == pytest.approx(x_k, rel=1e-4)
    x235, x238 = _uranium_mass_ratios(0.0072045, 0.9927955)
    assert defaults.radio_U235_abund == pytest.approx(x235, rel=1e-4)
    assert defaults.radio_U238_abund == pytest.approx(x238, rel=1e-6)
    assert x235 + x238 == pytest.approx(1.0, abs=1e-12)
    # Mercury inputs at formation: 26Al and 60Fe (56Fe is 91.754 % of Fe atoms)
    merc = _config_from_toml(_INPUT / 'planets' / 'mercury.toml').interior_energetics
    assert merc.radio_Al26_abund == pytest.approx(x_al, rel=1e-4)
    x_fe = mass_ratio_from_atom_ratio('fe60', 1e-8 * 0.91754, element_mmw['Fe'])
    assert merc.radio_Fe60_abund == pytest.approx(x_fe, rel=1e-4)


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_mercury_formation_inputs_reproduce_present_day_heating():
    """Cross-epoch check: K, U and Th given at Solar System formation (mercury.toml)
    give the heating history of the present-day Mercury mantle of Jaggi et al. (2021,
    Appendix A: K 403 ppmw, U 28 ppbw, Th 49 ppbw, atom ratios 40K/K 1.17e-4,
    235U/U 0.007, 238U/U 0.993) at every stellar age. The 5 significant digits of the
    config values set the 5e-5 tolerance."""
    merc = _config_from_toml(_INPUT / 'planets' / 'mercury.toml', age_ini=0.0)
    pres = _config_from_toml(_INPUT / 'all_options.toml', age_ini=0.0)
    pres_ie = pres.interior_energetics
    pres_ie.radio_K, pres_ie.radio_U, pres_ie.radio_Th = 403.0, 0.028, 0.049  # ppmw
    pres_ie.radio_K40_abund = mass_ratio_from_atom_ratio('k40', 1.17e-4, element_mmw['K'])
    pres_ie.radio_U235_abund, pres_ie.radio_U238_abund = _uranium_mass_ratios(0.007, 0.993)
    ages = np.array([0.0, 1.0e7, 1.0e9, 4.567e9])  # stellar age [yr]
    h_merc = radiogenic_heating(_kut(radionuclide_inventory(merc)), radio_t0_years(merc), ages)
    h_pres = radiogenic_heating(_kut(radionuclide_inventory(pres)), radio_t0_years(pres), ages)
    np.testing.assert_allclose(h_merc, h_pres, rtol=5e-5)
    # Today K 1.386e-12 + U 2.750e-12 + Th 1.292e-12 = 5.428e-12 W/kg, 6.33 times more at formation
    assert h_pres[-1] == pytest.approx(5.428e-12, rel=1e-3)
    assert h_pres[0] / h_pres[-1] == pytest.approx(6.331, rel=1e-3)

    # Wrong set-up 1: formation ratios with present-day element totals (U, Th not grown)
    ie = merc.interior_energetics
    wrong = _config_from_toml(_INPUT / 'planets' / 'mercury.toml', age_ini=0.0)
    wrong.interior_energetics.radio_U = pres_ie.radio_U
    wrong.interior_energetics.radio_Th = pres_ie.radio_Th
    h_wrong = radiogenic_heating(_kut(radionuclide_inventory(wrong)), 0.0, ages)
    assert np.all(np.abs(h_wrong / h_pres - 1.0) > 0.05)
    # Wrong set-up 2: present-day values read as formation values (no back-scaling)
    h_wrong2 = radiogenic_heating(_kut(radionuclide_inventory(pres)), 0.0, ages)
    # 6.3 times too low at formation, still 0.6 of the right value today
    assert np.all(np.abs(h_wrong2 / h_pres - 1.0) > 0.3)
    # Wrong set-up 3: the 40K/K atom ratio used as a mass ratio; 0.55 % off at every age
    atom_k = SimpleNamespace(**vars(pres_ie))
    atom_k.radio_K40_abund = 1.17e-4
    h_k = radiogenic_heating(
        _kut(radionuclide_inventory(SimpleNamespace(interior_energetics=atom_k))),
        radio_t0_years(pres),
        ages,
    )
    assert np.all(np.abs(h_k / h_pres - 1.0) > 2e-3)
    # The formation 235U/U + 238U/U still sums to 1
    assert ie.radio_U235_abund + ie.radio_U238_abund == pytest.approx(1.0, abs=1e-9)


@pytest.mark.physics_invariant
def test_heating_above_the_limit_is_clipped_at_the_model_start(caplog):
    """26Al at its formation ratio but with radio_tref left at the present day
    overflows (about 2^6200 times the formation value): the inputs move to model time
    0 with H(0) = RADIO_HEATING_MAX, a warning names radio_tref, and the clipped
    heating still decays with the 26Al half-life."""
    cfg = _config(radio_tref=4.567, radio_Al=2.35e4, radio_Al26_abund=5.0563e-5)
    cfg.interior_energetics.radio_K, cfg.interior_energetics.radio_K40_abund = 310.0, 1.1926e-4
    raw = radionuclide_inventory(cfg)
    with np.errstate(over='ignore'):
        assert np.isinf(radiogenic_heating(raw, radio_t0_years(cfg), 0.0))
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.interior_energetics.radiogenic'):
        inv, t0 = radiogenic_inputs(cfg)
    assert 'radio_tref' in caplog.text
    assert t0 == pytest.approx(0.0, abs=1e-30)
    assert [e[0] for e in inv] == ['al26', 'k40']
    h = radiogenic_heating(inv, t0, np.array([0.0, radnuc_data['al26']['halflife']]))
    assert np.all(np.isfinite(h))
    assert h[0] == pytest.approx(RADIO_HEATING_MAX, rel=1e-12)
    # 26Al dominates by about 1e1870, so one half-life halves the clipped value
    assert h[1] / h[0] == pytest.approx(0.5, rel=1e-9)


@pytest.mark.physics_invariant
def test_heating_below_the_limit_is_left_unchanged(caplog):
    """The Mercury inputs (2e-10 W/kg at 8 Myr) and an empty inventory pass through
    unchanged and without a warning; a limit just below H(0) clips to the limit."""
    merc = _config_from_toml(_INPUT / 'planets' / 'mercury.toml')
    raw, t0_raw = radionuclide_inventory(merc), radio_t0_years(merc)
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.interior_energetics.radiogenic'):
        inv, t0 = radiogenic_inputs(merc)
        assert limit_radiogenic_heating([], 5.0) == ([], 5.0)
    assert inv == raw
    assert t0 == pytest.approx(t0_raw, rel=1e-12)
    assert caplog.text == ''
    h0 = radiogenic_heating(raw, t0_raw, 0.0)
    assert 1e-10 < h0 < RADIO_HEATING_MAX
    # Edge: a limit at half of H(0) clips to it and keeps each isotope's share
    inv_c, t0_c = limit_radiogenic_heating(raw, t0_raw, h_max=0.5 * h0)
    assert radiogenic_heating(inv_c, t0_c, 0.0) == pytest.approx(0.5 * h0, rel=1e-12)
    share = [radiogenic_heating([e], t0_c, 0.0) for e in inv_c]
    share_raw = [radiogenic_heating([e], t0_raw, 0.0) for e in raw]
    np.testing.assert_allclose(np.array(share) / np.array(share_raw), 0.5, rtol=1e-12)
