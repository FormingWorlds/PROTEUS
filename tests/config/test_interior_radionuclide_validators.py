"""Unit tests for the radionuclide fields of ``proteus.config._interior.Interior``.

Covers the [0, 1] bounds on each isotope/element mass ratio [kg/kg] and the
check that 235U/U + 238U/U does not exceed 1.
"""

from __future__ import annotations

import pytest

from proteus.config._interior import Interior

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_ABUND_FIELDS = (
    'radio_Al26_abund',
    'radio_Fe60_abund',
    'radio_K40_abund',
    'radio_U235_abund',
    'radio_U238_abund',
    'radio_Th232_abund',
)


@pytest.mark.parametrize('name', _ABUND_FIELDS)
def test_isotope_ratio_rejects_values_outside_unit_interval(name):
    """A mass ratio of an isotope to its element lies in [0, 1]; the bounds pass."""
    # The other uranium ratio is set to 0 so only this field's own bound is exercised
    other = {'radio_U235_abund': 'radio_U238_abund', 'radio_U238_abund': 'radio_U235_abund'}
    base = {other[name]: 0.0} if name in other else {}
    for bad in (-1e-6, 1.5):
        with pytest.raises(ValueError):
            Interior(module='aragog', **base, **{name: bad})
    for good in (0.0, 1e-8, 1.0):
        i = Interior(module='aragog', **base, **{name: good})
        assert getattr(i, name) == pytest.approx(good, abs=1e-30)


def test_uranium_ratios_summing_above_one_are_rejected():
    """235U/U + 238U/U above 1 is rejected"""
    i = Interior(module='aragog', radio_U235_abund=0.24274, radio_U238_abund=0.75726)
    assert i.radio_U235_abund == pytest.approx(0.24274, rel=1e-12)
    i = Interior(module='aragog', radio_U235_abund=0.0072045, radio_U238_abund=0.9927955)
    assert i.radio_U238_abund == pytest.approx(0.9927955, rel=1e-12)
    with pytest.raises(ValueError, match='must not exceed 1'):
        Interior(module='aragog', radio_U235_abund=0.3, radio_U238_abund=0.75)
