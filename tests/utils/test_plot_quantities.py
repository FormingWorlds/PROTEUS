"""Unit tests for ``proteus.utils.plot_quantities``.

Checks the plotting presets for grid quantities against physics, not against a copy
of the table:
- each unit scale converts one display unit back to the helpfile SI value
  (R_earth, AU, day, GPa, Gyr, g cm-3)
- every scaled quantity names its display unit in the label
- every scale is finite and positive, so no axis flips sign or collapses

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import math

import pytest

from proteus.utils.constants import AU, M_earth, R_earth, secs_per_day
from proteus.utils.plot_quantities import PRESET_QUANTITIES

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.parametrize(
    'quantity, si_per_display_unit',
    [
        ('R_int', R_earth),  # m per R_earth
        ('M_planet', M_earth),  # kg per M_earth
        ('semimajorax', AU),  # m per AU
        ('orbital_period', secs_per_day),  # s per day
        ('P_cmb', 1.0e9),  # Pa per GPa
        ('age_star', 1.0e9),  # yr per Gyr
        ('rho_obs', 1.0e3),  # kg m-3 per g cm-3
        ('Phi_global', 0.01),  # fraction per percent
    ],
)
def test_scale_converts_one_display_unit_back_to_si(quantity, si_per_display_unit):
    """scale * (SI value of one display unit) == 1: the helpfile value of exactly one
    display unit plots as 1. An inverted scale would give (SI value)**2 instead.
    """
    scale = PRESET_QUANTITIES[quantity].scale
    assert scale * si_per_display_unit == pytest.approx(1.0, rel=1e-12)
    assert scale / si_per_display_unit != pytest.approx(1.0)  # inverted-scale guard


def test_every_preset_has_a_finite_positive_scale_and_a_label():
    """No preset flips the sign or zeroes an axis, and every label is non-empty."""
    bad = [
        k for k, q in PRESET_QUANTITIES.items() if not (math.isfinite(q.scale) and q.scale > 0)
    ]
    assert bad == []
    assert all(q.label.strip() for q in PRESET_QUANTITIES.values())


def test_scaled_quantities_state_their_display_unit():
    """A quantity whose value is converted names the unit it is converted to, so a
    reader of the axis knows the conversion happened.
    """
    unlabelled = [
        k for k, q in PRESET_QUANTITIES.items() if q.scale != 1.0 and '[' not in q.label
    ]
    assert unlabelled == []
    assert '[g cm$^{-3}$]' in PRESET_QUANTITIES['rho_obs'].label
    assert PRESET_QUANTITIES['atm_kg_per_mol'].label == r'MMW [g mol$^{-1}$]'
