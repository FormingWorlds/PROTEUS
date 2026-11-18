"""Unit tests for planet config validators (config/_planet.py).

Covers the initial entropy gradient rules: ini_dsdr must be finite in every
temperature mode, and a liquidus_super initial condition takes no positive
gradient.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import pytest

from proteus.config._planet import Planet

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.parametrize('dsdr', [4.698e-6, 1e-12])
def test_positive_ini_dsdr_is_rejected_with_liquidus_super(dsdr):
    """A positive ini_dsdr, however small, lowers the deep entropy below the
    certified molten adiabat, so liquidus_super rejects it and the message
    names both fields."""
    with pytest.raises(ValueError) as exc:
        Planet(temperature_mode='liquidus_super', ini_dsdr=dsdr)

    msg = str(exc.value)
    assert 'planet.ini_dsdr' in msg
    assert 'liquidus_super' in msg


@pytest.mark.parametrize(
    ('mode', 'dsdr'),
    [
        ('liquidus_super', -4.698e-6),
        ('liquidus_super', 0.0),
        ('isentropic', 4.698e-6),
        ('adiabatic_from_cmb', 4.698e-6),
    ],
)
def test_ini_dsdr_accepted_where_it_cannot_erode_the_margin(mode, dsdr):
    """Zero and negative gradients keep liquidus_super valid (the boundary
    case 0 included), and other temperature modes keep any sign."""
    planet = Planet(temperature_mode=mode, ini_dsdr=dsdr)

    assert planet.ini_dsdr == pytest.approx(dsdr, abs=0.0)
    assert planet.temperature_mode == mode


@pytest.mark.parametrize('mode', ['liquidus_super', 'isentropic', 'adiabatic_from_cmb'])
@pytest.mark.parametrize('dsdr', [float('nan'), float('inf'), float('-inf')])
def test_non_finite_ini_dsdr_is_rejected_in_every_mode(mode, dsdr):
    """A non-finite ini_dsdr passes every comparison as False and gives a
    non-finite initial entropy profile in every mode, so it is rejected."""
    with pytest.raises(ValueError, match='is not finite') as exc:
        Planet(temperature_mode=mode, ini_dsdr=dsdr)
    assert 'planet.ini_dsdr' in str(exc.value)
