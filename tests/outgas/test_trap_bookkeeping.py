"""Unit tests for ``proteus.outgas.trap_bookkeeping``.

The bookkeeping both trapping schemes share. These tests exercise:

* the mass a step returns to the melt, taken per species from its net moves,
  and the ``trap_kg_returned`` column escape reads,
* the defaults of the layer-only columns, written as NaN by a front step,
* the front speed from the crystallised mass, its closed form and its guards.

The reservoir moves themselves are tested in ``test_trapping.py``.

See ``docs/How-to/testing.md`` and ``docs/Explanations/test_framework.md``
for the test framework.
"""

from __future__ import annotations

import numpy as np
import pytest

from proteus.outgas.trap_bookkeeping import TrappingStep, front_speed, record_step

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.physics_invariant
def test_the_returned_mass_is_the_negative_part_of_the_net_moves():
    """A step that traps H2O and returns CO2 and N2 returns exactly the CO2
    and N2, which ``trap_kg_returned`` sums; the net step column keeps the
    sign of the sum, so a net trapping step can still return a species."""
    step = TrappingStep(mode='layers', dm_rm=1.0e22, f_tl=0.0, melt_mass=1.0e24)
    step.trapped_kg = {'H2O': 5.0e18, 'CO2': -2.0e17, 'N2': -3.0e15}
    assert step.returned_kg == {'CO2': 2.0e17, 'N2': 3.0e15}
    row = {'trap_kg_cumulative': 1.0e19}
    record_step(row, step)
    assert row['trap_kg_returned'] == pytest.approx(2.03e17, rel=1e-15)
    assert row['trap_kg_step'] == pytest.approx(5.0e18 - 2.03e17, rel=1e-15)
    # Discrimination: the net step is positive, so a return read from its sign
    # would be zero.
    assert row['trap_kg_step'] > 0.0
    assert row['trap_kg_cumulative'] == pytest.approx(1.0e19 + 5.0e18 - 2.03e17, rel=1e-15)

    nothing = TrappingStep(mode='front', dm_rm=1.0e22, f_tl=0.1, melt_mass=1.0e24)
    nothing.trapped_kg = {'H2O': 1.0e18}
    record_step(row, nothing)
    assert row['trap_kg_returned'] == pytest.approx(0.0, abs=0.0)


def test_a_front_step_writes_the_layer_only_columns_as_nan():
    """The mush-node count and the ocean melt fraction belong to the layer
    scheme; a front step leaves them NaN, which the chemistry reads as no
    ocean melt to substitute."""
    row = {}
    record_step(row, TrappingStep(mode='front', dm_rm=0.0, f_tl=0.0, melt_mass=0.0))
    assert np.isnan(row['trap_n_mush'])
    assert np.isnan(row['trap_Phi_ocean'])
    step = TrappingStep(mode='layers', dm_rm=0.0, f_tl=0.0, melt_mass=0.0)
    step.n_mush, step.phi_ocean = 12.0, 0.4
    record_step(row, step)
    assert row['trap_n_mush'] == pytest.approx(12.0, rel=0.0)
    assert row['trap_Phi_ocean'] == pytest.approx(0.4, rel=0.0)


@pytest.mark.physics_invariant
def test_front_speed_sweeps_the_crystallised_mass_through_the_base_area():
    """A front at 5000 km under rock of 4000 kg m-3 that crystallises 1e22 kg
    in 1e10 s advances at dM / (4 pi r^2 rho dt) = 7.96e-7 m s-1, whatever the
    sign of dM; a step of no length or a front of no area gives zero."""
    v = front_speed(1.0e22, 5.0e6, 4000.0, 1.0e10)
    expected = 1.0e22 / (4.0 * np.pi * 5.0e6**2 * 4000.0 * 1.0e10)
    assert v == pytest.approx(expected, rel=1e-14)
    assert v == pytest.approx(7.957747e-7, rel=1e-6)
    assert front_speed(-1.0e22, 5.0e6, 4000.0, 1.0e10) == pytest.approx(v, rel=1e-15)
    # Discrimination: r instead of r^2 is off by 5e6, and a missing 4 pi by 12.6.
    assert abs(1.0e22 / (5.0e6 * 4000.0 * 1.0e10) - v) > 1.0e3 * v
    assert abs(1.0e22 / (5.0e6**2 * 4000.0 * 1.0e10) - v) > 10.0 * v
    for args in ((1.0e22, 5.0e6, 4000.0, 0.0), (1.0e22, 0.0, 4000.0, 1.0e10)):
        assert front_speed(*args) == pytest.approx(0.0, abs=0.0)
    assert front_speed(1.0e22, float('nan'), 4000.0, 1.0e10) == pytest.approx(0.0, abs=0.0)
