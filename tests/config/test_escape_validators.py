"""Unit tests for the escape configuration schema in ``proteus.config._escape``.

Covers the Hill-clip fields: the enabled-by-default state, the (0, 1] bounds
on ``hill_clamp_frac``, and the interaction with module selection.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import pytest

from proteus.config._escape import Escape

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def test_hill_clamp_defaults_to_enabled_with_full_hill_radius():
    """A fresh escape config clips at the full Hill radius by default.

    The default matters physically: gas beyond the Hill radius is not bound
    to the planet, so an unclipped default would size escape from material
    the planet does not hold. The v2-to-v3 migration pins the field off for
    older configs, so this default only reaches configs written against the
    current schema.
    """
    cfg = Escape(module='zephyrus')
    assert cfg.hill_clamp is True
    assert cfg.hill_clamp_frac == pytest.approx(1.0, rel=1e-12)
    # The default holds regardless of the escape module chosen.
    assert Escape(module='dummy').hill_clamp is True
    assert Escape(module=None).hill_clamp is True


def test_hill_clamp_frac_rejects_values_outside_unit_interval():
    """The fraction must sit in (0, 1]: zero would clip every level to the
    solid body, and above one would place the limit outside the bound region,
    defeating the clip while appearing enabled.
    """
    for bad in (0.0, -0.5, 1.0001, 2.0):
        with pytest.raises(ValueError):
            Escape(module='zephyrus', hill_clamp_frac=bad)
    # The boundaries of validity are accepted: just inside zero, and exactly one.
    assert Escape(module='zephyrus', hill_clamp_frac=1e-6).hill_clamp_frac == pytest.approx(
        1e-6, rel=1e-12
    )
    assert Escape(module='zephyrus', hill_clamp_frac=1.0).hill_clamp_frac == pytest.approx(
        1.0, rel=1e-12
    )


def test_hill_clamp_can_be_disabled_explicitly():
    """An explicit ``hill_clamp = false`` survives construction, since A/B
    comparisons against unclipped behaviour rely on it; the fraction stays
    validated even while inert.
    """
    cfg = Escape(module='zephyrus', hill_clamp=False, hill_clamp_frac=0.5)
    assert cfg.hill_clamp is False
    assert cfg.hill_clamp_frac == pytest.approx(0.5, rel=1e-12)
    with pytest.raises(ValueError):
        Escape(module='zephyrus', hill_clamp=False, hill_clamp_frac=1.5)


def test_step_max_frac_rejects_values_outside_unit_interval():
    """The per-step cap must sit in (0, 1]: zero would forbid escape from
    removing anything at all, and above one would admit a step that asks for
    more mass than the reservoir holds, which is the overshoot the cap exists
    to bound.
    """
    for bad in (0.0, -0.25, 1.0001, 2.0):
        with pytest.raises(ValueError):
            Escape(module='zephyrus', step_max_frac=bad)
    # The boundaries of validity are accepted: just inside zero, and exactly one.
    assert Escape(module='zephyrus', step_max_frac=1e-6).step_max_frac == pytest.approx(
        1e-6, rel=1e-12
    )
    assert Escape(module='zephyrus', step_max_frac=1.0).step_max_frac == pytest.approx(
        1.0, rel=1e-12
    )
    # The default is the measured value, not an end of the valid range.
    assert Escape(module='zephyrus').step_max_frac == pytest.approx(0.25, rel=1e-12)


def test_step_dt_floor_frac_rejects_values_outside_unit_interval():
    """The floor fraction must sit in (0, 1]: zero would let a capped step
    shorten without bound, asking for a step no run finishes, and above one
    would raise the escape floor above ``params.dt.minimum``, tightening the
    global floor from inside the escape schema.
    """
    for bad in (0.0, -1e-3, 1.0001, 10.0):
        with pytest.raises(ValueError):
            Escape(module='zephyrus', step_dt_floor_frac=bad)
    # Exactly one is valid and makes the exemption inert, leaving the ordinary
    # floor in charge; the small end is where the exemption does its work.
    assert Escape(
        module='zephyrus', step_dt_floor_frac=1.0
    ).step_dt_floor_frac == pytest.approx(1.0, rel=1e-12)
    assert Escape(
        module='zephyrus', step_dt_floor_frac=1e-6
    ).step_dt_floor_frac == pytest.approx(1e-6, rel=1e-12)
    assert Escape(module='zephyrus').step_dt_floor_frac == pytest.approx(1e-3, rel=1e-12)
