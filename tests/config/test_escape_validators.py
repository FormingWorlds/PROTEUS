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


def test_hill_clamp_default_follows_the_escape_scheme():
    """Unset, the Hill clip is off under the dispatcher and on otherwise.

    The dispatcher screens the flow radius against the Roche lobe and carries
    an L1 overflow rate, so clipping the XUV level as well would hide the
    geometry it treats. Energy-limited escape has no such screen and keeps the
    clip, at the full Hill radius.
    """
    from proteus.config._escape import Zephyrus

    assert Escape(module='zephyrus').hill_clamp is False
    el = Escape(module='zephyrus', zephyrus=Zephyrus(scheme='el'))
    assert el.hill_clamp is True
    assert el.hill_clamp_frac == pytest.approx(1.0, rel=1e-12)
    assert Escape(module='dummy').hill_clamp is True
    assert Escape(module=None).hill_clamp is True
    # An explicit setting wins under either scheme.
    assert Escape(module='zephyrus', hill_clamp=True).hill_clamp is True


def test_scheme_dependent_defaults_resolve_once_at_construction():
    """The reservoir and the tidal factor follow the scheme when left unset.

    The dispatcher returns a rate per element, so its reservoir is ``pxuv``;
    energy-limited escape splits by atmospheric abundance (``outgas``). The
    tidal reduction of the barrier is on under the dispatcher, as in ZEPHYRUS,
    and off under energy-limited escape, as it was before the dispatcher.
    """
    from proteus.config._escape import Zephyrus

    d = Escape(module='zephyrus')
    assert d.uses_dispatcher is True
    assert d.reservoir == 'pxuv'
    assert d.zephyrus.tidal is True
    el = Escape(module='zephyrus', zephyrus=Zephyrus(scheme='el'))
    assert el.uses_dispatcher is False
    assert el.reservoir == 'outgas'
    assert el.zephyrus.tidal is False
    assert Escape(module='boreas').reservoir == 'outgas'
    assert Escape(module='zephyrus', zephyrus=Zephyrus(tidal=False)).zephyrus.tidal is False


def test_reservoir_must_match_the_escape_scheme():
    """``pxuv`` needs the dispatcher, and the dispatcher refuses an abundance split.

    The dispatcher sets the escaping composition at the wind base, so an
    explicit ``outgas`` or ``bulk`` beside it would describe a different
    composition from the one escaping; ``pxuv`` without it has no
    per-element rates to read.
    """
    from proteus.config._escape import Zephyrus

    for res in ('outgas', 'bulk'):
        with pytest.raises(ValueError, match='sets the escaping composition'):
            Escape(module='zephyrus', reservoir=res)
        el = Escape(module='zephyrus', zephyrus=Zephyrus(scheme='el'), reservoir=res)
        assert el.reservoir == res
    for module, scheme in (('zephyrus', 'el'), ('dummy', 'dispatcher'), ('boreas', 'el')):
        with pytest.raises(ValueError, match='"pxuv" requires'):
            Escape(module=module, zephyrus=Zephyrus(scheme=scheme), reservoir='pxuv')
    with pytest.raises(ValueError, match='must be one of'):
        Escape(module='zephyrus', reservoir='wind')
    assert Escape(module='zephyrus', reservoir='pxuv').reservoir == 'pxuv'


def test_dispatcher_settings_are_checked_by_zephyrus_at_load():
    """A dispatcher setting ZEPHYRUS refuses fails at load, naming the field.

    The hydrostatic refinement needs a ceiling of at least four times the
    first grid; the efficiency must be positive for the dispatcher (zero is
    allowed only for energy-limited escape, where it switches escape off).
    """
    from proteus.config._escape import Zephyrus

    with pytest.raises(ValueError, match='hydrostatic_levels_max'):
        Escape(
            module='zephyrus',
            zephyrus=Zephyrus(hydrostatic_levels_min=200, hydrostatic_levels_max=400),
        )
    with pytest.raises(ValueError, match='escape.zephyrus: efficiency'):
        Escape(module='zephyrus', zephyrus=Zephyrus(efficiency=0.0))
    el = Escape(module='zephyrus', zephyrus=Zephyrus(scheme='el', efficiency=0.0))
    assert el.zephyrus.efficiency == pytest.approx(0.0, abs=1e-15)
    ok = Escape(module='zephyrus', zephyrus=Zephyrus(hydrostatic_levels_max=800))
    assert ok.zephyrus.hydrostatic_levels_max == 800


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
