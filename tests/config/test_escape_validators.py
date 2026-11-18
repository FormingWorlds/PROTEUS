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

    assert Escape(module='zephyrus').effective_hill_clamp is False
    el = Escape(module='zephyrus', zephyrus=Zephyrus(scheme='el'))
    assert el.effective_hill_clamp is True
    assert el.hill_clamp_frac == pytest.approx(1.0, rel=1e-12)
    assert Escape(module='dummy').effective_hill_clamp is True
    assert Escape(module=None).effective_hill_clamp is True
    # An explicit setting wins under either scheme, and for any module.
    assert Escape(module='zephyrus', hill_clamp=True).effective_hill_clamp is True
    for module in ('zephyrus', 'dummy'):
        off = Escape(module=module, zephyrus=Zephyrus(scheme='el'), hill_clamp=False)
        assert off.effective_hill_clamp is False, module

    from helpers import PROTEUS_ROOT

    from proteus.config import read_config_object

    toi = read_config_object(PROTEUS_ROOT / 'input' / 'planets' / 'toi561b.toml').escape
    assert toi.zephyrus.scheme == 'el' and toi.effective_hill_clamp is False


def test_scheme_dependent_defaults_follow_the_scheme_without_being_stored():
    """The reservoir and the tidal factor follow the scheme when left unset.

    The dispatcher returns a rate per element, so its reservoir is ``pxuv``;
    energy-limited escape splits by atmospheric abundance (``outgas``). The
    tidal reduction of the barrier is on under the dispatcher, as in ZEPHYRUS,
    and off under energy-limited escape, as it was before the dispatcher. The
    fields stay unset, so a later change of module or scheme moves the
    defaults with it.
    """
    from proteus.config._escape import Zephyrus

    d = Escape(module='zephyrus')
    assert d.uses_dispatcher is True
    assert d.effective_reservoir == 'pxuv'
    assert d.zephyrus.effective_tidal is True
    assert d.reservoir is None and d.hill_clamp is None and d.zephyrus.tidal is None
    el = Escape(module='zephyrus', zephyrus=Zephyrus(scheme='el'))
    assert el.uses_dispatcher is False
    assert el.effective_reservoir == 'outgas'
    assert el.zephyrus.effective_tidal is False
    assert Escape(module='boreas').effective_reservoir == 'outgas'
    no_tide = Escape(module='zephyrus', zephyrus=Zephyrus(tidal=False))
    assert no_tide.zephyrus.effective_tidal is False

    d.module = 'dummy'
    assert d.effective_reservoir == 'outgas' and d.effective_hill_clamp is True


def test_a_case_switched_away_from_the_dispatcher_writes_and_reloads(tmp_path):
    """A case switched to dummy escape or to "el" reloads from the file written for it.

    This is what a grid does to each case: it loads the reference config,
    sets the case value, writes the file and reads it back. The reference
    leaves the reservoir, the Hill clip and the tidal factor unset, so the
    written case leaves them unset too and they take the case's defaults.
    """
    from helpers import PROTEUS_ROOT

    from proteus.config import read_config_object
    from proteus.utils.helper import recursive_setattr

    reference = PROTEUS_ROOT / 'input' / 'all_options.toml'
    for key, value in (('escape.module', 'dummy'), ('escape.zephyrus.scheme', 'el')):
        case = read_config_object(reference)
        assert case.escape.effective_reservoir == 'pxuv'
        recursive_setattr(case, key, value)
        path = tmp_path / f'{value}.toml'
        case.write(str(path))
        back = read_config_object(path)
        assert back.escape.reservoir is None
        assert back.escape.effective_reservoir == 'outgas'
        assert back.escape.effective_hill_clamp is True
        assert back.escape.zephyrus.tidal is None
    assert back.escape.zephyrus.effective_tidal is False


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
        assert el.effective_reservoir == res
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


def test_an_unknown_scheme_is_refused():
    """A misspelt scheme fails at load instead of falling back to a default."""
    from proteus.config._escape import Zephyrus

    with pytest.raises(ValueError, match='scheme'):
        Zephyrus(scheme='dispatch')
    assert Zephyrus(scheme='el').scheme == 'el'
