"""Unit tests for star config fields (config/_star.py).

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import cattrs
import pytest

from proteus.config._star import Star

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
def test_star_bol_scale_window_defaults_disabled():
    """By default the bolometric-scaling window is disabled: no start age
    and zero duration, matching the pre-existing (unwindowed) behaviour
    when a config only sets ``bol_scale``."""
    star = Star()
    assert star.bol_scale_start is None
    assert star.bol_scale_duration == pytest.approx(0.0)
    assert star.bol_scale == pytest.approx(1.0)


@pytest.mark.unit
def test_star_bol_scale_start_accepts_numeric_gyr_value():
    """A numeric ``bol_scale_start`` (Gyr) round-trips through both the
    direct attrs constructor and cattrs structuring (the TOML path)
    without being coerced or clamped."""
    star_direct = Star(bol_scale_start=0.5)
    assert star_direct.bol_scale_start == pytest.approx(0.5)

    star_cattrs = cattrs.structure({'bol_scale_start': 0.5, 'bol_scale_duration': 0.2}, Star)
    assert star_cattrs.bol_scale_start == pytest.approx(0.5)
    assert star_cattrs.bol_scale_duration == pytest.approx(0.2)
    assert star_direct.bol_scale_start == pytest.approx(star_cattrs.bol_scale_start)


@pytest.mark.unit
def test_star_bol_scale_duration_rejects_negative():
    """``bol_scale_duration`` uses the ``ge(0.0)`` validator: a negative
    window length is physically meaningless (time does not run backwards)
    and must raise rather than silently clamp."""
    with pytest.raises(Exception, match='bol_scale_duration'):
        Star(bol_scale_duration=-1.0)
    star = Star(bol_scale_duration=1.0)
    assert star.bol_scale_duration == pytest.approx(1.0)


@pytest.mark.unit
def test_star_bol_scale_duration_accepts_zero_boundary():
    """Exactly zero is the inclusive lower boundary of ``ge(0.0)`` and is
    also the default; it must be accepted, not rejected as a degenerate
    (zero-width) window."""
    star = Star(bol_scale_duration=0.0)
    assert star.bol_scale_duration == pytest.approx(0.0, abs=1e-15)


@pytest.mark.unit
def test_star_bol_scale_nonunity_requires_start():
    """A non-unity ``bol_scale`` with no ``bol_scale_start`` would silently
    apply nowhere (the window is undefined), so the combination is rejected
    outright rather than left as a config that quietly does nothing."""
    with pytest.raises(Exception, match='bol_scale_start'):
        Star(bol_scale=2.0, bol_scale_start=None)

    # The default bol_scale=1.0 is exempt: no window is needed for a no-op.
    star_default = Star(bol_scale_start=None)
    assert star_default.bol_scale == pytest.approx(1.0)

    # Providing a start and a positive duration with a non-unity scale is
    # accepted.
    star_windowed = Star(bol_scale=2.0, bol_scale_start=0.5, bol_scale_duration=0.1)
    assert star_windowed.bol_scale_start == pytest.approx(0.5)


@pytest.mark.unit
def test_star_bol_scale_nonunity_requires_positive_duration():
    """A non-unity ``bol_scale`` with ``bol_scale_start`` set but
    ``bol_scale_duration`` left at its zero default would define a window
    that never opens (start == end), silently scaling nothing anywhere.
    That combination must be rejected, not just the missing-start case."""
    with pytest.raises(Exception, match='bol_scale_duration'):
        Star(bol_scale=2.0, bol_scale_start=0.5)

    with pytest.raises(Exception, match='bol_scale_duration'):
        Star(bol_scale=2.0, bol_scale_start=0.5, bol_scale_duration=0.0)

    # The default bol_scale=1.0 is exempt: a zero duration is a real no-op.
    star_default = Star(bol_scale_duration=0.0)
    assert star_default.bol_scale == pytest.approx(1.0)
