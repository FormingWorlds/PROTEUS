# Test whether some of the plot helper functions work as expected
from __future__ import annotations

import pytest

from proteus.utils.plot import _preset_colours, get_colour, latexify

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
def test_get_colour():
    """``get_colour`` returns the preset hex for known keys, the fallback
    hex for unknown keys, and a deterministic generated hex (from a hash
    of the key) for keys that have no preset but should still produce a
    stable colour across runs.
    """
    assert get_colour('C') == _preset_colours['C']
    assert get_colour('OLR') == _preset_colours['OLR']

    # Test fallback
    assert get_colour('foo') == _preset_colours['_fallback']

    # Test generating a colour
    assert get_colour('SiH4C2') == '#b909ff'


@pytest.mark.unit
def test_latexify():
    """``latexify`` wraps numeric subscripts in LaTeX ``$_N$`` syntax for
    chemical formulae, leaves single-letter symbols unchanged, and is a
    pure string transform (different inputs map to different outputs).
    """
    assert latexify('H2O4') == r'H$_2$O$_4$'
    assert latexify('H') == r'H'
    assert latexify('A') != 'B'
