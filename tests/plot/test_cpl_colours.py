from __future__ import annotations

from proteus.utils.plot import get_colour, _preset_colours, latexify

def test_get_colour():
    # Test default colours
    assert get_colour("C")   == _preset_colours["C"]
    assert get_colour("OLR") == _preset_colours["OLR"]

    # Test fallback
    assert get_colour("foo") == _preset_colours["_fallback"]

    # Test generating a colour
    assert get_colour("SiH4C2") == '#d90aff'

def test_latexify():
    assert latexify("H2O4") == r"H$_2$O$_4$"
    assert latexify("H") == r"H"
    assert latexify("A") != "B"
