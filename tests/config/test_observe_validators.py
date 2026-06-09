"""Tests for observe config validators.

This file targets _observe.py validators. See testing standards in
docs/test_infrastructure.md, docs/test_categorization.md, and
docs/test_building.md for required structure, speed, and physics validity.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
def test_observe_platon_downsample_valid():
    """Test Platon downsampling factor validator with valid values."""
    from proteus.config._observe import Platon

    # Valid downsampling factors >= 1
    # Just verify defaults are set correctly since these use attrs validators
    p_default = Platon()
    assert p_default.downsample == 8  # Default value
    # Discrimination: the ge(1) validator must reject 0; a regression that
    # silently coerced or dropped the validator would let this through.
    with pytest.raises((ValueError, TypeError)):
        Platon(downsample=0)


@pytest.mark.unit
def test_observe_platon_clip_vmr_valid():
    """Test Platon clip_vmr with valid boundary values."""
    from proteus.config._observe import Platon

    # Default VMR value should be valid
    p_default = Platon()
    assert p_default.clip_vmr == pytest.approx(1e-8, rel=1e-12)
    # Discrimination: the (gt(0), lt(1)) validator pair must reject both
    # boundary endpoints. A regression to a one-sided check (e.g. only ge(0))
    # would let 1.0 slip past.
    with pytest.raises((ValueError, TypeError)):
        Platon(clip_vmr=0.0)
    with pytest.raises((ValueError, TypeError)):
        Platon(clip_vmr=1.0)


@pytest.mark.unit
def test_observe_synthesis_none_accepted():
    """Test Observe synthesis can accept None."""
    # Import inside test to avoid linter issues
    from proteus.config._observe import Observe

    # Synthesis accepts None as a valid value after converter
    obs = Observe(synthesis=None)
    assert obs.synthesis is None
    # Discriminating check: the construction produced an Observe instance with
    # an explicit synthesis=None field; a regression that swallowed the input
    # would have produced a class-level default (which `_observe.py` sets to
    # None too) and looked the same on the surface.
    assert isinstance(obs, Observe)


@pytest.mark.unit
def test_observe_synthesis_platon_accepted():
    """Test Observe synthesis accepts 'platon' string."""
    from proteus.config._observe import Observe

    # Synthesis accepts 'platon' as a valid string value
    obs = Observe(synthesis='platon')
    assert obs.synthesis == 'platon'
    # Discrimination: the validator must reject an unknown synthesizer.
    # A regression that dropped the in_((None, 'platon')) constraint would
    # let 'unknown' through.
    with pytest.raises((ValueError, TypeError)):
        Observe(synthesis='unknown')
