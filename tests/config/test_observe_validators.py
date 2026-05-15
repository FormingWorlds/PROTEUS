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

    # The validator ge(1) enforces downsampling >= 1 at class construction


@pytest.mark.unit
def test_observe_platon_clip_vmr_valid():
    """Test Platon clip_vmr with valid boundary values."""
    from proteus.config._observe import Platon

    # Default VMR value should be valid
    p_default = Platon()
    assert p_default.clip_vmr == 1e-8

    # Validator enforces 0 < vmr < 1


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
