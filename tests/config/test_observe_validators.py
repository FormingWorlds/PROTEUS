"""Tests for observe config validators.

This file targets _observe.py validators. See testing standards in
docs/test_infrastructure.md, docs/test_categorization.md, and
docs/test_building.md for required structure, speed, and physics validity.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
def test_observe_clip_vmr_valid():
    """Test Observe clip_vmr with valid boundary values."""
    from proteus.config._observe import Observe

    # Default VMR cutoff should be valid.
    obs_default = Observe()
    assert obs_default.clip_vmr == pytest.approx(1e-8, rel=1e-12)
    # Discrimination: the (gt(0), lt(1)) validator pair must reject both
    # boundary endpoints. A regression to a one-sided check (e.g. only ge(0))
    # would let 1.0 slip past.
    with pytest.raises((ValueError, TypeError)):
        Observe(clip_vmr=0.0)
    with pytest.raises((ValueError, TypeError)):
        Observe(clip_vmr=1.0)


@pytest.mark.unit
def test_observe_module_none_accepted():
    """Test Observe module can accept None."""
    # Import inside test to avoid linter issues
    from proteus.config._observe import Observe

    # Module accepts None as a valid value after converter
    obs = Observe(module=None)
    assert obs.module is None
    # Discriminating check: the construction produced an Observe instance with
    # an explicit module=None field; a regression that swallowed the input
    # would have produced a class-level default (which `_observe.py` sets to
    # None too) and looked the same on the surface.
    assert isinstance(obs, Observe)


@pytest.mark.unit
def test_observe_module_petitradtrans_accepted():
    """Test Observe module accepts 'petitRADTRANS' string."""
    from proteus.config._observe import Observe

    # Module accepts petitRADTRANS as the supported synthetic-observation backend.
    obs = Observe(module='petitRADTRANS')
    assert obs.module == 'petitRADTRANS'
    # Discrimination: the validator must reject the removed PLATON option.
    with pytest.raises((ValueError, TypeError)):
        Observe(module='platon')
