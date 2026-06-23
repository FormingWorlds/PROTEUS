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
def test_observe_remove_one_gas_default_and_override():
    """Top-level Observe.remove_one_gas defaults to True and is user-overridable."""
    from proteus.config._observe import Observe

    assert Observe().remove_one_gas is True
    assert Observe(remove_one_gas=False).remove_one_gas is False


@pytest.mark.unit
def test_petitradtrans_silent_default_and_override():
    """PetitRADTRANS.silent defaults to False and can be enabled."""
    from proteus.config._observe import PetitRADTRANS

    assert PetitRADTRANS().silent is False
    assert PetitRADTRANS(silent=True).silent is True


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


@pytest.mark.unit
def test_petitradtrans_defaults():
    """Test the nested PetitRADTRANS config defaults."""
    from proteus.config._observe import PetitRADTRANS

    prt = PetitRADTRANS()
    assert prt.line_opacity_mode == 'c-k'
    assert prt.include_rayleigh is True
    assert prt.include_cia is True


@pytest.mark.unit
def test_petitradtrans_line_opacity_mode_validator():
    """Test PetitRADTRANS line_opacity_mode accepts only the supported modes."""
    from proteus.config._observe import PetitRADTRANS

    assert PetitRADTRANS(line_opacity_mode='lbl').line_opacity_mode == 'lbl'
    with pytest.raises((ValueError, TypeError)):
        PetitRADTRANS(line_opacity_mode='invalid')


@pytest.mark.unit
def test_observe_source_validator():
    """Observe.source accepts all and individual source names only."""
    from proteus.config._observe import Observe

    assert Observe().source == 'all'
    assert Observe(source='outgas').source == 'outgas'
    assert Observe(source='profile').source == 'profile'
    assert Observe(source='offchem').source == 'offchem'

    with pytest.raises((ValueError, TypeError)):
        Observe(source='invalid_source')


@pytest.mark.unit
def test_observe_spectrum_type_validator():
    """Observe.spectrum_type accepts both, transit, and eclipse only."""
    from proteus.config._observe import Observe

    assert Observe().spectrum_type == 'both'
    assert Observe(spectrum_type='transit').spectrum_type == 'transit'
    assert Observe(spectrum_type='eclipse').spectrum_type == 'eclipse'

    with pytest.raises((ValueError, TypeError)):
        Observe(spectrum_type='invalid')
