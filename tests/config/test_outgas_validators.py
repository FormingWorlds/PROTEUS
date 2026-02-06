"""Tests for outgas config validators.

This file targets _outgas.py (Calliope module parameters). See testing standards in
docs/test_infrastructure.md, docs/test_categorization.md, and
docs/test_building.md for required structure, speed, and physics validity.
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_calliope_defaults():
    """Test Calliope module initializes with valid defaults."""
    from proteus.config._outgas import Calliope

    c = Calliope()
    assert c.T_floor == 700.0  # Default temperature floor
    assert c.include_H2O is True
    assert c.include_CO2 is True
    assert c.include_N2 is True
    assert c.rtol == 1e-4  # Default relative tolerance
    assert c.xtol == 1e-6  # Default absolute tolerance
    assert c.solubility is True


@pytest.mark.unit
def test_calliope_is_included_h2o():
    """Test Calliope is_included method returns volatile inclusion status."""
    from proteus.config._outgas import Calliope

    c = Calliope()
    assert c.is_included('H2O') is True
    assert c.is_included('CO2') is True
    assert c.is_included('N2') is True


@pytest.mark.unit
def test_calliope_custom_temperature_floor():
    """Test Calliope accepts custom temperature floor."""
    from proteus.config._outgas import Calliope

    c = Calliope(T_floor=500.0)
    assert c.T_floor == 500.0

    c2 = Calliope(T_floor=1200.0)
    assert c2.T_floor == 1200.0


@pytest.mark.unit
def test_calliope_tolerance_parameters():
    """Test Calliope tolerance parameters."""
    from proteus.config._outgas import Calliope

    c = Calliope(rtol=1e-5, xtol=1e-7)
    assert c.rtol == 1e-5
    assert c.xtol == 1e-7
