"""Tests for outgas config validators.

This file targets _outgas.py (Outgas, Calliope, Atmodeller parameters).
See testing standards in docs/test_infrastructure.md, docs/test_categorization.md,
and docs/test_building.md for required structure, speed, and physics validity.
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_calliope_defaults():
    """Test Calliope module initializes with valid defaults."""
    from proteus.config._outgas import Calliope

    c = Calliope()
    assert c.include_H2O is True
    assert c.include_CO2 is True
    assert c.include_N2 is True
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
def test_outgas_shared_solver_defaults():
    """Test shared solver parameters on Outgas (T_floor, tolerances)."""
    from proteus.config._outgas import Outgas

    o = Outgas(module='calliope', fO2_shift_IW=4.0)
    assert o.T_floor == 700.0
    assert o.solver_rtol == 1e-4
    assert o.solver_atol == 1e-6


@pytest.mark.unit
def test_outgas_custom_solver_params():
    """Test Outgas accepts custom shared solver parameters."""
    from proteus.config._outgas import Outgas

    o = Outgas(module='calliope', fO2_shift_IW=4.0, T_floor=500.0, solver_rtol=1e-5, solver_atol=1e-7)
    assert o.T_floor == 500.0
    assert o.solver_rtol == 1e-5
    assert o.solver_atol == 1e-7
