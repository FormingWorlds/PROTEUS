"""Tests for outgas config validators.

This file targets _outgas.py (Outgas, Calliope, Atmodeller parameters).
See testing standards in docs/test_infrastructure.md, docs/test_categorization.md,
and docs/test_building.md for required structure, speed, and physics validity.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
def test_calliope_defaults():
    """Test Calliope module initializes with valid defaults."""
    from proteus.config._outgas import Calliope

    c = Calliope()
    assert c.include_H2O is True
    assert c.include_CO2 is True
    assert c.include_N2 is True
    assert c.solubility is True
    # The cold-start pressure ceiling defaults to CALLIOPE's own 1e5 bar, so
    # the default PROTEUS config leaves the solver guess range unchanged.
    assert c.p_guess_max == pytest.approx(1.0e5)


@pytest.mark.unit
def test_calliope_p_guess_max_validator_and_override():
    """p_guess_max accepts a raised positive value (sub-Neptune use) and
    rejects a non-positive ceiling, which would invert the log-uniform draw
    range and make every cold-start guess invalid."""
    from proteus.config._outgas import Calliope

    # A raised ceiling is accepted and stored verbatim.
    c = Calliope(p_guess_max=1.0e8)
    assert c.p_guess_max == pytest.approx(1.0e8)
    # Discrimination: the override is honoured, not silently reset to the default.
    assert c.p_guess_max != pytest.approx(1.0e5)

    # Non-positive ceilings are rejected by the gt(0) validator.
    for bad in (0.0, -1.0e5):
        with pytest.raises(ValueError):
            Calliope(p_guess_max=bad)


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
    assert o.T_floor == pytest.approx(700.0, rel=1e-12)
    assert o.solver_rtol == pytest.approx(1e-4, rel=1e-12)
    assert o.solver_atol == pytest.approx(1e-6, rel=1e-12)


@pytest.mark.unit
def test_outgas_custom_solver_params():
    """Test Outgas accepts custom shared solver parameters."""
    from proteus.config._outgas import Outgas

    o = Outgas(
        module='calliope', fO2_shift_IW=4.0, T_floor=500.0, solver_rtol=1e-5, solver_atol=1e-7
    )
    assert o.T_floor == pytest.approx(500.0, rel=1e-12)
    assert o.solver_rtol == pytest.approx(1e-5, rel=1e-12)
    assert o.solver_atol == pytest.approx(1e-7, rel=1e-12)
