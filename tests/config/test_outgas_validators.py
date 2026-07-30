"""Tests for outgas config validators.

This file targets _outgas.py (Outgas, Calliope, Atmodeller parameters).
See testing standards in docs/How-to/testing.md and
docs/Explanations/test_framework.md for required structure, speed, and
physics validity.
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
    """p_guess_max accepts a raised ceiling within CALLIOPE's solver box and
    rejects values outside (0, 1e7]: a non-positive ceiling would invert the
    log-uniform draw range, and a ceiling above the 1e7 bar box is unreachable
    (the solver rejects roots beyond the box), so it must fail loudly."""
    from proteus.config._outgas import Calliope

    # A raised ceiling within the box is accepted and stored verbatim.
    c = Calliope(p_guess_max=5.0e6)
    assert c.p_guess_max == pytest.approx(5.0e6)
    # Discrimination: the override is honoured, not silently reset to the default.
    assert c.p_guess_max != pytest.approx(1.0e5)

    # Out-of-range ceilings are rejected: non-positive, above the 1e7 box, and
    # the non-finite inf that a bare gt(0) would have let through.
    for bad in (0.0, -1.0e5, 1.0e8, float('inf')):
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


@pytest.mark.unit
def test_lavatmos_defaults():
    """Lavatmos initializes with the documented rock-vapour defaults."""
    from proteus.config._outgas import Lavatmos

    lav = Lavatmos()
    assert lav.T_min == pytest.approx(1500.0, rel=1e-12)
    assert lav.melt_comp_name == 'BSE_palm'
    assert lav.P_melt == pytest.approx(0.01, rel=1e-12)
    assert lav.xatol == pytest.approx(1e-5, rel=1e-12)


@pytest.mark.unit
def test_lavatmos_custom_melt_pressure_and_tolerance():
    """
    P_melt and xatol are honoured verbatim and must stay strictly positive.
    """
    from proteus.config._outgas import Lavatmos

    # Custom melt pressure and solver tol.
    lav = Lavatmos(P_melt=0.05, xatol=2e-6)
    assert lav.P_melt == pytest.approx(0.05, rel=1e-12)
    assert lav.xatol == pytest.approx(2e-6, rel=1e-12)

    # Discrimination: overrides are stored, not reset to the defaults.
    assert lav.P_melt != pytest.approx(0.01, rel=1e-12)
    assert lav.xatol != pytest.approx(1e-5, rel=1e-12)

    # Check melt pressure and tol. are positive.
    for bad in (0.0, -0.01):
        with pytest.raises(ValueError):
            Lavatmos(P_melt=bad)
        with pytest.raises(ValueError):
            Lavatmos(xatol=bad)


def test_h2_binodal_enabled_is_rejected():
    """`h2_binodal = true` is rejected at config load.

    The H2-silicate binodal partitioning is not production ready, so
    enabling it must fail loudly instead of running an unsettled
    parameterisation. The default (off) constructs, pinning that the
    rejection is the flag value and not the field itself.
    """
    from proteus.config._outgas import Outgas

    with pytest.raises(ValueError, match='h2_binodal'):
        Outgas(h2_binodal=True)
    o = Outgas()
    assert o.h2_binodal is False
