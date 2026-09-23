"""Pressure term for Fe disproportionation (``src/proteus/interior_chem/eos_deng.py``).

Covers the tabulated integral of the reaction volume change for
3FeO = 2FeO1.5 + Fe, built from the Deng et al. (2020) silicate melt equation
of state together with liquid FeO and liquid Fe.

Invariants and contract clauses exercised here:

* Analytic identity of the fitted form: the fourth-order Birch-Murnaghan
  pressure returns the reference pressure exactly at the reference volume,
  which is what fixes the foot of every integral.
* The thermal-pressure fit is a parabola in V/V0 whose minimum sits just below
  the reference volume. Beyond that minimum it rises with expanding volume,
  which no thermal pressure does, so the module caps the liquid at V0 and
  splices below the corresponding pressure.
* Sign structure: the reaction volume change is positive at low pressure and
  negative at depth, which is why disproportionation is unfavourable near the
  surface and favoured in a deep magma ocean.
* The validity contract: points above the temperature ceiling or beyond the
  pressure range the source equation of state was exercised over are reported
  as invalid rather than silently extrapolated.

See docs/How-to/testing.md and docs/Explanations/test_framework.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from proteus.interior_chem.eos_deng import (
    _B,
    _C,
    _P0,
    _T0,
    _V0,
    P_EXERCISED,
    T_CEILING,
    _bh,
    _bm4_pressure,
    _int_V_FeO,
    _solve_V,
    int_dV_dP,
    p_splice,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.physics_invariant
@pytest.mark.parametrize('endmember', [0, 1],
                         ids=['ferrous endmember', 'ferric endmember'])
def test_birch_murnaghan_returns_the_reference_pressure_at_the_reference_volume(endmember):
    """At the reference volume and reference temperature the fitted equation
    of state reproduces the reference pressure exactly.

    This is an analytic identity of the fourth-order Birch-Murnaghan form with
    the published coefficients, and it is what anchors the foot of the
    pressure integral. A transposed coefficient or a wrong strain definition
    would move it by orders of magnitude.
    """
    P = _bm4_pressure(_V0[endmember], _T0, endmember)

    assert P == pytest.approx(_P0, rel=1e-9)
    # Sign guard: compression is positive. A flipped strain sign lands at -1e-4.
    assert P > 0.0
    # Scale guard: 1e-4 GPa is 1 bar. A bar/GPa confusion would give 1e5.
    assert 1e-6 < P < 1e-2


@pytest.mark.physics_invariant
@pytest.mark.parametrize('endmember', [0, 1],
                         ids=['ferrous endmember', 'ferric endmember'])
def test_thermal_pressure_fit_turns_upward_just_below_the_reference_volume(endmember):
    """The thermal-pressure coefficient decreases with expanding volume only
    up to V/V0 near 0.97, then rises, so the fit is usable on compressed
    states alone.

    A thermal pressure must fall as a liquid expands. The published
    coefficients form a parabola in V/V0 with a positive leading term, so
    beyond its minimum the fit is unphysical. Locating that minimum is what
    sets the splice the module applies, and it is why the equation has no
    solution at low pressure once the liquid is hot enough to expand past V0.
    """
    minimum_at = _B[endmember] / (2.0 * _C[endmember])
    below = float(_bh(np.array(minimum_at - 0.1), endmember))
    at = float(_bh(np.array(minimum_at), endmember))
    above = float(_bh(np.array(minimum_at + 0.4), endmember))

    assert minimum_at == pytest.approx(0.97, abs=0.02)
    assert at < below
    assert at < above
    # The coefficient stays positive throughout, so the sign of the thermal
    # term is set by (T - T0) alone.
    assert at > 0.0


@pytest.mark.physics_invariant
def test_no_solution_exists_below_the_splice_pressure_at_high_temperature():
    """Above the reference temperature the liquid would have to expand past
    the reference volume at low pressure, where the fit has no root, and the
    splice pressure marks that boundary.

    The module solves on a bracket capped at V0, so the absence of a root
    below the splice is the expected behaviour rather than a solver failure.
    """
    T = 4000.0
    boundary = p_splice(T)
    below = _solve_V(0.5 * boundary, T, 1)
    above = _solve_V(2.0 * boundary, T, 1)

    assert not np.isfinite(below)
    assert np.isfinite(above)
    # The solved volume sits inside the compressed branch the fit covers.
    assert above < _V0[1]
    # Scale guard: a few GPa at 4000 K, not a fraction of a bar nor 100 GPa.
    assert 0.1 < boundary < 10.0


def test_splice_pressure_vanishes_at_the_reference_temperature():
    """At the reference temperature the thermal term is zero, so the liquid
    sits at its reference volume at the reference pressure and no splice is
    needed.

    This is the limit-input contract: the splice exists only to cover thermal
    expansion past V0, which does not occur at T0.
    """
    assert p_splice(_T0) == pytest.approx(_P0, rel=1e-9)
    # Below the reference temperature the liquid is already compressed, so the
    # boundary moves to negative pressure and never binds.
    assert p_splice(2000.0) < 0.0
    assert p_splice(4000.0) > p_splice(_T0)


@pytest.mark.physics_invariant
def test_reaction_volume_integral_vanishes_at_the_reference_pressure():
    """Integrating from the reference pressure to itself gives zero, so the
    standard state carries no pressure correction."""
    value, valid = int_dV_dP(3000.0, _P0)

    assert float(value) == pytest.approx(0.0, abs=1.0)
    assert bool(valid)


@pytest.mark.reference_pinned
def test_reaction_volume_change_is_positive_near_the_surface_and_negative_at_depth():
    """Disproportionation is volumetrically unfavourable in a shallow melt and
    favourable in a deep one, so the pressure term changes sign.

    Anchor: Schaefer et al. (2024) JGR Planets 129, e2023JE008262, whose
    metal-saturation results have no metal forming in the 500 km magma ocean
    models but a large metal event at the base of the whole-mantle models.
    That contrast requires the integral to change sign with depth, and it is
    the reason a version without this term inverts the depth trend.
    """
    shallow, shallow_valid = int_dV_dP(2500.0, 2.0)
    deep, deep_valid = int_dV_dP(2500.0, 20.0)

    assert float(shallow) > 0.0
    assert float(deep) < 0.0
    assert bool(shallow_valid) and bool(deep_valid)
    # Scale guard: tens of kJ/mol, not tens of J/mol nor tens of MJ/mol. The
    # deep value must also dominate, since it is what drives saturation.
    assert 1.0e3 < abs(float(deep)) < 1.0e6
    assert abs(float(deep)) > abs(float(shallow))


@pytest.mark.parametrize('T_kelvin, P_gpa, reason', [
    (T_CEILING + 100.0, 50.0, 'above the temperature ceiling'),
    (3000.0, P_EXERCISED + 50.0, 'beyond the calibrated pressure range'),
    (np.nan, 50.0, 'undefined temperature'),
])
def test_points_outside_the_equation_of_state_domain_are_reported_invalid(
        T_kelvin, P_gpa, reason):
    """Conditions the source equation of state does not cover are flagged
    rather than extrapolated, and return a zero the caller can distinguish
    from a computed zero through the validity flag.

    A silent zero would be indistinguishable from the pressure term being
    switched off, which are very different states for the caller.
    """
    value, valid = int_dV_dP(T_kelvin, P_gpa)

    assert not bool(valid), reason
    assert float(value) == pytest.approx(0.0, abs=1e-30)


def test_conditions_inside_the_domain_are_reported_valid():
    """Magma-ocean conditions well inside both limits are accepted, so the
    validity mask is not rejecting everything."""
    value, valid = int_dV_dP(3000.0, 50.0)

    assert bool(valid)
    # A real, non-zero correction: the guard against a mask that passes but a
    # table that silently returns zeros.
    assert abs(float(value)) > 1.0e3


@pytest.mark.physics_invariant
def test_liquid_iron_oxide_volume_integral_is_linear_at_low_pressure():
    """The Murnaghan integral of the FeO liquid reduces to volume times
    pressure when compression is negligible.

    Analytic limit: for P much smaller than the bulk modulus the integrand is
    effectively constant, so doubling the pressure doubles the integral.
    """
    T = 2500.0
    V_cm3 = 13650.0 + 2.92 * (T - 1673.0)          # cm3/mol times 1e3
    small, smaller = _int_V_FeO(T, 0.02), _int_V_FeO(T, 0.01)

    assert small == pytest.approx(2.0 * smaller, rel=1e-3)
    # Pin the absolute scale against the uncompressed volume: 1 cm3/mol GPa
    # is 1e3 J/mol, so V(T) times 0.01 GPa lands near V_cm3 * 1e-2 J/mol.
    assert smaller == pytest.approx(V_cm3 * 1e-2, rel=1e-2)
    assert smaller > 0.0
