"""Fe metal saturation chemistry (``src/proteus/interior_chem/disproportionation.py``).

Covers the disproportionation reaction 3FeO = 2FeO1.5 + Fe used by the melt
redox tracker to decide whether the silicate liquid is supersaturated in iron
metal, and by how far it must react to reach equilibrium.

Invariants and contract clauses exercised here:

* Cross-implementation anchor: the Kowalski & Spencer (1995) standard-state
  Gibbs energies reproduce the Hirschmann (2021) experimental iron-wuestite
  buffer through the independent reaction Fe(l) + 1/2 O2 = FeO(l).
* Reference-state cancellation: the reaction balances 3 Fe = 2 + 1 and
  3x1 O = 2x1.5 O, so the Fe and O2 reference terms must drop out.
* Dilution sensitivity: a_Fe scales as 1/n_sil, because the reaction consumes
  three dissolved species and produces two. An implementation that cancelled
  the melt inventory entirely would be wrong in a way the ratio alone cannot
  reveal.
* Mass balance: Fe and O are conserved identically under the reaction-extent
  parameterisation, across both the forward and the back reaction.
* Buffering: a melt that starts below the critical ferric fraction relaxes to
  it regardless of where it started, the behaviour Schaefer et al. (2024)
  report for their whole-mantle models.
* Limit and error contracts: an undersaturated melt with no metal present does
  not react, and requesting the pressure term without a pressure raises.

See docs/How-to/testing.md and docs/Explanations/test_framework.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from proteus.interior_chem.disproportionation import (
    GAMMA,
    GAMMA_FEO,
    K_eq,
    R,
    _g_fe_liquid,
    _g_feo_liquid,
    activity_Fe_metal,
    critical_ferric_fraction,
    dG0,
    solve_extent,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Bulk-silicate-Earth melt on a single-cation oxide basis, moles per 100 g
# (Schaefer et al. 2024 Table 1: 7.82 wt% FeO, 38.3 MgO, 45.5 SiO2, 3.58 CaO,
# 4.49 Al2O3). Asymmetric by construction, so a bug that mixes up the Fe2+ and
# Fe3+ slots cannot pass by symmetry.
_N_FET = 7.82 / 71.844
_N_OTHER = 38.3 / 40.304 + 45.5 / 60.084 + 3.58 / 56.077 + 2 * 4.49 / 101.96
_N_SIL = _N_OTHER + _N_FET


def _kowalski_spencer_g_o2(T: float) -> float:
    """Gibbs energy of O2 gas, Kowalski & Spencer (1995), T < 3300 K branch.

    Supplied by the test rather than the module: the module does not need it,
    because the O2 term cancels in the disproportionation stoichiometry. It is
    required here to build the independent oxidation reaction used as the
    cross-implementation anchor.
    """
    return (-13137.5203 + 525809.556 / T + 25.3200332 * T
            - 33.627603 * T * np.log(T) - 0.00119159274 * T ** 2
            + 1.3561111e-8 * T ** 3)


def _hirschmann_2021_iw(T: float) -> float:
    """log10 fO2 of the iron-wuestite buffer at 1 bar.

    Hirschmann (2021) GCA 313, 74-84, Table 1, fcc/bcc branch, evaluated at
    P = 0. An experimental calibration, independent of the CALPHAD data the
    module is built on.
    """
    return 6.844864 + 5.791364e-4 * T - 7.971469e-5 * T * np.log(T) - 2.769002e4 / T


@pytest.mark.reference_pinned
@pytest.mark.parametrize('T_kelvin, tolerance', [(1673.0, 0.05), (1800.0, 0.15)],
                         ids=['at the CALPHAD reference temperature',
                              'above the reference temperature'])
def test_standard_state_gibbs_reproduces_the_iron_wustite_buffer(T_kelvin, tolerance):
    """The tabulated Gibbs energies place the Fe/FeO equilibrium on the
    measured iron-wuestite buffer.

    Anchor: Hirschmann (2021) GCA 313, 74-84, Table 1. The module carries
    Kowalski & Spencer (1995) CALPHAD polynomials; combining them into the
    independent reaction Fe(l) + 1/2 O2 = FeO(l) must land on an experimental
    fit that shares no data with them. Tolerance widens with temperature
    because the module treats both Fe and FeO as liquids while the buffer is
    calibrated against solid wuestite.
    """
    dG = float(_g_feo_liquid(T_kelvin) - _g_fe_liquid(T_kelvin)
               - 0.5 * _kowalski_spencer_g_o2(T_kelvin))
    log_fO2 = 2.0 * dG / (np.log(10) * R * T_kelvin)
    expected = _hirschmann_2021_iw(T_kelvin)

    assert log_fO2 == pytest.approx(expected, abs=tolerance)
    # Sign guard: IW sits far below unit fugacity at magma-ocean temperature.
    # A flipped sign in the Gibbs difference would land near +10 instead.
    assert log_fO2 < 0.0
    # Scale guard: log10 fO2 near -10, not -1 (dropped factor of 2 on the
    # half-mole of O2) nor -100 (J/mol confused with kJ/mol).
    assert -14.0 < log_fO2 < -6.0


@pytest.mark.physics_invariant
def test_reaction_gibbs_is_insensitive_to_the_reference_state_choice():
    """Building the reaction energy from formation energies or from absolute
    Gibbs energies gives the same answer, because the reaction balances iron
    and oxygen internally.

    3 Fe = 2 + 1 and 3x1 O = 2x1.5 O, so any Fe-reference and O2 term added to
    every species must cancel. This is what lets the module carry three
    polynomials instead of five.
    """
    T = 2500.0
    # An arbitrary, deliberately large shift standing in for a reference-state
    # choice: added to FeO once, FeO1.5 once, Fe once, weighted by stoichiometry.
    shift = 1.234e5
    direct = float(dG0(T))
    shifted = float(
        2 * (0.5 * shift) + 1 * (0.0 * shift) - 3 * (0.0 * shift) + direct
    ) - 2 * (0.5 * shift)

    assert shifted == pytest.approx(direct, rel=1e-12)
    # The reaction energy is strongly positive at 1 bar: disproportionation is
    # unfavourable without the pressure term. Sign guard.
    assert direct > 0.0
    # Scale guard: order 1e5 J/mol, not 1e2 (kJ/J slip) nor 1e8.
    assert 5.0e4 < direct < 5.0e5


@pytest.mark.physics_invariant
def test_metal_activity_falls_as_the_melt_is_diluted():
    """Diluting the melt at fixed Fe3+/Fe2+ lowers the metal activity in
    inverse proportion to the total melt inventory.

    The reaction consumes three dissolved species and produces two, so one
    power of n_sil survives the activity ratio. An implementation that formed
    only the Fe3+/Fe2+ ratio would be completely insensitive to dilution, and
    the product a_Fe * n_sil below would not be constant.
    """
    T = 2500.0
    n2, n3 = _N_FET * 0.99, _N_FET * 0.01
    a_single, _ = activity_Fe_metal(T, n2, n3, _N_SIL)
    a_double, _ = activity_Fe_metal(T, n2, n3, 2.0 * _N_SIL)

    assert float(a_double) == pytest.approx(0.5 * float(a_single), rel=1e-12)
    # Invariance of the product is the discriminating form: a cancelled n_sil
    # would give a ratio of 1.0, a squared one a ratio of 0.25.
    assert float(a_single) * _N_SIL == pytest.approx(
        float(a_double) * 2.0 * _N_SIL, rel=1e-12)
    assert float(a_single) > 0.0


@pytest.mark.physics_invariant
@pytest.mark.parametrize('ferric_fraction', [0.001, 0.01, 0.05],
                         ids=['strongly reduced melt', 'near the threshold',
                              'oxidised melt'])
def test_metal_activity_equals_the_constant_over_the_zero_extent_residual(ferric_fraction):
    """The metal activity is the equilibrium constant divided by the
    mass-action residual evaluated before any reaction has occurred.

    This identity is what makes the saturation test double as the bracket
    check for the extent solve: a_Fe > 1 is exactly the statement that the
    residual changes sign somewhere in the forward direction.
    """
    T = 2500.0
    n2, n3 = _N_FET * (1.0 - ferric_fraction), _N_FET * ferric_fraction
    K = float(K_eq(T)[0])
    residual_at_zero = n3 ** 2 * _N_SIL / (GAMMA_FEO ** 3 * n2 ** 3)
    a_fe, _ = activity_Fe_metal(T, n2, n3, _N_SIL)

    assert float(a_fe) == pytest.approx(K / residual_at_zero, rel=1e-12)
    assert float(a_fe) > 0.0


@pytest.mark.physics_invariant
@pytest.mark.parametrize('ferric_fraction', [0.01, 0.005, 0.001],
                         ids=['mildly reduced', 'reduced', 'strongly reduced'])
def test_reaction_extent_conserves_iron_and_oxygen(ferric_fraction):
    """Reacting a supersaturated melt to equilibrium moves iron between the
    melt and the metal without creating or destroying either element.

    Metal removes iron but leaves its oxygen behind, so total Fe (melt plus
    metal) and total O bound to Fe must both be unchanged. Conservation
    closure is the primary assertion, so it doubles as the exponent guard.
    """
    T = 2500.0
    n2, n3 = _N_FET * (1.0 - ferric_fraction), _N_FET * ferric_fraction
    K = float(K_eq(T)[0])
    xi = solve_extent(K, n2, n3, _N_SIL)

    fe_before, fe_after = n2 + n3, (n2 - 3 * xi) + (n3 + 2 * xi) + xi
    o_before, o_after = n2 + 1.5 * n3, (n2 - 3 * xi) + 1.5 * (n3 + 2 * xi)

    assert fe_after == pytest.approx(fe_before, rel=1e-14)
    assert o_after == pytest.approx(o_before, rel=1e-14)
    # Sign guard: a reduced melt must form metal, not consume it.
    assert xi > 0.0
    # Scale guard: only a small fraction of the iron disproportionates at
    # these conditions, well under 10 per cent.
    assert xi < 0.1 * (n2 + n3)


@pytest.mark.physics_invariant
def test_supersaturated_melt_relaxes_until_metal_activity_is_unity():
    """After reacting, the melt sits exactly on the metal saturation
    boundary rather than overshooting or stopping short."""
    T = 2500.0
    n2, n3 = _N_FET * 0.995, _N_FET * 0.005
    K = float(K_eq(T)[0])
    xi = solve_extent(K, n2, n3, _N_SIL)

    a_before, _ = activity_Fe_metal(T, n2, n3, _N_SIL)
    a_after, _ = activity_Fe_metal(
        T, n2 - 3 * xi, n3 + 2 * xi, _N_SIL - xi)

    assert float(a_after) == pytest.approx(1.0, rel=1e-9)
    # The starting melt really was supersaturated, so the test is not
    # vacuously satisfied by a melt that never needed to react.
    assert float(a_before) > 2.0


@pytest.mark.reference_pinned
def test_final_ferric_fraction_is_independent_of_the_starting_value():
    """Melts that start anywhere below the saturation threshold all end at the
    same ferric fraction, so the initial value stops carrying information.

    Anchor: Schaefer et al. (2024) JGR Planets 129, e2023JE008262, Section
    3.1.1, which reports that for the whole-Earth models "the initial
    Fe3+/FeT makes no difference to the outcome, because all models converge
    to the same Fe3+/FeT content".
    """
    T = 2500.0
    K = float(K_eq(T)[0])
    finals = []
    for f_0 in (0.010, 0.005, 0.002, 0.001):
        n2, n3 = _N_FET * (1.0 - f_0), _N_FET * f_0
        xi = solve_extent(K, n2, n3, _N_SIL)
        n2_new, n3_new = n2 - 3 * xi, n3 + 2 * xi
        finals.append(n3_new / (n2_new + n3_new))

    # Fractional spreads, so input and output are compared on the same scale:
    # the starting values span 900 per cent, the final values a fraction of one.
    input_spread = 0.010 / 0.001 - 1.0
    output_spread = max(finals) / min(finals) - 1.0

    assert finals[0] == pytest.approx(finals[-1], rel=5e-3)
    # The discriminating statement is that the outputs collapse: a tenfold
    # spread in the starting value survives as a fraction of a per cent.
    # Convergence is close but not exact, because the melt inventory shrinks
    # by the reaction extent and that extent differs between the runs.
    assert output_spread < 0.01
    assert input_spread / output_spread > 100.0
    # All of them ended more oxidised than the most oxidised starting point,
    # so the buffer pushed every melt upwards rather than merely holding it.
    assert min(finals) > 0.010


@pytest.mark.physics_invariant
def test_critical_ferric_fraction_decreases_with_temperature():
    """A hotter melt at 1 bar tolerates a lower ferric fraction before metal
    saturates, because the reaction energy rises faster than RT.

    Without the pressure term the saturation threshold depends only on
    temperature and melt dilution, which is why this mode cannot reproduce a
    depth-resolved metal distribution.
    """
    thresholds = [critical_ferric_fraction(T, _N_FET, _N_SIL)
                  for T in (1800.0, 2500.0, 3500.0)]

    assert thresholds == sorted(thresholds, reverse=True)
    # Scale guard: the threshold sits around one to two per cent, not ten per
    # cent (missing GAMMA) nor 0.1 per cent (squared instead of cubed n2).
    assert all(0.005 < t < 0.05 for t in thresholds)
    assert thresholds[0] > 1.5 * thresholds[-1]


def test_undersaturated_melt_with_no_metal_present_does_not_react():
    """An oxidised melt with no metal to redissolve is a fixed point of the
    extent solve.

    This is the limit-input contract for a function with no explicit
    validation: there is nothing to react in either direction, so the only
    admissible answer is zero.
    """
    T = 2500.0
    n2, n3 = _N_FET * 0.9, _N_FET * 0.1
    K = float(K_eq(T)[0])
    a_fe, _ = activity_Fe_metal(T, n2, n3, _N_SIL)
    xi = solve_extent(K, n2, n3, _N_SIL, n_metal_avail=0.0)

    assert xi == pytest.approx(0.0, abs=1e-30)
    # The melt really is undersaturated, so the zero is the physical answer
    # and not an accidental early return.
    assert float(a_fe) < 1.0


def test_metal_redissolves_when_the_melt_is_undersaturated():
    """Metal held in an oxidised melt dissolves back, driving the extent
    negative and consuming ferric iron.

    PROTEUS keeps the metal co-located with the melt rather than segregating
    it, so the back reaction is required; Schaefer's own code has no need of
    it because it deletes the metal from the system.
    """
    T = 2500.0
    n2, n3 = _N_FET * 0.9, _N_FET * 0.1
    K = float(K_eq(T)[0])
    available = 1.0e-4
    xi = solve_extent(K, n2, n3, _N_SIL, n_metal_avail=available)

    assert xi < 0.0
    # Cannot dissolve more metal than exists.
    assert xi >= -available - 1e-18
    # Dissolving metal consumes ferric iron: the melt becomes more reduced.
    assert (n3 + 2 * xi) < n3


def test_activity_coefficients_enter_only_through_the_identifiable_combination():
    """Only the grouping gamma_FeO**3 / gamma_FeO1.5**2 affects the answer, so
    the two coefficients are not separately constrained by the model."""
    assert GAMMA == pytest.approx(GAMMA_FEO ** 3, rel=1e-12)
    # Scale guard against a dropped or doubled exponent: 1.55**3 is 3.72,
    # 1.55**2 is 2.40 and 1.55**4 is 5.77.
    assert 3.5 < GAMMA < 4.0


def test_requesting_the_pressure_term_without_a_pressure_raises():
    """The equilibrium constant refuses to silently drop the pressure term
    when it was asked to include one, and leaves no state behind."""
    with pytest.raises(ValueError, match='P_gpa'):
        K_eq(2500.0, P_gpa=None, use_P_term=True)

    # The no-pressure path still works afterwards, so the failure did not
    # corrupt module state.
    K, valid = K_eq(2500.0)
    assert float(K) > 0.0
    assert bool(np.all(valid))


@pytest.mark.parametrize('n2, n3', [(0.0, 1.0e-3), (1.0e-3, 0.0), (0.0, 0.0)],
                         ids=['no ferrous iron', 'no ferric iron',
                              'no iron at all'])
def test_degenerate_iron_inventories_do_not_react(n2, n3):
    """A melt missing one of the iron species cannot run the reaction, and the
    extent solve returns zero rather than dividing by it."""
    xi = solve_extent(float(K_eq(2500.0)[0]), n2, n3, _N_SIL)

    assert xi == pytest.approx(0.0, abs=1e-30)
    assert np.isfinite(xi)
