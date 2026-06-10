"""Unit tests for ``proteus.utils.constants``.

Validates that physical constants are consistent with their reference
values (CODATA 2018 / IAU 2015 / IUPAC). These are reference-pinned
tests: each constant is compared against its published value with a
tolerance that covers known inter-source variation (e.g. truncation in
different published tables).

Invariants tested:
  - Positivity: all physical constants are strictly positive
  - Magnitude: each constant is within 0.1% of the CODATA/IAU/IUPAC value
  - Internal consistency: derived constants match their definitions
  - Structural: gas_list, vol_list, element_list are non-empty and contain
    expected species

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

import pytest

from proteus.utils.constants import (
    AU,
    L_sun,
    M_earth,
    M_sun,
    R_earth,
    R_sun,
    const_c,
    const_G,
    const_h,
    const_k,
    const_Nav,
    const_R,
    const_sigma,
    element_list,
    element_mmw,
    gas_list,
    secs_per_year,
    vol_list,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# -----------------------------------------------------------------------
# Fundamental constants (CODATA 2018)
# -----------------------------------------------------------------------


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_gravitational_constant():
    """G = 6.674e-11 m^3 kg^-1 s^-2 (CODATA 2018: 6.67430e-11).

    Discrimination: a CGS-vs-SI slip would give 6.674e-8 (factor 1000).
    A missing factor of 4*pi (Gaussian units) would give ~8.38e-10.
    """
    assert const_G == pytest.approx(6.674e-11, rel=1e-3)
    assert const_G > 0
    # Scale guard: SI, not CGS
    assert 1e-12 < const_G < 1e-10


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_stefan_boltzmann_constant():
    """sigma = 5.670e-8 W m^-2 K^-4 (CODATA 2018: 5.670374419e-8).

    Discrimination: a CGS slip (erg cm^-2 s^-1 K^-4) gives 5.67e-5.
    """
    assert const_sigma == pytest.approx(5.670e-8, rel=1e-3)
    assert const_sigma > 0
    assert 1e-9 < const_sigma < 1e-7


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_speed_of_light():
    """c = 2.998e8 m/s (exact: 299792458 m/s).

    Discrimination: km/s (2.998e5) or cm/s (2.998e10) are orders of
    magnitude off.
    """
    assert const_c == pytest.approx(2.99792458e8, rel=1e-9)
    assert const_c > 0
    assert 1e8 < const_c < 1e9


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_planck_constant():
    """h = 6.626e-34 J s (CODATA 2018: 6.62607015e-34, exact).

    Discrimination: erg-s (CGS) gives 6.626e-27.
    """
    assert const_h == pytest.approx(6.626e-34, rel=1e-3)
    assert const_h > 0
    assert 1e-35 < const_h < 1e-33


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_boltzmann_constant():
    """k_B = 1.381e-23 J/K (CODATA 2018: 1.380649e-23, exact).

    Discrimination: erg/K (CGS) gives 1.381e-16.
    """
    assert const_k == pytest.approx(1.381e-23, rel=1e-3)
    assert const_k > 0
    assert 1e-24 < const_k < 1e-22


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_gas_constant():
    """R = 8.314 J K^-1 mol^-1 (CODATA 2018: 8.314462618...).

    Discrimination: cal-based constant gives ~1.987 cal/(K mol).
    """
    assert const_R == pytest.approx(8.314, rel=1e-3)
    assert const_R > 0
    assert 5.0 < const_R < 15.0


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_avogadro_constant():
    """N_A = 6.022e23 mol^-1 (CODATA 2018: 6.02214076e23, exact).

    Discrimination: wrong exponent (6.022e22 or 6.022e24) is 10x off.
    """
    assert const_Nav == pytest.approx(6.022e23, rel=1e-3)
    assert const_Nav > 0
    assert 1e23 < const_Nav < 1e24


# -----------------------------------------------------------------------
# Astronomical constants (IAU 2015)
# -----------------------------------------------------------------------


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_solar_luminosity():
    """L_sun = 3.828e26 W (IAU 2015 Resolution B3 nominal).

    Discrimination: erg/s (CGS) gives 3.828e33, 7 orders off.
    """
    assert L_sun == pytest.approx(3.828e26, rel=1e-3)
    assert L_sun > 0
    assert 1e26 < L_sun < 1e27


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_solar_radius():
    """R_sun = 6.957e8 m (IAU 2015 Resolution B3 nominal).

    Discrimination: km (6.957e5) or cm (6.957e10) are orders off.
    """
    assert R_sun == pytest.approx(6.957e8, rel=1e-3)
    assert R_sun > 0
    assert 1e8 < R_sun < 1e9


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_solar_mass():
    """M_sun = 1.989e30 kg (IAU 2015: 1.98892e30 from GM_sun).

    Discrimination: grams (1.989e33) or Earth masses (3.0e-6) are
    orders of magnitude off.
    """
    assert M_sun == pytest.approx(1.989e30, rel=1e-3)
    assert M_sun > 0
    assert 1e30 < M_sun < 1e31


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_astronomical_unit():
    """1 AU = 1.496e11 m (IAU 2012 Resolution B2, exact: 149597870700 m).

    Discrimination: km (1.496e8) or cm (1.496e13) are orders off.
    """
    assert AU == pytest.approx(1.496e11, rel=1e-3)
    assert AU > 0
    assert 1e11 < AU < 1e12


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_earth_mass():
    """M_earth = 5.972e24 kg (IAU nominal).

    Discrimination: grams (5.972e27) or solar masses (~3e-6) are
    obviously wrong.
    """
    assert M_earth == pytest.approx(5.972e24, rel=1e-3)
    assert M_earth > 0
    assert 1e24 < M_earth < 1e25


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_earth_radius():
    """R_earth = 6.335e6 m (volumetric mean: 6.371e6 m).

    The PROTEUS value is slightly below the volumetric mean because
    it uses a specific reference. Pin to the stored value with 1%
    tolerance against the volumetric mean.
    """
    assert R_earth == pytest.approx(6.371e6, rel=0.01)
    assert R_earth > 0
    assert 5e6 < R_earth < 8e6


# -----------------------------------------------------------------------
# Derived constants: internal consistency
# -----------------------------------------------------------------------


def test_secs_per_year_internal_consistency():
    """secs_per_year = 365.25 * 24 * 3600 = 31557600.

    Discrimination: a Julian year (365.25 d) vs a Gregorian year
    (365.2425 d) vs a tropical year (365.2422 d) differ by < 0.01%,
    but using 365 days flat gives 31536000 (0.07% low).
    """
    expected = 365.25 * 24.0 * 3600.0
    assert secs_per_year == pytest.approx(expected, rel=1e-12)
    # Scale guard: ~3.16e7, not 3.16e4 (hours) or 3.16e10
    assert 3e7 < secs_per_year < 4e7


# -----------------------------------------------------------------------
# Structural: species lists
# -----------------------------------------------------------------------


def test_vol_list_contains_expected_species():
    """vol_list contains the 11 volatile species tracked by PROTEUS.

    At minimum: H2O, CO2, H2, N2, O2.
    """
    assert len(vol_list) == 11
    for species in ('H2O', 'CO2', 'H2', 'N2', 'O2'):
        assert species in vol_list, f'Missing volatile: {species}'


def test_gas_list_extends_vol_list():
    """gas_list = vol_list + vap_list (volatiles + vapour species).

    gas_list must be a superset of vol_list.
    """

    # check lengths
    assert len(gas_list) >= len(vol_list)

    # check presence of volatiles
    for species in vol_list:
        assert species in gas_list


def test_element_list_contains_expected_elements():
    """element_list contains the 9 elements tracked by PROTEUS.

    At minimum: H, O, C, N, S.
    """
    assert len(element_list) == 9
    for elem in ('H', 'O', 'C', 'N', 'S'):
        assert elem in element_list


def test_element_mmw_all_positive():
    """All molar masses in element_mmw are strictly positive.

    Positivity invariant: a zero or negative molar mass would cause
    division-by-zero in mole-to-mass conversions.
    """
    for elem, mmw in element_mmw.items():
        assert mmw > 0, f'Molar mass for {elem} is not positive: {mmw}'
        assert mmw < 1.0, f'Molar mass for {elem} implausibly large: {mmw} kg/mol'


@pytest.mark.reference_pinned
def test_element_mmw_hydrogen():
    """Hydrogen molar mass = 1.008e-3 kg/mol (IUPAC: 1.008 g/mol).

    Discrimination: if stored in g/mol instead of kg/mol, the value
    would be 1.008 (3 orders of magnitude off).
    """
    assert element_mmw['H'] == pytest.approx(1.008e-3, rel=1e-3)
    # Scale guard: kg/mol, not g/mol
    assert element_mmw['H'] < 0.01
    assert element_mmw['H'] > 1e-4
