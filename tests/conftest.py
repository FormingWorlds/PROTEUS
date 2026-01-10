"""
Pytest configuration and shared fixtures for PROTEUS test suite.

This module provides:
  1. Parameter classes for three representative exoplanet scenarios spanning
     habitability → magma ocean → extreme irradiation gradient
  2. Pytest fixtures exposing these parameters for use in tests
  3. Configuration path fixtures for accessing input files
  4. Pytest marker registration for test categorization

All physical parameters use SI units and are sourced from proteus.utils.constants
for consistency across the PROTEUS ecosystem.

**Test Scenarios**:
  - EarthLikeParams: Modern Earth (habitable reference)
  - UltraHotSuperEarthParams: TOI-561 b (ultra-hot stripped planet)
  - IntermediateSuperEarthParams: L 98-59 d (volatile-rich magma ocean)

**Usage**:
  def test_with_earth(earth_params):
      assert earth_params.planet_mass == earth_params.planet_mass
      assert earth_params.equilibrium_temperature > 273  # Above freezing

See test_infrastructure.md for full documentation of conftest.py role in the
testing framework.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Prevent oversubscription when using VULCAN
os.environ['OMP_NUM_THREADS'] = '2'  # noqa

sys.path.append(os.path.join(os.path.dirname(__file__), 'helpers'))

# Import physical constants from PROTEUS
from proteus.utils.constants import (
    AU,
    L_sun,
    M_earth,
    M_sun,
    R_earth,
    R_sun,
    const_G,
    const_sigma,
    secs_per_year,
)

# =============================================================================
# Physical Parameters for Earth-like Test Scenarios
# =============================================================================


class EarthLikeParams:
    """
    Physical parameters for Earth-like planet around Sun-like star.

    Represents a habitable reference scenario: modern Earth around the Sun
    with solid interior, thin outgassed atmosphere (N₂/O₂), and moderate
    surface temperature suitable for life.

    **Scenario Use**: Unit tests, smoke tests, baseline habitability metrics

    **Physical Basis**: Current Earth (4.543 Gyr old) with modern surface
    conditions (288 K, 1 bar N₂/O₂ atmosphere, 0.31 albedo).

    All values in SI units unless otherwise noted.
    """

    # --- Star Parameters (Sun-like) ---
    star_mass = M_sun  # kg; Solar mass
    star_radius = R_sun  # m; Solar radius
    star_luminosity = L_sun  # W; Solar luminosity
    star_age = 4.567e9  # yr; Current age of the Sun

    # --- Orbital Parameters (Earth-like) ---
    orbital_semimajor = AU  # m; 1 AU (Earth's orbital distance)
    orbital_eccentricity = 0.0167  # dimensionless; Modern Earth eccentricity
    orbital_period = 1.0  # yr; 365.25 days
    insolation_factor = 0.375  # dimensionless; S₀/4 factor for standard calculation

    # --- Planet Parameters (Earth-like) ---
    planet_mass = M_earth  # kg; Modern Earth mass
    planet_radius = R_earth  # m; Modern Earth radius
    planet_surface_temp = 288.0  # K; Current mean surface temperature
    planet_albedo = 0.31  # dimensionless; Cloudy Earth albedo
    planet_age = 4.543e9  # yr; Age of Earth

    # --- Atmosphere Parameters ---
    surface_pressure = 1.0e5  # Pa; Modern sea-level pressure
    surface_pressure_bar = 1.0  # bar; Same as above (PROTEUS convention)

    # --- Interior Parameters ---
    mantle_temp_initial = 3000.0  # K; Early magma ocean temperature
    core_fraction = 0.55  # dimensionless; Core radius as fraction of total

    # --- Physical Constants (from PROTEUS) ---
    gravitational_constant = const_G  # m³ kg⁻¹ s⁻²
    stefan_boltzmann = const_sigma  # W m⁻² K⁻⁴
    seconds_per_year = secs_per_year  # s/yr; Seconds in a year


# =============================================================================
# Fixtures for Common Test Scenarios
# =============================================================================


@pytest.fixture(scope='session')
def earth_params():
    """
    Provide Earth-like physical parameters for test scenarios.

    **Scope**: session (cached across all tests for efficiency)

    **Returns**:
        EarthLikeParams: Instance with SI-based physical constants for a
        habitable, Earth-like planet.

    **Usage**:
        def test_habitability(earth_params):
            assert earth_params.planet_surface_temp > 273  # Above freezing
            assert earth_params.orbital_semimajor == AU  # 1 AU orbit

    **Notes**:
        - All parameters sourced from proteus.utils.constants
        - Suitable for unit, smoke, and integration tests
        - Baseline scenario for comparative tests
    """
    return EarthLikeParams()


@pytest.fixture(scope='session')
def proteus_root() -> Path:
    """
    Return absolute path to PROTEUS root directory.

    **Scope**: session (cached; same value for all tests)

    **Returns**:
        Path: Absolute path to repository root (parent of tests/ directory).

    **Usage**:
        def test_input_files(proteus_root):
            config = proteus_root / 'input' / 'minimal.toml'
            assert config.exists()
    """
    return Path(__file__).parent.parent


@pytest.fixture(scope='session')
def config_earth(proteus_root: Path):
    """
    Load Earth configuration file for integration tests.

    **Depends on**: proteus_root fixture

    **Returns**:
        Path: Path to input/planets/earth.toml (modern Earth scenario)

    **Raises**:
        AssertionError: If config file not found (PROTEUS not installed properly)

    **Notes**:
        - Requires PROTEUS installed in editable mode
        - Fails fast if file missing (safety check)
    """
    config_path = proteus_root / 'input' / 'planets' / 'earth.toml'
    assert config_path.exists(), f'Earth config not found at {config_path}'
    return config_path


@pytest.fixture(scope='session')
def config_minimal(proteus_root: Path):
    """
    Load minimal configuration file for quick validation tests.

    **Depends on**: proteus_root fixture

    **Returns**:
        Path: Path to input/minimal.toml (fast, minimal setup)

    **Use Case**: Smoke tests and quick integration validation

    **Notes**:
        - Faster initialization than full Earth config
        - Minimal timesteps, low resolution
        - Preferred for smoke tests (@pytest.mark.smoke)
    """
    config_path = proteus_root / 'input' / 'minimal.toml'
    assert config_path.exists(), f'Minimal config not found at {config_path}'
    return config_path


@pytest.fixture(scope='session')
def config_dummy(proteus_root: Path):
    """
    Load dummy configuration file for isolated unit tests.

    **Depends on**: proteus_root fixture

    **Returns**:
        Path: Path to input/demos/dummy.toml (fastest initialization)

    **Use Case**: Unit tests validating Python logic only (no heavy physics)

    **Notes**:
        - Fastest to load (instant initialization)
        - Uses only dummy/placeholder modules
        - No heavy Fortran computations
        - Preferred for @pytest.mark.unit tests
    """
    config_path = proteus_root / 'input' / 'demos' / 'dummy.toml'
    assert config_path.exists(), f'Dummy config not found at {config_path}'
    return config_path


# =============================================================================
# Ultra-Hot Exoplanet Parameters (TOI-561 b)
# =============================================================================


class UltraHotSuperEarthParams:
    """
    Physical parameters for TOI-561 b: ultra-hot super-Earth.

    **Physical Scenario**: Ultra-short-period (0.45 day orbit) exoplanet
    experiencing extreme stellar irradiation (~100× Earth). Ultra-low density
    (4.3 g/cm³) suggests thick volatile envelope despite extreme heating.

    **Astrophysical Significance**: Challenges atmospheric loss models;
    represents extreme end of exoplanet parameter space for testing escape
    physics and magma ocean dynamics.

    **Data Source**: Teske et al. (2025) arXiv:2509.17231
    **Additional References**: Lacedelli et al. (2021, 2022), Weiss et al. (2021)

    **Test Use Cases**:
      - Atmospheric escape physics (extreme irradiation)
      - Magma ocean cooling dynamics
      - High-temperature chemistry and photochemistry
      - Volatile loss timescales and mechanisms

    All values in SI units unless otherwise noted.
    """

    # Star parameters (Thick-disk, metal-poor star)
    # ~10 Gyr old, iron-poor ([Fe/H] = -0.40), alpha-rich ([alpha/Fe] = 0.23)
    star_mass = M_sun * 0.98  # ~1 M_sun (solar-like but metal-poor)
    star_radius = 6.576e8  # m (0.843 R_sun from Teske et al.)
    star_luminosity = L_sun * 0.57  # W (estimated from T_eff and radius)
    star_teff = 5372.0  # K (effective temperature)
    star_age = 11.0e9  # yr (11 Gyr, thick-disk population)

    # Orbital parameters (Ultra-short period)
    orbital_period = 0.4465689683  # days (Teske et al.)
    orbital_semimajor = 3.099e9  # m (~0.0207 AU, computed from Kepler's 3rd law)
    orbital_eccentricity = 0.0  # dimensionless (assumed circular)

    # Planet parameters (Ultra-low density super-Earth)
    planet_mass = 2.24 * M_earth  # kg (2.24 ± 2.0 M_earth, large uncertainty)
    planet_radius = 1.4195 * R_earth  # m (1.4195 R_earth from TESS/Cheops)
    planet_bulk_density = 4.3049e3  # kg/m^3 (4.3 ± 0.4 g/cm^3; ultra-low for a super-Earth)

    # Atmosphere parameters (Thick volatile envelope)
    # Observations indicate H2O + O2 or pure H2O compositions
    dayside_brightness_temp = 1800.0  # K (brightest observational estimate, Eureka! reduction)
    dayside_brightness_temp_alt = 2150.0  # K (higher estimate, ExoTiC JEDI 2)
    bare_rock_equilibrium_temp = 2950.0  # K (theoretical maximum without atmosphere)
    irradiation_temperature = 3274.0  # K (Teske et al.; see Table A.1)

    # Physical constants
    gravitational_constant = const_G  # m3 kg−1 s−2
    stefan_boltzmann = const_sigma  # W m−2 K−4
    seconds_per_year = secs_per_year  # s/yr


@pytest.fixture(scope='session')
def ultra_hot_params():
    """
    Provide TOI-561 b parameters for testing extreme irradiation scenarios.

    **Scope**: session (cached; same instance for all tests)

    **Returns**:
        UltraHotSuperEarthParams: Physical constants for an ultra-short-period,
        ultra-low-density exoplanet with volatile atmosphere.

    **Test Categories**:
        @pytest.mark.unit: Fast logic tests (mocked atmospheres)
        @pytest.mark.smoke: Binary initialization tests
        @pytest.mark.integration: Multi-module coupling with escape physics
        @pytest.mark.slow: Full atmospheric evolution over Gyr timescales

    **Use Cases**:
        - Atmospheric escape physics (extreme XUV irradiation)
        - Magma ocean cooling and volatile outgassing
        - High-temperature chemistry and photochemistry
        - Testing limits of volatile retention mechanisms
        - Benchmarking against observed density anomalies

    **Physical Significance**:
        - Represents extreme end of exoplanet parameter space
        - Bulk density anomalously low: suggests thick volatile envelope
        - Challenges standard atmospheric loss models

    **References**: Teske et al. (arXiv:2509.17231); Lacedelli et al. (2021, 2022)
    """
    return UltraHotSuperEarthParams()


class IntermediateSuperEarthParams:
    """
    Physical parameters for L 98-59 d: volatile-rich super-Earth with magma ocean.

    **Physical Scenario**: Intermediate-density super-Earth (M=2.14 M⊕, R=1.52 R⊕)
    with H₂-rich atmosphere (MMW~9 u) and permanent magma ocean. Bridges
    habitability and atmospheric loss regimes: retains volatiles in deep magma
    ocean despite Gyr-scale evolution and photochemistry (unlike outgassed Earth);
    maintains interior structure for tidal heating (unlike stripped ultra-hots).

    **Astrophysical Significance**: M-dwarf system (0.273 M☉, 4.94 Gyr); 3.7-day
    orbit enables tidal heating. Permanent 45% melt fraction sustained by tidal
    heating + greenhouse atmospheric blanketing. Key feature: photochemical SO₂
    production in reducing H₂ atmosphere, with sulfur dissolved in magma ocean.

    **Data Source**: Nicholls et al. (2025) arXiv:2507.02656
    **Additional References**: Cloutier et al. (2019), Demangeon et al. (2021),
    Gressier et al. (2024), Rajpaul et al. (2024)

    **Test Use Cases**:
      - Volatile retention in magma oceans over Gyr timescales
      - Photochemical chemistry in H₂-rich reducing atmospheres
      - Tidal heating as sustained interior energy source
      - Magma-atmosphere volatile partitioning (S, H, C, N)
      - Intermediate case between habitable and stripped exoplanets

    All values in SI units unless otherwise noted.
    """

    # --- Star Parameters (M-dwarf in L 98-59 system) ---
    star_mass = 0.273 * M_sun  # kg; ~0.27 M☉ (low-mass M-dwarf)
    star_rotation_period = 80.9  # days; at present epoch
    star_age = 4.94e9  # yr; System age per Engle & Guinan (2023)

    # --- Planet Parameters (L 98-59 d: volatile-rich super-Earth) ---
    planet_mass = 2.14 * M_earth  # kg; 2.14 M⊕ (Nicholls et al.)
    planet_radius = 1.52 * R_earth  # m; 1.52 R⊕ (transit radius at 20 mbar photosphere)
    planet_bulk_density = 3.45e3  # kg/m³; From Rajpaul et al. (2024)
    planet_age = 4.94e9  # yr; Assumes co-formation with star

    # --- Orbital Parameters (3.7-day orbit; enables tidal heating) ---
    orbital_period = 3.688  # days; From Cloutier et al. (2019), Table 1
    orbital_eccentricity = 0.0  # dimensionless; Assumed circular per paper
    equilibrium_temperature = 416.0  # K; Zero-albedo estimate, Demangeon et al. (2021)

    # --- Atmosphere Composition (H₂-rich with photochemical SO₂) ---
    # JWST retrieval: MMW = 9.18 -2.41/+2.51 u (free-chemistry fit, 5.6σ detection)
    atmosphere_mmw = 9.18  # g/mol; Mean molecular weight (H₂-dominated)
    h2s_volume_mixing_ratio = 10.0 ** (-0.74)  # dimensionless; Gressier et al. (2024)
    so2_formation_mechanism = 'photochemical'  # SO₂ produced via H₂S + OH chemistry

    # --- Interior Structure (Permanent magma ocean with volatile dissolution) ---
    # From Nicholls et al. model grid, representative case study
    mantle_melt_fraction = 0.45  # dimensionless; 45% melt → permanent magma ocean
    volatile_hydrogen_inventory = 7150  # ppmw; Total H budget in planet
    sulfur_hydrogen_ratio = 8.0  # dimensionless; S/H mass ratio in volatile reservoir
    magma_ocean_redox_state = 'IW-3'  # log(fO₂); Relative to iron-wüstite buffer
    surface_temperature = 1830.0  # K; Present-day (Gyr-scale cooling from formation)

    @classmethod
    def bulk_density_check(cls):
        """Verify bulk density from M and R matches paper value."""
        volume = (4 / 3) * 3.14159265 * cls.planet_radius**3  # m³
        calculated_density = cls.planet_mass / volume  # kg/m³
        paper_density = cls.planet_bulk_density
        agreement = abs(calculated_density - paper_density) / paper_density
        assert agreement < 0.01, (
            f'Density mismatch: {calculated_density:.0f} vs {paper_density:.0f}'
        )
        return calculated_density


@pytest.fixture(scope='session')
def intermediate_params():
    """
    Fixture returning L 98-59 d parameters for volatile-rich magma ocean testing.

    **Scope**: session (cached; same instance for all tests)

    **Returns**:
        IntermediateSuperEarthParams: Physical constants for a volatile-rich
        super-Earth with H₂-rich atmosphere and permanent magma ocean.

    **Test Categories**:
        @pytest.mark.unit: Fast logic tests (mocked interior/atmosphere coupling)
        @pytest.mark.smoke: Binary initialization tests
        @pytest.mark.integration: Tidal heating + volatile outgassing coupling
        @pytest.mark.slow: Multi-Gyr volatile retention and magma ocean cooling

    **Use Cases**:
        - Volatile retention in magma oceans over Gyr timescales
        - Photochemical SO₂ production in H₂-rich reducing atmospheres
        - Tidal heating as sustained interior energy source
        - Magma-atmosphere volatile partitioning (S, H, C, N)
        - Intermediate case: bridges habitable and stripped exoplanet regimes

    **Physical Significance**:
        - Permanent 45% melt fraction sustained by tidal heating + greenhouse
        - H₂-rich atmosphere (MMW ~9 u) with dissolved sulfur in magma
        - Retains volatiles despite Gyr-scale evolution

    **References**: Nicholls et al. (arXiv:2507.02656); Cloutier et al. (2019);
    Gressier et al. (2024)
    """
    return IntermediateSuperEarthParams()


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Register custom pytest markers for test categorization."""
    config.addinivalue_line(
        'markers',
        'unit: fast unit tests with mocked physics (target <100ms each)',
    )
    config.addinivalue_line(
        'markers',
        'smoke: quick validation tests with real binaries (target <30s each)',
    )
    config.addinivalue_line(
        'markers',
        'integration: multi-module coupling tests (minutes to hours)',
    )
    config.addinivalue_line(
        'markers',
        'slow: comprehensive physics validation tests (hours)',
    )
    config.addinivalue_line(
        'markers',
        'skip: placeholder tests not yet implemented',
    )
